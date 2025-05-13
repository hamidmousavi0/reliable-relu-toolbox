  
import argparse
import os
import torch.backends.cudnn
import torch.nn as nn
from fxpmath import Fxp
# import horovod.torch as hvd
from rrelu.setup import build_data_loader, build_model, replace_act
from metrics import eval_fault, eval
import random
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np 
from rrelu.utils.distributed import set_running_statistics
import  torch.distributed as dist
parser = argparse.ArgumentParser() 
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_worker", type=int, default=32)
parser.add_argument("--iterations", type=int, default=100)
parser.add_argument("--n_word", type=int, default=32)
parser.add_argument("--n_frac", type=int, default=16)
parser.add_argument("--n_int", type=int, default=15)
parser.add_argument("--dataset", type=str, default="imagenet", choices=["mnist", "cifar10", "cifar100", "imagenet"])
parser.add_argument("--data_path", type=str, default="./dataset/imagenet")
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument("--model", type=str, default="AlexNet")
parser.add_argument("--save_path", type=str, default=None)
parser.add_argument("--name_relu_bound", type=str, default="fitact", choices=["zero", "tresh", "fitact", "proact", "none"])
parser.add_argument("--name_serach_bound", type=str, default="fitact", choices=["ranger", "ftclip", "fitact", "proact"])
parser.add_argument("--bounds_type", type=str, default="neuron", choices=["layer", "neuron"])
parser.add_argument("--bitflip", type=str, default="fixed", choices=["fixed", "float"])
parser.add_argument("--fault_rates", type=list, default=[1e-7,3e-7,1e-6,3e-6,1e-5,3e-5])
parser.add_argument("--pretrained_model", action='store_true',  default=False)
def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

if __name__ == "__main__":
    args = parser.parse_args()
    path = 'pretrained_models/{}/{}'.format(args.dataset,args.model)  
    if not os.path.exists(path):
        os.makedirs(path) 
    # Initialize Horovod
    # hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(hvd.local_rank())
    local_rank, rank, world_size = setup_distributed()
    # args.num_gpus = hvd.size()
    # args.rank = hvd.rank()
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    data_loader_dict, n_classes = build_data_loader(
        args.dataset,
        args.image_size,
        args.batch_size,
        args.n_worker,
        args.data_path,
        num_replica=world_size,
        rank= rank
    )
    model = build_model(args.model, args.dataset, n_classes, 0.0, pretrained=True).cuda()
    model = DDP(model, device_ids=[local_rank])
    print(f"original Model accuracy in {args.bitflip} is : {eval(model, data_loader_dict)}") 
    if args.bitflip == 'fixed':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param is not None:
                    param.copy_(torch.tensor(Fxp(param.clone().cpu().numpy(), True, n_word=args.n_word, n_frac=args.n_frac, n_int=args.n_int).get_val(),dtype=torch.float32,device='cuda').cuda())        
    if args.name_relu_bound!='none':
        model = replace_act(model, args.name_relu_bound, args.name_serach_bound, data_loader_dict, args.bounds_type, args.bitflip,args.pretrained_model,args.dataset,is_root=(dist.get_rank() == 0))
        if args.pretrained_model:
            model.load_state_dict(torch.load('pretrained_models/{}/{}/{}_{}_{}_{}.pth'.format(args.dataset,args.model,args.name_relu_bound,args.name_serach_bound,args.bounds_type,args.bitflip),map_location='cuda:0'))
        else:
            torch.save(model.state_dict(), 'pretrained_models/{}/{}/{}_{}_{}_{}.pth'.format(args.dataset,args.model,args.name_relu_bound,args.name_serach_bound,args.bounds_type,args.bitflip))    

    print(f"{args.dataset} {args.model} {args.name_relu_bound} {args.name_serach_bound} {args.bounds_type} {args.bitflip} {args.iterations}{args.pretrained_model}")
    
    if args.pretrained_model or args.name_relu_bound=='none':
        # if args.name_relu_bound == "proact":
        #     set_running_statistics(model,data_loader_dict["sub_train"],distributed=False)    
        print(f"Model accuracy in {args.bitflip} format after replacing ReLU activation functions: {eval(model, data_loader_dict)}")
        for fault_rate in args.fault_rates:
            val_results_fault = eval_fault(model, data_loader_dict, fault_rate, args.iterations, args.bitflip, args.n_word, args.n_frac, args.n_int)
            print(f"top1 = {val_results_fault['val_top1']}, top5 = {val_results_fault['val_top1']}, Val_loss = {val_results_fault['val_loss']}, fault_rate = {val_results_fault['fault_rate']}")

    













