import argparse
import os
import torch.backends.cudnn
import torch.nn as nn
from fxpmath import Fxp
from rrelu.setup import build_data_loader, build_model, replace_act
from metrics import eval_fault, eval
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from rrelu.pytorchfi.weight_error_models import multi_weight_inj_fixed,multi_weight_inj_float,multi_weight_inj_int
from rrelu.relu_bound.bound_relu import Relu_bound
from rrelu.pytorchfi.core import FaultInjection
import numpy as np 
from rrelu.utils.distributed import set_running_statistics
from rrelu.relu_bound.bound_relu import Relu_bound
parser = argparse.ArgumentParser() 
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--n_worker", type=int, default=8)
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
parser.add_argument("--name_relu_bound", type=str, default="zero", choices=["zero", "tresh", "fitact", "proact", "none"])
parser.add_argument("--name_serach_bound", type=str, default="ranger", choices=["ranger", "ftclip", "fitact", "proact"])
parser.add_argument("--bounds_type", type=str, default="layer", choices=["layer", "neuron"])
parser.add_argument("--bitflip", type=str, default="fixed", choices=["fixed", "float"])
parser.add_argument("--fault_rates", type=list, default=[1e-7,3e-7,1e-6,3e-6,1e-5,3e-5])
parser.add_argument("--pretrained_model", action='store_true',  default=False)

activation = {}

# Hook function to capture activations
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()  # Ensure detachment to save memory
    return hook

# Register hooks for all ReLU layers in the model
def relu_hooks(model: nn.Module, prefix=''):
    for name, layer in model.named_children():
        layer_name = f"{prefix}.{name}" if prefix else name
        if isinstance(layer, nn.ReLU) or isinstance(layer, Relu_bound)or isinstance(layer, nn.ReLU6):
            layer.register_forward_hook(get_activation(layer_name))
        elif len(list(layer.children())) > 0:  # Recursively check nested modules
            relu_hooks(layer, layer_name)

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()
if __name__ == "__main__":
    args = parser.parse_args()
    path = 'pretrained_models/{}/{}'.format(args.dataset,args.model)  
    local_rank, rank, world_size = setup_distributed()
    # Initialize Horovod
    # hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(hvd.local_rank())
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
    model = build_model(args.model, args.dataset, n_classes, 0.0, pretrained=True)
    model = DDP(model, device_ids=[local_rank])
    print(f"original Model accuracy in {args.bitflip} is : {eval(model, data_loader_dict)}") 
    if args.bitflip == 'fixed':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param is not None:
                    param.copy_(torch.tensor(Fxp(param.clone().cpu().numpy(), True, n_word=args.n_word, n_frac=args.n_frac, n_int=args.n_int).get_val(),dtype=torch.float32,device='cuda').cuda())     
    model = replace_act(model, args.name_relu_bound, args.name_serach_bound, data_loader_dict, args.bounds_type, args.bitflip,True,args.dataset,is_root=(dist.get_rank() == 0))                
    model.load_state_dict(torch.load('pretrained_models/{}/{}/{}_{}_{}_{}.pth'.format(args.dataset,args.model,args.name_relu_bound,args.name_serach_bound,args.bounds_type,args.bitflip),map_location='cuda:0'))
    model.eval()
    relu_hooks(model)
    with torch.no_grad():
        first_batch = True
        for data, label in data_loader_dict['sub_train']:
            data = data.to('cuda:0', non_blocking=True)

            # Forward pass
            _ = model(data)
    for key, val in activation.items():
        torch.save(val,f"relus/Orig_{args.name_serach_bound}_{key}_{args.model}_{args.dataset}.pt") 
    inputs, classes = next(iter(data_loader_dict['val'])) 
    pfi_model = FaultInjection(model, 
                            inputs.shape[0],
                            input_shape=[inputs.shape[1],inputs.shape[2],inputs.shape[3]],
                            layer_types=[torch.nn.Conv2d, torch.nn.Linear ,Relu_bound],
                            total_bits= args.n_word,
                            n_frac = args.n_frac, 
                            n_int = args.n_int, 
                            use_cuda=True,
                            )
    print(pfi_model.print_pytorchfi_layer_summary())    
    corrupted_model = multi_weight_inj_fixed(pfi_model,3e-7)
    corrupted_model.eval()
    relu_hooks(corrupted_model)
    with torch.no_grad():
        first_batch = True
        for data, label in data_loader_dict['sub_train']:
            data = data.to('cuda:0', non_blocking=True)

            # Forward pass
            _ = corrupted_model(data)
    for key, val in activation.items():
        torch.save(val,f"relus/Fault_{args.name_serach_bound}_{key}_{args.model}_{args.dataset}.pt") 
