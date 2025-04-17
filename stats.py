import argparse
import os
import torch.backends.cudnn
import torch.nn as nn
from fxpmath import Fxp
import horovod.torch as hvd
from rrelu.setup import build_data_loader, build_model, replace_act
from metrics import eval_fault, eval
import random
from rrelu.pytorchfi.weight_error_models import multi_weight_inj_fixed,multi_weight_inj_float,multi_weight_inj_int
from rrelu.relu_bound.bound_relu import Relu_bound
from rrelu.pytorchfi.core import FaultInjection
import numpy as np 
from rrelu.utils.distributed import set_running_statistics
from rrelu.relu_bound.bound_relu import Relu_bound
parser = argparse.ArgumentParser() 
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_worker", type=int, default=8)
parser.add_argument("--n_word", type=int, default=32)
parser.add_argument("--n_frac", type=int, default=16)
parser.add_argument("--n_int", type=int, default=15)
parser.add_argument("--dataset", type=str, default="imagenet", choices=["cifar10", "cifar10", "cifar100", "imagenet"])
parser.add_argument("--data_path", type=str, default="./dataset/imagenet")
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--manual_seed", type=int, default=0)
parser.add_argument("--model", type=str, default="ResNet50")
parser.add_argument("--save_path", type=str, default=None)
parser.add_argument("--name_relu_bound", type=str, default="zero", choices=["zero", "tresh", "fitact", "proact", "none"])
parser.add_argument("--name_serach_bound", type=str, default="ranger", choices=["ranger", "ftclip", "fitact", "proact"])
parser.add_argument("--bounds_type", type=str, default="neuron", choices=["layer", "neuron"])
parser.add_argument("--bitflip", type=str, default="fixed", choices=["fixed", "float"])
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
def count_weights(net):
    total_weights = sum(p.numel() for n,p in net.named_parameters() if p.requires_grad and "bound" not in n )
    return total_weights
def count_tresh(net):
    total_tresh = sum(p.numel() for n,p in net.named_parameters() if p.requires_grad and "bound" in n )
    return total_tresh
def count_neuron_layer (layer):
    total_neuron = layer.numel()
    return total_neuron
if __name__ == "__main__":
    args = parser.parse_args()
    # Initialize Horovod
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
        num_replica=None,
        rank= None
    )
    model = build_model(args.model, args.dataset, n_classes, 0.0, pretrained=True).cuda()  
    if args.bitflip == 'fixed':
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param is not None:
                    param.copy_(torch.tensor(Fxp(param.clone().cpu().numpy(), True, n_word=args.n_word, n_frac=args.n_frac, n_int=args.n_int).get_val(),dtype=torch.float32,device='cuda').cuda())     
    model = replace_act(model, args.name_relu_bound, args.name_serach_bound, data_loader_dict, args.bounds_type, args.bitflip,True,args.dataset,is_root=True)                
    model.eval()
    relu_hooks(model)
    with torch.no_grad():
        first_batch = True
        for data, label in data_loader_dict['sub_train']:
            data = data.to('cuda:0', non_blocking=True)

            # Forward pass
            _ = model(data)
    print(count_weights(model))
    print(count_tresh(model))

    for key,val in activation.items():
        print(key, ":", count_neuron_layer(val))
    print(len(activation.keys()))
