import copy
import torchvision.models as models
import math
import os.path
from PIL import Image
from typing import Dict, Optional, Tuple, Type
from rrelu.utils.metric import accuracy
from tqdm import tqdm
from rrelu.pytorchfi.weight_error_models import multi_weight_inj_fixed,multi_weight_inj_float,multi_weight_inj_int
from rrelu.relu_bound.bound_relu import Relu_bound
from rrelu.pytorchfi.core import FaultInjection
from rrelu.utils.distributed import DistributedMetric,DistributedTensor
from rrelu.utils.metric import AverageMeter
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from rrelu.relu_bound import (
    bounded_relu_zero,
    bounded_relu_tresh,
    bounded_relu_fitact,
    bounded_hyrelu_proact

)
from rrelu.models_cifar import *
from rrelu.models_imagenet import *
from rrelu.search_bound import (
    FtClipAct_bounds,
    Ranger_bounds,
    fitact_bounds,
    proact_bounds

)
__all__ = ["build_data_loader", "build_model", "replace_act","find_bounds"]








def build_data_loader(dataset: str,image_size: int,batch_size: int,
                      n_worker: int = 8,data_path: Optional[str] = None,
                      num_replica: Optional[int] = None,rank: Optional[int] = None,) -> Tuple[Dict, int]:
    dataset_info_dict = {
        "imagenet": (os.path.expanduser("./dataset/imagenet"), 1000),
        "cifar10": (os.path.expanduser("./dataset/cifar10"), 10),
        "cifar100": (os.path.expanduser("./dataset/cifar100"), 100),
        "mnist":(os.path.expanduser("./dataset/mnist"), 10)
    }
    assert dataset in dataset_info_dict, f"Do not support {dataset}"

    data_path = data_path or dataset_info_dict[dataset][0]
    n_classes = dataset_info_dict[dataset][1]

    # build datasets
    if dataset=="mnist":
        train_dataset  =datasets.MNIST(
            os.path.join(data_path, "MNIST"),
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                
                
            ]
            )
        )
        val_dataset = datasets.MNIST(
            os.path.join(data_path, "MNIST"),
            train=False,
            download=True,
            transform=transforms.Compose([
                 transforms.ToTensor(),
            ]     
            )
        )
    elif dataset=="cifar10":
        train_dataset  =datasets.CIFAR10(
            os.path.join(data_path, "CIFAR10"),
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                
                
            ]
            )
        )
        val_dataset = datasets.CIFAR10(
            os.path.join(data_path, "CIFAR10"),
            train=False,
            download=True,
            transform=transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ]     
            )
        )
    elif dataset=="cifar100":
        train_dataset  =datasets.CIFAR100(
            os.path.join(data_path, "CIFAR100"),
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32,padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
                
            ]
            )
        )
        val_dataset = datasets.CIFAR100(
            os.path.join(data_path, "CIFAR100"),
            train=False,
            download=True,
            transform=transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            ]     
            )
        )
    else:    
        train_dataset = datasets.ImageFolder(
            os.path.join(data_path, "train"),
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(image_size),

                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )

        val_dataset = datasets.ImageFolder(
            os.path.join(data_path, "val"),
            transforms.Compose(
                [
                    transforms.Resize(int(math.ceil(image_size / 0.875))),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )

    sub_train_dataset = copy.deepcopy(train_dataset)  # used for resetting bn statistics
    sub_train_dataset.transform = val_dataset.transform
    if len(sub_train_dataset) * batch_size > 16000:
        g = torch.Generator()
        g.manual_seed(937162211)
        rand_indexes = torch.randperm(len(sub_train_dataset), generator=g).tolist()
        rand_indexes = rand_indexes[:1024] # for alexnet and  vgg use 3000 for ftclip 1000
        if  dataset=="cifar10":
            sub_train_dataset.data = [
                sub_train_dataset.data[idx] for idx in rand_indexes
            ]
            sub_train_dataset.targets = [
                sub_train_dataset.targets[idx] for idx in rand_indexes
            ]
        elif  dataset=="cifar100":
            sub_train_dataset.data = [
                sub_train_dataset.data[idx] for idx in rand_indexes
            ]    
            sub_train_dataset.targets = [
                sub_train_dataset.targets[idx] for idx in rand_indexes
            ]
        elif  dataset=="mnist":
            sub_train_dataset.data = [
                sub_train_dataset.data[idx] for idx in rand_indexes
            ]    
            sub_train_dataset.targets = [
                sub_train_dataset.targets[idx] for idx in rand_indexes
            ]    
        else:   
            sub_train_dataset.samples = [
                sub_train_dataset.samples[idx] for idx in rand_indexes
            ]
            

        # build data loader
    if num_replica is None:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_worker,
            pin_memory=True,
            drop_last=True,
        )
        sub_train_loader = torch.utils.data.DataLoader(
            dataset=sub_train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_worker,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_worker,
            pin_memory=True,
            drop_last=False,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replica, rank
            ),
            num_workers=n_worker,
            pin_memory=True,
            drop_last=True,
        )
        sub_train_loader = torch.utils.data.DataLoader(
            dataset=sub_train_dataset,
            batch_size=batch_size,
            num_workers=n_worker,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replica, rank
            ),
            num_workers=n_worker,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

    # prefetch sub_train
    sub_train_loader = [data for data in sub_train_loader]

    data_loader_dict = {
        "train": train_loader,
        "val": valid_loader,
        "sub_train": sub_train_loader,
    }

    return data_loader_dict, n_classes


def build_model(
    name: str,
    dataset:str,
    n_classes=10,
    dropout_rate=0.0,
    pretrained = True,
    **kwargs,
) -> nn.Module:

    if dataset == "imagenet":
        if pretrained:
            model_dict = {
                "ResNet50":resnet50(ResNet50_Weights.DEFAULT),
                "VGG16" : models.vgg16_bn(models.VGG16_BN_Weights.DEFAULT),
                "AlexNet": models.alexnet(models.AlexNet_Weights.DEFAULT),
                "MobileNet": models.mobilenet_v2(models.MobileNet_V2_Weights.DEFAULT)
            }
        else:
            model_dict = {
                "ResNet50": resnet50(ResNet50_Weights.DEFAULT),
                "VGG16" : models.vgg16_bn(),
                "AlexNet": models.alexnet(),
                "MobileNet": models.mobilenet_v2()
            }
        print(name)    
        model = model_dict[name]
    else:  
        '''
        support models : [resnet20,resnet32,resnet44,resnet56,vgg11_bn,vgg13_bn,vgg16_bn,
                          vgg19_bn, mobilenetv2_x0_5,mobilenetv2_x0_75,shufflenetv2_x1_5]
                          all models are in https://github.com/chenyaofo/pytorch-cifar-models/tree/master
        '''
        support_models = ["resnet20","resnet32","resnet44","resnet56","vgg11_bn,vgg13_bn","vgg16_bn",
                          'vgg19_bn', "mobilenetv2_x0_5","mobilenetv2_x0_75","shufflenetv2_x1_5"]
        if name in support_models:
            if 'resnet' in name.lower():
                model = torch.hub.load("rrelu/", dataset + '_' + name.lower(), pretrained=pretrained,source='local')
            else:      
                model = torch.hub.load("chenyaofo/pytorch-cifar-models", dataset + '_' + name.lower(), pretrained=pretrained)    
        else:
            if 'alexnet' in name.lower():
                model = AlexNet_model(n_classes=n_classes, dropout_rate=dropout_rate)
            if 'resnet50' in name.lower():
                model = ResNet50(n_classes,dropout_rate)     
    return model

# Hook function to capture activations
activation={}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach() # Ensure detachment to save memory
    return hook

# Register hooks for all ReLU layers in the model
def relu_hooks(model: nn.Module, prefix=''):
    for name, layer in model.named_children():
        layer_name = f"{prefix}.{name}" if prefix else name
        if isinstance(layer, nn.ReLU) or isinstance(layer, Relu_bound) or isinstance(layer, nn.ReLU6) :
            layer.register_forward_hook(get_activation(layer_name))
        elif len(list(layer.children())) > 0:  # Recursively check nested modules
            relu_hooks(layer, layer_name)

def find_bounds(model:nn.Module, data_loader, name:str,bound_type:str,bitflip:str,is_root:bool,device:str):
    search_bounds_dict={
        "ranger" : Ranger_bounds,
        "ftclip" : FtClipAct_bounds,
        'fitact' : fitact_bounds,
        'proact'  : proact_bounds
    }
    return search_bounds_dict[name](model,data_loader,bound_type=bound_type,bitflip=bitflip,is_root = is_root,device=device)

def replace_act_all(model:nn.Module,relu_bound,bounds,tresh,alpha=None, prefix='',device='cuda')->nn.Module:
    for name,layer in model.named_children():
        layer_name = f"{prefix}.{name}" if prefix else name
        if isinstance(layer, nn.ReLU) or isinstance(layer, nn.ReLU6):
            if alpha == None:
                model._modules[name] = relu_bound(bounds[layer_name].detach(),tresh[layer_name].detach(),alpha,device=device)
            else:
                model._modules[name] = relu_bound(bounds[layer_name].detach(),tresh[layer_name].detach(),alpha[layer_name].detach(),device=device)
                    
        elif len(list(layer.children())) > 0:
            replace_act_all(layer,relu_bound,bounds,tresh,alpha,layer_name,device=device)               
    return model 


def replace_act(model:nn.Module, name_relu_bound:str, name_serach_bound:str,data_loader,bound_type:str,bitflip:str,pretrained:bool,dataset:str,is_root:False,device='cpu')->nn.Module:
    replace_act_dict={
        'zero' : bounded_relu_zero,
        'tresh': bounded_relu_tresh,
        'fitact': bounded_relu_fitact,
        'proact' : bounded_hyrelu_proact
    }
    if pretrained:
        model.eval()

        bounds = {}
        tresh = {}
        alpha=None
        relu_hooks(model)
        if dataset=="imagenet":
            dummy_input = torch.randn((1,3,224,224)).to(device)
        else:
            dummy_input = torch.randn((1,3,32,32)).to(device)  
        # Use torch.no_grad() for inference efficiency
        with torch.no_grad():
            _ = model(dummy_input)
            # Initialize results and tresh with the first batch activations
            for key, val in activation.items():
                bounds[key] = val.clone()
                bounds[key] = bounds[key].squeeze()
                tresh[key] = val.clone()
                tresh[key] = tresh[key].squeeze()
            # Compute max and min values for layer-level bounds if required
            if bound_type == "layer":
                if name_relu_bound!='proact':
                    for key in bounds.keys():
                        bounds[key] = torch.max(bounds[key])
                        # Scalar for easier handling
                        tresh[key] = torch.min(tresh[key])
                else:
                    for i,(key, val) in enumerate(bounds.items()):
                        if i<len(bounds) - 1:
                            bounds[key] = torch.max(val)  
                            tresh[key] = torch.min(tresh[key])         
    else:
        bounds,tresh,alpha = find_bounds(copy.deepcopy(model),data_loader,name_serach_bound,bound_type,bitflip,is_root,device) 
    model = replace_act_all(model,replace_act_dict[name_relu_bound],bounds,tresh,alpha,prefix='',device=device)
    return model                
                         
    



