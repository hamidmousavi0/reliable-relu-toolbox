import copy
import math
import os.path
from typing import Dict, Optional, Tuple, Type
from rrelu.utils.metric import accuracy
from torchpack import distributed as dist
from tqdm import tqdm
from rrelu.pytorchfi.weight_error_models import multi_weight_inj_fixed,multi_weight_inj_float,multi_weight_inj_int
from rrelu.relu_bound.bound_relu import Relu_bound
from rrelu.pytorchfi.core import FaultInjection
from rrelu.utils.distributed import DistributedMetric
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from rrelu.models.vgg import make_layers,cfg
from rrelu.models import (
   Lenet,
   VGG,
   AlexNet_model,
   ResNet50,
   VGG16,
   LeNet_cifar,
   AlexNet_cifar100,
   VGG16_cifar100,
   ResNet50_cifar100,
)
from rrelu.relu_bound import (
    bounded_relu_zero,
    bounded_relu_tresh,
    bounded_relu_fitact,
    bounded_hyrelu_proact

)

from rrelu.search_bound import (
    FtClipAct_bounds,
    Ranger_bounds,
    fitact_bounds,
    proact_bounds

)
__all__ = ["build_data_loader", "build_model", "replace_act","find_bounds","eval_fault","eval"]

def eval_fault(model:nn.Module,data_loader_dict, fault_rate,iterations=500,bitflip=None,total_bits = 32 , n_frac = 16 , n_int = 15 )-> Dict:
    inputs, classes = next(iter(data_loader_dict['val'])) 
    pfi_model = FaultInjection(model, 
                            inputs.shape[0],
                            input_shape=[inputs.shape[1],inputs.shape[2],inputs.shape[3]],
                            layer_types=[torch.nn.Conv2d, torch.nn.Linear ,Relu_bound],
                            total_bits= total_bits,
                            n_frac = n_frac, 
                            n_int = n_int, 
                            use_cuda=True,
                            )
    print(pfi_model.print_pytorchfi_layer_summary())
    test_criterion = nn.CrossEntropyLoss().cuda()

    val_loss = DistributedMetric()
    val_top1 = DistributedMetric()
    val_top5 = DistributedMetric()

    pfi_model.original_model.eval()
    with torch.no_grad():
        with tqdm(
            total= iterations,
            desc="Eval",
            disable=not dist.is_master(),
        ) as t:
            for i in range(iterations):
                if bitflip=='float':
                    corrupted_model = multi_weight_inj_float(pfi_model,fault_rate)
                elif bitflip=='fixed':    
                    corrupted_model = multi_weight_inj_fixed(pfi_model,fault_rate)
                elif bitflip =="int":
                    corrupted_model = multi_weight_inj_int (pfi_model,fault_rate)
                    # corrupted_model = multi_weight_inj_int(pfi_model,fault_rate)          
                for images, labels in data_loader_dict["val"]:
                    images, labels = images.cuda(), labels.cuda()
                    output = corrupted_model(images)
                    loss = test_criterion(output, labels)
                    val_loss.update(loss, images.shape[0])
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    val_top5.update(acc5[0], images.shape[0])
                    val_top1.update(acc1[0], images.shape[0])
                    
                ####        
                t.set_postfix(
                    {
                        "loss": val_loss.avg.item(),
                        "top1": val_top1.avg.item(),
                        "top5": val_top5.avg.item(),
                        "#samples": val_top1.count.item(),
                        "batch_size": images.shape[0],
                        "img_size": images.shape[2],
                        "fault_rate": fault_rate,
                    }
                )
                t.update()
                # pfi_model.original_model = corrupted_model    
        val_results = {
            "val_top1": val_top1.avg.item(),
            "val_top5": val_top5.avg.item(),
            "val_loss": val_loss.avg.item(),
            "fault_rate": fault_rate,
        }
    return val_results

def eval(model: nn.Module, data_loader_dict) -> Dict:

    test_criterion = nn.CrossEntropyLoss().cuda()

    val_loss = DistributedMetric()
    val_top1 = DistributedMetric()
    val_top5 = DistributedMetric()

    model.eval()
    with torch.no_grad():
        with tqdm(
            total=len(data_loader_dict["val"]),
            desc="Eval",
            disable=not dist.is_master(),
        ) as t:
            for images, labels in data_loader_dict["val"]:
                images, labels = images.cuda(), labels.cuda()
                # compute output
                output = model(images)
                loss = test_criterion(output, labels)
                val_loss.update(loss, images.shape[0])
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                val_top5.update(acc5[0], images.shape[0])
                val_top1.update(acc1[0], images.shape[0])

                t.set_postfix(
                    {
                        "loss": val_loss.avg.item(),
                        "top1": val_top1.avg.item(),
                        "top5": val_top5.avg.item(),
                        "#samples": val_top1.count.item(),
                        "batch_size": images.shape[0],
                        "img_size": images.shape[2],
                    }
                )
                t.update()

    val_results = {
        "val_top1": val_top1.avg.item(),
        "val_top5": val_top5.avg.item(),
        "val_loss": val_loss.avg.item(),
    }
    return val_results


def build_data_loader(
    dataset: str,
    image_size: int,
    batch_size: int,
    n_worker: int = 8,
    data_path: Optional[str] = None,
    num_replica: Optional[int] = None,
    rank: Optional[int] = None,
) -> Tuple[Dict, int]:
    # build dataset
    dataset_info_dict = {
        "imagenet21k_winter_p": (
            os.path.expanduser("./dataset/imagenet21k_winter_p"),
            10450,
        ),
        "imagenet": (os.path.expanduser("./dataset/imagenet"), 1000),
        "car": (os.path.expanduser("./dataset/fgvc/stanford_car"), 196),
        "flowers102": (os.path.expanduser("./dataset/fgvc/flowers102"), 102),
        "food101": (os.path.expanduser("./dataset/fgvc/food101"), 101),
        "cub200": (os.path.expanduser("./dataset/fgvc/cub200"), 200),
        "pets": (os.path.expanduser("./dataset/fgvc/pets"), 37),
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
                
                
            ]
            )
        )
        val_dataset = datasets.CIFAR10(
            os.path.join(data_path, "CIFAR10"),
            train=False,
            download=True,
            transform=transforms.Compose([
                 transforms.ToTensor(),
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
                
            ]
            )
        )
        val_dataset = datasets.CIFAR100(
            os.path.join(data_path, "CIFAR100"),
            train=False,
            download=True,
            transform=transforms.Compose([
                 transforms.ToTensor(),
            ]     
            )
        )
    else:    
        train_dataset = datasets.ImageFolder(
            os.path.join(data_path, "train"),
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
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
    if len(sub_train_dataset) > 16000:
        g = torch.Generator()
        g.manual_seed(937162211)
        rand_indexes = torch.randperm(len(sub_train_dataset), generator=g).tolist()
        rand_indexes = rand_indexes[:3000] # for alexnet and  vgg use 3000 for ftclip 1000
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
            pin_memory=False,
            drop_last=True,
        )
        sub_train_loader = torch.utils.data.DataLoader(
            dataset=sub_train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.distributed.DistributedSampler(
                sub_train_dataset, num_replica, rank
            ),
            num_workers=n_worker,
            pin_memory=False,
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
            pin_memory=False,
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
    n_classes=10,
    dropout_rate=0.0,
    **kwargs,
) -> nn.Module:

    model_dict = {
        "lenet": Lenet,
        "lenet_cifar10": LeNet_cifar,
        "vgg16": VGG16,
        "resnet50": ResNet50,
        "alexnet": AlexNet_model,
        "alexnet_cifar100" : AlexNet_cifar100, 
        "vgg16_cifar100":VGG16_cifar100,
        "resnet50_cifar100": ResNet50_cifar100,
    }

    name = name.split("-")
    if len(name) > 1:
        kwargs["width_mult"] = float(name[1])
    name = name[0]

    return model_dict[name](n_classes=n_classes, dropout_rate=dropout_rate, **kwargs)


def find_bounds(model:nn.Module, data_loader, name:str,bound_type:str,bitflip:str):
    search_bounds_dict={
        "ranger" : Ranger_bounds,
        "ftclip" : FtClipAct_bounds,
        'fitact' : fitact_bounds,
        'proact'  : proact_bounds
    }
    return search_bounds_dict[name](model,data_loader,bound_type=bound_type,bitflip=bitflip)

def replace_act_all(model:nn.Module,relu_bound,bounds,tresh,alpha=None, name='')->nn.Module:
    for name1,layer in model.named_children():
        if list(layer.children()) == []:
            if isinstance(layer,nn.ReLU) and 'last' not in name1:
                name_ = name1 + name
                if alpha == None:
                    # print(name_)
                    model._modules[name1] = relu_bound(bounds[name_].detach(),tresh[name_].detach(),alpha)
                else:
                    model._modules[name1] = relu_bound(bounds[name_].detach(),tresh[name_].detach(),alpha[name_].detach())    
            elif isinstance(layer,nn.ReLU) and 'last' in name1:
                name_ = name1 + name
                if alpha == None:
                    # print(name_)
                    model._modules[name1] = relu_bound(bounds[name_].detach(),tresh[name_].detach(),alpha,k=-20.0)
                else:
                    model._modules[name1] = relu_bound(bounds[name_].detach(),tresh[name_].detach(),alpha[name_].detach(),k=-20.0) 
                    
        else:
            name+=name1
            replace_act_all(layer,relu_bound,bounds,tresh,alpha,name)               
    return model  
def replace_act(model:nn.Module, name_relu_bound:str, name_serach_bound:str,data_loader,bound_type:str,bitflip:str)->nn.Module:
    replace_act_dict={
        'zero' : bounded_relu_zero,
        'tresh': bounded_relu_tresh,
        'fitact': bounded_relu_fitact,
        'proact' : bounded_hyrelu_proact
    }
    bounds,tresh,alpha = find_bounds(copy.deepcopy(model),data_loader,name_serach_bound,bound_type,bitflip) 
    print(bounds)
    model = replace_act_all(model,replace_act_dict[name_relu_bound],bounds,tresh,alpha,name='')
    return model                
                         
    



