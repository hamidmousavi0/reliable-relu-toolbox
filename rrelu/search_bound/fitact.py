import sys
import torch.nn as nn
import torch
import copy
import numpy as np
import warnings
import sys; 
from rrelu.relu_bound.bound_fitact import bounded_relu_fitact
import os
from rrelu.relu_bound.bound_relu import Relu_bound
import argparse
from rrelu.utils.metric import accuracy,AverageMeter
from rrelu.utils.lr_scheduler import CosineLRwithWarmup
from rrelu.utils.distributed import DistributedMetric
from rrelu.search_bound.ranger import Ranger_bounds
import random
import time
from tqdm import tqdm
import horovod.torch as hvd


def eval(model: nn.Module, data_loader_dict,is_root) :

    test_criterion = nn.CrossEntropyLoss().cuda()

    val_loss = DistributedMetric()
    val_top1 = DistributedMetric()
    val_top5 = DistributedMetric()

    model.eval()
    with torch.no_grad():
        with tqdm(
            total=len(data_loader_dict["val"]),
            desc="Eval",
            disable= not is_root,
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


activation={}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

def relu_hooks(model: nn.Module, prefix=''):
    for name, layer in model.named_children():
        layer_name = f"{prefix}.{name}" if prefix else name
        if isinstance(layer, nn.ReLU) or isinstance(layer,Relu_bound) or isinstance(layer, nn.ReLU6):
            layer.register_forward_hook(get_activation(layer_name))
        elif len(list(layer.children())) > 0:  # Recursively check nested modules
            relu_hooks(layer, layer_name)

def replace_act(model:nn.Module,bounds,tresh, prefix='')->nn.Module:
    for name,layer in model.named_children():
        layer_name = f"{prefix}.{name}" if prefix else name
        if isinstance(layer, nn.ReLU) or isinstance(layer,Relu_bound) or isinstance(layer, nn.ReLU6):
            model._modules[name] = bounded_relu_fitact(bounds[layer_name].detach(),tresh[layer_name].detach(),-20)
        elif len(list(layer.children())) > 0:
            replace_act(layer,bounds,tresh,layer_name)               
    return model





def fitact_bounds(model:nn.Module,train_loader, device="cuda", bound_type='layer',bitflip='float',is_root=False):
    results,tresh,_ = Ranger_bounds(copy.deepcopy(model),train_loader,device,bound_type)
    model = replace_act(model,results,tresh)
    torch.backends.cudnn.benchmark = True

    for name, param in model.named_parameters():
        if np.any([key in name for key in ["weight", "norm","bias"]]):
            param.requires_grad=False
        else:
            param.requires_grad=True
    for name, param in model.named_parameters():
        if np.any([key in name for key in ["weight", "norm","bias"]]):
            param.requires_grad=False
        else:
            param.requires_grad=True
    model = train(model, train_loader,is_root)
    bounds_dict = {}
    keys=[]
    i=0
   
    for key,val in results.items():
        keys.append(key)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if np.any([key in name for key in ["bounds_param"]]):
                bounds_dict[keys[i]]=param
                i+=1
    return bounds_dict,bounds_dict,None

def train(
    model: nn.Module,
    data_provider,
    is_root,
    base_lr=0.001,
    warmup_epochs = 0 ,
    n_epochs = 100,
    weight_decay = 4e-11

):
    
    params_without_wd = []
    params_with_wd = []
    for name, param in model.named_parameters():
        if param.requires_grad:

            if np.any([key in name for key in ["bias", "norm"]]):
                params_without_wd.append(param)
            else:
                # print(name)
                params_with_wd.append(param)
    net_params = [
        {"params": params_without_wd, "weight_decay": 0},
        {
            "params": params_with_wd,
            "weight_decay": weight_decay,
        },
    ]
    # build optimizer
    optimizer = torch.optim.Adam(
        net_params,
        lr=base_lr * hvd.size(),
    )
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    # build lr scheduler
    lr_scheduler = CosineLRwithWarmup(
        optimizer,
        warmup_epochs * len(data_provider['train']),
        base_lr,
        n_epochs * len(data_provider['train']),
    )
    
    # train criterion
    train_criterion = nn.CrossEntropyLoss().cuda()
    # init
    best_val = 0.0
    start_epoch = 0

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    # start training
    for epoch in range(
        start_epoch,
        n_epochs
        + warmup_epochs,
    ):
        train_info_dict = train_one_epoch(
            model,
            data_provider,
            is_root,
            epoch,
            optimizer,
            train_criterion,
            lr_scheduler,
        )
    
        val_info_dict = eval(model, data_provider,is_root)
        is_best = val_info_dict["val_top1"] > best_val
        best_val = max(best_val, val_info_dict["val_top1"])
        # log
        if is_root:
            epoch_log = f"[{epoch + 1 - warmup_epochs}/{n_epochs}]"
            epoch_log += f"\tval_top1={val_info_dict['val_top1']:.2f} ({best_val:.2f})"
            epoch_log += f"\ttrain_top1={train_info_dict['train_top1']:.2f}\tlr={optimizer.param_groups[0]['lr']:.2E}"
         
    return model




def train_one_epoch(
    model: nn.Module,
    data_provider,
    is_root,
    epoch: int,
    optimizer,
    criterion,
    lr_scheduler,

):
    train_loss = DistributedMetric()
    train_top1 = DistributedMetric()

    model.train()
    data_provider['train'].sampler.set_epoch(epoch)

    data_time = AverageMeter()
    with tqdm(
        total=len(data_provider["train"]),
        desc="Train Epoch #{}".format(epoch + 1),
        disable=not is_root,
    ) as t:
        end = time.time()
        for _, (images, labels) in enumerate(data_provider['train']):
            data_time.update(time.time() - end)
            images, labels = images.cuda(), labels.cuda()
            l2_bounds = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad==True:
                    if "bounds" in name:
                        l2_bounds += torch.mean(torch.pow(param, 2))
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels) + 4e-11 * l2_bounds
            loss.backward()
            top1 = accuracy(output, labels, topk=(1,))[0][0]
            optimizer.step()
            lr_scheduler.step()

            train_loss.update(loss, images.shape[0])
            train_top1.update(top1, images.shape[0])

            t.set_postfix(
                {
                    "loss": train_loss.avg.item(),
                    "top1": train_top1.avg.item(),
                    "batch_size": images.shape[0],
                    "img_size": images.shape[2],
                    "lr": optimizer.param_groups[0]["lr"],
                    "data_time": data_time.avg,
                }
            )
            t.update()

            end = time.time()
    return {
        "train_top1": train_top1.avg.item(),
        "train_loss": train_loss.avg.item(),
    }



