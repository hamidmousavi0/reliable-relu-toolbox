import sys
import torch.nn as nn
import torch
import copy
import numpy as np
import warnings
import sys;
from rrelu.relu_bound.bound_proact import bounded_hyrelu_proact
from rrelu.relu_bound.bound_zero import bounded_relu_zero
import os
import argparse
from typing import Dict, Optional
from rrelu.relu_bound.bound_relu import Relu_bound
from rrelu.pytorchfi.weight_error_models import multi_weight_inj_float,multi_weight_inj_fixed,multi_weight_inj_int
from rrelu.utils.metric import accuracy,AverageMeter
from rrelu.utils.lr_scheduler import CosineLRwithWarmup
from rrelu.utils.distributed import DistributedMetric
import random
from rrelu.pytorchfi.core import FaultInjection
import torch.nn.functional as F
import time
from tqdm import tqdm
import torch.distributed as dist
# import horovod.torch as hvd
from rrelu.relu_bound.bound_relu import Relu_bound
from rrelu.search_bound.ranger import Ranger_bounds

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
def eval_(model: nn.Module, data_loader_dict,is_root,device) :

    test_criterion = nn.CrossEntropyLoss()

    val_loss = AverageMeter()
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()

    model.eval()
    model.to(device)
    with torch.no_grad():
        with tqdm(
            total=len(data_loader_dict["val"]),
            desc="Eval",
        ) as t:
            for images, labels in data_loader_dict["val"]:
                images, labels = images.to(device), labels.to(device)
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
                        "#samples": val_top1.count,
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
        if isinstance(layer, nn.ReLU) or isinstance(layer,Relu_bound)or isinstance(layer, nn.ReLU6):
            layer.register_forward_hook(get_activation(layer_name))
        elif len(list(layer.children())) > 0:  # Recursively check nested modules
            relu_hooks(layer, layer_name)

def replace_act_all(model:nn.Module,bounds,tresh, prefix='',device='cuda')->nn.Module:
    for name,layer in model.named_children():
        layer_name = f"{prefix}.{name}" if prefix else name
        if isinstance(layer, nn.ReLU) or isinstance(layer,Relu_bound)or isinstance(layer, nn.ReLU6):
            model._modules[name] = bounded_hyrelu_proact(bounds[layer_name].detach(),tresh[layer_name].detach(),device=device)
        elif len(list(layer.children())) > 0:
            replace_act_all(layer,bounds,tresh,layer_name,device=device)               
    return model  
  
def load_state_dict_from_file(file: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(file, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint
def proact_bounds(model:nn.Module, train_loader, bound_type='layer', bitflip='float',is_root=False,device='cpu'):
    original_model  = copy.deepcopy(model)
    results,tresh,_ = Ranger_bounds(copy.deepcopy(model),train_loader,device,'neuron',bitflip)
    len_relu = len(results)
    if bound_type =="layer":
        for i,(key, val) in enumerate(results.items()):
                if i<len_relu - 1:
                    results[key] = torch.max(val)  
                    tresh[key] = torch.min(tresh[key]) 
    model = replace_act_all(model,results,tresh,device=device)
    torch.save(model.state_dict(), "temp_{}_{}_{}.pth".format(bound_type,bitflip,original_model.__class__.__name__))      
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True
    for name, param in model.named_parameters():
        if np.any([key in name for key in ["weight", "norm","bias"]]):
            param.requires_grad=False
        else:
            param.requires_grad=True      
     
    weight_decay_list =[4e-5,4e-6,4e-7,4e-8,4e-9,4e-10,4e-11,4e-12,4e-13,4e-14,4e-15]
    model = train(model=model,original_model=original_model,data_provider=train_loader,weight_decay_list=weight_decay_list,bound_type=bound_type,bitflip=bitflip,is_root=is_root,device=device)
    model.load_state_dict(torch.load("temp_{}_{}_{}.pth".format(bound_type,bitflip,original_model.__class__.__name__)))  
    for name, param in model.named_parameters():
        if np.any([key in name for key in ["weight", "norm","bias"]]):
            param.requires_grad=False
        else:
            param.requires_grad=True   
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
    return bounds_dict,tresh,None



def distillation_loss(feat_s, feat_t,inputs,device):
    
    cosine_loss =  nn.CosineSimilarity()
    
    return torch.mean(cosine_loss(feat_s.view(feat_s.size(0),-1), feat_t.view(feat_t.size(0),-1))) #oss.sum()
    
def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss
def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))
def L1_reg(model):
    nweights = 0
    for name,param in model.named_parameters():
        if param.requires_grad==True:
            if 'bounds'  in name:
                nweights = nweights + param.numel()
    L1_term = torch.tensor(0., requires_grad=True)
    for name, param in model.named_parameters():
        if param.requires_grad==True:
            if 'bounds'  in name:
                weights_sum = torch.sum(torch.abs(param))
                L1_term = L1_term + weights_sum
    L1_term = L1_term / nweights
    return L1_term


def train(model,original_model,data_provider,weight_decay_list,base_lr=0.01,warmup_epochs=5,n_epochs=5 , treshold=torch.tensor(0.2),bound_type = "layer" , bitflip = "fixed",is_root=False,device='cpu'):
    params_with_wd = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_with_wd.append(param)
    net_params = [
        {
            "params": params_with_wd,
            "weight_decay": weight_decay_list[0],
        },
    ]
    # build optimizer
    if torch.cuda.device_count()>1:
        optimizer = torch.optim.Adam(
                net_params,
                lr=base_lr * dist.get_world_size(),
                weight_decay=4e-11
            )
    else:
        optimizer = torch.optim.Adam(
                net_params,
                lr=base_lr ,
                weight_decay=4e-11
            )   
    # optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    # build lr scheduler
    lr_scheduler = CosineLRwithWarmup(
        optimizer,
        warmup_epochs * len(data_provider['train']),
        base_lr,
        n_epochs * len(data_provider['train']),
    )
    for name, param in model.named_parameters():
        if np.any([key in name for key in ["weight", "norm","bias"]]):
            param.requires_grad=False
        else:
            param.requires_grad=False 
    train_criterion = nn.CrossEntropyLoss().to(device)
    test_criterion = nn.CrossEntropyLoss().to(device)
    # init
    # hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    if torch.cuda.device_count()>1:
        val_info_dict = eval(model, data_provider,is_root)
    else:
        val_info_dict = eval_(model, data_provider,True)   
    print(val_info_dict["val_top1"])
    best_acc =torch.tensor(val_info_dict["val_top1"])
    print(f"the best accuracy is :{best_acc}")
    for name, param in reversed(list(model.named_parameters())):
        if np.any([key in name for key in ["weight", "norm","bias"]]):
            param.requires_grad=False
            continue
        else:
            print(name)
            param.requires_grad=True 
            if param.nelement() > 3 : 
                base_lr = 0.001
                n_epochs = 100
                warmup_epochs=5
            else:
                base_lr = 0.01   
                n_epochs = 100
                warmup_epochs=5
            
        for wd in weight_decay_list:
            print(wd)
            WD_Break = False
            optimizer.param_groups[0]['weight_decay'] = wd
            start_epoch = 0
            best_val = 0.0
            for epoch in range(
                start_epoch,
                n_epochs
                + warmup_epochs,
            ):
                
                train_info_dict = train_one_epoch(
                    model,
                    original_model,
                    data_provider,
                    is_root,
                    epoch,
                    optimizer,
                    train_criterion,
                    lr_scheduler,
                    device
                )
                if torch.cuda.device_count()>1:
                    val_info_dict = eval(model, data_provider,is_root)
                else:
                    val_info_dict = eval_(model, data_provider,True)    
                is_best = val_info_dict["val_top1"] > best_val
                best_val = max(best_val, val_info_dict["val_top1"])
                if is_root:
                    epoch_log = f"[{epoch + 1 - warmup_epochs}/{n_epochs}]"
                    epoch_log += f"\tval_top1={val_info_dict['val_top1']:.2f} ({best_val:.2f})"
                    epoch_log += f"\ttrain_top1={train_info_dict['train_top1']:.2f}\tlr={optimizer.param_groups[0]['lr']:.2E}"
                if torch.abs(best_acc - val_info_dict["val_top1"]) >=10:
                    model.load_state_dict(torch.load("temp_{}_{}_{}.pth".format(bound_type,bitflip,original_model.__class__.__name__)))  
                    break

                if (epoch + 1 ) % 1 == 0:    
                    if torch.abs(best_acc - val_info_dict["val_top1"]) <=treshold:
                        torch.save(model.state_dict(), "temp_{}_{}_{}.pth".format(bound_type,bitflip,original_model.__class__.__name__)) 
                    else:
                        model.load_state_dict(torch.load("temp_{}_{}_{}.pth".format(bound_type,bitflip,original_model.__class__.__name__)))  
                        break       
        param.requires_grad = False       
    return model



def train_one_epoch(
    model: nn.Module,
    original_model,
    data_provider,
    is_root,
    epoch: int,
    optimizer,
    criterion,
    lr_scheduler,
    device,
):
    if torch.cuda.device_count()>1:
        train_loss = DistributedMetric()
        train_top1 = DistributedMetric()
    else:
        train_loss = AverageMeter()
        train_top1 = AverageMeter()    
    model.train()
    if torch.cuda.device_count()>1:
        data_provider['train'].sampler.set_epoch(epoch)

    data_time = AverageMeter()
    with tqdm(
        total=len(data_provider["train"]) ,
        desc="Train Epoch #{}".format(epoch + 1),
        disable= not is_root,
    ) as t:
        end = time.time()         
        for _, (images, labels) in enumerate(data_provider['train']):
            data_time.update(time.time() - end)
            images, labels = images.to(device), labels.to(device)
            l2_bounds = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad==True:
                    if "bounds" in name:
                        l2_bounds += torch.mean(torch.pow(param, 2))
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_output = original_model(images).detach()
                teacher_logits = F.softmax(teacher_output, dim=1)
            nat_logits = model(images)
            kd_loss = cross_entropy_loss_with_soft_target(
                        nat_logits,teacher_logits
                    )
            nat_logits = model(images)
            loss =   criterion(nat_logits,labels) + kd_loss #  +  wd * l2_bounds #+ 0.1 * k_loss #
            loss.backward()
            top1 = accuracy(nat_logits, labels, topk=(1,))[0][0]
            
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
