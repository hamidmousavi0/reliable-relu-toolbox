import torch.nn as nn 
import torch
from typing import Dict
from rrelu.utils.distributed import DistributedMetric
import torch.distributed as dist
from rrelu.utils.metric import AverageMeter
import numpy as np
from tqdm import tqdm
from rrelu.utils.metric import accuracy
from rrelu.pytorchfi.weight_error_models import multi_weight_inj_fixed,multi_weight_inj_float,multi_weight_inj_int
from rrelu.relu_bound.bound_relu import Relu_bound
from rrelu.pytorchfi.core import FaultInjection
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
            disable=not dist.get_rank() == 0,
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

def eval_(model: nn.Module, data_loader_dict,device) -> Dict:

    test_criterion = nn.CrossEntropyLoss().to(device)

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
                images, labels = images.to(device), labels.device()
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
            disable=not dist.get_rank() == 0,
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


def eval_fault_(model:nn.Module,data_loader_dict, fault_rate,iterations=500,bitflip=None,total_bits = 32 , n_frac = 16 , n_int = 15,device='cpu' )-> Dict:
    inputs, classes = next(iter(data_loader_dict['val'])) 
    pfi_model = FaultInjection(model, 
                            inputs.shape[0],
                            input_shape=[inputs.shape[1],inputs.shape[2],inputs.shape[3]],
                            layer_types=[torch.nn.Conv2d, torch.nn.Linear ,Relu_bound],
                            total_bits= total_bits,
                            n_frac = n_frac, 
                            n_int = n_int, 
                            use_cuda= device=='cuda',
                            )
    print(pfi_model.print_pytorchfi_layer_summary())
    test_criterion = nn.CrossEntropyLoss()

    val_loss = AverageMeter()
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()

    pfi_model.original_model.eval()
    with torch.no_grad():
        with tqdm(
            total= iterations,
            desc="Eval",
        ) as t:
            for i in range(iterations):
                if bitflip=='float':
                    corrupted_model = multi_weight_inj_float(pfi_model,fault_rate,device=device)
                elif bitflip=='fixed':    
                    corrupted_model = multi_weight_inj_fixed(pfi_model,fault_rate,device=device)
                elif bitflip =="int":
                    corrupted_model = multi_weight_inj_int (pfi_model,fault_rate,device=device)
                    # corrupted_model = multi_weight_inj_int(pfi_model,fault_rate)          
                for images, labels in data_loader_dict["val"]:
                    images, labels = images.to(device), labels.to(device)
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
                        "#samples": val_top1.count,
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
