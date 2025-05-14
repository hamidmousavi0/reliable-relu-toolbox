import sys
import torch
import numpy as np
import torch
import copy
import torch.nn as nn 
from rrelu.pytorchfi.core import FaultInjection
from rrelu.utils.distributed import DistributedMetric
from rrelu.utils.metric import accuracy
from rrelu.pytorchfi.weight_error_models import bit_flip_weight_IEEE,bit_flip_weight_fixed,bit_flip_weight_int
from rrelu.relu_bound.bound_zero import bounded_relu_zero 
from rrelu.relu_bound.bound_relu import Relu_bound
from rrelu.search_bound.ranger import Ranger_bounds
from rrelu.utils.metric import AverageMeter
import random
activation={}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

# Register hooks for all ReLU layers in the model
def relu_hooks(model: nn.Module, prefix=''):
    for name, layer in model.named_children():
        layer_name = f"{prefix}.{name}" if prefix else name
        if isinstance(layer, nn.ReLU) or isinstance(layer,Relu_bound) or isinstance(layer, nn.ReLU6):
            layer.register_forward_hook(get_activation(layer_name))
        elif len(list(layer.children())) > 0:  # Recursively check nested modules
            relu_hooks(layer, layer_name)
def replace_act_all(model:nn.Module,bounds,tresh, prefix='',device='cuda')->nn.Module:
    for name,layer in model.named_children():
        layer_name = f"{prefix}.{name}" if prefix else name
        if isinstance(layer, nn.ReLU) or isinstance(layer,Relu_bound) or isinstance(layer, nn.ReLU6):
            model._modules[name] = bounded_relu_zero(bounds[layer_name].detach(),tresh[layer_name].detach(),device=device)
        elif len(list(layer.children())) > 0:
            replace_act_all(layer,bounds,tresh,layer_name)               
    return model


# def replace_act(model:nn.Module,bounds,key,tresh, prefix='')->nn.Module:
#     for name,layer in model.named_children():
#         layer_name = f"{prefix}.{name}" if prefix else name
#         if isinstance(layer, nn.ReLU) or isinstance(layer,Relu_bound):
#             if layer_name == key:
#                 model._modules[name] = bounded_relu_zero(bounds[layer_name].detach(),tresh[layer_name].detach())
#         elif len(list(layer.children())) > 0:
#             replace_act(layer,bounds,key,tresh,layer_name)               
#     return model

def replace_act(model:nn.Module,bounds,key,tresh,name='')->nn.Module:
    for name1,layer in model.named_children():
        if list(layer.children()) == []:
            if isinstance(layer,nn.ReLU)or isinstance(layer,Relu_bound) or isinstance(layer, nn.ReLU6):
                name_ = name1 + name
                if name_==key:
                    model._modules[name1] = bounded_relu_zero(bounds,tresh)
        else:
            name+=name1
            replace_act(layer,bounds,key,tresh,name)               
    return model  
def FtClipAct_bounds(model:nn.Module, data_loader, device="cuda",bound_type='layer',bitflip='fixed',is_root=False,N=4, M = 3,fault_rates=[1e-7,1e-6,3e-6,1e-5,3e-5], delta_b=0.2):
    model.eval()
    if bound_type=="neuron":
        raise AssertionError("FTClip Act doesnt work with neuron wise")
    else:
        bounds_ftclip,tresh_ftclip,_ = Ranger_bounds(copy.deepcopy(model),data_loader)               
     
    model = replace_act_all(model,bounds_ftclip,tresh_ftclip,prefix='',device=device).to(device)
    for key,val in bounds_ftclip.items():
        print(key,val)
        counter = 1
        i = 1 
        act_max = val.item()
        tresh_val = tresh_ftclip[key].item()
        while counter<=N:
            if i==1:
                S = torch.tensor([0,act_max])
                AUC,T = AUC_Calculation(S,copy.deepcopy(model),fault_rates,data_loader,key,device,tresh_val,bitflip)
                S,T_result =Interval_Search(T,AUC)
            else:
                AUC, T = AUC_Calculation(S, copy.deepcopy(model) , fault_rates, data_loader,key ,device,tresh_val,bitflip)
                S,T_result =Interval_Search(T,AUC)
            Delta=[]
            for j in range(3):
                Delta.append(torch.abs(AUC[j+1]-AUC[j]))
            Delta = torch.tensor(Delta)
            if torch.max(Delta)<=delta_b and counter>=M:
                bounds_ftclip[key] = T_result
                break
            counter+=1
            i+=1
        model = replace_act(model,T_result,key,tresh_val)    
        bounds_ftclip[key] = T_result
        print("the final bounded is {}\n".format(T_result))
        bounds_ftclip,tresh_ftclip,_ = Ranger_bounds(copy.deepcopy(model),data_loader)
    return bounds_ftclip,tresh_ftclip,None  




def AUC_Calculation(S,model,fault_rates,data_loader,key,device,tresh,bitflip):
    model.eval()
    T1 = torch.min(S)
    T2 = T1 + (torch.max(S)-torch.min(S))/3
    T3 = T2 + (torch.max(S)-torch.min(S))/3
    T4 = torch.max(S)
    T=[T1,T2,T3,T4]
    AUC=[]
    orig_model = copy.deepcopy(model)
    for i in range(4):
        model = replace_act(model,T[i],key,tresh) 
        auc = auc_compute(model , fault_rates, data_loader,device,bitflip)
        AUC.append(auc)
        model = copy.deepcopy(orig_model)
    return torch.tensor(AUC),torch.tensor(T)




      

def auc_compute(model,fault_rates,data_loader,device,bitflip):
    model.eval()
    acc_list=[]
    acc = 0 
    orig_model = copy.deepcopy(model)
    for fault_rate in fault_rates:
        acc = accuracy_vs_faultrate(model,fault_rate,data_loader,bitflip,device)
        acc_list.append(acc)
        acc=0
        model = copy.deepcopy(orig_model)
    fault_rates = np.array([0.2,0.4,0.6,0.8,1])
    acc_list = np.array(acc_list)
    auc = np.trapz(acc_list,fault_rates)
    return  auc
def Interval_Search(T,AUC):
    '''
    interval serach in ftclip act paper to serahc better treshold
    ------------------------------------------------------------
    T : list of the bounds
    AUC : AUC
    '''
    index = torch.argmax(AUC)
    if index==3:
        S = torch.tensor([T[2],T[3]])
    elif index==0:
        S = torch.tensor([T[0], T[1]])
    else:
        S = torch.tensor([T[index-1],T[index+1]])
    T_prime = T[index]
    return S,T_prime





def accuracy_vs_faultrate(model,fault_rate,data_loader,bitflip,fault_simulation_epochs=5,device='cpu'):
    inputs, classes = next(iter(data_loader['sub_train']))
    if device == 'cuda':
        use_cuda = True
    pfi_model = FaultInjection(copy.deepcopy(model), 
                inputs.shape[0],
                input_shape=[inputs.shape[1],inputs.shape[2],inputs.shape[3]],
                layer_types=[torch.nn.Conv2d, torch.nn.Linear],
                use_cuda=use_cuda,
                )
    model.to(device)
    if device=='cuda':
        val_loss = DistributedMetric()
        val_top1 = DistributedMetric()
        val_top5 = DistributedMetric()
    else:
        val_loss = AverageMeter()
        val_top1 = AverageMeter()
        val_top5 = AverageMeter()
    test_criterion = nn.CrossEntropyLoss().to(device)
    pfi_model.original_model.eval()
    with torch.no_grad():
        for i in range(fault_simulation_epochs):  
            if bitflip=='float':
                    corrupted_model = multi_weight_inj_float(pfi_model,fault_rate,device=device)
            elif bitflip=='fixed':    
                    corrupted_model = multi_weight_inj_fixed(pfi_model,fault_rate,device=device)
            elif bitflip =="int":
                    corrupted_model = multi_weight_inj_int (pfi_model,fault_rate,device=device)
            for images, labels in data_loader["sub_train"]:
                images, labels = images.to(device), labels.to(device)
                output = corrupted_model(images)
                loss = test_criterion(output, labels)
                val_loss.update(loss, images.shape[0])
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                val_top5.update(acc5[0], images.shape[0])
                val_top1.update(acc1[0], images.shape[0])
    return val_top1.avg.item()


######################### Fault Injection ##################################
def multi_weight_inj_int(pfi, sdc_p=1e-5, function1=bit_flip_weight_int,function2=bit_flip_weight_IEEE,device='cpu'):
    corrupt_idx = [[], [], [], [], []]
    corrupt_bit_idx = []
    corrupt_idx_bias = [[], [], [], [], []]
    corrupt_bit_idx_bias = []
    total_bits,n_frac,n_int = pfi.get_total_bits()
    layer_idx = 0
    for layer in pfi.original_model.modules():
        if isinstance(layer, tuple(pfi._inj_layer_types)):
            shape = list(pfi.get_weights_size(layer_idx))
            shape_bias = list(pfi.get_bias_size(layer_idx))
            dim_bias = len(shape_bias)
            dim_len = len(shape)  
            shape.extend([1 for i in range(4 - len(shape))])
            nunmber_fault_weight = int(shape[0] * shape[1] * shape[2] * shape [3] * total_bits * sdc_p)
            shape_bias.extend([1 for i in range(4 - len(shape_bias))])
            if shape_bias[0] !=None : 
                nunmber_fault_bias = int(shape_bias[0] * shape_bias[1] * shape_bias[2] * shape_bias [3] * total_bits * sdc_p) 
            if nunmber_fault_weight !=0: 
                k_w = torch.randint(shape[0],(nunmber_fault_weight,), device=device)
                dim1_w = torch.randint(shape[1],(nunmber_fault_weight,), device=device)
                dim2_w = torch.randint(shape[2],(nunmber_fault_weight,), device=device)
                dim3_w = torch.randint(shape[3],(nunmber_fault_weight,), device=device)
                dim4_w = torch.randint(total_bits,(nunmber_fault_weight,), device=device)
            if shape_bias[0]!=None:
                if nunmber_fault_bias!=0:
                    k_b = torch.randint(shape[0],(nunmber_fault_bias,), device=device)
                    dim1_b = torch.randint(shape[1],(nunmber_fault_bias,), device=device)
                    dim2_b = torch.randint(shape[2],(nunmber_fault_bias,), device=device)
                    dim3_b = torch.randint(shape[3],(nunmber_fault_bias,), device=device)
                    dim4_b = torch.randint(total_bits,(nunmber_fault_bias,), device=device)
            for fault in range(nunmber_fault_weight):
                idx = [layer_idx, k_w[fault].item(), dim1_w[fault].item(), dim2_w[fault].item(), dim3_w[fault].item()]
                for i in range(dim_len + 1):
                    corrupt_idx[i].append(idx[i])
                for i in range(dim_len + 1, 5): 
                    corrupt_idx[i].append(None)  
                corrupt_bit_idx.append(dim4_w[fault]) 
            if shape_bias[0]!= None:                                                                       
                for fault in range(nunmber_fault_bias):
                    idx = [layer_idx, k_b[fault].item(), dim1_b[fault].item(), dim2_b[fault].item(), dim3_b[fault].item()]
                    for i in range(dim_bias + 1):
                        corrupt_idx_bias[i].append(idx[i])
                    for i in range(dim_bias + 1, 5): 
                        corrupt_idx_bias[i].append(None)  
                    corrupt_bit_idx_bias.append(dim4_b[fault])       
        layer_idx+=1                                                             
    return pfi.declare_weight_fault_injection(
        layer_num=[corrupt_idx[0],corrupt_idx_bias[0]],
        k=[corrupt_idx[1],corrupt_idx_bias[1]],
        dim1=[corrupt_idx[2],corrupt_idx_bias[2]],
        dim2=[corrupt_idx[3],corrupt_idx_bias[3]],
        dim3=[corrupt_idx[4],corrupt_idx_bias[4]],
        dim4 = [corrupt_bit_idx,corrupt_bit_idx_bias],
        function=[function1,function2],
        total_bits = total_bits,
        n_frac = None,
        n_int = None
    )



def multi_weight_inj_float(pfi, sdc_p=1e-5, function1=bit_flip_weight_IEEE,function2=bit_flip_weight_IEEE,device='cpu'):
    corrupt_idx = [[], [], [], [], []]
    corrupt_bit_idx = []
    corrupt_idx_bias = [[], [], [], [], []]
    corrupt_bit_idx_bias = []
    total_bits,n_frac,n_int = pfi.get_total_bits()
    layer_idx = 0
    for layer in pfi.original_model.modules():
        if isinstance(layer, tuple(pfi._inj_layer_types)):
            shape = list(pfi.get_weights_size(layer_idx))
            shape_bias = list(pfi.get_bias_size(layer_idx))
            dim_bias = len(shape_bias)
            dim_len = len(shape)  
            shape.extend([1 for i in range(4 - len(shape))])
            nunmber_fault_weight = int(shape[0] * shape[1] * shape[2] * shape [3] * total_bits * sdc_p)
            shape_bias.extend([1 for i in range(4 - len(shape_bias))])
            if shape_bias[0] !=None : 
                nunmber_fault_bias = int(shape_bias[0] * shape_bias[1] * shape_bias[2] * shape_bias [3] * total_bits * sdc_p) 
            if nunmber_fault_weight !=0:   
                k_w = torch.randint(shape[0],(nunmber_fault_weight,), device=device)
                dim1_w = torch.randint(shape[1],(nunmber_fault_weight,), device=device)
                dim2_w = torch.randint(shape[2],(nunmber_fault_weight,), device=device)
                dim3_w = torch.randint(shape[3],(nunmber_fault_weight,), device=device)
                dim4_w = torch.randint(total_bits,(nunmber_fault_weight,), device=device)
            if shape_bias[0]!=None:
                if nunmber_fault_bias!=0:
                    k_b = torch.randint(shape[0],(nunmber_fault_bias,), device=device)
                    dim1_b = torch.randint(shape[1],(nunmber_fault_bias,), device=device)
                    dim2_b = torch.randint(shape[2],(nunmber_fault_bias,), device=device)
                    dim3_b = torch.randint(shape[3],(nunmber_fault_bias,), device=device)
                    dim4_b = torch.randint(total_bits,(nunmber_fault_bias,), device=device)
            for fault in range(nunmber_fault_weight):
                idx = [layer_idx, k_w[fault].item(), dim1_w[fault].item(), dim2_w[fault].item(), dim3_w[fault].item()]
                for i in range(dim_len + 1):
                    corrupt_idx[i].append(idx[i])
                for i in range(dim_len + 1, 5): 
                    corrupt_idx[i].append(None)  
                corrupt_bit_idx.append(dim4_w[fault]) 
            if shape_bias[0]!= None:                                                                       
                for fault in range(nunmber_fault_bias):
                    idx = [layer_idx, k_b[fault].item(), dim1_b[fault].item(), dim2_b[fault].item(), dim3_b[fault].item()]
                    for i in range(dim_bias + 1):
                        corrupt_idx_bias[i].append(idx[i])
                    for i in range(dim_bias + 1, 5): 
                        corrupt_idx_bias[i].append(None)  
                    corrupt_bit_idx_bias.append(dim4_b[fault])       
        layer_idx+=1
    return pfi.declare_weight_fault_injection(
        layer_num=[corrupt_idx[0],corrupt_idx_bias[0]],
        k=[corrupt_idx[1],corrupt_idx_bias[1]],
        dim1=[corrupt_idx[2],corrupt_idx_bias[2]],
        dim2=[corrupt_idx[3],corrupt_idx_bias[3]],
        dim3=[corrupt_idx[4],corrupt_idx_bias[4]],
        dim4 = [corrupt_bit_idx,corrupt_bit_idx_bias],
        function=[function1,function2],
        total_bits = total_bits,
        n_frac = None,
        n_int = None
    )



def multi_weight_inj_fixed(pfi, sdc_p=1e-5, function1=bit_flip_weight_fixed,function2=bit_flip_weight_fixed):
    corrupt_idx = [[], [], [], [], []]
    corrupt_bit_idx = []
    corrupt_idx_bias = [[], [], [], [], []]
    corrupt_bit_idx_bias = []
    total_bits,n_frac,n_int = pfi.get_total_bits()
    layer_idx = 0
    for layer in pfi.original_model.modules():
        if isinstance(layer, tuple(pfi._inj_layer_types)):
            shape = list(pfi.get_weights_size(layer_idx))
            shape_bias = list(pfi.get_bias_size(layer_idx))
            dim_bias = len(shape_bias)
            dim_len = len(shape)  
            shape.extend([1 for i in range(4 - len(shape))])
            nunmber_fault_weight = int(shape[0] * shape[1] * shape[2] * shape [3] * total_bits * sdc_p)
            shape_bias.extend([1 for i in range(4 - len(shape_bias))])
            if shape_bias[0] !=None : 
                nunmber_fault_bias = int(shape_bias[0] * shape_bias[1] * shape_bias[2] * shape_bias [3] * total_bits * sdc_p) 
            if nunmber_fault_weight !=0:      
                k_w = torch.randint(shape[0],(nunmber_fault_weight,), device='cuda')
                dim1_w = torch.randint(shape[1],(nunmber_fault_weight,), device='cuda')
                dim2_w = torch.randint(shape[2],(nunmber_fault_weight,), device='cuda')
                dim3_w = torch.randint(shape[3],(nunmber_fault_weight,), device='cuda')
                dim4_w = torch.randint(total_bits,(nunmber_fault_weight,), device='cuda')
            if shape_bias[0]!=None:
                if nunmber_fault_bias!=0:
                    k_b = torch.randint(shape[0],(nunmber_fault_weight,), device='cuda')
                    dim1_b = torch.randint(shape[1],(nunmber_fault_weight,), device='cuda')
                    dim2_b = torch.randint(shape[2],(nunmber_fault_weight,), device='cuda')
                    dim3_b = torch.randint(shape[3],(nunmber_fault_weight,), device='cuda')
                    dim4_b = torch.randint(total_bits,(nunmber_fault_weight,), device='cuda')
            for fault in range(nunmber_fault_weight):
                idx = [layer_idx, k_w[fault].item(), dim1_w[fault].item(), dim2_w[fault].item(), dim3_w[fault].item()]
                for i in range(dim_len + 1):
                    corrupt_idx[i].append(idx[i])
                for i in range(dim_len + 1, 5): 
                    corrupt_idx[i].append(None)  
                corrupt_bit_idx.append(dim4_w[fault]) 
            if shape_bias[0]!= None:                                                                       
                for fault in range(nunmber_fault_bias):
                    idx = [layer_idx, k_b[fault].item(), dim1_b[fault].item(), dim2_b[fault].item(), dim3_b[fault].item()]
                    for i in range(dim_bias + 1):
                        corrupt_idx_bias[i].append(idx[i])
                    for i in range(dim_bias + 1, 5): 
                        corrupt_idx_bias[i].append(None)  
                    corrupt_bit_idx_bias.append(dim4_b[fault])           
            layer_idx+=1                                                         
    return pfi.declare_weight_fault_injection(
        layer_num=[corrupt_idx[0],corrupt_idx_bias[0]],
        k=[corrupt_idx[1],corrupt_idx_bias[1]],
        dim1=[corrupt_idx[2],corrupt_idx_bias[2]],
        dim2=[corrupt_idx[3],corrupt_idx_bias[3]],
        dim3=[corrupt_idx[4],corrupt_idx_bias[4]],
        dim4 = [corrupt_bit_idx,corrupt_bit_idx_bias],
        function=[function1,function2],
        total_bits = total_bits,
        n_frac = n_frac, 
        n_int = n_int, 
    )




###############################################################################################
