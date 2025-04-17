
import torch
import torch.nn as nn
from rrelu.relu_bound.bound_relu import Relu_bound
import torch.nn.functional as F

class bounded_hyrelu_proact(nn.Module,Relu_bound):
    def __init__(self,bounds=None,tresh=None,alpha_param = None,k=-20):
        super().__init__()
        bounds_param={}
        param_name1= "bounds_param"
        self.tresh = tresh
        if tresh ==None:
            bounds_param[param_name1] = nn.Parameter(data=torch.zeros_like(bounds).cuda(), requires_grad=True) 
        else:
            bounds_param[param_name1] = nn.Parameter(data=bounds.cuda(), requires_grad=True) 
          
        self.k = k 
        for name, param in bounds_param.items():
            self.register_parameter(name, param)   
        self.bounds =  self.__getattr__("bounds_param")  
    def forward(self,input):
        # input = torch.nan_to_num(input)
        output =   input - input * torch.sigmoid(-self.k* (input-self.__getattr__("bounds_param"))) # + self.__getattr__("bounds_param")
        # print(output)
        return torch.maximum(torch.tensor(0.0),output)   
