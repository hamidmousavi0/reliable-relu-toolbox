
import torch
import torch.nn as nn
from rrelu.relu_bound.bound_relu import Relu_bound
class bounded_relu_tresh(nn.Module,Relu_bound):
    '''
    Bound the relu activatoin and back the values to treshold
    ------------------------------------
    bound : the bound for the activation
    -------------------------------------
    pytorch module with forward function
    '''
    def __init__(self, bounds ,tresh=None,alpha=None,k=-20):
        super().__init__()
        bounds_param={}
        tresh_param={}
        param_name= "bounds_param"
        tres_name = 'tresh_param'
        bounds_param[param_name] = nn.Parameter(data=bounds.cuda(), requires_grad=True)  
        tresh_param[param_name] = nn.Parameter(data=tresh.cuda(), requires_grad=True)  
        for name, param in bounds_param.items():
            self.register_parameter(name, param) 
        for name, param in tresh_param.items():
            self.register_parameter(name, param)     
        self.bounds =  self.__getattr__("bounds_param")   
        self.tresh = self.__getattr__("tresh_param")  
        
    def forward(self, input):
        # input = torch.nan_to_num(input)
        output = torch.ones_like(input) * input
        output[torch.gt(input,self.__getattr__("bounds_param")   )] = self.__getattr__("tresh_param")  
        return torch.maximum(torch.tensor(0.0),output)       
