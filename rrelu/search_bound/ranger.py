import sys
import torch.nn as nn
import torch
import sys;
from rrelu.relu_bound.bound_relu import Relu_bound
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
# Check batch size go through two loop in data loader
# Ranger bounds function for activation tracking and boundary computation
def Ranger_bounds(model: nn.Module, train_loader, device="cuda", bound_type='layer', bitflip='float',is_root=False):
    model = model.to(device)
    model.eval()

    results = {}
    tresh = {}

    relu_hooks(model)
    # Use torch.no_grad() for inference efficiency
    with torch.no_grad():
        first_batch = True
        for data, label in train_loader['sub_train']:
            data = data.to(device, non_blocking=True)

            # Forward pass
            _ = model(data)

            if first_batch:
                # Initialize results and tresh with the first batch activations
                for key, val in activation.items():
                    results[key] = val.clone()
                    tresh[key] = val.clone()
                first_batch = False
            else:
                # Update results and tresh efficiently
                for key, val in activation.items():
                    prev_max = torch.max(results[key],dim=0)[0]
                    prev_mean = torch.mean(tresh[key],dim=0)
                    curr_max = torch.max(activation[key],dim=0)[0]
                    curr_mean = torch.mean(activation[key],dim=0)
                    results[key] = torch.maximum(prev_max,curr_max)
                    tresh[key] = torch.minimum(prev_mean,curr_mean)  
                    
    # Compute max and min values for layer-level bounds if required
    if bound_type == "layer":
        for key in results.keys():
            results[key] = torch.max(results[key])  # Scalar for easier handling
            tresh[key] = torch.min(tresh[key])

    return results, tresh, None
