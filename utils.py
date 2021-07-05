import math
import torch

def measure_module_sparsity(module, threshold=0, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(torch.abs(param) < threshold).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(torch.abs(param) < threshold).item()
                num_elements += param.nelement()
                
    if num_elements == 0:
        return 0, 0, 0

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def measure_global_sparsity(model, threshold=0, weight=True, bias=False, use_mask=True):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():
        module_num_zeros, module_num_elements, _ = measure_module_sparsity(
            module, weight=weight, bias=bias, use_mask=use_mask)
        num_zeros += module_num_zeros
        num_elements += module_num_elements

    if num_elements == 0:
        return 0,0,0
    
    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def compute_final_pruning_rate(pruning_rate, num_iterations):
    final_pruning_rate = 1 - (1 - pruning_rate)**num_iterations

    return final_pruning_rate

def compute_iterative_prune_rate(target_prune_rate, num_iterations):
    iterative_prune_rate = 1 - (1 - target_prune_rate) ** (1 / float(num_iterations))
    
    return iterative_prune_rate

def compute_num_param_to_prune(model, target_prune_rate):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_param_to_prune = math.ceil(num_parameters * target_prune_rate)
    
    return num_parameters, num_param_to_prune
