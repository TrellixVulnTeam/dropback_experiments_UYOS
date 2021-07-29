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
    """
    This function calculate the final pruning rate, given pruning rate of a single iteration 
    and number of iterations.
    """
    final_pruning_rate = 1 - (1 - pruning_rate)**num_iterations

    return final_pruning_rate

def compute_number_prune_iteration(prune_rate_per_round, target_prune_rate):
    '''
    Find the number of iterations needed to achieve a targer_prune_rate given a pruning rate 
    per iteration.

    Example:
    print(compute_number_prune_iteration(0.1, 0.95))

    >>> 29
    '''
    for x in range(100):
        if compute_final_pruning_rate(prune_rate_per_round, x) >= target_prune_rate:
            break
    
    return x

def compute_iterative_prune_rate(target_prune_rate, num_iterations):
    """
    This function is in conjunction with the function before, it takes in the targat final 
    pruning rate and number of iterations and calculate the pruning rate of a single iteration

    Example:
    print(compute_iterative_prune_rate(0.95, 20))

    >>> 0.139
    """
    iterative_prune_rate = 1 - (1 - target_prune_rate) ** (1 / float(num_iterations))
    
    return iterative_prune_rate

def compute_num_param_to_keep(model, target_prune_rate):
    '''
    Given a model and target pruneing rate, calculate the parameters to keep.
    Used for deciding dropback experiment parameters.

    Examples:
    model = models.mobilenet_v2(num_classes=10)
    print(compute_num_param_to_keep(model, 0.05))

    >>> (2236682, 111835)
    '''
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_param_to_prune = math.ceil(num_parameters * target_prune_rate)
    
    return num_parameters, num_param_to_prune
