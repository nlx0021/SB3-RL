import torch
import numpy as np

def phi_exp_factory(p=1, q=1):
    assert p % 2 == 1 & q % 2 == 1
    return lambda x: torch.exp(torch.pow(torch.abs(x), p/q) * (torch.tensor(x>0, dtype=torch.float32) * 2 - 1))

def phi_power_factory(p=1, q=1):
    assert p % 2 == 1 & q % 2 == 1
    return lambda x: torch.pow(torch.abs(x), p/q) * (torch.tensor(x>0, dtype=torch.float32) * 2 - 1)

def phi_fixed_power_factory(p=1, q=1):
    # power(p, q) in [-1,1], power(q, p) outside [-1,1].
    assert p % 2 == 1 & q % 2 == 1
    power = phi_power_factory(p, q)
    inv_power = phi_power_factory(q, p)
    
    def phi_fixed_power(x):
        flag = (torch.abs(x) > 1).float()
        return inv_power(x) * flag + power(x) * (1-flag)    
    
    return phi_fixed_power

def phi_log_factory(p=1, q=None):
    assert p % 2 == 0
    return lambda x: torch.log(1 + p * torch.abs(x)) * (torch.tensor(x>0, dtype=torch.float32) * 2 - 1)

def phi_log_sigmoid_factory(p=None, q=None):
    return lambda x: torch.log((1 / (1 + torch.exp(-x))) + 1e-13)

def phi_sigmoid_factory(p=None, q=None):
    return lambda x: 1 / (1 + torch.exp(-x))
    

def get_phi(name="exp", p=1, q=1):
    
    if name == "exp":
        return phi_exp_factory(p, q)
    
    elif name == "power":
        return phi_power_factory(p, q)
    
    elif name == "fixed-power":
        return phi_fixed_power_factory(p, q)
    
    elif name == "log":
        return phi_log_factory(p, q)
    
    elif name == "log-sigmoid":
        return phi_log_sigmoid_factory(p, q)
    
    elif name == "sigmoid":
        return phi_sigmoid_factory(p, q)