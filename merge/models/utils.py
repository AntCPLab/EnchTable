import torch
import copy
import gc
import logging
import subprocess
import numpy as np
import torch.nn as nn

def calculate_state_dict_norm(model, norm_type=2):

    if isinstance(model, torch.nn.Module):
        state = model.state_dict()
    else:
        state = model

    diffs = []

    for key in state:
        if state[key] is not None:
            # print(state[key].sum(), state[key].shape)
            diffs.append(state[key].view(-1))
    all_diffs = torch.cat(diffs)
    if norm_type == 1:
        norm = torch.norm(all_diffs, p=1).item()
    elif norm_type == 2:
        norm = torch.norm(all_diffs, p=2).item()
    else:
        raise ValueError("only support norm_type=1 (L1) or 2 (L2)")

    return norm

def obtain_delta(model1, model2):

    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    assert state_dict1.keys() == state_dict2.keys(), "diffent state_dict keys"

    param_diff = {}
    for name in state_dict1.keys():
        param1 = state_dict1[name]
        param2 = state_dict2[name]
        
        if param1.shape == param2.shape:
            diff = param1 - param2
            param_diff[name] = diff
        else:
            param_diff[name] = None

    return param_diff