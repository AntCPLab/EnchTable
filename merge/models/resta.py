import copy
import torch

from models.utils import obtain_delta, calculate_state_dict_norm
# from models.compute_fisher_information_matrix import compute_importance_score

import torch

def resta_merge(task_model, taske_pre_model, safety_model, safety_pre_model, adaptive=False, model_name=None):
    task_vector = obtain_delta(task_model, taske_pre_model)
    safety_vector = obtain_delta(safety_model, safety_pre_model)
    
    # if model_name is not None:
    #     importance_score = compute_importance_score(model_name, batch_size=4, num_samples=1000)
    #     for key in safety_vector:
    #         if task_vector[key] is not None and safety_vector[key] is not None:
    #             beta = importance_score[key]
    #             safety_vector[key] = safety_vector[key] * beta

    if adaptive:
        task_norm = calculate_state_dict_norm(task_vector)
        safety_norm = calculate_state_dict_norm(safety_vector)
        alpha = task_norm / safety_norm * 0.1
        print(alpha)
    else:
        alpha = 1

    merge_tv = {}
    for key in task_vector:
        if task_vector[key] is not None and safety_vector[key] is not None:

            merge_tv[key] = task_vector[key] + alpha * safety_vector[key] * 0.

    merged_params = {}
    for name in task_model.state_dict().keys():
        if name in merge_tv:
            param1 = taske_pre_model.state_dict()[name]
            param2 = merge_tv[name]
            if param1.shape == param2.shape:
                merged_params[name] = param1 + param2
                continue
        print("{} if skipped".format(name))
        merged_params[name] = task_model.state_dict()[name]
    task_model = copy.deepcopy(task_model)
    task_model.load_state_dict(merged_params)
    return task_model   