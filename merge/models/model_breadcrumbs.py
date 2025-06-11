import os
import torch
import copy

from models.utils import obtain_delta
from mergekit.sparsify import RescaleNorm, SparsificationMethod, sparsify

def breadcrumbs_merge(task_model, taske_pre_model, safety_model, safety_pre_model):
    task_vector = obtain_delta(task_model, taske_pre_model)
    safety_vector = obtain_delta(safety_model, safety_pre_model)

    merge_tv = breadcrumbs(task_vector, safety_vector)

    merged_params = {}
    for name in taske_pre_model.state_dict().keys():
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

def breadcrumbs(task_vector, safety_vector, task_weight=1.0, safety_weight=1.0, density=0.5) -> torch.Tensor:
    # collect task vectors
    # sparsify
    for key in task_vector:
        if task_vector[key] is not None:
            kwargs = {}
            task_vector[key] = sparsify(
                task_vector[key].cuda(),
                density=density,
                method="magnitude_outliers",
                rescale_norm=None,
                **kwargs,
            ).cpu()

    for key in safety_vector:
        if safety_vector[key] is not None:
            kwargs = {}
            safety_vector[key] = sparsify(
                safety_vector[key].cuda(),
                density=density,
                method="magnitude_outliers",
                rescale_norm=None,
                **kwargs,
            ).cpu()
    merge_tv = {}
    for key in task_vector:
        if task_vector[key] is not None and safety_vector[key] is not None and task_vector[key].shape == safety_vector[key].shape:
            deltas = torch.cat([task_vector[key].unsqueeze(0), safety_vector[key].unsqueeze(0)], dim=0)
            weights = torch.tensor([task_weight, safety_weight], dtype=deltas.dtype, device=deltas.device)
            while len(deltas.shape) > len(weights.shape):
                weights.unsqueeze_(-1)

            weighted_deltas = deltas * weights
            mixed_delta = weighted_deltas.sum(dim=0)

            merge_tv[key] = mixed_delta

    return merge_tv


def get_mask(
    delta: torch.Tensor,
    method="sum",
    mask_dtype=None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.

    For the methodology described in the TIES paper use 'sum'. For a
    simpler naive count of signs, use 'count'."""
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = delta.sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign