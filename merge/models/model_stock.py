import os
import torch
import copy

from models.utils import obtain_delta
def stock_merge(task_model, taske_pre_model, safety_model, safety_pre_model):
    task_vector = obtain_delta(task_model, taske_pre_model)
    safety_vector = obtain_delta(safety_model, safety_pre_model)

    merge_tv = stock(task_vector, safety_vector)

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


def stock(task_vector, safety_vector) -> torch.Tensor:

    merge_tv = {}
    for key in task_vector:
        if task_vector[key] is not None and safety_vector[key] is not None and task_vector[key].shape == safety_vector[key].shape:
            out_shape = task_vector[key].shape
            offsets = [task_vector[key].view(-1), safety_vector[key].view(-1)]

            cos_thetas = []
            for i, w_0_offset in enumerate(offsets):
                for j in range(i + 1, len(offsets)):
                    w_1_offset = offsets[j]

                    norm_product = torch.norm(w_0_offset, dim=-1) * torch.norm(
                        w_1_offset, dim=-1
                    )
                    cos_theta = (
                        (w_0_offset * w_1_offset).sum(dim=-1) / norm_product.clamp(min=1e-6)
                    ).clamp(-1, 1)
                    cos_thetas.append(cos_theta)

            cos_theta = torch.stack(cos_thetas).mean(dim=0).unsqueeze(-1)
            N = 2
            t = (N * cos_theta) / (1 + (N - 1) * cos_theta)

            w_avg = (task_vector[key] + safety_vector[key]) / 2 * t

            merge_tv[key] = w_avg

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