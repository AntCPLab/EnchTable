import copy
import torch

from models.utils import obtain_delta, calculate_state_dict_norm


def enchtable_merge(task_model, taske_pre_model, safety_model, safety_pre_model, learn_mask=False, m0=0., adaptive=False, ada_alpha=0.1):
    task_vector = obtain_delta(task_model, taske_pre_model)
    safety_vector = obtain_delta(safety_model, safety_pre_model)

    if adaptive:
        task_norm = calculate_state_dict_norm(task_vector)
        safety_norm = calculate_state_dict_norm(safety_vector)
        alpha = task_norm / safety_norm * ada_alpha
        print(alpha, task_norm, safety_norm)
    else:
        alpha = 1
    # return task_vector
    
    for key in safety_vector:
        if task_vector[key] is not None and safety_vector[key] is not None:
            safety_vector[key] = safety_vector[key] * alpha

    safety_vector = adjust_safety([task_vector, safety_vector], learn_mask=learn_mask, m0=m0)

    merge_tv = {}
    for key in task_vector:
        if task_vector[key] is not None and safety_vector[key] is not None:
            merge_tv[key] = task_vector[key] + safety_vector[key]

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


def adjust_safety(task_vectors, learn_mask=False, m0=0.0):

    # with torch.no_grad():
    new_vector = {}
    for key in task_vectors[0]:
        if task_vectors[0][key] is None or task_vectors[1][key] is None:
            continue

        if task_vectors[1][key].sum() < 1e-8:
            new_vector[key] = task_vectors[1][key]

        if len(task_vectors[0][key].shape) != 2:
            new_vector[key] = task_vectors[1][key]
        else:
            if learn_mask:
                new_vector[key] = compute_interference_learn(task_vectors[0][key].cuda(), task_vectors[1][key].cuda(), m0=m0)
            else:
                weight, _ = compute_interference(task_vectors[0][key].cuda(), task_vectors[1][key].cuda())
                new_vector[key] = task_vectors[1][key] * weight

    return new_vector


def compute_interference(vec1, vec2, ratio1=0.5, ratio2=0.5):
    u1, s1, v1 = torch.linalg.svd(vec1, full_matrices=False)
    reduced_index_1 = int(s1.shape[0] * ratio1)
    u2, s2, v2 = torch.linalg.svd(vec2, full_matrices=False)
    reduced_index_2 = int(s2.shape[0] * ratio2)

    sum_u = torch.cat([u1[:, :reduced_index_1], u2[:, :reduced_index_2]], dim=1)
    sum_s = torch.cat([s1[:reduced_index_1], s2[:reduced_index_2]], dim=0)
    sum_v = torch.cat([v1[:reduced_index_1, :], v2[:reduced_index_2, :]], dim=0)

    UUt = torch.matmul(sum_u.t(), sum_u)
    VtV = torch.matmul(sum_v, sum_v.t())
    UUt_minus_I = UUt - torch.eye(sum_s.shape[0], device=sum_u.device)
    VtV_minus_I = VtV - torch.eye(sum_s.shape[0], device=sum_u.device)
    result_matrix = torch.matmul(torch.matmul(UUt_minus_I, torch.diag(sum_s)), VtV_minus_I)
    print(result_matrix.shape, vec2.shape)
    interference = torch.norm(result_matrix, p=1) / result_matrix.shape[0] / result_matrix.shape[1] * 200
    weight = (-interference).exp().item()
    print(weight)

    return weight, (sum_u, sum_s, sum_v)

def compute_interference_learn(vec1, vec2, ratio1=0.5, ratio2=0.5, ite=40, lr=1e3, m0=0.0):
    m = torch.tensor([0.5]).float().cuda()
    m_copy = copy.deepcopy(m).cpu()
    m.requires_grad_(True)
    optimizer = torch.optim.SGD([m], lr=lr)
    u1, s1, v1 = torch.linalg.svd(vec1, full_matrices=False)
    reduced_index_1 = int(s1.shape[0] * ratio1)

    for i in range(ite):
        u2, s2, v2 = torch.linalg.svd(vec2.cuda() * (m0 + m), full_matrices=False)
        reduced_index_2 = int(s2.shape[0] * ratio2)
        sum_u = torch.cat([u1[:, :reduced_index_1].cuda(), u2[:, :reduced_index_2]], dim=1)
        sum_s = torch.cat([s1[:reduced_index_1].cuda(), s2[:reduced_index_2]], dim=0)
        sum_v = torch.cat([v1[:reduced_index_1, :].cuda(), v2[:reduced_index_2, :]], dim=0)

        UUt = torch.matmul(sum_u.t(), sum_u)
        VtV = torch.matmul(sum_v, sum_v.t())
        UUt_minus_I = UUt - torch.eye(sum_s.shape[0], device=sum_u.device)
        VtV_minus_I = VtV - torch.eye(sum_s.shape[0], device=sum_u.device)
        result_matrix = torch.matmul(torch.matmul(UUt_minus_I, torch.diag(sum_s)), VtV_minus_I)

        interference = torch.norm(result_matrix, p=1) / result_matrix.shape[0] / result_matrix.shape[1]
        if torch.isnan(interference).any() or torch.isinf(interference).any():

            print("loss contains NaN or Inf")

        optimizer.zero_grad()
        interference.backward()
        print(i, interference.item(), m.grad.item(), m.item())
        # torch.nn.utils.clip_grad_norm_(m, max_norm=1.0)
        if torch.isnan(m.grad).any() or torch.isinf(m.grad).any():
            print("grad contains NaN or Inf")
            break
        optimizer.step()
    mask = m
    diff = (m_copy - mask.cpu()).abs()
    print(mask.min().item(), mask.max().item(), mask.mean().item(), diff.max().item(), diff.mean().item())
    return (vec2 * (m0+mask)).detach().cpu()
