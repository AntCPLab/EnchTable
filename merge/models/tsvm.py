import copy
import torch

from models.utils import obtain_delta


def tsvm_merge(task_model, taske_pre_model, safety_model, safety_pre_model, alpha=1.0, mode=1):
    task_vector = obtain_delta(task_model, taske_pre_model)
    safety_vector = obtain_delta(safety_model, safety_pre_model)

    if mode == 1:
        merge_tv = compute_and_sum_svd_mem_reduction([task_vector, safety_vector])
    elif mode == 2:
        merge_tv = compute_and_sum_svd_mem_reduction_learnable2([task_vector, safety_vector])
    elif mode == 3:
        merge_tv = compute_and_sum_svd_mem_reduction_learnable3([task_vector, safety_vector])
    else:
        raise NotImplementedError
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


def compute_and_sum_svd_mem_reduction(task_vectors, saliency_maps=None):
    """
    Computes the Singular Value Decomposition (SVD) for each vector in the task_vectors,
    reduces the dimensionality of the vectors based on the sv_reduction factor, and concatenate
    the low-rank matrices. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    Computation of the SVD is performed also for the second operation.

    Args:
        task_vectors (list): A list of task vector objects, where each object contains a
                             dictionary of vectors.
        config (object): Configuration object containing the following attributes:
                         - DATASETS (list): List of datasets.
                         - device (torch.device): The device to perform computations on.

    Returns:
        dict: A dictionary containing the new vectors after SVD computation and merging.
    """
    sv_reduction = 1 / len(task_vectors)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0]:
            if task_vectors[0][key] is None or task_vectors[1][key] is None:
                continue

            new_vector[key] = {}
            for i, task_vector in enumerate(task_vectors
            ):
                vec = task_vector[key].to(device)

                if (
                    len(task_vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    reduced_index_s = int(s.shape[0] * sv_reduction)

                    # select only the first reduced_index_s columns of u and place them
                    sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                        :, :reduced_index_s
                    ]
                    sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                        :reduced_index_s
                    ]
                    # select only the first reduced_index_s rows of v and place them
                    sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                        :reduced_index_s, :
                    ]

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector[key].shape) == 2 and "text_projection" not in key:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        u_u,
                        v_u,
                        torch.diag(sum_s),
                        u_v,
                        v_v,
                    )
                )
            new_vector[key] = new_vector[key].cpu()

    return new_vector


def compute_and_sum_svd_mem_reduction_learnable2(task_vectors, saliency_maps=None):
    sv_reduction = 1 / len(task_vectors)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computing SVD...")
    new_vector = {}
    n_steps = 20
    lr = 1e-6
    beta = 1.0
    for key in task_vectors[0]:
        if task_vectors[0][key] is None or task_vectors[1][key] is None:
            continue

        new_vector[key] = {}
        vectors = []
        for i, task_vector in enumerate(task_vectors
        ):
            vec = task_vector[key].to(device)

            if (len(task_vector[key].shape) == 2 and "text_projection" not in key):
                vectors.append(vec)

            else:
                if i == 0:
                    new_vector[key] = vec.clone()
                else:
                    new_vector[key] += (vec - new_vector[key]) / (i + 1)

        if len(task_vector[key].shape) == 2 and "text_projection" not in key:
            ratio1 = ratio2 = 0.5
            u1, s1, v1 = torch.linalg.svd(vectors[0], full_matrices=False)
            reduced_index_1 = int(s1.shape[0] * ratio1)
            u2, s2, v2 = torch.linalg.svd(vectors[1], full_matrices=False)
            reduced_index_2 = int(s2.shape[0] * ratio2)
            sum_u = torch.cat([u1[:, :reduced_index_1], u2[:, :reduced_index_2]], dim=1)
            sum_s = torch.cat([s1[:reduced_index_1], s2[:reduced_index_2]], dim=0)
            sum_v = torch.cat([v1[:reduced_index_1, :], v2[:reduced_index_2, :]], dim=0)
            sum_u.requires_grad_(True)
            sum_s.requires_grad_(True)
            sum_v.requires_grad_(True)

            parameter = [sum_u, sum_s, sum_v]
            optimizer = torch.optim.Adam(parameter, lr=lr)
            parameter_copy = copy.deepcopy(parameter)
            for step_i in range(n_steps):
                optimizer.zero_grad()
                loss_interference = compute_interference_2(parameter[0], parameter[1], parameter[2])
                loss_reg = torch.nn.functional.mse_loss(parameter[0], parameter_copy[0].detach()) + torch.nn.functional.mse_loss(parameter[1], parameter_copy[1].detach()) + torch.nn.functional.mse_loss(parameter[2], parameter_copy[2].detach())
                loss = loss_interference + loss_reg * beta
                loss.backward() # regularization 
                torch.nn.utils.clip_grad_norm_(parameter, max_norm=1.0)
                # print(0, parameter[0].grad)
                # print(1, parameter[1].grad)
                print(key, loss.item(), loss_interference.item(), loss_reg.item())
                optimizer.step()
            # needed ? or just
            # new_vector[key] = torch.linalg.multi_dot((sum_u, torch.diag(sum_s), sum_v,))
            with torch.no_grad():
                sum_u, sum_s, sum_v = parameter
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        u_u,
                        v_u,
                        torch.diag(sum_s),
                        u_v,
                        v_v,
                    )
                )
            del parameter, parameter_copy
            # torch.cuda.empty_cache()
        new_vector[key] = new_vector[key].cpu()

    return new_vector

def compute_and_sum_svd_mem_reduction_learnable3(task_vectors, saliency_maps=None):
    sv_reduction = 1 / len(task_vectors)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computing SVD...")
    new_vector = {}
    n_steps = 20
    lr = 1e-6
    beta = 1.0
    for key in task_vectors[0]:
        if task_vectors[0][key] is None or task_vectors[1][key] is None:
            continue

        new_vector[key] = {}
        vectors = []
        for i, task_vector in enumerate(task_vectors
        ):
            vec = task_vector[key].to(device)

            if (len(task_vector[key].shape) == 2 and "text_projection" not in key):
                vectors.append(vec)

            else:
                if i == 0:
                    new_vector[key] = vec.clone()
                else:
                    new_vector[key] += (vec - new_vector[key]) / (i + 1)

        if len(task_vector[key].shape) == 2 and "text_projection" not in key:
            ratio1 = ratio2 = 0.5
            u1, s1, v1 = torch.linalg.svd(vectors[0], full_matrices=False)
            reduced_index_1 = int(s1.shape[0] * ratio1)
            u2, s2, v2 = torch.linalg.svd(vectors[1], full_matrices=False)
            reduced_index_2 = int(s2.shape[0] * ratio2)
            sum_u = torch.cat([u1[:, :reduced_index_1], u2[:, :reduced_index_2]], dim=1)
            sum_s = torch.cat([s1[:reduced_index_1], s2[:reduced_index_2]], dim=0)
            sum_v = torch.cat([v1[:reduced_index_1, :], v2[:reduced_index_2, :]], dim=0)
            sum_u.requires_grad_(True)
            sum_s.requires_grad_(True)
            sum_v.requires_grad_(True)

            parameter = [sum_u, sum_s, sum_v]
            optimizer = torch.optim.Adam(parameter, lr=lr)
            parameter_copy = copy.deepcopy(parameter)
            for step_i in range(n_steps):
                optimizer.zero_grad()
                loss_interference = compute_interference_2(parameter[0], parameter[1], parameter[2])
                loss_reg = torch.nn.functional.mse_loss(parameter[0], parameter_copy[0].detach()) + torch.nn.functional.mse_loss(parameter[1], parameter_copy[1].detach()) + torch.nn.functional.mse_loss(parameter[2], parameter_copy[2].detach())
                loss = loss_interference + loss_reg * beta
                loss.backward() # regularization ?
                torch.nn.utils.clip_grad_norm_(parameter, max_norm=1.0)
                # print(0, parameter[0].grad)
                # print(1, parameter[1].grad)
                print(key, loss.item(), loss_interference.item(), loss_reg.item())
                optimizer.step()
            with torch.no_grad():
                sum_u, sum_s, sum_v = parameter
                new_vector[key] = torch.linalg.multi_dot((sum_u, torch.diag(sum_s), sum_v,))
            del parameter, parameter_copy
            # torch.cuda.empty_cache()
        new_vector[key] = new_vector[key].cpu()

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
    interference = torch.norm(result_matrix, p=1)

    return interference, (sum_u, sum_s, sum_v)

def compute_interference_2(sum_u, sum_s, sum_v):

    UUt = torch.matmul(sum_u.t(), sum_u)
    VtV = torch.matmul(sum_v, sum_v.t())

    # 计算 U^T U - I 和 V^T V - I
    UUt_minus_I = UUt - torch.eye(sum_s.shape[0], device=sum_u.device)
    VtV_minus_I = VtV - torch.eye(sum_s.shape[0], device=sum_u.device)

    # 计算 (U^T U - I) Sigma (V^T V - I)
    result_matrix = torch.matmul(torch.matmul(UUt_minus_I, torch.diag(sum_s)), VtV_minus_I)
    # print(result_matrix)

    # 计算结果矩阵的 1-范数
    interference = torch.norm(result_matrix, p=1)

    return interference