import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time


'''pcagrad'''
def cagrad( grad_vec, num_tasks):
    """
        grad_vec: [num_tasks, dim]
        """
    cagrad_c = 0.5
    num_tasks = 2
    grads = grad_vec
    GG = grads.mm(grads.t()).cpu()
    scale = (torch.diag(GG) + 1e-4).sqrt().mean()
    GG = GG / scale.pow(2)
    Gg = GG.mean(1, keepdims=True)
    gg = Gg.mean(0, keepdims=True)
    '''avg gradient'''
    # g0=torch.mean(grads,dim=0,keepdim=True)
    # g0_norm=g0.norm()
    w = torch.zeros(num_tasks, 1, requires_grad=True)
    # if num_tasks == 50:
    #     w_opt = torch.optim.SGD([w], lr=50, momentum=0.5)
    # else:
    w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)
    c = (gg + 1e-4).sqrt() * cagrad_c

    w_best = None
    obj_best = np.inf
    for i in range(21):
        w_opt.zero_grad()
        ww = torch.softmax(w, 0)
        obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
        if obj.item() < obj_best:
            obj_best = obj.item()
            w_best = w.clone()
        if i < 20:
            obj.backward()
            w_opt.step()
    ww = torch.softmax(w_best, 0)
    gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()

    lmbda = c.view(-1) / (gw_norm + 1e-4)
    g = ((1 / num_tasks + ww * lmbda).view(
        -1, 1).to(grads.device) * grads).sum(0) / (1 + cagrad_c ** 2)
    return g

def CAgrad_backward(grads, shapes, has_grads):
    '''cagrad'''
    ca_grad, num_task = copy.deepcopy(grads), len(grads)
    total_grad = torch.stack(ca_grad)
    g = cagrad(total_grad, num_tasks=num_task)
    merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
    merged_grad = g
    return merged_grad

def PCGrad_backward(grads, shapes, has_grads):
    '''pcagrad'''
    pc_grad, num_task = copy.deepcopy(grads), len(grads)
    total_grad = torch.stack(pc_grad)
    g = cagrad(total_grad, num_tasks=num_task)
    merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
    merged_grad = g
    return merged_grad

