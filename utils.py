import torch
import torch.nn as nn
import numpy as np


def add_perturbation(net, radius):
    rand_orient = torch.randn(sum(p.numel() for p in net.parameters()), device=net.device)
    denom = random_orient.norm() / radius
    rand_vector = rand_orient / denom
    start_idx = 0
    for param in net.parameters():
        num_param = param.numel()
        shape_param = param.data.shape
        perturbation = rand_vector[start_idx : start_idx + num_param]
        perturbation = perturbation.view(shape_param)
        param.data += perturbation

    return

