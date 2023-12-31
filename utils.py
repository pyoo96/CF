import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import copy
from torch.nn.modules.batchnorm import _BatchNorm


#######################################
# Perturbation Related Functions
#######################################

def add_perturbation(net, radius, device='cuda'):
    """
    Add random perturbation to the network with L2-norm being equal to radius.
    """
    rand_orient = torch.randn(sum(p.numel() for p in net.parameters()), device=device)
    denom = rand_orient.norm() / radius
    rand_vector = rand_orient / denom
    start_idx = 0
    for param in net.parameters():
        num_param = param.numel()
        shape_param = param.data.shape
        perturbation = rand_vector[start_idx : start_idx + num_param]
        perturbation = perturbation.view(shape_param)
        param.data += perturbation


def calc_perturbed_loss(net, radius, num_samples, criterion, trainloader, testloader=None, detailed=False, pbar=None):
    """
    Calculates the average loss value L(θ + Δ) w.r.t. random perturbation |Δ| = radius.
    """
    device = next(net.parameters()).device
    theta = copy.deepcopy(net.state_dict())
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for _ in range(num_samples):
        with torch.no_grad():
            add_perturbation(net, radius, device)

            train_loss, train_acc = evaluate(net, trainloader, criterion, report_acc=detailed)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            if (detailed):
                test_loss, test_acc = evaluate(net, testloader, criterion, report_acc=True)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
        
        net.load_state_dict(theta)

        if (pbar):
            pbar.update()


    train_losses, test_losses, train_accs, test_accs = \
            np.array(train_losses), np.array(test_losses), np.array(train_accs), np.array(test_accs)

    if (not detailed):
        return (train_losses.mean(), train_losses.std())
    else:
        return (train_losses.mean(), train_losses.std(), train_accs.mean(), train_accs.std(), \
                test_losses.mean(), test_losses.std(), test_losses.mean(), test_losses.std())


def calc_solution_set_radius(net, trainloader, testloader=None, tau=1.0, max_radius=100., num_samples=50, num_iter=10, \
                                criterion=nn.CrossEntropyLoss(), use_tqdm=False, verbose=False):
    """
    Return |Δ| such that E[L(θ + Δ)] ≅ τ.

    max_radius: The maximum radius size that binary search will be performed on.
    num_samples: The number of perturbation samples to estimate the expected loss value.
    num_iter: The number of binary search iterations.
    """
    min_radius, min_loss, max_loss = 0., None, None

    pbar = None if not use_tqdm else tqdm(range(int(num_samples * (num_iter + 1))))

    min_loss, _ = evaluate(net, trainloader, criterion, report_acc=False)
    max_loss, _ = calc_perturbed_loss(net, max_radius, num_samples, criterion, trainloader, testloader, pbar=pbar)

    mid_radius, mid_loss = None, None

    anomaly_radii = []

    for i in range(num_iter):
        mid_radius = (min_radius + max_radius) / 2.0
        mid_loss, _ = calc_perturbed_loss(net, mid_radius, num_samples, criterion, trainloader, testloader, pbar=pbar)
        if (verbose):
            # (radius, loss) format
            print("[%d/%d] (τ = %.2f) lower, upper, mid = (r=%.3f, l=%.3f), (r=%.3f, l=%.3f), (r=%.3f, l=%.3f)" % \
                    (i + 1, num_iter, tau, min_radius, min_loss, max_radius, max_loss, mid_radius, mid_loss))
        if (mid_loss < tau):
            if (mid_loss < min_loss):
                anomaly_radii.append(mid_radius)
            min_radius, min_loss = mid_radius, mid_loss
        else:
            if (mid_loss < min_loss):
                anomaly_radii.append(mid_radius)
            max_radius, max_loss = mid_radius, mid_loss

    if (len(anomaly_radii) > 0):
        print("%d incident(s) with mid_loss < min_loss or mid_loss > high_loss have been occured." % (len(anomaly_radii)))
        print("radii = " + str(anomaly_radii))
        print("Consider increasing num_samples to get more accurate estimate of the expected loss.")

    if (pbar):
        pbar.close()

    return (min_radius + max_radius) / 2.0



#######################################
# Train, Test, Evaluate
#######################################

def evaluate(net, dataloader, criterion, report_acc=False):
    """
    Returns the network's loss (and accuracy) statistics for the given dataloader and criterion.
    if-else design so that a process does not have to encounter the branching condition inside the loop.
    """
    device = next(net.parameters()).device

    if (not report_acc):
        loss = 0.
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()

        loss /= len(dataloader)
        return (loss, -1.0)
    else:
        loss, correct, total = 0., 0, 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss, acc = loss / len(dataloader), 100 * correct / (total + 1e-8)
        return (loss, acc)



#######################################
# Functions for SAM
#######################################
"""
Codes from https://github.com/davda54/sam/blob/main/example/utility/bypass_bn.py.
"""
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

