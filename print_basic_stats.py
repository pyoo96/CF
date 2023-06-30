import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models import SmallCNN
import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--dir-name', '--dir', '--path', type=str)
parser.add_argument('--reps', type=int)
parser.add_argument('--batch-size', default=2**14, type=int)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--center-point', default=0, type=float)
parser.add_argument('--dataset', default='mnist', choices=['mnist'], type=str)
parser.add_argument('--tau', default=1.0, type=float)
parser.add_argument('--cores', action='store_true', help='Summarize only the important statistics')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

net = SmallCNN().cuda()
criterion = nn.CrossEntropyLoss()

center = args.center_point
l1_reg = lambda net: sum(abs(p - center).sum() for p in net.parameters())
l2_reg = lambda net: sum((p - center).pow(2).sum() for p in net.parameters())

# Dataloaders and their corresponding transformations
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('/home/weebum/data/MNIST', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

testset = datasets.MNIST('/home/weebum/data/MNIST', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

path_prefix = args.dir_name

# DEBUGGING CODE BLOCK---------------------------------------------
"""
Find argmin(5n) s.t. 5n <= 100 and calc_solution_set_radius(5n) == calc_solution_set_radius(5n+5)
"""
if (args.debug):
    model_name = path_prefix + '/0.pt'
    net.load_state_dict(torch.load(model_name))

    prev_radius = -1.0
    for num_samples in range(5, 101, 5):
        current_radius = calc_solution_set_radius(net, trainloader, tau=args.tau, num_samples=num_samples, num_iter=10, use_tqdm=False, verbose=True)
        print("num_samples: %d, prev_radius: %.3f, current_radius: %.3f" % (num_samples, prev_radius, current_radius))
        if (prev_radius == current_radius):
            print("argmin(5n): %d" % (num_samples - 5))
            break
        else:
            prev_radius = current_radius
    print ("argmin(5n): %d" % (num_samples - 5))
    raise Exception("breakpoint")
# -----------------------------------------------------------------

radii = []

for i in range(args.reps):
    model_name = path_prefix + '/%d.pt' % (i)
    print("Calculating the R(%.2f)'s radius w.r.t %s..." % (args.tau, model_name))
    if (args.debug):
        approx_radius = calc_solution_set_radius(net, trainloader, tau=args.tau, num_samples=50, num_iter=10, use_tqdm=True)
        radii.append(approx_radius)
        break
    approx_radius = calc_solution_set_radius(net, trainloader, tau=args.tau, use_tqdm=True)
    radii.append(approx_radius)

radii = np.array(radii)

raise Exception("breakpoint")

train_losses, test_losses, test_accs, L1_distances, L2_distances = [], [], [], [], []
for i in tqdm(range(args.reps)):
    net.load_state_dict(torch.load(path_prefix + '/%d.pt' % (i)))
    L1_distances.append(l1_reg(net).item())
    L2_distances.append(l2_reg(net).item())
    with torch.no_grad():
        train_loss = 0
        for images, labels in trainloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            train_loss += criterion(outputs, labels).item()
        train_losses.append(train_loss / len(trainloader))

        test_loss, correct, total = 0, 0, 0
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            test_loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_losses.append(test_loss / len(testloader))
        test_accs.append(100 * correct / total)


L1s, L2s = np.array(L1_distances), np.array(L2_distances)
L1_mean, L1_std, L2_mean, L2_std = L1s.mean(), L1s.std(), L2s.mean(), L2s.std()
train_losses, test_losses, test_accs = np.array(train_losses), np.array(test_losses), np.array(test_accs)
train_losses_mean, train_losses_std = train_losses.mean(), train_losses.std()
test_losses_mean, test_losses_std = test_losses.mean(), test_losses.std()
test_accs_mean, test_accs_std = test_accs.mean(), test_accs.std()



# Create a dictionary where keys are column names and values are the corresponding row values
data = {
    'config': args.dir_name,
    'L1 (mean)': L1_mean,
    'L1 (std)': L1_std,
    'L2 (mean)': L2_mean,
    'L2 (std)': L2_std,
    'train loss (mean)': train_losses_mean,
    'train loss (std)': train_losses_std,
    'test loss (mean)': test_losses_mean,
    'test loss (std)': test_losses_std,
    'test acc (mean)': test_accs_mean,
    'test acc (std)': test_accs_std
}

# Convert the dictionary to a dataframe
df = pd.DataFrame([data])

if os.path.exists('basic_stats.csv'):
    df_prev = pd.read_csv('basic_stats.csv')
    if ((df_prev['config'] == args.dir_name).any()):
        if args.overwrite:
            print("Config %s already exists. Overwriting the config..." % (args.dir_name))
            index = df.index[df['config'] == args.dir_name]
            df.loc[index] = data
        else:
            print("Config %s already exists. Skipping the config..." % (args.dir_name))
    else:
        df = pd.concat([df, df_prev])
df.to_csv('basic_stats.csv')

