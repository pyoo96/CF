import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models import SmallCNN
import argparse
from utils import *
from sam.sam import SAM
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--cpu-only', action='store_true')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--decay', default=0.0005, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=0.0, type=float)
parser.add_argument('--L1', default=0.0, type=float)
parser.add_argument('--L2', default=0.0, type=float)
parser.add_argument('--eps', default=0.0, type=float)
parser.add_argument('--b', default=0.0, type=float, help='b-flat minima')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--model-name', default='dummy.pth', type=str)
parser.add_argument('--center-point', default=0, type=float)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--print-freq', default=1, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--optimizer', default='default', choices=['default', 'sam'], type=str)

args = parser.parse_args()

assert(args.print_freq > 0)

# Dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST('/home/weebum/data/MNIST', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

testset = datasets.MNIST('/home/weebum/data/MNIST', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# Model, loss function and optimizer
net = SmallCNN()
criterion = nn.CrossEntropyLoss()
optimizer = None
if (args.optimizer == 'default'):
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
elif (args.optimizer == 'sam'):
    base_optimizer = torch.optim.SGD
    optimizer = SAM(net.parameters(), base_optimizer, lr=args.lr, momentum=args.momentum)

device = "cuda" if not args.cpu_only else "cpu"
net.to(device)

# Simple functions
l1_reg, l2_reg = lambda net: 0.0, lambda net: 0.0
lambda1, lambda2, center = args.L1, args.L2, args.center_point
if (lambda1 > 0.0):
    l1_reg = lambda net: lambda1 * sum(abs(p - center).sum() for p in net.parameters())
if (lambda2 > 0.0):
    l2_reg = lambda net: lambda2 * sum((p - center).pow(2).sum() for p in net.parameters())

l1_respective, l2_respective = lambda net_cur, net_ref: 0.0, lambda net_cur, net_ref: 0.0
lambda1, lambda2 = args.L1, args.L2
if (lambda1 > 0.0):
    l1_respective = lambda net_cur, net_ref: lambda1 * sum(abs(p - q).sum() for (p, q) in zip(net_cur.parameters(), net_ref.parameters()))
if (lambda2 > 0.0):
    l2_respective = lambda net_cur, net_ref: lambda2 * sum((p - q).pow(2).sum() for (p, q) in zip(net_cur.parameters(), net_ref.parameters()))

enforce_box = lambda net, eps: None
if (args.eps > 0.0):
    enforce_box = lambda net, eps: list(map(lambda p: p.data.clamp_(-eps, eps), net.parameters()))

perturb = lambda net, b: None
if (args.b > 0.0):
    perturb = lambda net, b: [p.data.add_(torch.empty(p.data.size()).uniform_(-b, b).to(p.device)) for p in net.parameters()]

# Statistics
num_param = sum(p.numel() for p in net.parameters())


# Train and test
if (args.optimizer == 'default'):
    for epoch in range(args.epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            perturb(net, args.b)
            outputs = net(images)
            loss = criterion(outputs, labels) + l1_reg(net) + l2_reg(net)
            loss.backward()
            optimizer.step()
            enforce_box(net, args.eps)

            running_loss += loss.item()

        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                test_loss += criterion(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        if (args.verbose and epoch % args.print_freq == 0):
            print("[%d/%d] Train loss: %.4f, Test loss: %.4f, Test acc: %.2f" % (epoch + 1, args.epochs, running_loss / len(trainloader), test_loss / len(testloader), 100 * correct / total))

elif (args.optimizer == 'sam'):
    for epoch in range(args.epochs):
        running_loss = 0
        for images, labels in trainloader:
            enable_running_stats(net)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels) + l1_reg(net) + l2_reg(net)
            running_loss += loss.item()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            disable_running_stats(net)
            outputs = net(images)
            loss = criterion(outputs, labels) + l1_reg(net) + l2_reg(net)
            loss.backward()
            optimizer.second_step(zero_grad=True)

        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                test_loss += criterion(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        if (args.verbose and epoch % args.print_freq == 0):
            print("[%d/%d] Train loss: %.4f, Test loss: %.4f, Test acc: %.2f" % (epoch + 1, args.epochs, running_loss / len(trainloader), test_loss / len(testloader), 100 * correct / total))

print("Train loss: %.4f, Test loss: %.4f, Test acc: %.2f" % (running_loss / len(trainloader), test_loss / len(testloader), 100 * correct / total))

# Remove coefficients for l1_reg and l2_reg
l1_reg = lambda net: sum(abs(p - center).sum() for p in net.parameters())
l2_reg = lambda net: sum((p - center).pow(2).sum() for p in net.parameters())
print("L1 norm: %.4f, L2 norm: %.4f" % (l1_reg(net), l2_reg(net)))

torch.save(net.state_dict(), args.model_name)



