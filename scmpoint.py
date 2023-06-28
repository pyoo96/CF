import os
import sys
import argparse
import time

parser = argparse.ArgumentParser(description='Evaluation of Models')
parser.add_argument('--gpu', type=int)
parser.add_argument('--cpu', type=int)
parser.add_argument('--num-cpu', default=4, type=int)
parser.add_argument('--cpu-only', action='store_true')
parser.add_argument('--reps', default=10, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--weight-decay', default='', type=str)
parser.add_argument('--L1', default='', type=str)
parser.add_argument('--L2', default='', type=str)
parser.add_argument('--eps', default='', type=str)
parser.add_argument('--b', default='', type=str)
parser.add_argument('--dataset', default='mnist', choices=['mnist'], type=str)
parser.add_argument('--center-point', '--center', default='0', type=str)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--print-freq', '--freq', default=10, type=int)
parser.add_argument('--sleep-hour', default='24', type=str)
parser.add_argument('--adjust-init', action='store_true')


args = parser.parse_args()

cpu_list = "%d-%d" % (args.cpu, args.cpu + args.num_cpu)

prefix = "CUDA_VISIBLE_DEVICES=%d taskset --cpu-list %s " % (args.gpu, cpu_list)
suffix = ""
if (args.cpu_only):
    suffix += " --cpu-only"

config = "weights/%s/" % (args.dataset)
conf_prefix_idx = len(config)
if (args.L1):
    suffix += " --L1 %s" % (args.L1)
    config += "L1_%s_" % (args.L1)
if (args.L2):
    suffix += " --L2 %s" % (args.L2)
    config += "L2_%s_" % (args.L2)
if (args.eps):
    suffix += " --eps %s" % (args.eps)
    config += "eps_%s_" % (args.eps)
if (args.weight_decay):
    suffix += " --weight-decay %s" % (args.weight_decay)
    config += "wd_%s_" % (args.weight_decay)
if (args.center_point):
    suffix += " --center-point %s" % (args.center_point)
    config += "c_%s_" % (args.center_point)
if (args.b):
    suffix += " --b %s" % (args.b)
    config += "b_%s_" % (args.b)


os.makedirs(config, exist_ok=True)

suffix += " --epochs %d" % (args.epochs)
if (not args.batch_size == 64):
    suffix += " --batch-size %d" % (args.batch_size)
if (args.adjust_init):
    raise NotImplementedError("Use initialization schemes with mean 0 (Performs severely bad.)")


print("Experiment Config:%s" % (suffix))
suffix += " 2>&1 | tee -a logs/mnist/%s.txt" % (config[conf_prefix_idx:])

start_time = time.time()
for i in range(args.reps):
    save_pth = config + "/%d.pt" % (i)
    command = prefix + "python point_exp.py --model-name %s" % (save_pth) + suffix
    print(command)
    os.system(command)
    if (i == 0):
        print("Elapsed Time per Trial: %.2f min" % ((time.time() - start_time) / 60.0))

command = "python idle.py --hr %s" % (args.sleep_hour)
print(command)
os.system(command)

"""
eps = float(sys.argv[1])
tau = float(sys.argv[2])
std = float(sys.argv[3])
xi = float(sys.argv[4])
for i in range(3):
    command = "python cifar.py --EINS 1 --eps %.2f --tau %.2f --std %.2f --xi %.2f --epoch 200 --no-jsd" % (eps, tau, std, xi)
    print(command)
    os.system(command)
"""
"""
start_eps = int(sys.argv[2])
for i in range(4):
    command = "taskset --cpu-list %s python cifar.py --EINS '' --eps %.1f --epoch 400" % (cpu_list, start_eps + float(i))
    print(command)
    os.system(command)
"""

"""
clf = str(sys.argv[3])
cuda = int(sys.argv[4])
"""
"""
for _ in range(3):
    command = "CUDA_VISIBLE_DEVICES=1 taskset --cpu-list 0,1,2,3,4,5 python cifar.py --epoch 200"
    print(command)
    os.system(command)
"""
"""
params = [(0.5, 1.0), (1.0, 2.0), (1.5, 3.0), (2.0, 4.0),
            (0.25, 1.0), (0.5, 2.0), (0.75, 3.0), (1.0, 4.0)]

for p in params:
    command = "taskset --cpu-list %s python main.py --eps 8.0 --std %s --tau %s" % (cpu_list, p[0], p[1])
    print(command)
    os.system(command)
"""
