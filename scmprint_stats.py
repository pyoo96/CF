import os
import sys
import argparse
import time

parser = argparse.ArgumentParser(description='Evaluation of Models')
parser.add_argument('--gpu', type=int)
parser.add_argument('--cpu', type=int)
parser.add_argument('--num-cpu', default=4, type=int)
parser.add_argument('--reps', default=10, type=int)
parser.add_argument('--weight-decay', default='', type=str)
parser.add_argument('--L1', default='', type=str)
parser.add_argument('--L2', default='', type=str)
parser.add_argument('--eps', default='', type=str)
parser.add_argument('--b', default='', type=str)
parser.add_argument('--center-point', '--center', default='0', type=str)
parser.add_argument('--sleep-hour', default='12', type=str)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--dataset', default='mnist', choices=['mnist'], type=str)
parser.add_argument('--tau', default=1.0, type=float)
parser.add_argument('--cores', action='store_true', help='Summarize only the important statistics')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--optimizer', default='default', choices=['default', 'sam'], type=str)

args = parser.parse_args()

if (args.cores):
    raise NotImplementedError("TBD")


cpu_list = "%d-%d" % (args.cpu, args.cpu + args.num_cpu)

prefix = "CUDA_VISIBLE_DEVICES=%d taskset --cpu-list %s " % (args.gpu, cpu_list)

config = "weights/%s/" % (args.dataset)
if (args.optimizer != 'default'):
    config += "%s_" % (args.optimizer)
if (args.L1):
    config += "L1_%s_" % (args.L1)
if (args.L2):
    config += "L2_%s_" % (args.L2)
if (args.eps):
    config += "eps_%s_" % (args.eps)
if (args.weight_decay):
    config += "wd_%s_" % (args.weight_decay)
if (args.center_point):
    config += "c_%s_" % (args.center_point)
if (args.b):
    config += "b_%s_" % (args.b)

if (not os.path.exists(config)):
    raise FileNotFoundError("directory with name %s does not exist." % (config))

suffix = ""
suffix += " --reps %d" % (args.reps)
suffix += " --dir-name %s" % (config)
suffix += " --dataset %s" % (args.dataset)
suffix += " --tau %.2f" % (args.tau)

if (args.overwrite):
    suffix += " --overwrite"
if (args.debug):
    suffix += " --debug"

if (args.debug):
    print("************************************Debugging Mode****************************************")

command = prefix + "python print_basic_stats.py" + suffix
print(command)
os.system(command)

print("Sleeping for %s hours..." % (args.sleep_hour))
secs = int(args.sleep_hour) * 3600
time.sleep(secs)
