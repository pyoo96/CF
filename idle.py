import time
import argparse

parser = argparse.ArgumentParser(description='Evaluation of Models')
parser.add_argument('--hr', default=24, type=float)
args = parser.parse_args()

sleep_time = args.hr * 3600

print("Sleeping for %.2f hrs..." % (args.hr))
time.sleep(sleep_time)
print("Woke up!")

