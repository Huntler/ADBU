import argparse
import os

parser = argparse.ArgumentParser(description="This script trains multiple models, one for each configuration file specified.")
parser.add_argument("--folder", dest="config_folder", help="Set path to config folder.")
args = parser.parse_args()

config_files = os.listdir(args.config_folder)
config_files.sort()
print(f"Found {len(config_files)} configurations.")
for config in config_files:
    print(f"Starting training for {config}.")
    path = os.path.join(args.config_folder, config)
    os.system(f"python main.py --config {path}")
    print("Training ended.\n")