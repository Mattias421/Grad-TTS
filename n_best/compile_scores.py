import os
import numpy as np
import yaml
import argparse
from tqdm import tqdm

# Define the directory containing the YAML files
parser = argparse.ArgumentParser(description="Process a directory path.")
parser.add_argument(
    "--directory_path",
    type=str,
    required=True,
    help="The directory path to process."
)
parser.add_argument('-N', type=int)
parser.add_argument('-I', type=int)

args = parser.parse_args()

directory_path = args.directory_path


score_list = np.zeros((args.I, args.N))

# Loop through files in the directory
for root, _, files in tqdm(os.walk(directory_path)):

    if '.hydra' in root:
        continue

    for filename in files:
        if filename.endswith('.yaml'):
            file_path = os.path.join(root, filename)
            
            # Read the YAML file
            with open(file_path, 'r') as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)

                score_list[yaml_data['i'], yaml_data['n']] = yaml_data['diffusion_score']
                

name = directory_path.split('/')[-1]

np.save(f'diffusion_scores/{name}.npy', score_list)