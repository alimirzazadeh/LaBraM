from tqdm import tqdm
import os
import pickle
from ipdb import set_trace as bp

import sys 

if sys.argv[1] == 'test':
    root = "/data/netmit/sleep_lab/EEG_FM/TUAB/data/v3.0.1/edf/processed/test_with_spec"
elif sys.argv[1] == 'train':
    root = "/data/netmit/sleep_lab/EEG_FM/TUAB/data/v3.0.1/edf/processed/train_with_spec"
elif sys.argv[1] == 'val':
    root = "/data/netmit/sleep_lab/EEG_FM/TUAB/data/v3.0.1/edf/processed/val_with_spec"
else:
    raise ValueError("Invalid dataset")

files = os.listdir(root)
unsuccessful_files = []
successful_files = []
for file in tqdm(files):
    if file.endswith('.pkl'):
        with open(os.path.join(root, file), 'rb') as f:
            sample = pickle.load(f)
            if 'spec_true_bw1' not in sample:
                unsuccessful_files.append(file)
                ## remove the file 
                print(f'Removing file: {file}')
                
            else:
                successful_files.append(file)

print(f'Number of unsuccessful files: {len(unsuccessful_files)}')
print(f'Number of successful files: {len(successful_files)}')

bp() 
for file in unsuccessful_files:
    os.remove(os.path.join(root, file))