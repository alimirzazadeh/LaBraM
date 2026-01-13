import os
import pickle
from ipdb import set_trace as bp
## multithread this 
from multiprocessing import Pool
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
def check_file(file):
    if not file.endswith('.pkl'):
        return (file, None)  # Not a pkl file, skip
    try:
        with open(os.path.join(root, file), 'rb') as f:
            sample = pickle.load(f)
            if 'spec_true_bw1' not in sample:
                return (file, False)  # Unsuccessful
            else:
                return (file, True)  # Successful
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return (file, False)  # Treat errors as unsuccessful

with Pool(processes=10) as pool:
    results = pool.map(check_file, files)
    
unsuccessful_files = [file for file, status in results if status is False]
successful_files = [file for file, status in results if status is True]

print(f'Number of unsuccessful files: {len(unsuccessful_files)}')
print(f'Number of successful files: {len(successful_files)}')

bp() 
for file in unsuccessful_files:
    os.remove(os.path.join(root, file))