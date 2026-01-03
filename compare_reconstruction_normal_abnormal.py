import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats

def remove_nan_inf(x):
    nan_mask = np.isnan(x)
    inf_mask = np.isinf(x)
    fill_value = np.nanmean(x[~nan_mask & ~inf_mask])
    if inf_mask.mean() > 0 or nan_mask.mean() > 0:
        print('total nan: ', np.mean(nan_mask))
        print('total inf: ', np.mean(inf_mask))
    x[nan_mask] = fill_value
    x[inf_mask] = fill_value
    return x
def calculate_reconstruction_l2(spec_true, spec_recon):
    spec_true = spec_true.reshape(-1)
    spec_true = remove_nan_inf(spec_true)
    spec_recon = spec_recon.reshape(-1)
    spec_recon = remove_nan_inf(spec_recon)
    return np.linalg.norm(spec_true - spec_recon)
def calculate_reconstruction_pearson(spec_true, spec_recon):
    spec_true = spec_true.reshape(-1)
    spec_true = remove_nan_inf(spec_true)
    spec_recon = spec_recon.reshape(-1)
    spec_recon = remove_nan_inf(spec_recon)
    return scipy.stats.pearsonr(spec_true, spec_recon)[0]

folder = '/data/netmit/sleep_lab/EEG_FM/TUAB/data/v3.0.1/edf/processed/train_with_spec'
files = os.listdir(folder)

all_patients_normal = {} 
all_patients_abnormal = {} 

for file in tqdm(files):
    patient_id = file.split('_')[0] + '_' + file.split('_')[1]
    with open(os.path.join(folder, file), 'rb') as f:
        data = pickle.load(f)
        is_normal = data['y'] == 0
        if is_normal:
            if patient_id not in all_patients_normal:
                all_patients_normal[patient_id] = []
            recon_l2 = calculate_reconstruction_l2(data['spec_true'], data['spec_recon'])
            recon_pearson = calculate_reconstruction_pearson(data['spec_true'], data['spec_recon'])
            all_patients_normal[patient_id].append([recon_l2, recon_pearson])
        else:
            if patient_id not in all_patients_abnormal:
                all_patients_abnormal[patient_id] = []
            recon_l2 = calculate_reconstruction_l2(data['spec_true'], data['spec_recon'])
            recon_pearson = calculate_reconstruction_pearson(data['spec_true'], data['spec_recon'])
            all_patients_abnormal[patient_id].append([recon_l2, recon_pearson])

l2_normal = [np.mean([item[0] for item in all_patients_normal[patient]]) for patient in all_patients_normal]
l2_abnormal = [np.mean([item[0] for item in all_patients_abnormal[patient]]) for patient in all_patients_abnormal]
pearson_normal = [np.mean([item[1] for item in all_patients_normal[patient]]) for patient in all_patients_normal]
pearson_abnormal = [np.mean([item[1] for item in all_patients_abnormal[patient]]) for patient in all_patients_abnormal]
print('--------------------------------')
print('Mean L2 Normal: ', np.mean(l2_normal))
print('Mean L2 Abnormal: ', np.mean(l2_abnormal))
print('p-value L2: ', scipy.stats.ttest_ind(l2_normal, l2_abnormal)[1])
print('--------------------------------')
print('Mean Pearson Normal: ', np.mean(pearson_normal))
print('Mean Pearson Abnormal: ', np.mean(pearson_abnormal))
print('p-value Pearson: ', scipy.stats.ttest_ind(pearson_normal, pearson_abnormal)[1])
print('--------------------------------')
print('Std L2 Normal: ', np.std(l2_normal))
print('Std L2 Abnormal: ', np.std(l2_abnormal))
print('Std Pearson Normal: ', np.std(pearson_normal))
print('Std Pearson Abnormal: ', np.std(pearson_abnormal))
print('--------------------------------')