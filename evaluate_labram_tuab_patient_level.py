# --------------------------------------------------------
# Evaluation script for LaBraM on TUAB dataset
# Computes AUROC at both sample level and patient level
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from einops import rearrange
from timm.models import create_model
import utils
import modeling_finetune
from pyhealth.metrics import binary_metrics_fn
from ipdb import set_trace as bp

def load_model_checkpoint(checkpoint_path, device):
    """Load the finetuned model from checkpoint"""
    # Model configuration (matching the finetuning setup)
    model = create_model(
        'labram_base_patch200_200',
        pretrained=False,
        num_classes=1,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=True,
        use_abs_pos_emb=False,
        init_values=0.1,
        qkv_bias=True,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict (handle different checkpoint formats)
    # Check for common checkpoint keys (from finetuning script)
    checkpoint_model = None
    model_key_options = ['model', 'module', 'model_ema', 'model_without_ddp']
    
    for model_key in model_key_options:
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print(f"Load state_dict by model_key = {model_key}")
            break
    
    if checkpoint_model is None:
        # If no standard key found, check if it's already a state dict
        if isinstance(checkpoint, dict) and any('weight' in k or 'bias' in k for k in checkpoint.keys()):
            checkpoint_model = checkpoint
        else:
            raise ValueError(f"Could not find model state dict in checkpoint. Available keys: {checkpoint.keys()}")
    
    # Remove keys that might cause issues
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    
    # Remove relative position index (will be recomputed)
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)
    
    # Load state dict
    utils.load_state_dict(model, checkpoint_model, prefix='')
    
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(model, data_loader, device, ch_names):
    """Run inference and collect predictions, labels, and patient IDs"""
    # Only use input_chans if the model has positional embeddings
    # Otherwise, pass None to avoid indexing None
    if hasattr(model, 'pos_embed') and model.pos_embed is not None:
        input_chans = utils.get_input_chans(ch_names)
    else:
        input_chans = None
    
    all_predictions = []
    all_labels = []
    all_pids = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 100 == 0:
                print(f"Processing batch {batch_idx}/{len(data_loader)}")
            
            # Handle different return formats
            if isinstance(batch[1], dict):
                # When return_pid=True, batch[1] is a dict with 'y', 'pid', 'session'
                # DataLoader batches dicts, so each value is already a list/tensor
                EEG = batch[0]
                labels = batch[1]['y']
                pids = batch[1]['pid']
                
                # Convert pids to list if needed
                if isinstance(pids, torch.Tensor):
                    pids = pids.cpu().tolist()
                elif not isinstance(pids, list):
                    pids = list(pids)
            else:
                # Fallback: extract pid from filename if available
                EEG = batch[0]
                labels = batch[1]
                # Get actual indices for this batch
                batch_size_actual = len(labels)
                start_idx = batch_idx * data_loader.batch_size
                end_idx = min(start_idx + batch_size_actual, len(data_loader.dataset))
                actual_indices = list(range(start_idx, end_idx))
                # Try to get pids from dataset files
                pids = [data_loader.dataset.files[i].split("_")[0] for i in actual_indices]
            
            EEG = EEG.float().to(device, non_blocking=True) / 100
            EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
            
            # Run inference
            with torch.cuda.amp.autocast():
                output = model(EEG, input_chans=input_chans)
            
            # Apply sigmoid for binary classification
            predictions = torch.sigmoid(output).cpu().numpy()
            
            # Convert labels to numpy
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            else:
                labels = np.array(labels)
            
            # Ensure labels are 1D
            if labels.ndim > 1:
                labels = labels.flatten()
            
            all_predictions.append(predictions)
            all_labels.append(labels)
            all_pids.extend(pids)
    
    # Concatenate all predictions and labels
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Flatten if needed
    if all_predictions.ndim > 1:
        all_predictions = all_predictions.flatten()
    if all_labels.ndim > 1:
        all_labels = all_labels.flatten()
    
    # Ensure same length
    assert len(all_predictions) == len(all_labels) == len(all_pids), \
        f"Mismatch in lengths: predictions={len(all_predictions)}, labels={len(all_labels)}, pids={len(all_pids)}"
    
    return all_predictions, all_labels, all_pids


def compute_patient_level_metrics(predictions, labels, pids):
    """Average predictions per patient and compute metrics"""
    # Group predictions and labels by patient ID
    patient_dict = defaultdict(lambda: {'predictions': [], 'labels': []})
    
    for pred, label, pid in zip(predictions, labels, pids):
        patient_dict[pid]['predictions'].append(pred)
        patient_dict[pid]['labels'].append(label)
    
    # Average predictions per patient
    patient_predictions = []
    patient_labels = []
    patient_ids = []
    
    for pid, data in patient_dict.items():
        # Average predictions for this patient
        avg_pred = np.mean(data['predictions'])
        # Labels should be the same for all samples from the same patient
        labels_for_patient = np.array(data['labels'])
        if not np.all(labels_for_patient == labels_for_patient[0]):
            print(f"Warning: Patient {pid} has inconsistent labels: {labels_for_patient}")
        label = labels_for_patient[0]  # Take first label
        
        patient_predictions.append(avg_pred)
        patient_labels.append(label)
        patient_ids.append(pid)
    
    patient_predictions = np.array(patient_predictions)
    patient_labels = np.array(patient_labels)
    
    return patient_predictions, patient_labels, patient_ids, patient_dict


def main():
    # Configuration
    checkpoint_path = "checkpoints/finetune_tuab_base_bs512/checkpoint-best.pth"
    dataset_root = "/data/netmit/sleep_lab/EEG_FM/TUAB/data/v3.0.1/edf/processed"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    
    # Channel names (matching finetuning setup)
    ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 
                'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF',
                'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 
                'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 
                'EEG T1-REF', 'EEG T2-REF']
    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
    
    print("=" * 60)
    print("LaBraM TUAB Patient-Level Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = load_model_checkpoint(checkpoint_path, device)
    print("Model loaded successfully!")
    
    # Load test dataset with return_pid=True to get patient IDs
    print(f"\nLoading TUAB test dataset from {dataset_root}...")
    _, test_dataset_with_pid, _ = utils.prepare_TUAB_dataset(dataset_root, return_pid=True)

    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset_with_pid,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
        shuffle=False
    )
    
    print(f"Test dataset size: {len(test_dataset_with_pid)}")
    print(f"Dataset return_pid: {test_dataset_with_pid.return_pid}")
    
    # Run evaluation
    print("\nRunning evaluation...")
    predictions, labels, pids = evaluate_model(model, test_loader, device, ch_names)
    
    print(f"\nTotal samples: {len(predictions)}")
    print(f"Unique patients: {len(set(pids))}")
    
    # Compute sample-level AUROC
    print("\n" + "=" * 60)
    print("Sample-Level Metrics")
    print("=" * 60)
    sample_metrics = binary_metrics_fn(
        labels,
        predictions,
        metrics=["roc_auc", "pr_auc", "accuracy", "balanced_accuracy"],
        threshold=0.5
    )
    
    print(f"Sample-Level AUROC: {sample_metrics['roc_auc']:.4f}")
    print(f"Sample-Level AUPRC: {sample_metrics['pr_auc']:.4f}")
    print(f"Sample-Level Accuracy: {sample_metrics['accuracy']:.4f}")
    print(f"Sample-Level Balanced Accuracy: {sample_metrics['balanced_accuracy']:.4f}")
    
    # Compute patient-level metrics
    print("\n" + "=" * 60)
    print("Patient-Level Metrics")
    print("=" * 60)
    patient_predictions, patient_labels, patient_ids, patient_dict = compute_patient_level_metrics(
        predictions, labels, pids
    )
    
    print(f"Total patients: {len(patient_predictions)}")
    
    # Compute patient-level AUROC
    patient_metrics = binary_metrics_fn(
        patient_labels,
        patient_predictions,
        metrics=["roc_auc", "pr_auc", "accuracy", "balanced_accuracy"],
        threshold=0.5
    )
    
    print(f"Patient-Level AUROC: {patient_metrics['roc_auc']:.4f}")
    print(f"Patient-Level AUPRC: {patient_metrics['pr_auc']:.4f}")
    print(f"Patient-Level Accuracy: {patient_metrics['accuracy']:.4f}")
    print(f"Patient-Level Balanced Accuracy: {patient_metrics['balanced_accuracy']:.4f}")
    
    # Print patient-level statistics
    print("\n" + "=" * 60)
    print("Patient-Level Statistics")
    print("=" * 60)
    print(f"Number of patients: {len(patient_dict)}")
    samples_per_patient = [len(data['predictions']) for data in patient_dict.values()]
    print(f"Average samples per patient: {np.mean(samples_per_patient):.2f}")
    print(f"Min samples per patient: {np.min(samples_per_patient)}")
    print(f"Max samples per patient: {np.max(samples_per_patient)}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

