# --------------------------------------------------------
# Evaluation script for LaBraM on TUAB dataset
# --------------------------------------------------------

import torch
import numpy as np
from einops import rearrange
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score
from timm.models import create_model
import utils
import modeling_finetune  # Required to register the model with timm
from run_class_finetuning import get_dataset


def load_model_checkpoint(checkpoint_path, device):
    """Load the finetuned model from checkpoint using auto_load_model"""
    # Load checkpoint to get the saved args (contains model configuration)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    saved_args = checkpoint['args']
    print(f"Loading model with configuration from checkpoint:")
    print(f"  - abs_pos_emb: {getattr(saved_args, 'abs_pos_emb', False)}")
    print(f"  - qkv_bias: {getattr(saved_args, 'qkv_bias', True)}")
    print(f"  - rel_pos_bias: {getattr(saved_args, 'rel_pos_bias', True)}")
    
    # Create model with configuration from checkpoint
    model = create_model(
        getattr(saved_args, 'model', 'labram_base_patch200_200'),
        pretrained=False,
        num_classes=getattr(saved_args, 'nb_classes', 1),
        drop_rate=getattr(saved_args, 'drop', 0.0),
        drop_path_rate=getattr(saved_args, 'drop_path', 0.1),
        attn_drop_rate=getattr(saved_args, 'attn_drop_rate', 0.0),
        drop_block_rate=None,
        use_mean_pooling=getattr(saved_args, 'use_mean_pooling', True),
        init_scale=getattr(saved_args, 'init_scale', 0.001),
        use_rel_pos_bias=getattr(saved_args, 'rel_pos_bias', True),
        use_abs_pos_emb=getattr(saved_args, 'abs_pos_emb', False),
        init_values=getattr(saved_args, 'layer_scale_init_value', 0.1),
        qkv_bias=getattr(saved_args, 'qkv_bias', True),
    )
    
    model.to(device)
    model_without_ddp = model  # For evaluation, model is not wrapped in DDP
    
    # Create minimal args object for auto_load_model
    class Args:
        output_dir = 'checkpoints/finetune_tuab_base_bs512'
        resume = checkpoint_path
        auto_resume = False
        enable_deepspeed = False
        model_ema = False
    
    args = Args()
    
    # Create dummy optimizer and loss_scaler objects (auto_load_model may try to load them)
    class DummyOptimizer:
        def load_state_dict(self, state_dict):
            pass  # Do nothing for evaluation
    
    class DummyLossScaler:
        def load_state_dict(self, state_dict):
            pass  # Do nothing for evaluation
    
    # Use the same loading function as training script
    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=DummyOptimizer(),
        loss_scaler=DummyLossScaler(),
        model_ema=None
    )
    
    model.eval()
    return model


def calculate_metrics(predictions, targets, metrics, pid=None, threshold=0.5):
    """
    Calculate metrics from predictions and targets.
    
    Args:
        predictions: numpy array of predictions (probabilities for binary classification)
        targets: numpy array of true labels
        metrics: list of metric names to calculate
        pid: optional numpy array of patient IDs (same length as predictions/targets)
        threshold: threshold for binary classification (default 0.5)
    
    Returns:
        dict with calculated metrics
    """
    # If pid is provided, average predictions per patient
    if pid is not None:
        # Group predictions and targets by patient ID
        patient_dict = defaultdict(lambda: {'predictions': [], 'targets': []})
        
        for pred, target, p in zip(predictions, targets, pid):
            patient_dict[p]['predictions'].append(pred)
            patient_dict[p]['targets'].append(target)
        
        # Average predictions per patient and get target (assumed same for all samples from same patient)
        patient_predictions = []
        patient_targets = []
        
        for p, data in patient_dict.items():
            avg_pred = np.mean(data['predictions'])
            # Target should be the same for all samples from the same patient
            target_val = data['targets'][0]  # Take first target (all should be same)
            patient_predictions.append(avg_pred)
            patient_targets.append(target_val)
        
        predictions = np.array(patient_predictions)
        targets = np.array(patient_targets)
    
    results = {}
    
    # Flatten arrays to ensure 1D
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Check if we have valid targets (not all 0 or all 1) for AUROC
    unique_targets = np.unique(targets)
    has_both_classes = len(unique_targets) > 1
    
    # Calculate requested metrics
    if 'roc_auc' in metrics:
        if has_both_classes:
            results['roc_auc'] = roc_auc_score(targets, predictions)
        else:
            results['roc_auc'] = 0.0
    
    if 'pr_auc' in metrics:
        if has_both_classes:
            results['pr_auc'] = average_precision_score(targets, predictions)
        else:
            results['pr_auc'] = 0.0
    
    if 'accuracy' in metrics:
        pred_binary = (predictions >= threshold).astype(int)
        results['accuracy'] = accuracy_score(targets, pred_binary)
    
    if 'balanced_accuracy' in metrics:
        pred_binary = (predictions >= threshold).astype(int)
        results['balanced_accuracy'] = balanced_accuracy_score(targets, pred_binary)
    
    return results


def evaluate_model(data_loader, model, device, ch_names=None):
    """Custom evaluation function that loops through dataloader and calculates metrics"""
    model.eval()
    
    # Setup input_chans if needed
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    
    # Loss function for binary classification
    criterion = torch.nn.BCEWithLogitsLoss()
    
    all_predictions = []
    all_targets = []
    all_pids = []
    total_loss = 0.0
    num_samples = 0
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 100 == 0:
                print(f"Processing batch {batch_idx}/{len(data_loader)}")
            
            EEG = batch[0]
            target = batch[-1]['y']
            pid = batch[-1]['pid']
            session = batch[-1]['session']
            
            # Preprocess data (matching engine_for_finetuning.py)
            EEG = EEG.float().to(device, non_blocking=True) / 100
            EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
            target = target.to(device, non_blocking=True).float().unsqueeze(-1)
            
            # Forward pass
            with torch.cuda.amp.autocast():
                output = model(EEG, input_chans=input_chans)
                loss = criterion(output, target)
            
            # Apply sigmoid for binary classification
            predictions = torch.sigmoid(output).cpu().numpy()
            targets = target.cpu().numpy()
            
            # Convert pid to list if it's a tensor
            if isinstance(pid, torch.Tensor):
                pid = pid.cpu().tolist()
            elif not isinstance(pid, list):
                pid = list(pid)
            
            all_predictions.append(predictions)
            all_targets.append(targets)
            all_pids.extend(pid)
            total_loss += loss.item() * len(target)
            num_samples += len(target)
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_pids = np.array(all_pids) if all_pids else None
    
    # Calculate average loss
    avg_loss = total_loss / num_samples
    
    # Calculate metrics using our custom function
    metrics = ['roc_auc', 'pr_auc', 'accuracy', 'balanced_accuracy']
    results = calculate_metrics(all_predictions, all_targets, metrics, pid=None, threshold=0.5)
    results['loss'] = avg_loss
    
    return results


def main():
    # Configuration
    checkpoint_path = "checkpoints/finetune_tuab_base_bs512/checkpoint-49.pth" #checkpoint-49.pth" #checkpoint-best.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    
    print("=" * 60)
    print("LaBraM TUAB Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = load_model_checkpoint(checkpoint_path, device)
    print("Model loaded successfully!")
    
    # Load dataset using the same function as finetuning script
    class Args:
        dataset = 'TUAB'
        
    args = Args()
    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args, return_pid=True)
    
    print(f"\nTest dataset size: {len(dataset_test)}")
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
        shuffle=False
    )
    
    # Only pass ch_names if the model has pos_embed (needed for input_chans)
    # If use_abs_pos_emb=False, pos_embed is None and we shouldn't pass ch_names
    ch_names_to_use = None
    if hasattr(model, 'pos_embed') and model.pos_embed is not None:
        ch_names_to_use = ch_names
    
    # Run evaluation
    print("\nRunning evaluation...")
    metrics_result = evaluate_model(
        test_loader,
        model,
        device,
        ch_names=ch_names_to_use
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Loss: {metrics_result['loss']:.4f}")
    print(f"AUROC: {metrics_result['roc_auc']:.4f}")
    print(f"AUPRC: {metrics_result['pr_auc']:.4f}")
    print(f"Accuracy: {metrics_result['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics_result['balanced_accuracy']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

