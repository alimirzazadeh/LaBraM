# --------------------------------------------------------
# Evaluation script for LaBraM on TUAB dataset
# --------------------------------------------------------

import torch
from collections import OrderedDict
from timm.models import create_model
import utils
from engine_for_finetuning import evaluate


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
    
    # Load checkpoint - match exact logic from run_class_finetuning.py
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("Load ckpt from %s" % checkpoint_path)
    checkpoint_model = None
    # Use same model_key logic as finetuning script: 'model|module' (default)
    model_key = 'model|module'
    for model_key_option in model_key.split('|'):
        if model_key_option in checkpoint:
            checkpoint_model = checkpoint[model_key_option]
            print("Load state_dict by model_key = %s" % model_key_option)
            break
    
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    
    # Apply model_filter_name filtering (default is 'gzp' in finetuning script)
    # This filters keys starting with 'student.' and removes the prefix
    model_filter_name = 'gzp'  # Match default from finetuning script
    if (checkpoint_model is not None) and (model_filter_name != ''):
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('student.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                pass  # Match exact logic from finetuning script
        # Only replace checkpoint_model if we found student keys
        if len(new_dict) > 0:
            checkpoint_model = new_dict
    
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
    
    # Load state dict with empty prefix (matching finetuning script)
    utils.load_state_dict(model, checkpoint_model, prefix='')
    
    model.to(device)
    model.eval()
    
    return model


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
    print("LaBraM TUAB Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = load_model_checkpoint(checkpoint_path, device)
    print("Model loaded successfully!")
    
    # Load test dataset
    print(f"\nLoading TUAB test dataset from {dataset_root}...")
    _, test_dataset, _ = utils.prepare_TUAB_dataset(dataset_root, return_pid=False)

    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
        shuffle=False
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Run evaluation
    print("\nRunning evaluation...")
    metrics = evaluate(
        test_loader,
        model,
        device,
        header='Test:',
        ch_names=ch_names,
        metrics=['roc_auc', 'pr_auc', 'accuracy', 'balanced_accuracy'],
        is_binary=True
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"AUROC: {metrics['roc_auc']:.4f}")
    print(f"AUPRC: {metrics['pr_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

