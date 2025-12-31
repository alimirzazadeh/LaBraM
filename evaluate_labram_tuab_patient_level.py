# --------------------------------------------------------
# Evaluation script for LaBraM on TUAB dataset
# --------------------------------------------------------

import torch
from collections import OrderedDict
from timm.models import create_model
import utils
import modeling_finetune  # Required to register the model with timm
from engine_for_finetuning import evaluate
from run_class_finetuning import get_dataset


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    
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
    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)
    
    print(f"\nTest dataset size: {len(dataset_test)}")
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
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
    metrics_result = evaluate(
        test_loader,
        model,
        device,
        header='Test:',
        ch_names=ch_names_to_use,
        metrics=metrics,
        is_binary=True
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Loss: {metrics_result['loss']:.4f}")
    for metric_name in metrics:
        if metric_name in metrics_result:
            print(f"{metric_name.capitalize()}: {metrics_result[metric_name]:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

