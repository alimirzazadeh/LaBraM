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


def load_model_checkpoint(checkpoint_path, device, load=True):
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
    if load:
        # Load checkpoint - match exact logic from utils.auto_load_model (line 636)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("Load ckpt from %s" % checkpoint_path)
        
        # Checkpoint saved during training has 'model' key directly (see utils.save_model line 588)
        # During training resume, they use: model_without_ddp.load_state_dict(checkpoint['model'])
        if 'model' not in checkpoint:
            raise ValueError(f"Checkpoint does not contain 'model' key. Available keys: {checkpoint.keys()}")
        
        # Load directly like training script does (utils.py line 636)
        model.load_state_dict(checkpoint['model'], strict=False)
        print("Model state dict loaded successfully")
    
    model.to(device)
    model.eval()
    
    return model


def main():
    # Configuration
    checkpoint_path = "checkpoints/finetune_tuab_base_bs512/checkpoint-49.pth" #checkpoint-best.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    
    print("=" * 60)
    print("LaBraM TUAB Evaluation")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = load_model_checkpoint(checkpoint_path, device, load=False)
    print("Model loaded successfully!")
    
    # Load dataset using the same function as finetuning script
    class Args:
        dataset = 'TUAB'
        
    args = Args()
    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)
    
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

