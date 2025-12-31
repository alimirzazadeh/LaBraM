# --------------------------------------------------------
# Evaluation script for LaBraM on TUAB dataset
# --------------------------------------------------------

import torch
from timm.models import create_model
import utils
import modeling_finetune  # Required to register the model with timm
from engine_for_finetuning import evaluate
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

