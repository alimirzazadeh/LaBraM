#!/usr/bin/env python3
"""
Run multiple training runs with different seeds in parallel across GPUs.
Usage: python run_parallel_seeds.py
"""

import torch
import torch.multiprocessing as mp
import argparse
import sys
import os
from types import SimpleNamespace

# Import your training function
from train_cnn import main


def run_single_seed(gpu_id, seed, base_args):
    """Run a single training job on a specific GPU with a specific seed."""
    
    # Set the GPU for this process
    torch.cuda.set_device(gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Create args namespace
    args = SimpleNamespace(**base_args)
    args.seed = seed
    args.resolution = 0.2  # Set as in original main
    
    # Set seeds
    torch.manual_seed(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Starting seed {seed} on GPU {gpu_id}")
    
    try:
        # Run training
        result = main(args)
        print(f"Completed seed {seed} on GPU {gpu_id}")
        return {'seed': seed, 'gpu_id': gpu_id, 'success': True, 'result': result}
    except Exception as e:
        print(f"Failed seed {seed} on GPU {gpu_id}: {str(e)}")
        return {'seed': seed, 'gpu_id': gpu_id, 'success': False, 'error': str(e)}


def main_parallel():
    parser = argparse.ArgumentParser(description='Run multiple seeds in parallel')
    
    # Base training arguments
    parser.add_argument('--dataset', type=str, default='TUAB', choices=['TUAB', 'TUEV'])
    parser.add_argument('--model_type', type=str, default='resnet', choices=['conv1d', 'conv2d', 'resnet'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--window_length', type=int, default=4)
    parser.add_argument('--resolution_factor', type=int, default=1)
    parser.add_argument('--stride_length', type=int, default=1)
    parser.add_argument('--bandwidth', type=float, default=2.0)
    parser.add_argument('--multitaper', type=bool, default=True)
    parser.add_argument('--load_spec_true', type=bool, default=True)
    parser.add_argument('--load_spec_recon', type=bool, default=False)
    parser.add_argument('--lr_warmup_prop', type=float, default=0.2)
    
    # Parallel execution arguments
    parser.add_argument('--seeds', type=int, nargs='+', default=[10, 20, 30, 40, 50, 60, 70, 80],
                       help='List of seeds to run')
    parser.add_argument('--num_gpus', type=int, default=8,
                       help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Convert args to dict for passing to workers
    base_args = {
        'dataset': args.dataset,
        'model_type': args.model_type,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'lr': args.lr,
        'epochs': args.epochs,
        'window_length': args.window_length,
        'resolution_factor': args.resolution_factor,
        'stride_length': args.stride_length,
        'bandwidth': args.bandwidth,
        'multitaper': args.multitaper,
        'load_spec_true': args.load_spec_true,
        'load_spec_recon': args.load_spec_recon,
        'lr_warmup_prop': args.lr_warmup_prop
    }
    
    seeds = args.seeds
    num_gpus = args.num_gpus
    
    print(f"Running {len(seeds)} seeds across {num_gpus} GPUs")
    print(f"Seeds: {seeds}")
    
    # Use multiprocessing Pool to run in parallel
    with mp.Pool(processes=num_gpus) as pool:
        # Create arguments for each job: (gpu_id, seed, base_args)
        job_args = [
            (i % num_gpus, seed, base_args) 
            for i, seed in enumerate(seeds)
        ]
        
        # Run all jobs
        results = pool.starmap(run_single_seed, job_args)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for result in results:
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"Seed {result['seed']} (GPU {result['gpu_id']}): {status}")
        if not result['success']:
            print(f"  Error: {result['error']}")
    
    # Check if all succeeded
    all_success = all(r['success'] for r in results)
    if all_success:
        print("\nAll runs completed successfully!")
    else:
        print("\nSome runs failed. Check the output above.")
        sys.exit(1)
    
    return results


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main_parallel()