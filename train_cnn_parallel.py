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


def run_single_seed(gpu_id, seed, base_args, return_dict):
    """Run a single training job on a specific GPU with a specific seed."""
    
    # Set the GPU for this process
    torch.cuda.set_device(gpu_id)
    
    # Pin this process and its children to specific CPU cores
    # Divide 55 CPUs among 8 GPUs: ~6-7 cores per GPU
    cores_per_gpu = 7
    start_core = gpu_id * cores_per_gpu
    end_core = min(start_core + cores_per_gpu, 55)
    cpu_affinity = list(range(start_core, end_core))
    os.sched_setaffinity(0, cpu_affinity)
    print(f"GPU {gpu_id} pinned to CPUs {cpu_affinity}")
    
    # Create a complete args namespace with all fields
    args = argparse.Namespace(**base_args)
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
        return_dict[seed] = {'seed': seed, 'gpu_id': gpu_id, 'success': True, 'result': result}
    except Exception as e:
        print(f"Failed seed {seed} on GPU {gpu_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return_dict[seed] = {'seed': seed, 'gpu_id': gpu_id, 'success': False, 'error': str(e), 'traceback': traceback.format_exc()}


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
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8],
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
    
    # Use Manager for shared dict to collect results
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Launch processes manually (non-daemonic)
    processes = []
    for i, seed in enumerate(seeds):
        gpu_id = i % num_gpus
        p = mp.Process(target=run_single_seed, args=(gpu_id, seed, base_args, return_dict))
        p.start()
        processes.append(p)
        
        # If we've filled all GPUs, wait for this batch to complete before starting more
        if len(processes) >= num_gpus:
            for p in processes:
                p.join()
            processes = []
    
    # Wait for any remaining processes
    for p in processes:
        p.join()
    
    # Convert return_dict to regular dict and sort by seed
    results = [return_dict[seed] for seed in sorted(return_dict.keys())]
    
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

def calculate_final_results(results):
    from ipdb import set_trace as bp
    import numpy as np
    bp() 
    print('Based on Final Epoch: ')
    results_test = [item['result'][2] for item in results]
    results_val = [item['result'][1] for item in results]
    results_train = [item['result'][0] for item in results]
    auroc_last = np.mean([result['auroc'] for result in results_test])
    auprc_last = np.mean([result['auprc'] for result in results_test])
    acc_last = np.mean([result['balanced_accuracy'] for result in results_test])
    print(f'Test AUROC: {auroc_last:.4f}, Test AUPRC: {auprc_last:.4f}, Test Accuracy: {acc_last:.4f}')
    
    auroc_last = np.mean([result['auroc'] for result in results_val])
    auprc_last = np.mean([result['auprc'] for result in results_val])
    acc_last = np.mean([result['balanced_accuracy'] for result in results_val])
    print(f'Val AUROC: {auroc_last:.4f}, Val AUPRC: {auprc_last:.4f}, Val Accuracy: {acc_last:.4f}')
    
    auroc_last = np.mean([result['auroc'] for result in results_train])
    auprc_last = np.mean([result['auprc'] for result in results_train])
    acc_last = np.mean([result['balanced_accuracy'] for result in results_train])
    print(f'Train AUROC: {auroc_last:.4f}, Train AUPRC: {auprc_last:.4f}, Train Accuracy: {acc_last:.4f}')
    

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    results = main_parallel()
    calculate_final_results(results)