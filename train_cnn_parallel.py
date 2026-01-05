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
import numpy as np
from ipdb import set_trace as bp
# Import your training function
from train_cnn import main
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

#time it 
import time
start_time = time.time()
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
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--window_length', type=int, default=4)
    parser.add_argument('--resolution_factor', type=int, default=1)
    parser.add_argument('--stride_length', type=int, default=1)
    parser.add_argument('--bandwidth', type=float, default=2.0)
    parser.add_argument('--multitaper', type=bool, default=True)
    parser.add_argument('--load_spec_true', default=False, action='store_true')
    parser.add_argument('--load_spec_recon', default=False, action='store_true')
    parser.add_argument('--lr_warmup_prop', type=float, default=0.2)
    parser.add_argument('--normalize_spec', default=False, action='store_true')
    parser.add_argument('--percentile_low', type=float, default=-20)
    parser.add_argument('--percentile_high', type=float, default=30)
    parser.add_argument('--drop_extra_channels', default=False, action='store_true')
    parser.add_argument('--custom_name', type=str, default='')
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
        'lr_warmup_prop': args.lr_warmup_prop,
        'normalize_spec': args.normalize_spec,
        'percentile_low': args.percentile_low,
        'percentile_high': args.percentile_high,
        'drop_extra_channels': args.drop_extra_channels,
        'custom_name': args.custom_name,
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
    
    return results, base_args

def calculate_final_results(results):

    print('Based on Final Epoch: ')
    exp_name = results[0]['result'][3]
    last_underscore = exp_name[::-1].index('_')
    exp_name = exp_name[:-last_underscore-1] + '_allseeds'
    
    
    results_test = [item['result'][2] for item in results]
    results_val = [item['result'][1] for item in results]
    results_train = [item['result'][0] for item in results]
    all_auroc_test = [[res['auroc'] for res in result] for result in results_test]
    all_auprc_test = [[res['auprc'] for res in result] for result in results_test]
    all_acc_test = [[res['balanced_accuracy'] for res in result] for result in results_test]
    
    all_auroc_val = [[res['auroc'] for res in result] for result in results_val]
    all_auprc_val = [[res['auprc'] for res in result] for result in results_val]
    all_acc_val = [[res['balanced_accuracy'] for res in result] for result in results_val]
    all_auroc_train = [[res['auroc'] for res in result] for result in results_train]
    all_auprc_train = [[res['auprc'] for res in result] for result in results_train]
    all_acc_train = [[res['balanced_accuracy'] for res in result] for result in results_train]

    auroc_last_test = np.mean([auroc[-1] for auroc in all_auroc_test])
    auprc_last_test = np.mean([auprc[-1] for auprc in all_auprc_test])
    acc_last_test = np.mean([acc[-1] for acc in all_acc_test])
    print(f'Last Test AUROC: {auroc_last_test:.4f}, Last Test AUPRC: {auprc_last_test:.4f}, Last Test Accuracy: {acc_last_test:.4f}')
    auroc_last_val = np.mean([auroc[-1] for auroc in all_auroc_val])
    auprc_last_val = np.mean([auprc[-1] for auprc in all_auprc_val])
    acc_last_val = np.mean([acc[-1] for acc in all_acc_val])
    print(f'Last Val AUROC: {auroc_last_val:.4f}, Last Val AUPRC: {auprc_last_val:.4f}, Last Val Accuracy: {acc_last_val:.4f}')
    auroc_last_train = np.mean([auroc[-1] for auroc in all_auroc_train])
    auprc_last_train = np.mean([auprc[-1] for auprc in all_auprc_train])
    acc_last_train = np.mean([acc[-1] for acc in all_acc_train])
    print(f'Last Train AUROC: {auroc_last_train:.4f}, Last Train AUPRC: {auprc_last_train:.4f}, Last Train Accuracy: {acc_last_train:.4f}')
    
    auroc_best_test = np.mean([np.max(auroc) for auroc in all_auroc_test])
    auprc_best_test = np.mean([np.max(auprc) for auprc in all_auprc_test])
    acc_best_test = np.mean([np.max(acc) for acc in all_acc_test])
    print(f'Best Test AUROC: {auroc_best_test:.4f}, Best Test AUPRC: {auprc_best_test:.4f}, Best Test Accuracy: {acc_best_test:.4f}')
    auroc_best_val = np.mean([np.max(auroc) for auroc in all_auroc_val])
    auprc_best_val = np.mean([np.max(auprc) for auprc in all_auprc_val])
    acc_best_val = np.mean([np.max(acc) for acc in all_acc_val])
    print(f'Best Val AUROC: {auroc_best_val:.4f}, Best Val AUPRC: {auprc_best_val:.4f}, Best Val Accuracy: {acc_best_val:.4f}')
    auroc_best_train = np.mean([np.max(auroc) for auroc in all_auroc_train])
    auprc_best_train = np.mean([np.max(auprc) for auprc in all_auprc_train])
    acc_best_train = np.mean([np.max(acc) for acc in all_acc_train])
    print(f'Best Train AUROC: {auroc_best_train:.4f}, Best Train AUPRC: {auprc_best_train:.4f}, Best Train Accuracy: {acc_best_train:.4f}')
    final_metrics = { 
                     "best/test/auroc": auroc_best_test,
                     "best/test/auprc": auprc_best_test,
                     "best/test/accuracy": acc_best_test,
                     "best/val/auroc": auroc_best_val,
                     "best/val/auprc": auprc_best_val,
                     "best/val/accuracy": acc_best_val,
                     "best/train/auroc": auroc_best_train,
                     "best/train/auprc": auprc_best_train,
                     "best/train/accuracy": acc_best_train,
                     "last/test/auroc": auroc_last_test,
                     "last/test/auprc": auprc_last_test,
                     "last/test/accuracy": acc_last_test,
                     "last/val/auroc": auroc_last_val,
                     "last/val/auprc": auprc_last_val,
                     "last/val/accuracy": acc_last_val,
                     "last/train/auroc": auroc_last_train,
                     "last/train/auprc": auprc_last_train,
                     "last/train/accuracy": acc_last_train}
    return final_metrics, exp_name

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    results, base_args = main_parallel()
    final_metrics, exp_name = calculate_final_results(results)
    total_time = time.time() - start_time
    print(f'Total time: {total_time / 3600:.2f} hours')
    hp = {
        'lr': base_args['lr'],
        'batch_size': base_args['batch_size'],
        'bandwidth': base_args['bandwidth'],
        'source': 'timeseries' if (not base_args['load_spec_true'] and not base_args['load_spec_recon']) else 'true_spec' if base_args['load_spec_true'] else 'recon_spec' if base_args['load_spec_recon'] else '',
        'window_length': base_args['window_length'],
        'model_type': base_args['model_type'],
        'dataset': base_args['dataset'],
        'total_time': np.round(total_time / 3600, 2),
        'num_workers': base_args['num_workers'],
        'normalize_spec': base_args['normalize_spec'],
        'percentile_low': base_args['percentile_low'],
        'percentile_high': base_args['percentile_high'],
        'logv2': True,
        'drop_extra_channels': base_args['drop_extra_channels'],
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
    }
    # Create writer
    writer = SummaryWriter(log_dir=exp_name)
    
    # 1) Make sure metric values are plain Python floats
    final_metrics_clean = {k: float(v) for k, v in final_metrics.items()}

    # 2) Write the hparams plugin metadata
    session_start, session_end, hparams_summary = hparams(hp, final_metrics_clean)
    writer.file_writer.add_summary(session_start, global_step=0)
    writer.file_writer.add_summary(session_end, global_step=0)
    writer.file_writer.add_summary(hparams_summary, global_step=0)

    # 3) IMPORTANT: write the metric scalars with the *same tags*
    for k, v in final_metrics_clean.items():
        writer.add_scalar(k, v, global_step=0)

    writer.flush()
    writer.close()
    
    # writer = SummaryWriter(log_dir=exp_name)
    # session_start, session_end, hparams_summary = hparams(hp, final_metrics)

    # # Add all three events
    # writer.file_writer.add_summary(session_start, global_step=0)
    # writer.file_writer.add_summary(session_end, global_step=0)
    # writer.file_writer.add_summary(hparams_summary, global_step=0)
    # writer.file_writer.flush()