"""
parse_results.py
Reads all slurm log files and prints a comparison table of results.
Usage: python parse_results.py --log_dir slurm_logs/
"""

import os
import re
import argparse
from pathlib import Path


def parse_log_file(log_path):
    """
    Parse a single log file and extract:
    - optimizer
    - learning rate
    - best validation loss
    - final training loss
    - final validation loss
    """
    result = {
        'file': log_path.name,
        'optimizer': None,
        'lr': None,
        'best_val_loss': None,
        'final_train_loss': None,
        'final_val_loss': None,
        'status': 'incomplete'
    }

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            # extract optimizer and lr from job info line
            if 'Running with optimizer' in line:
                match = re.search(r'optimizer (\w+) and learning rate ([\d.]+)', line)
                if match:
                    result['optimizer'] = match.group(1)
                    result['lr'] = float(match.group(2))

            # extract best val loss
            if 'reach best val loss' in line:
                match = re.search(r'reach best val loss\s*:\s*([\d.]+)', line)
                if match:
                    result['best_val_loss'] = float(match.group(1))

            # extract final losses from last epoch line
            # looks for: train_loss=X.XXXXXX, val_loss=X.XXX
            if 'train_loss=' in line and 'val_loss=' in line:
                train_match = re.search(r'train_loss=([\d.]+)', line)
                val_match = re.search(r'val_loss=([\d.]+)', line)
                if train_match and val_match:
                    result['final_train_loss'] = float(train_match.group(1))
                    result['final_val_loss'] = float(val_match.group(1))

            # check if training completed
            if 'Training completed' in line:
                result['status'] = 'completed'

    except Exception as e:
        result['status'] = f'error: {e}'

    return result


def print_table(results):
    """Print a formatted comparison table."""

    # sort by best val loss
    completed = [r for r in results if r['status'] == 'completed' and r['best_val_loss'] is not None]
    incomplete = [r for r in results if r['status'] != 'completed']

    completed.sort(key=lambda x: x['best_val_loss'])

    # header
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Rank':<6} {'Optimizer':<10} {'LR':<10} {'Best Val Loss':<16} {'Final Train':<14} {'Final Val':<12} {'Status'}")
    print("-" * 80)

    # completed jobs sorted by best val loss
    for i, r in enumerate(completed, 1):
        print(f"{i:<6} {r['optimizer']:<10} {str(r['lr']):<10} "
              f"{r['best_val_loss']:<16.6f} "
              f"{r['final_train_loss']:<14.6f} "
              f"{r['final_val_loss']:<12.6f} "
              f"{r['status']}")

    # incomplete/failed jobs
    if incomplete:
        print("\n--- Incomplete / Failed Jobs ---")
        for r in incomplete:
            print(f"  {r['file']:<40} optimizer={r['optimizer']} lr={r['lr']} status={r['status']}")

    print("=" * 80)

    # best result summary
    if completed:
        best = completed[0]
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Optimizer    : {best['optimizer']}")
        print(f"   Learning Rate: {best['lr']}")
        print(f"   Best Val Loss: {best['best_val_loss']:.6f}")
        print("=" * 80)

    # summary by optimizer
    print("\n--- BEST PER OPTIMIZER ---")
    optimizers = {}
    for r in completed:
        opt = r['optimizer']
        if opt not in optimizers or r['best_val_loss'] < optimizers[opt]['best_val_loss']:
            optimizers[opt] = r

    print(f"{'Optimizer':<10} {'Best LR':<10} {'Best Val Loss'}")
    print("-" * 35)
    for opt, r in sorted(optimizers.items(), key=lambda x: x[1]['best_val_loss']):
        print(f"{opt:<10} {str(r['lr']):<10} {r['best_val_loss']:.6f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Parse SLURM log files and compare results')
    parser.add_argument('--log_dir', type=str, default='slurm_logs',
                        help='Directory containing .out log files')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        print(f"Error: log directory '{log_dir}' not found")
        return

    # find all .out files
    log_files = sorted(log_dir.glob('*.out'))

    if not log_files:
        print(f"No .out files found in {log_dir}")
        return

    print(f"Found {len(log_files)} log files in {log_dir}")

    # parse each log file
    results = []
    for log_path in log_files:
        result = parse_log_file(log_path)
        results.append(result)

    # print table
    print_table(results)


if __name__ == '__main__':
    main()