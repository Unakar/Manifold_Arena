#!/usr/bin/env python3
"""Plot learning rate sweep results for Spectral Ball vs Manifold Muon."""

import argparse
import os
import pickle
import glob
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str) -> Dict:
    """Load pickled results from file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def extract_lr_results(results_dir: str, seed: int) -> Tuple[List[float], Dict[str, List[float]]]:
    """Extract learning rate sweep results.

    Returns:
        lr_values: List of learning rates
        results_dict: Dict with keys 'manifold_muon' and 'spectral_ball',
                     each containing lists of test accuracies
    """
    # Find all result files
    pattern = os.path.join(results_dir, f"update-*-lr-*-wd-0.0-seed-{seed}.pkl")
    files = glob.glob(pattern)

    results_dict = {
        'manifold_muon': {},
        'spectral_ball': {}
    }

    for filepath in files:
        filename = os.path.basename(filepath)

        # Parse filename to extract update method and lr
        # Format: update-{method}-lr-{lr}-wd-{wd}-seed-{seed}.pkl
        parts = filename.split('-')

        # Find update method (between 'update' and 'lr')
        update_idx = parts.index('update')
        lr_idx = parts.index('lr')
        method = '-'.join(parts[update_idx+1:lr_idx])

        # Extract learning rate
        lr = float(parts[lr_idx + 1])

        # Load results
        results = load_results(filepath)
        test_acc = results['test_acc']

        if method in results_dict:
            results_dict[method][lr] = test_acc

    # Sort by learning rate
    lr_values = sorted(set(list(results_dict['manifold_muon'].keys()) +
                           list(results_dict['spectral_ball'].keys())))

    # Build aligned lists
    muon_accs = [results_dict['manifold_muon'].get(lr, None) for lr in lr_values]
    ball_accs = [results_dict['spectral_ball'].get(lr, None) for lr in lr_values]

    return lr_values, {'manifold_muon': muon_accs, 'spectral_ball': ball_accs}


def plot_lr_sweep(lr_values: List[float],
                  results: Dict[str, List[float]],
                  output_path: str = None):
    """Plot learning rate vs accuracy."""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Manifold Muon
    muon_accs = results['manifold_muon']
    muon_valid = [(lr, acc) for lr, acc in zip(lr_values, muon_accs) if acc is not None]
    if muon_valid:
        lrs, accs = zip(*muon_valid)
        ax.plot(lrs, accs, 'o-', label='Manifold Muon (Stiefel)',
                linewidth=2.5, markersize=8, color='#2E86AB')

    # Plot Spectral Ball
    ball_accs = results['spectral_ball']
    ball_valid = [(lr, acc) for lr, acc in zip(lr_values, ball_accs) if acc is not None]
    if ball_valid:
        lrs, accs = zip(*ball_valid)
        ax.plot(lrs, accs, 's-', label='Spectral Ball',
                linewidth=2.5, markersize=8, color='#A23B72')

    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Learning Rate Sweep: Spectral Ball vs Manifold Muon',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=11)

    # Add annotation for best LR
    if muon_valid:
        lrs, accs = zip(*muon_valid)
        best_idx = np.argmax(accs)
        best_lr, best_acc = lrs[best_idx], accs[best_idx]
        ax.annotate(f'Best: {best_acc:.2f}%\n(lr={best_lr})',
                   xy=(best_lr, best_acc), xytext=(20, 20),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', fc='#2E86AB', alpha=0.7, ec='none'),
                   color='white', ha='left',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                 color='#2E86AB', lw=1.5))

    if ball_valid:
        lrs, accs = zip(*ball_valid)
        best_idx = np.argmax(accs)
        best_lr, best_acc = lrs[best_idx], accs[best_idx]
        ax.annotate(f'Best: {best_acc:.2f}%\n(lr={best_lr})',
                   xy=(best_lr, best_acc), xytext=(20, -35),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', fc='#A23B72', alpha=0.7, ec='none'),
                   color='white', ha='left',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                 color='#A23B72', lw=1.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def print_summary_table(lr_values: List[float], results: Dict[str, List[float]]):
    """Print summary table of results."""
    print("\n" + "="*70)
    print("Learning Rate Sweep Results Summary")
    print("="*70)
    print(f"{'Learning Rate':<15} {'Manifold Muon':<20} {'Spectral Ball':<20} {'Difference':<15}")
    print("-"*70)

    for lr, muon_acc, ball_acc in zip(lr_values,
                                      results['manifold_muon'],
                                      results['spectral_ball']):
        muon_str = f"{muon_acc:.2f}%" if muon_acc is not None else "N/A"
        ball_str = f"{ball_acc:.2f}%" if ball_acc is not None else "N/A"

        if muon_acc is not None and ball_acc is not None:
            diff = ball_acc - muon_acc
            diff_str = f"{diff:+.2f}%"
        else:
            diff_str = "N/A"

        print(f"{lr:<15.4f} {muon_str:<20} {ball_str:<20} {diff_str:<15}")

    print("="*70)

    # Print best results
    muon_accs = [acc for acc in results['manifold_muon'] if acc is not None]
    ball_accs = [acc for acc in results['spectral_ball'] if acc is not None]

    if muon_accs:
        best_muon_idx = np.argmax([acc if acc is not None else -np.inf
                                   for acc in results['manifold_muon']])
        best_muon_lr = lr_values[best_muon_idx]
        best_muon_acc = results['manifold_muon'][best_muon_idx]
        print(f"\nBest Manifold Muon: {best_muon_acc:.2f}% at lr={best_muon_lr}")

    if ball_accs:
        best_ball_idx = np.argmax([acc if acc is not None else -np.inf
                                   for acc in results['spectral_ball']])
        best_ball_lr = lr_values[best_ball_idx]
        best_ball_acc = results['spectral_ball'][best_ball_idx]
        print(f"Best Spectral Ball: {best_ball_acc:.2f}% at lr={best_ball_lr}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Plot learning rate sweep results for Spectral Ball vs Manifold Muon'
    )
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed used in experiments')
    parser.add_argument('--results-dir', type=str, default='src/results',
                       help='Directory containing results')
    parser.add_argument('--output', type=str, default='lr_sweep_comparison.png',
                       help='Output filename for plot')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting (only print table)')
    args = parser.parse_args()

    # Extract results
    try:
        lr_values, results = extract_lr_results(args.results_dir, args.seed)
    except Exception as e:
        print(f"Error loading results: {e}")
        print(f"\nMake sure you have run the learning rate sweep first:")
        print(f"  ./lr_sweep.sh --epochs 10 --seed {args.seed}")
        return

    if not lr_values:
        print("No results found!")
        print(f"Expected files in {args.results_dir} with seed={args.seed}")
        return

    # Print summary
    print_summary_table(lr_values, results)

    # Plot results
    if not args.no_plot:
        plot_lr_sweep(lr_values, results, args.output)


if __name__ == '__main__':
    main()
