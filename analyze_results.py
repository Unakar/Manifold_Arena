#!/usr/bin/env python3
"""Analyze and compare results from Spectral Ball vs Manifold Muon experiments."""

import argparse
import os
import pickle
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str) -> Dict[str, Any]:
    """Load pickled results from file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def print_summary(name: str, results: Dict[str, Any]):
    """Print summary statistics for a single run."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Update method:     {results['update']}")
    print(f"Learning rate:     {results['lr']}")
    print(f"Epochs:            {results['epochs']}")
    print(f"Seed:              {results['seed']}")
    print(f"\nFinal Results:")
    print(f"  Test accuracy:   {results['test_acc']:.2f}%")
    print(f"  Train accuracy:  {results['train_acc']:.2f}%")
    print(f"  Final loss:      {results['epoch_losses'][-1]:.4f}")
    print(f"  Avg epoch time:  {np.mean(results['epoch_times']):.2f}s")

    # Spectral norm statistics
    norms = [norm.item() for norm in results['norms']]
    print(f"\nSpectral Norms (per layer):")
    for i, norm in enumerate(norms):
        print(f"  Layer {i+1}: {norm:.4f}")

    # Singular value statistics
    print(f"\nSingular Values (largest per layer):")
    for i, sv in enumerate(results['singular_values']):
        max_sv = sv[0].item() if len(sv) > 0 else 0.0
        print(f"  Layer {i+1}: {max_sv:.4f}")


def compare_results(muon_results: Dict[str, Any], ball_results: Dict[str, Any]):
    """Compare Manifold Muon vs Spectral Ball results."""
    print(f"\n{'='*60}")
    print("Comparison: Manifold Muon vs Spectral Ball")
    print(f"{'='*60}")

    muon_acc = muon_results['test_acc']
    ball_acc = ball_results['test_acc']

    print(f"\nTest Accuracy:")
    print(f"  Manifold Muon:   {muon_acc:.2f}%")
    print(f"  Spectral Ball:   {ball_acc:.2f}%")
    print(f"  Difference:      {ball_acc - muon_acc:+.2f}%")

    muon_loss = muon_results['epoch_losses'][-1]
    ball_loss = ball_results['epoch_losses'][-1]

    print(f"\nFinal Loss:")
    print(f"  Manifold Muon:   {muon_loss:.4f}")
    print(f"  Spectral Ball:   {ball_loss:.4f}")
    print(f"  Difference:      {ball_loss - muon_loss:+.4f}")

    muon_time = np.mean(muon_results['epoch_times'])
    ball_time = np.mean(ball_results['epoch_times'])

    print(f"\nAverage Epoch Time:")
    print(f"  Manifold Muon:   {muon_time:.2f}s")
    print(f"  Spectral Ball:   {ball_time:.2f}s")
    print(f"  Speedup:         {muon_time/ball_time:.2f}x")


def plot_comparison(muon_results: Dict[str, Any], ball_results: Dict[str, Any], output_path: str = None):
    """Plot training curves comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss curves
    ax = axes[0]
    epochs = range(1, len(muon_results['epoch_losses']) + 1)
    ax.plot(epochs, muon_results['epoch_losses'], 'o-', label='Manifold Muon', linewidth=2)
    ax.plot(epochs, ball_results['epoch_losses'], 's-', label='Spectral Ball', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot spectral norms comparison
    ax = axes[1]
    muon_norms = [norm.item() for norm in muon_results['norms']]
    ball_norms = [norm.item() for norm in ball_results['norms']]
    layers = range(1, len(muon_norms) + 1)

    width = 0.35
    x = np.arange(len(layers))
    ax.bar(x - width/2, muon_norms, width, label='Manifold Muon', alpha=0.8)
    ax.bar(x + width/2, ball_norms, width, label='Spectral Ball', alpha=0.8)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Spectral Norm', fontsize=12)
    ax.set_title('Final Spectral Norms per Layer', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in layers])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze Spectral Ball vs Manifold Muon results')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate used in experiments')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used in experiments')
    parser.add_argument('--results-dir', type=str, default='src/results', help='Directory containing results')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    parser.add_argument('--output', type=str, default='comparison.png', help='Output filename for plots')
    args = parser.parse_args()

    # Construct filenames
    muon_file = f"update-manifold_muon-lr-{args.lr}-wd-0.0-seed-{args.seed}.pkl"
    ball_file = f"update-spectral_ball-lr-{args.lr}-wd-0.0-seed-{args.seed}.pkl"

    muon_path = os.path.join(args.results_dir, muon_file)
    ball_path = os.path.join(args.results_dir, ball_file)

    # Check files exist
    if not os.path.exists(muon_path):
        print(f"Error: Manifold Muon results not found at {muon_path}")
        return
    if not os.path.exists(ball_path):
        print(f"Error: Spectral Ball results not found at {ball_path}")
        return

    # Load results
    muon_results = load_results(muon_path)
    ball_results = load_results(ball_path)

    # Print summaries
    print_summary("Manifold Muon (Stiefel Manifold)", muon_results)
    print_summary("Spectral Ball", ball_results)

    # Compare results
    compare_results(muon_results, ball_results)

    # Plot comparison
    if args.plot:
        plot_comparison(muon_results, ball_results, args.output)


if __name__ == '__main__':
    main()
