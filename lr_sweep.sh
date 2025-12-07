#!/bin/bash

# Learning Rate Sweep Script for Spectral Ball vs Manifold Muon
# Sweeps learning rates from 1e-3 to 1.0 and compares accuracy

set -e  # Exit on error

# Default settings
EPOCHS=10
SEED=42
LR_VALUES=(0.001 0.005 0.01 0.05 0.1 0.5 1.0)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

echo "================================================"
echo "Learning Rate Sweep Experiment"
echo "Epochs: $EPOCHS, Seed: $SEED"
echo "LR values: ${LR_VALUES[@]}"
echo "================================================"

cd src

# Run experiments for each learning rate
for lr in "${LR_VALUES[@]}"; do
    echo ""
    echo "================================================"
    echo "Running with LR = $lr"
    echo "================================================"

    # Run Manifold Muon
    echo ""
    echo "1/2: Manifold Muon (Stiefel)..."
    python main.py \
        --update manifold_muon \
        --epochs $EPOCHS \
        --lr $lr \
        --seed $SEED

    # Run Spectral Ball
    echo ""
    echo "2/2: Spectral Ball..."
    python main.py \
        --update spectral_ball \
        --epochs $EPOCHS \
        --lr $lr \
        --seed $SEED
done

echo ""
echo "================================================"
echo "Learning Rate Sweep Complete!"
echo "Results saved in src/results/"
echo ""
echo "To analyze results, run:"
echo "  python ../plot_lr_sweep.py --seed $SEED"
echo "================================================"
