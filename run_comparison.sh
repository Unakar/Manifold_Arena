#!/bin/bash

# Comparison script for Spectral Ball vs Manifold Muon
# This script runs both optimizers with the same hyperparameters and compares results

set -e  # Exit on error

# Default hyperparameters
EPOCHS=10
LR=0.1
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
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
echo "Running Spectral Ball vs Manifold Muon Comparison"
echo "Epochs: $EPOCHS, Learning Rate: $LR, Seed: $SEED"
echo "================================================"

cd src

# Run Manifold Muon (Stiefel)
echo ""
echo "1/2: Running Manifold Muon (Stiefel manifold)..."
python main.py \
    --update manifold_muon \
    --epochs $EPOCHS \
    --lr $LR \
    --seed $SEED

# Run Spectral Ball
echo ""
echo "2/2: Running Spectral Ball..."
python main.py \
    --update spectral_ball \
    --epochs $EPOCHS \
    --lr $LR \
    --seed $SEED

echo ""
echo "================================================"
echo "Comparison Complete!"
echo "Results saved in src/results/"
echo ""
echo "Result files:"
echo "  - update-manifold_muon-lr-${LR}-wd-0.0-seed-${SEED}.pkl"
echo "  - update-spectral_ball-lr-${LR}-wd-0.0-seed-${SEED}.pkl"
echo "================================================"
