#!/bin/bash

# Learning Rate Sweep Script for Spectral Ball vs Manifold Muon
# Sweeps learning rates from 1e-3 to 1.0 and compares accuracy

set -e  # Exit on error

# Default settings
EPOCHS=10
SEED=42
LR_VALUES=(0.001 0.005 0.01 0.05 0.1 0.5 1.0 10)

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

# Run experiments for each learning rate in parallel
# Each GPU runs one learning rate
pids=()
for i in "${!LR_VALUES[@]}"; do
    lr="${LR_VALUES[$i]}"
    gpu_id=$i

    echo "Starting GPU $gpu_id with LR = $lr"

    # Run both manifold_muon and spectral_ball on the same GPU sequentially, in background
    (
        echo "[GPU $gpu_id] Running Manifold Muon with LR = $lr"
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
            --update manifold_muon \
            --epochs $EPOCHS \
            --lr $lr \
            --seed $SEED

        echo "[GPU $gpu_id] Running Spectral Ball with LR = $lr"
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
            --update spectral_ball \
            --epochs $EPOCHS \
            --lr $lr \
            --seed $SEED

        echo "[GPU $gpu_id] Completed LR = $lr"
    ) &

    pids+=($!)
done

echo ""
echo "================================================"
echo "All jobs launched. Waiting for completion..."
echo "PIDs: ${pids[@]}"
echo "================================================"

# Wait for all background processes to complete
for pid in "${pids[@]}"; do
    wait $pid
    echo "Process $pid completed"
done

echo ""
echo "================================================"
echo "Learning Rate Sweep Complete!"
echo "Results saved in src/results/"
echo ""
echo "To analyze results, run:"
echo "  python ../plot_lr_sweep.py --seed $SEED"
echo "================================================"
