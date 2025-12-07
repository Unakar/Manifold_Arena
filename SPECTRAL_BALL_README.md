# Spectral Ball vs Manifold Muon Comparison

This directory contains an implementation of **Spectral Ball** optimizer for comparison with **Manifold Muon** (Stiefel manifold).

## Key Differences

### Manifold Muon (Stiefel Manifold)
- **Constraint**: All singular values equal 1 → orthogonal matrices
- **Manifold**: Stiefel manifold (orthogonal matrices)
- **Method**: Dual ascent to solve tangent space constraint `W^T @ A + A^T @ W = 0`
- **Projection**: `msign(W)` - matrix sign function

### Spectral Ball
- **Constraint**: Largest singular value = R → spectral norm constraint
- **Manifold**: Spectral ball `{W : ||W||_2 = R}`
- **Method**: Rank-1 projection `Θ = uv^T` + Lagrange multiplier `λ`
- **Projection**: Power iteration + scaling

## Files

- `src/spectral_ball.py` - Spectral Ball optimizer implementation
- `src/manifold_muon.py` - Manifold Muon (Stiefel) implementation (original)
- `src/main.py` - Training script (updated to support both)
- `run_comparison.sh` - Automated comparison script
- `analyze_results.py` - Result analysis and visualization

## Quick Start

### 1. Learning Rate Sweep (Recommended)

```bash
# Make script executable
chmod +x lr_sweep.sh

# Run learning rate sweep from 1e-3 to 1.0
./lr_sweep.sh --epochs 10 --seed 42

# Plot results: Learning Rate vs Test Accuracy
python plot_lr_sweep.py --seed 42 --output lr_sweep.png

# View results table without plotting
python plot_lr_sweep.py --seed 42 --no-plot
```

This will:
- Train both optimizers at multiple learning rates: [0.001, 0.01, 0.1, 1.0]
- Save all results to `src/results/`
- Generate a comparison plot showing accuracy vs learning rate

### 2. Single Learning Rate Comparison

```bash
# Make script executable
chmod +x run_comparison.sh

# Run comparison with default settings (10 epochs, lr=0.1)
./run_comparison.sh

# Run with custom settings
./run_comparison.sh --epochs 20 --lr 0.05 --seed 42
```

### 3. Run Individual Experiments

```bash
cd src

# Run Manifold Muon (Stiefel)
python main.py --update manifold_muon --epochs 10 --lr 0.1 --seed 42

# Run Spectral Ball
python main.py --update spectral_ball --epochs 10 --lr 0.1 --seed 42

# Run baseline AdamW
python main.py --update adam --epochs 10 --lr 0.001 --wd 0.01 --seed 42
```

### 4. Analyze Results

```bash
# Print summary statistics
python analyze_results.py --lr 0.1 --seed 42

# Generate comparison plots
python analyze_results.py --lr 0.1 --seed 42 --plot --output comparison.png
```

## Implementation Details

### Spectral Ball Algorithm

```python
def spectral_ball(W, G, eta):
    """
    Args:
        W: Weight matrix (modified in-place)
        G: Gradient tensor
        eta: Learning rate

    Returns:
        Updated weight matrix on spectral sphere
    """
    # 1. Power iteration: compute σ, u, v
    sigma, u, v = power_iteration(W)

    # 2. Retract to spectral sphere: W ← (R/σ) * W
    W = (R / sigma) * W

    # 3. Form rank-1 matrix: Θ = u @ v^T
    Theta = u @ v.T

    # 4. Solve λ: <Θ, msign(G + λΘ)> = 0
    lambda_star = solve_lambda_bisection(G, Theta)

    # 5. Update: Φ = msign(G + λΘ)
    A = msign(G + lambda_star * Theta)

    # 6. Apply update and retract
    new_W = W - eta * A
    new_W = (R / ||new_W||_2) * new_W

    return new_W
```

### Key Parameters

**Spectral Ball**:
- `radius_mode`: "spectral_mup" (R = √(n_out/n_in)) or "identity" (R = 1)
- `power_iter_steps`: Power iteration steps (default: 50)
- `msign_steps`: Matrix sign iterations (default: 10)
- `lambda_tolerance`: Convergence tolerance for λ solver (default: 1e-6)

**Manifold Muon**:
- `alpha`: Dual variable step size (default: 0.01)
- `steps`: Dual ascent iterations (default: 100)
- `tol`: Convergence tolerance (default: 1e-6)

## Expected Results

On CIFAR-10 with a simple MLP:

| Method | Test Acc | Train Acc | Spectral Norms |
|--------|----------|-----------|----------------|
| Manifold Muon | ~50-55% | ~60-65% | ≈ 1.0 (all layers) |
| Spectral Ball | ~50-55% | ~60-65% | ≈ R (varies by layer) |
| AdamW | ~50% | ~55% | Unconstrained |

**Key Observations**:
1. **Spectral Ball** maintains only the top singular value, allowing other singular values to vary
2. **Manifold Muon** constrains all singular values ≈ 1 (orthogonality)
3. **Spectral Ball** may be faster due to rank-1 projection vs full orthogonalization
4. Both methods provide implicit regularization through manifold constraints

## Tuning Tips

### Learning Rate
- Spectral Ball: Start with `lr=0.1`
- Manifold Muon: Start with `lr=0.1`
- AdamW: Start with `lr=0.001`

### Radius Mode
```bash
# Spectral MuP scaling (recommended for depth)
python main.py --update spectral_ball --lr 0.1

# Identity scaling (R=1 for all layers)
# Modify spectral_ball.py: radius_mode="identity"
```

### Convergence Tolerance
For faster experiments, reduce tolerance:
```python
# In spectral_ball.py
lambda_tolerance = 1e-4  # Faster but less precise
msign_steps = 5          # Fewer iterations
```

## Citation

If you use this implementation, please cite:

```bibtex
@article{modular_manifolds_2024,
  title={Modular Duality in Deep Learning},
  journal={arXiv preprint arXiv:2410.21265},
  year={2024}
}

@article{spectral_ball_2025,
  title={Spectral Ball Optimization for Deep Learning},
  note={Implementation based on Megatron-LM},
  year={2025}
}
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in `main.py`:
```python
train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
```

### Slow Convergence
- Increase `power_iter_steps` (e.g., 100)
- Increase `msign_steps` (e.g., 15)
- Adjust learning rate

### NaN/Inf Values
- Check gradient normalization in `spectral_ball.py`
- Reduce learning rate
- Increase numerical stability constants (eps)

## Future Work

- [ ] Add support for convolutional layers
- [ ] Implement distributed training
- [ ] Compare with PSGD and Shampoo
- [ ] Ablation studies on radius modes
- [ ] Memory-efficient power iteration
