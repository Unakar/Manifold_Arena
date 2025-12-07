"""Spectral Ball optimizer implementation.

This implements optimization on the spectral ball manifold, where the spectral norm
(largest singular value) of the weight matrix is constrained to a fixed radius R.

The key difference from Stiefel manifold methods (manifold_muon):
- Spectral ball: constrains only the largest singular value ||W||_2 = R
- Stiefel manifold: constrains all singular values to be 1 (orthogonal matrices)

Algorithm:
1. Power iteration to compute σ (spectral norm) and top singular vectors (u, v)
2. Retract W to spectral sphere: W ← (R/σ) * W
3. Form rank-1 matrix Θ = u @ v^T (pointing in top singular direction)
4. Solve for λ: <Θ, msign(G + λΘ)> = 0 using bisection
5. Return update: W_new = W - eta * msign(G + λΘ)

Reference:
    - Spectral Ball optimization for deep learning
"""

import torch
from msign import msign


@torch.no_grad()
def power_iteration(W: torch.Tensor, steps: int = 50, eps: float = 1e-20):
    """Compute leading singular triplet (σ, u, v) via bilateral power iteration.

    Args:
        W: Weight matrix of shape (m, n)
        steps: Number of power iteration steps
        eps: Small constant for numerical stability

    Returns:
        sigma: Leading singular value (spectral norm)
        u: Left singular vector of shape (m, 1)
        v: Right singular vector of shape (n, 1)
    """
    # Initialize v randomly
    v = torch.ones(W.shape[1], 1, dtype=W.dtype, device=W.device)

    # Power iteration
    for _ in range(steps):
        # v ← W^T @ (W @ v) / ||W^T @ (W @ v)||
        Wv = W @ v
        v = W.T @ Wv
        v = v / (v.norm() + eps)

    # Compute u and σ
    u = W @ v
    sigma = u.norm()
    u = u / (sigma + eps)

    return sigma.item(), u, v


@torch.no_grad()
def compute_f(G: torch.Tensor, Theta: torch.Tensor, lambda_val: float, msign_steps: int = 10) -> float:
    """Compute f(λ) = <Θ, msign(G + λΘ)>.

    The root of this function gives the optimal Lagrange multiplier λ.

    Args:
        G: Gradient/momentum tensor
        Theta: Rank-1 matrix u @ v^T
        lambda_val: Current λ value
        msign_steps: Number of msign iteration steps

    Returns:
        f(λ): Inner product <Θ, msign(G + λΘ)>
    """
    Z = G + lambda_val * Theta
    Phi = msign(Z, steps=msign_steps)
    f_val = (Theta * Phi).sum().item()
    return f_val


@torch.no_grad()
def solve_lambda_bisection(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_step: float = 1e-3,
    tolerance: float = 1e-6,
    max_iterations: int = 20,
    max_expansions: int = 10,
    msign_steps: int = 10,
):
    """Solve for λ such that f(λ) = <Θ, msign(G + λΘ)> = 0 using bisection.

    Args:
        G: Gradient/momentum tensor
        Theta: Rank-1 matrix u @ v^T
        initial_step: Initial step size for bracketing
        tolerance: Convergence tolerance for |f(λ)|
        max_iterations: Maximum bisection iterations
        max_expansions: Maximum bracket expansion iterations
        msign_steps: Number of msign iteration steps

    Returns:
        lambda_star: Optimal Lagrange multiplier
    """
    # Start from λ = 0
    lambda_curr = 0.0
    f_curr = compute_f(G, Theta, lambda_curr, msign_steps)

    # If already near zero, return
    if abs(f_curr) < tolerance:
        return lambda_curr

    # Find bracket: expand in the direction indicated by f_curr
    step = initial_step if f_curr < 0 else -initial_step
    lambda_prev = lambda_curr
    f_prev = f_curr

    # Bracket expansion
    for _ in range(max_expansions):
        lambda_new = lambda_prev + step
        f_new = compute_f(G, Theta, lambda_new, msign_steps)

        # Check for sign change
        if f_prev * f_new <= 0:
            # Found bracket
            if f_prev <= 0 and f_new >= 0:
                lambda_L, lambda_R = lambda_prev, lambda_new
                f_L, f_R = f_prev, f_new
            else:
                lambda_L, lambda_R = lambda_new, lambda_prev
                f_L, f_R = f_new, f_prev
            break

        # Expand step
        step *= 2.0
        lambda_prev, f_prev = lambda_new, f_new
    else:
        # Bracket not found, return λ=0 (fallback to standard Muon)
        return 0.0

    # Bisection
    for _ in range(max_iterations):
        lambda_mid = 0.5 * (lambda_L + lambda_R)
        f_mid = compute_f(G, Theta, lambda_mid, msign_steps)

        # Converged
        if abs(f_mid) < tolerance:
            return lambda_mid

        # Update bracket (f is monotone increasing)
        if f_mid < 0:
            lambda_L, f_L = lambda_mid, f_mid
        else:
            lambda_R, f_R = lambda_mid, f_mid

    # Return best estimate
    return lambda_mid


@torch.no_grad()
def spectral_ball(
    W: torch.Tensor,
    G: torch.Tensor,
    eta: float = 0.1,
    radius_mode: str = "spectral_mup",
    power_iter_steps: int = 50,
    msign_steps: int = 10,
    lambda_tolerance: float = 1e-6,
    lambda_max_iter: int = 20,
):
    """Spectral Ball constrained optimization.

    Performs optimization on the spectral ball manifold: {W : ||W||_2 = R}.

    Algorithm:
    1. Power iteration: compute σ, u, v
    2. Retract to spectral sphere: W ← (R/σ) * W
    3. Form Θ = u @ v^T
    4. Solve λ: <Θ, msign(G + λΘ)> = 0
    5. Update: W_new = W - eta * msign(G + λΘ)
    6. Retract: W_new ← (R/||W_new||_2) * W_new

    Args:
        W: Weight matrix (will be modified in-place for retraction)
        G: Gradient tensor
        eta: Learning rate
        radius_mode: Target radius mode ("spectral_mup" or "identity")
        power_iter_steps: Number of power iteration steps
        msign_steps: Number of msign iteration steps
        lambda_tolerance: Convergence tolerance for λ solver
        lambda_max_iter: Maximum iterations for λ solver

    Returns:
        W_new: Updated weight matrix on the spectral sphere
    """
    # Handle transpose for tall matrices (same as manifold_muon)
    should_transpose = W.shape[0] < W.shape[1]
    if should_transpose:
        W = W.T
        G = G.T

    # Compute target radius R
    if radius_mode == "spectral_mup":
        n_out, n_in = W.shape
        R = (n_out / n_in) ** 0.5
    elif radius_mode == "identity":
        R = 1.0
    else:
        raise ValueError(f"Invalid radius_mode: {radius_mode}")

    # Step 1: Power iteration to get σ, u, v
    sigma, u, v = power_iteration(W, steps=power_iter_steps)

    # Step 2: Retract W to spectral sphere
    if sigma > 1e-8:
        W.mul_(R / sigma)

    # Step 3: Form Theta = u @ v^T (rank-1 matrix)
    Theta = u @ v.T

    # Normalize gradient
    G_normalized = G / (G.norm() + 1e-8)

    # Step 4: Solve for λ
    lambda_star = solve_lambda_bisection(
        G_normalized,
        Theta,
        tolerance=lambda_tolerance,
        max_iterations=lambda_max_iter,
        msign_steps=msign_steps,
    )

    # Step 5: Compute update direction
    Z = G_normalized + lambda_star * Theta
    A = msign(Z, steps=msign_steps)

    # Step 6: Update and retract
    new_W = W - eta * A

    # Retract to spectral sphere
    sigma_new, _, _ = power_iteration(new_W, steps=power_iter_steps)
    if sigma_new > 1e-8:
        new_W.mul_(R / sigma_new)

    # Restore shape if transposed
    return new_W.T if should_transpose else new_W
