import argparse
import math
import os
import pickle
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from hyperspherical_descent import hyperspherical_descent
from manifold_muon import manifold_muon
from spectral_ball import spectral_ball
from torch.optim import AdamW
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)


def get_mup_lr_scale_factor(d_out, d_in, scale_mode="spectral_mup"):
    """Get the MuP learning rate scale factor for a layer.

    Args:
        d_out: Output dimension (rows)
        d_in: Input dimension (columns)
        scale_mode: Scaling mode
            - "spectral_mup": (d_out / d_in) ** 0.5
            - "none": 1.0 (no scaling)

    Returns:
        Scale factor to multiply with base learning rate.
    """
    if scale_mode == "spectral_mup":
        return math.sqrt(d_out / d_in)
    elif scale_mode == "none":
        return 1.0
    else:
        raise ValueError(f"Invalid scale_mode: {scale_mode}")


def spectral_mup_init_method_normal(sigma=0.02):
    """Spectral MuP initialization: σ * √(d_out/d_in) / ||W'||₂ * W'

    This initialization method applies spectral normalization and MuP scaling to linear layers.
    For 2D weight matrices W ∈ R^(d_out × d_in):
    1. Initialize W' ~ N(0, σ)
    2. Compute spectral norm s = ||W'||₂
    3. Apply scaling: W = σ * √(d_out/d_in) / s * W'

    Args:
        sigma: Standard deviation for the initial normal distribution.

    Returns:
        Initialization function that can be applied to tensors.
    """
    def init_(tensor):
        # Skip non-2D parameters (bias, layernorm, etc.)
        if len(tensor.shape) != 2:
            return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

        d_out, d_in = tensor.shape

        # Step 1: Initialize W' ~ N(0, σ)
        torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

        # Step 2: Compute spectral norm s = ||W'||₂
        spectral_norm = torch.linalg.matrix_norm(tensor, ord=2)

        # Step 3: Apply MuP scaling: W = σ * √(d_out/d_in) / s * W'
        mup_scale = math.sqrt(d_out / d_in) / spectral_norm
        tensor.data.mul_(mup_scale)

        return tensor

    return init_


class ResidualBlock(nn.Module):
    """Residual block with two linear layers."""
    def __init__(self, dim, use_spectral_mup_init=True):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)

        if use_spectral_mup_init:
            init_fn = spectral_mup_init_method_normal(sigma=0.02)
            init_fn(self.fc1.weight)
            init_fn(self.fc2.weight)

    def forward(self, x):
        residual = x
        out = torch.relu(self.fc1(x))
        out = self.fc2(out)
        out = out + residual  # Residual connection
        out = torch.relu(out)
        return out


class ResNet5(nn.Module):
    """5-layer ResNet: 1 input layer + 3 residual blocks + 1 output layer."""
    def __init__(self, hidden_dim=256, use_spectral_mup_init=True):
        super(ResNet5, self).__init__()
        # Input layer
        self.fc_in = nn.Linear(32 * 32 * 3, hidden_dim, bias=False)

        # 3 residual blocks (each has 2 linear layers)
        self.res_block1 = ResidualBlock(hidden_dim, use_spectral_mup_init)
        self.res_block2 = ResidualBlock(hidden_dim, use_spectral_mup_init)
        self.res_block3 = ResidualBlock(hidden_dim, use_spectral_mup_init)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, 10, bias=False)

        # Apply spectral MuP initialization to input and output layers
        if use_spectral_mup_init:
            init_fn = spectral_mup_init_method_normal(sigma=0.02)
            init_fn(self.fc_in.weight)
            init_fn(self.fc_out.weight)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = torch.relu(self.fc_in(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.fc_out(x)
        return x


def train(epochs, initial_lr, update, wd, use_mup_lr=True):
    model = ResNet5(hidden_dim=256).cuda()
    criterion = nn.CrossEntropyLoss()

    if update == AdamW:
        optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=wd)
    else:
        assert update in [manifold_muon, hyperspherical_descent, spectral_ball]
        optimizer = None

    steps = epochs * len(train_loader)
    step = 0

    # Compute MuP LR scale factors for each parameter
    mup_lr_scales = {}
    if optimizer is None and use_mup_lr:
        for name, p in model.named_parameters():
            if p.ndim == 2:  # Only for weight matrices
                d_out, d_in = p.shape
                mup_lr_scales[name] = get_mup_lr_scale_factor(d_out, d_in, scale_mode="spectral_mup")
            else:
                mup_lr_scales[name] = 1.0
    else:
        # No MuP scaling for AdamW or if disabled
        for name, p in model.named_parameters():
            mup_lr_scales[name] = 1.0

    if optimizer is None:
        # Project the weights to the manifold
        for p in model.parameters():
            p.data = update(p.data, torch.zeros_like(p.data), eta=0)

    epoch_losses = []
    epoch_times = []

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            base_lr = initial_lr * (1 - step / steps)
            with torch.no_grad():
                if optimizer is None:
                    # Apply MuP LR scaling for each parameter
                    for name, p in model.named_parameters():
                        lr = base_lr * mup_lr_scales[name]
                        p.data = update(p, p.grad, eta=lr)
                else:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = base_lr
                    optimizer.step()
            step += 1

            running_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        end_time = time.time()
        epoch_loss = running_loss / len(train_loader)
        epoch_time = end_time - start_time
        epoch_losses.append(epoch_loss)
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Time: {epoch_time:.4f} seconds")
    return model, epoch_losses, epoch_times


def eval(model):
    # Test the model
    model.eval()
    with torch.no_grad():
        accs = []
        for dataloader in [test_loader, train_loader]:
            correct = 0
            total = 0
            for images, labels in dataloader:
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accs.append(100 * correct / total)

    print(f"Accuracy of the network on the {len(test_loader.dataset)} test images: {accs[0]} %")
    print(f"Accuracy of the network on the {len(train_loader.dataset)} train images: {accs[1]} %")
    return accs

def weight_stats(model):
    singular_values = []
    norms = []
    for p in model.parameters():
        u,s,v = torch.svd(p)
        singular_values.append(s)
        norms.append(p.norm())
    return singular_values, norms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate.")
    parser.add_argument("--update", type=str, default="manifold_muon", choices=["manifold_muon", "spectral_ball", "hyperspherical_descent", "adam"], help="Update rule to use.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay for AdamW.")
    args = parser.parse_args()

    # determinism flags
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    update_rules = {
        "manifold_muon": manifold_muon,
        "spectral_ball": spectral_ball,
        "hyperspherical_descent": hyperspherical_descent,
        "adam": AdamW
    }

    update = update_rules[args.update]

    print(f"Training with: {args.update}")
    print(f"Epochs: {args.epochs} --- LR: {args.lr}", f"--- WD: {args.wd}" if args.update == "adam" else "")

    model, epoch_losses, epoch_times = train(
        epochs=args.epochs,
        initial_lr=args.lr,
        update=update,
        wd=args.wd
    )
    test_acc, train_acc = eval(model)
    singular_values, norms = weight_stats(model)

    results = {
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "wd": args.wd,
        "update": args.update,
        "epoch_losses": epoch_losses,
        "epoch_times": epoch_times,
        "test_acc": test_acc,
        "train_acc": train_acc,
        "singular_values": singular_values,
        "norms": norms
    }

    filename = f"update-{args.update}-lr-{args.lr}-wd-{args.wd}-seed-{args.seed}.pkl"
    os.makedirs("results", exist_ok=True)

    print(f"Saving results to {os.path.join("results", filename)}")
    with open(os.path.join("results", filename), "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {os.path.join("results", filename)}")
