"""M4 residual network: train a PyTorch MLP on actuator torque-gap data.

Architecture: Linear(144,128) -> ELU -> Linear(128,128) -> ELU ->
              Linear(128,64)  -> ELU -> Linear(64,12)
Input:  4-step history of (qpos_err, qvel_err, tau_cmd) = 144 floats
Output: tau_delta = tau_cpu - tau_mjx per joint (12 floats)

Train/val split is episode-aware (split by substep blocks, not random shuffle)
to avoid temporal leakage between train and validation.

Run from repo root:
    python train/actuator_residual/residual_network.py \\
        --data train/actuator_residual/data/residual.npz \\
        --epochs 100 --batch 2048 \\
        --ckpt checkpoints/actuator_residual.pt

Smoke test:
    python train/actuator_residual/residual_network.py \\
        --data train/actuator_residual/data/residual.npz \\
        --epochs 1 --batch 256
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import INPUT_DIM, OUTPUT_DIM

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA = str(_REPO_ROOT / "train/actuator_residual/data/residual.npz")
_DEFAULT_CKPT = str(_REPO_ROOT / "checkpoints/actuator_residual.pt")
_DEFAULT_CURVE = str(_REPO_ROOT / "checkpoints/actuator_residual_curves.png")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResidualMLP(nn.Module):
    """3-hidden-layer MLP that predicts per-joint torque corrections."""

    def __init__(self, in_dim: int = INPUT_DIM, out_dim: int = OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ELU(),
            nn.Linear(128, 128),   nn.ELU(),
            nn.Linear(128, 64),    nn.ELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ResidualDataset(Dataset):
    """Wraps pre-computed (features, labels) arrays with input normalization."""

    def __init__(
        self,
        features: np.ndarray,  # (N, 144)
        labels:   np.ndarray,  # (N, 12)
        in_mean:  np.ndarray,
        in_std:   np.ndarray,
        out_mean: np.ndarray,
        out_std:  np.ndarray,
    ):
        features_norm = (features - in_mean) / np.maximum(in_std, 1e-8)
        labels_norm   = (labels   - out_mean) / np.maximum(out_std, 1e-8)
        self.x = torch.from_numpy(features_norm.astype(np.float32))
        self.y = torch.from_numpy(labels_norm.astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Train / val split (episode-aware)
# ---------------------------------------------------------------------------

def episode_aware_split(
    features: np.ndarray,
    labels:   np.ndarray,
    n_substeps: int,
    ctrl_steps: int,
    n_episodes: int,
    val_frac:   float = 0.2,
    seed:       int   = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by episode index to avoid temporal leakage.

    Returns (train_feat, train_lbl, val_feat, val_lbl).
    """
    rng = np.random.default_rng(seed)
    ep_size = n_substeps * ctrl_steps   # samples per episode (approximate)

    ep_indices = np.arange(n_episodes)
    rng.shuffle(ep_indices)
    n_val = max(1, int(n_episodes * val_frac))
    val_eps  = set(ep_indices[:n_val])
    train_eps = set(ep_indices[n_val:])

    def gather(eps_set):
        idx = []
        for ep in sorted(eps_set):
            start = ep * ep_size
            end   = min(start + ep_size, len(features))
            idx.extend(range(start, end))
        return np.array(idx, dtype=int)

    train_idx = gather(train_eps)
    val_idx   = gather(val_eps)
    return (
        features[train_idx], labels[train_idx],
        features[val_idx],   labels[val_idx],
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    out_std:    np.ndarray,
    device:     torch.device,
) -> dict:
    """Compute MSE and R² in original (un-normalized) units."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu())
        targets.append(y)
    preds   = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    # Denormalize.
    std = np.maximum(out_std, 1e-8)
    preds_dn   = preds   * std
    targets_dn = targets * std

    per_joint_mse = np.mean((preds_dn - targets_dn)**2, axis=0)  # (12,)
    per_joint_var = np.var(targets_dn, axis=0)
    per_joint_r2  = 1 - per_joint_mse / np.maximum(per_joint_var, 1e-8)

    return {
        "overall_mse":    float(per_joint_mse.mean()),
        "per_joint_mse":  per_joint_mse.tolist(),
        "per_joint_r2":   per_joint_r2.tolist(),
        "mean_r2":        float(per_joint_r2.mean()),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_curves(history: dict, per_joint_mse: list, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss (normalized)")
    axes[0].set_title("Train / Val Loss")
    axes[0].legend()

    axes[1].bar(range(len(per_joint_mse)), per_joint_mse)
    axes[1].set_xlabel("Joint index")
    axes[1].set_ylabel("MSE [N·m²]")
    axes[1].set_title("Per-joint Val MSE (original units)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Curves saved to {out_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    npz_path:   str,
    ckpt_path:  str,
    curve_path: str,
    epochs:     int   = 100,
    batch:      int   = 2048,
    lr:         float = 3e-4,
    val_frac:   float = 0.2,
    seed:       int   = 0,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading data from {npz_path} ...")
    npz = np.load(npz_path, allow_pickle=True)
    features = npz["features"]   # (N, 144)
    labels   = npz["labels"]     # (N, 12)
    meta     = npz["meta"][0]    # dict
    print(f"  {len(features)} samples  features={features.shape}  labels={labels.shape}")

    n_substeps  = int(meta.get("n_substeps",  5))
    ctrl_steps  = int(meta.get("ctrl_steps",  1000))
    n_episodes  = int(meta.get("n_episodes",  50))

    tr_feat, tr_lbl, va_feat, va_lbl = episode_aware_split(
        features, labels, n_substeps, ctrl_steps, n_episodes, val_frac, seed
    )
    print(f"  Train: {len(tr_feat)}  Val: {len(va_feat)}")

    # Normalization stats (computed on training split only).
    in_mean  = tr_feat.mean(axis=0).astype(np.float32)
    in_std   = tr_feat.std(axis=0).astype(np.float32)
    out_mean = tr_lbl.mean(axis=0).astype(np.float32)
    out_std  = tr_lbl.std(axis=0).astype(np.float32)

    train_ds = ResidualDataset(tr_feat, tr_lbl, in_mean, in_std, out_mean, out_std)
    val_ds   = ResidualDataset(va_feat, va_lbl, in_mean, in_std, out_mean, out_std)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  drop_last=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, drop_last=False, num_workers=2)

    model = ResidualMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    history    = {"train_loss": [], "val_loss": []}
    best_val   = float("inf")
    best_epoch = 0
    best_state = None

    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        # Train
        model.train()
        tr_losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        # Validate
        model.eval()
        va_losses = []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                va_losses.append(loss_fn(model(x), y).item())

        tr_loss = np.mean(tr_losses)
        va_loss = np.mean(va_losses)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if va_loss < best_val:
            best_val   = va_loss
            best_epoch = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:4d}/{epochs}  train={tr_loss:.6f}  val={va_loss:.6f}"
                  f"  best={best_val:.6f} (ep {best_epoch})")

    # Reload best weights and evaluate.
    model.load_state_dict(best_state)
    val_metrics = evaluate(model, val_dl, out_std, device)

    print(f"\nBest val MSE (normalized): {best_val:.6f} at epoch {best_epoch}")
    print(f"Val MSE (original units):  {val_metrics['overall_mse']:.6f} N·m²")
    print(f"Mean val R²:               {val_metrics['mean_r2']:.4f}")
    print("Per-joint val MSE [N·m²]:")
    for j, v in enumerate(val_metrics["per_joint_mse"]):
        print(f"  joint {j:2d}: {v:.6f}   R²={val_metrics['per_joint_r2'][j]:.4f}")

    # Save checkpoint.
    torch.save({
        "state_dict": best_state,
        "arch": {
            "in_dim":  INPUT_DIM,
            "out_dim": OUTPUT_DIM,
            "hidden":  [128, 128, 64],
            "activation": "elu",
        },
        "norm": {
            "in_mean":  in_mean,
            "in_std":   in_std,
            "out_mean": out_mean,
            "out_std":  out_std,
        },
        "val_metrics": val_metrics,
        "best_epoch":  best_epoch,
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")

    plot_curves(history, val_metrics["per_joint_mse"], curve_path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train M4 actuator residual MLP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",   default=_DEFAULT_DATA,  help="Path to residual.npz")
    p.add_argument("--ckpt",   default=_DEFAULT_CKPT,  help="Output checkpoint path")
    p.add_argument("--curves", default=_DEFAULT_CURVE, help="Output plot path")
    p.add_argument("--epochs", type=int,   default=100)
    p.add_argument("--batch",  type=int,   default=2048)
    p.add_argument("--lr",     type=float, default=3e-4)
    p.add_argument("--seed",   type=int,   default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        npz_path=args.data,
        ckpt_path=args.ckpt,
        curve_path=args.curves,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
    )
