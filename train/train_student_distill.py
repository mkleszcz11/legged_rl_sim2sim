"""DAgger distillation: proprio-only GRU student trained from the frozen M2 Spot teacher.

Observation spaces:
  teacher input — state (81-dim):  noisy_gyro[0:3], noisy_gravity[3:6],
                                   noisy_joint_angles[6:18], qpos_error_history[18:54],
                                   noisy_feet_pos[54:66], last_act[66:78], command[78:81]
  student input — proprio (69-dim): same but indices [54:66] (feet_pos) are dropped.

Training loop (DAgger):
  1. Student acts in the environment using proprio observations.
  2. Teacher labels every transition with the action it would take given the full state.
  3. Student is updated via supervised MSE loss against teacher labels.

Usage:
  python train/train_student_distill.py [--config configs/distill_spot_proprio.yaml] [--wandb]
"""

import argparse
import importlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Env vars before any JAX/XLA import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "mujoco_playground"))
sys.path.insert(0, str(_PROJECT_ROOT / "train"))

from mujoco_playground import registry
from mujoco_playground._src.wrapper_torch import RSLRLBraxWrapper, _jax_to_torch
from utils.jax_oracle import load_teacher


class GRUStudent(nn.Module):
    """Proprioceptive GRU policy: proprio(69) → GRU(256) → MLP([128]) → actions(12).

    The GRU accumulates temporal history that substitutes for the teacher's explicit
    feet_pos observations.  Hidden state is reset to zero on episode termination.
    """

    def __init__(self, obs_dim: int, action_dim: int, rnn_hidden_dim: int, hidden_dims: list):
        super().__init__()
        self.gru = nn.GRU(obs_dim, rnn_hidden_dim, num_layers=1, batch_first=False)
        self.mlp = self._build_mlp(rnn_hidden_dim, hidden_dims, action_dim)

    @staticmethod
    def _build_mlp(in_dim: int, hidden_dims: list, out_dim: int) -> nn.Sequential:
        layers = nn.ModuleList()
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        """One-step inference.

        Args:
            obs:    (num_envs, obs_dim)       — current proprio observation.
            hidden: (1, num_envs, hidden_dim) — GRU hidden state.

        Returns:
            actions: (num_envs, action_dim)
            hidden:  (1, num_envs, hidden_dim)  updated hidden state.
        """
        out, hidden = self.gru(obs.unsqueeze(0), hidden)  # out: (1, N, H)
        actions = self.mlp(out.squeeze(0))                 # (N, action_dim)
        return actions, hidden

    def init_hidden(self, num_envs: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, num_envs, self.gru.hidden_size, device=device)


def extract_proprio(state_obs: torch.Tensor) -> torch.Tensor:
    """Drop feet_pos [54:66] from the 81-dim state to produce 69-dim proprio."""
    return torch.cat([state_obs[:, :54], state_obs[:, 66:]], dim=-1)


@dataclass
class RolloutStorage:
    """Pre-allocated tensors holding one DAgger iteration of experience."""
    proprio: torch.Tensor        # (T, N, 69)
    teacher_acts: torch.Tensor   # (T, N, 12)
    dones: torch.Tensor          # (T, N)  float: 1.0 = episode terminated

    def mini_batch_iter(self, num_mini_batches: int) -> Iterator[dict]:
        """Yield mini-batches split along the environment dimension.

        Environments are randomly shuffled each call, so each epoch sees a
        different partition.
        """
        num_envs = self.proprio.shape[1]
        batch_size = num_envs // num_mini_batches
        perm = torch.randperm(num_envs, device=self.proprio.device)
        for start in range(0, num_envs, batch_size):
            idx = perm[start : start + batch_size]
            yield {
                "proprio":      self.proprio[:, idx],       # (T, B, 69)
                "teacher_acts": self.teacher_acts[:, idx],  # (T, B, 12)
                "dones":        self.dones[:, idx],         # (T, B)
            }


def collect_rollout(
    env: RSLRLBraxWrapper,
    student: GRUStudent,
    oracle,
    obs,                  # TensorDict from previous step (or env.reset())
    hidden: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> tuple:
    """Run student in the environment for num_steps_per_env steps.

    Teacher is queried on the CURRENT observation before env.step(), so each
    label answers "what should I do in this state?" (not the next state).

    GRU hidden state is zeroed for any environment that terminates mid-rollout,
    since the auto-reset wrapper returns the first observation of the new episode.

    Args:
        obs:    TensorDict with key "state" (N, 81) — observation entering this rollout.
        hidden: (1, N, 256) — GRU hidden state entering this rollout.

    Returns:
        (obs, hidden, storage, metrics)
        obs and hidden are the values at the END of the rollout, ready for the next iteration.
    """
    num_steps: int  = cfg["training"]["num_steps_per_env"]
    num_envs:  int  = cfg["env"]["num_envs"]
    ctrl_dt:   float = cfg["env"]["ctrl_dt"]

    storage = RolloutStorage(
        proprio      = torch.zeros(num_steps, num_envs, cfg["student"]["obs_dim"],    device=device),
        teacher_acts = torch.zeros(num_steps, num_envs, cfg["student"]["action_dim"], device=device),
        dones        = torch.zeros(num_steps, num_envs,                               device=device),
    )

    # Gait-frequency tracking: count foot liftoffs via feet_air_time edge detection.
    # feet_air_time == 0  →  foot is in contact;  > 0  →  foot is in the air.
    prev_feet_air = torch.zeros(num_envs, 4, device=device)
    swing_counts  = torch.zeros(num_envs,    device=device)

    # Episode survival tracking.
    ep_steps       = torch.zeros(num_envs, device=device)
    survival_count = torch.zeros(num_envs, device=device)
    total_episodes = torch.zeros(num_envs, device=device)

    student.eval()
    with torch.no_grad():
        for t in range(num_steps):
            state_obs = obs["state"]                      # (N, 81)
            proprio   = extract_proprio(state_obs)        # (N, 69)

            # Query teacher BEFORE stepping — labels "what to do in this state".
            teacher_acts         = oracle.query(state_obs)          # (N, 12)
            student_acts, hidden = student(proprio, hidden)         # (N, 12)

            storage.proprio[t]      = proprio
            storage.teacher_acts[t] = teacher_acts

            obs, _reward, done, info = env.step(student_acts)
            storage.dones[t] = done

            # done[i] == 1 means env i auto-reset; its next obs starts a new episode.
            hidden = hidden * (1.0 - done).view(1, -1, 1)

            # Gait frequency: detect foot liftoffs from feet_air_time in env state.
            # This is a per-env JAX array in env.env_state.info, not the wrapper's
            # info_ret which stores only scalar means.
            feet_air      = _jax_to_torch(env.env_state.info["feet_air_time"])  # (N, 4)
            liftoffs      = (feet_air > 0) & (prev_feet_air == 0)
            swing_counts += liftoffs.float().sum(dim=1)
            prev_feet_air = feet_air.clone()

            # Track episode survival: survived = reached ≥90% of the max episode length.
            ep_steps += 1.0
            terminated = done > 0.5
            if terminated.any():
                survived = (ep_steps[terminated] >= env.max_episode_length * 0.9).float()
                survival_count[terminated] += survived
                total_episodes[terminated] += 1.0
                ep_steps[terminated] = 0.0

    # Mean liftoffs per foot per second, averaged over all envs.
    gait_hz = swing_counts.mean().item() / (num_steps * ctrl_dt) / 4

    survival_rate = (
        survival_count.sum() / total_episodes.sum()
    ).item() if total_episodes.sum() > 0 else 0.0

    metrics = {
        "perf/gait_hz":       gait_hz,
        "perf/survival_rate": survival_rate,
    }
    # Append per-step mean reward components logged by the Brax wrapper.
    for k, v in info["log"].items():
        metrics[f"env/{k}"] = v

    student.train()
    return obs, hidden, storage, metrics


# ── Student update ─────────────────────────────────────────────────────────────

def optimize_student(
    student:   GRUStudent,
    storage:   RolloutStorage,
    optimizer: torch.optim.Optimizer,
    cfg:       dict,
    device:    torch.device,
) -> float:
    """Run supervised DAgger updates over the collected rollout.

    Uses PyTorch's batched nn.GRU call (cuDNN kernel) instead of a Python-level
    per-step loop.  This is ~100× faster and is the standard approach for RNN
    training.  Episode-boundary hidden-state masking is skipped here (acceptable
    approximation — masking is applied correctly in collect_rollout where it
    matters for behavioral correctness).

    Returns:
        Mean MSE (or Huber) loss across all gradient updates.
    """
    num_epochs       = cfg["training"]["num_learning_epochs"]
    num_mini_batches = cfg["training"]["num_mini_batches"]
    loss_type        = cfg["training"]["loss_type"]
    grad_clip        = cfg["training"]["grad_clip_norm"]

    total_loss, n_updates = 0.0, 0

    for _ in range(num_epochs):
        for batch in storage.mini_batch_iter(num_mini_batches):
            proprio      = batch["proprio"]        # (T, B, 69)
            teacher_acts = batch["teacher_acts"]   # (T, B, 12)
            batch_envs   = proprio.shape[1]

            # Single batched GRU call over the full T-step sequence.
            # proprio is already (T, B, obs_dim) — the format nn.GRU expects
            # with batch_first=False.
            hidden      = student.init_hidden(batch_envs, device)
            gru_out, _  = student.gru(proprio, hidden)    # (T, B, rnn_hidden_dim)
            student_acts = student.mlp(gru_out)            # (T, B, action_dim)

            if loss_type == "huber":
                loss = F.huber_loss(student_acts, teacher_acts)
            else:
                loss = F.mse_loss(student_acts, teacher_acts)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=grad_clip)
            optimizer.step()

            total_loss += loss.item()
            n_updates  += 1

    return total_loss / n_updates


# ── Checkpointing ──────────────────────────────────────────────────────────────

def save_checkpoint(student: GRUStudent, iteration: int, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"iteration": iteration, "model": student.state_dict()}
    torch.save(payload, directory / f"student_iter{iteration:07d}.pt")
    torch.save(payload, directory / f"{directory.name}.pt")  # rolling alias keyed by run
    print(f"  [ckpt] saved iteration {iteration:,}  →  {directory}")


# ── Environment builder ────────────────────────────────────────────────────────

def _resolve_randomization_fn(spec: str | None):
    """Import a randomization function from a 'module:function' spec string.

    Example: "mujoco_playground._src.locomotion.spot.randomize:domain_randomize"
    Returns None when spec is None or empty.
    """
    if not spec:
        return None
    module_name, fn_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module_name), fn_name)


def build_env(cfg: dict) -> RSLRLBraxWrapper:
    env_name   = cfg["env"]["name"]
    num_envs   = cfg["env"]["num_envs"]
    seed       = cfg["env"]["seed"]
    env_config = registry.get_default_config(env_name)
    raw_env    = registry.load(env_name, config=env_config)
    randomization_fn = _resolve_randomization_fn(cfg["env"].get("randomization_fn"))
    if randomization_fn is not None:
        print(f"  Physics-DR enabled: {cfg['env']['randomization_fn']}")
    return RSLRLBraxWrapper(
        raw_env,
        num_actors        = num_envs,
        seed              = seed,
        episode_length    = env_config.episode_length,
        action_repeat     = 1,
        randomization_fn  = randomization_fn,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DAgger student distillation for Spot locomotion")
    parser.add_argument(
        "--config", default="configs/distill_spot_proprio.yaml",
        help="YAML config path (relative to project root or absolute)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"])

    # ── W&B ───────────────────────────────────────────────────────────────────
    run_name = cfg["logging"]["run_name"] or f"distill_spot_proprio_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.wandb:
        import wandb
        wandb.init(project=cfg["logging"]["project"], name=run_name, config=cfg)
    print(f"Run: {run_name}")

    # ── Environment ───────────────────────────────────────────────────────────
    print("\nBuilding environment...")
    env = build_env(cfg)

    # ── Teacher ───────────────────────────────────────────────────────────────
    print("Loading teacher oracle...")
    teacher_ckpt = _PROJECT_ROOT / cfg["teacher"]["checkpoint_dir"]
    oracle = load_teacher(checkpoint_dir=teacher_ckpt, env_name=cfg["teacher"]["env_name"])

    # ── Student ───────────────────────────────────────────────────────────────
    student_cfg = cfg["student"]
    student = GRUStudent(
        obs_dim        = student_cfg["obs_dim"],
        action_dim     = student_cfg["action_dim"],
        rnn_hidden_dim = student_cfg["rnn_hidden_dim"],
        hidden_dims    = student_cfg["hidden_dims"],
    ).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {n_params:,}")

    optimizer = torch.optim.Adam(student.parameters(), lr=cfg["training"]["learning_rate"])

    # ── DAgger loop ───────────────────────────────────────────────────────────
    max_iter      = cfg["training"]["max_iterations"]
    log_interval  = cfg["logging"]["log_interval"]
    save_interval = cfg["logging"]["save_interval"]
    ckpt_dir      = _PROJECT_ROOT / cfg["logging"]["checkpoint_dir"]
    num_envs      = cfg["env"]["num_envs"]
    num_steps     = cfg["training"]["num_steps_per_env"]

    # Initial env reset and hidden state — both are maintained across iterations so
    # that the environment and GRU context carry over between DAgger iterations.
    obs    = env.reset()
    hidden = student.init_hidden(num_envs, device)

    total_steps = max_iter * num_envs * num_steps
    steps_str   = f"{total_steps/1e9:.1f}B" if total_steps >= 1e9 else f"{total_steps/1e6:.0f}M"
    print(f"\nStarting DAgger training: {max_iter:,} iterations  (≈{steps_str} env steps)\n")

    iter_times: list[float] = []

    for iteration in range(max_iter):
        t0 = time.perf_counter()

        obs, hidden, storage, env_metrics = collect_rollout(
            env, student, oracle, obs, hidden, cfg, device
        )
        loss = optimize_student(student, storage, optimizer, cfg, device)

        iter_time = time.perf_counter() - t0
        iter_times.append(iter_time)

        if iteration % log_interval == 0:
            window     = iter_times[-log_interval:]
            avg_time   = sum(window) / len(window)
            remaining  = (max_iter - iteration) * avg_time
            gait_hz    = env_metrics["perf/gait_hz"]
            survival   = env_metrics["perf/survival_rate"]
            print(
                f"[{iteration:7,}/{max_iter:,}]  loss={loss:.5f}  "
                f"gait={gait_hz:.2f}Hz  survival={survival:.2f}  "
                f"t={iter_time:.1f}s  remaining≈{remaining/3600:.1f}h"
            )
            if args.wandb:
                import wandb
                wandb.log({"train/loss": loss, **env_metrics}, step=iteration)

        if iteration % save_interval == 0:
            save_checkpoint(student, iteration, ckpt_dir)

    save_checkpoint(student, max_iter, ckpt_dir)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
