"""Shared utilities: GRU student loading and environment building for M3 eval."""

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "mujoco_playground"))

import torch
import torch.nn as nn

from mujoco_playground import registry
from mujoco_playground._src.wrapper_torch import RSLRLBraxWrapper

# Default paths relative to repo root (legged_rl_sim2sim/).
DEFAULT_STUDENT_CKPT = "checkpoints/student_spot_proprio/student_spot_proprio.pt"
TEACHER_CKPT_DIR     = "mujoco_playground/logs/SpotJoystickRoughTerrain-20260428-123447/checkpoints"
ENV_NAME             = "SpotJoystickRoughTerrain"


class GRUStudent(nn.Module):
    """Proprio-only GRU policy: proprio(69) → GRU(256) → MLP([128]) → actions(12)."""

    def __init__(self, obs_dim: int, action_dim: int, rnn_hidden_dim: int, hidden_dims: list):
        super().__init__()
        self.gru = nn.GRU(obs_dim, rnn_hidden_dim, num_layers=1, batch_first=False)
        self.mlp = self._build_mlp(rnn_hidden_dim, hidden_dims, action_dim)

    @staticmethod
    def _build_mlp(in_dim: int, hidden_dims: list, out_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        """obs: (N, obs_dim), hidden: (1, N, hidden_dim) → actions (N, act_dim), hidden."""
        out, hidden = self.gru(obs.unsqueeze(0), hidden)
        return self.mlp(out.squeeze(0)), hidden

    def init_hidden(self, num_envs: int, device) -> torch.Tensor:
        return torch.zeros(1, num_envs, self.gru.hidden_size, device=device)


def extract_proprio(state_obs: torch.Tensor) -> torch.Tensor:
    """Drop feet_pos [54:66] from 81-dim state → 69-dim proprio."""
    return torch.cat([state_obs[:, :54], state_obs[:, 66:]], dim=-1)


def load_student(checkpoint_path: str | Path, device: str = "cuda:0") -> GRUStudent:
    """Load GRU student from a .pt checkpoint; infers architecture from saved weights."""
    p = Path(checkpoint_path)
    if not p.is_absolute():
        p = _REPO_ROOT / p

    payload = torch.load(p, map_location=device, weights_only=True)
    sd = payload["model"]

    # Infer architecture from weight shapes (no config file needed).
    obs_dim        = sd["gru.weight_ih_l0"].shape[1]   # (3H, obs_dim)
    rnn_hidden_dim = sd["gru.weight_hh_l0"].shape[1]   # (3H, H)
    linear_keys    = [k for k in sd if k.startswith("mlp.") and k.endswith(".weight")]
    hidden_dims    = [sd[k].shape[0] for k in linear_keys[:-1]]
    action_dim     = sd[linear_keys[-1]].shape[0]

    student = GRUStudent(obs_dim, action_dim, rnn_hidden_dim, hidden_dims).to(device)
    student.load_state_dict(sd)
    student.eval()

    iteration = payload.get("iteration", "?")
    print(f"[load_student] {p.name}  iter={iteration}  "
          f"obs={obs_dim}  rnn={rnn_hidden_dim}  mlp={hidden_dims}  act={action_dim}")
    return student


def build_env_wrapper(
    env_name: str = ENV_NAME,
    num_envs: int = 128,
    seed: int = 1,
) -> RSLRLBraxWrapper:
    """RSLRLBraxWrapper for metrics evaluation (same config as training)."""
    cfg     = registry.get_default_config(env_name)
    raw_env = registry.load(env_name, config=cfg)
    return RSLRLBraxWrapper(
        raw_env,
        num_actors     = num_envs,
        seed           = seed,
        episode_length = cfg.episode_length,
        action_repeat  = 1,
    )


def build_env_raw(env_name: str = ENV_NAME, residual_ckpt: str | None = None):
    """Raw Brax/JAX env for single-env rendering and fixed-command evaluation.

    If residual_ckpt is provided, build the M4 SpotJoystickResidualEnv with the
    trained residual injected at every physics substep. This lets the M3 student
    drive an env whose dynamics approximate accurate-CPU physics WITHOUT any
    PPO finetune -- a direct test of whether the residual closes the sim2sim
    gap at inference time.
    """
    cfg = registry.get_default_config(env_name)

    if residual_ckpt is None:
        return registry.load(env_name, config=cfg)

    if env_name != ENV_NAME:
        raise ValueError(
            f"Residual injection is only wired up for {ENV_NAME}, got {env_name}."
        )

    # Lazy import to avoid pulling Flax + torch into every eval script.
    sys.path.insert(0, str(_REPO_ROOT / "train" / "actuator_residual"))
    from finetune_teacher import SpotJoystickResidualEnv, port_weights

    params, in_mean, in_std, out_mean, out_std = port_weights(residual_ckpt)
    print(f"[build_env_raw] residual injected from {residual_ckpt}")
    return SpotJoystickResidualEnv(
        config=cfg,
        residual_params=params,
        in_mean=in_mean, in_std=in_std,
        out_mean=out_mean, out_std=out_std,
    )
