"""JAX teacher oracle bridge.

Loads the frozen M2 JAX/Brax PPO teacher and exposes a single .query() method
that accepts a PyTorch tensor, runs the teacher inference in JAX, and returns
a PyTorch tensor — all via DLPack zero-copy.
"""

import os
import sys
from pathlib import Path

# Env vars must be set before JAX is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

# Make evaluate/m2/load_policy.py and the mujoco_playground package importable.
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "mujoco_playground"))
sys.path.insert(0, str(_PROJECT_ROOT / "evaluate" / "m2"))

import jax
from load_policy import load_policy  # from evaluate/m2/
from mujoco_playground._src.wrapper_torch import _jax_to_torch, _torch_to_jax


class JaxOracleBridge:
    """Wraps the frozen JAX teacher inference function for use inside a PyTorch loop.

    Usage:
        oracle = JaxOracleBridge(inference_fn)
        teacher_actions = oracle.query(state_obs_torch)  # (N, 81) → (N, 12)
    """

    def __init__(self, inference_fn, seed: int = 0):
        self._inference_fn = inference_fn
        self._rng = jax.random.PRNGKey(seed)

    def query(self, state_torch):
        """Query teacher for deterministic actions given the 81-dim state observation.

        Args:
            state_torch: float32 CUDA tensor of shape (num_envs, 81).

        Returns:
            float32 CUDA tensor of shape (num_envs, 12).
        """
        state_jax = _torch_to_jax(state_torch)
        # Split rng for stationarity even though the policy is deterministic.
        self._rng, rng = jax.random.split(self._rng)
        actions_jax, _ = self._inference_fn({"state": state_jax}, rng)
        return _jax_to_torch(actions_jax)


def load_teacher(checkpoint_dir: str | Path, env_name: str) -> JaxOracleBridge:
    """Load the M2 teacher checkpoint and return a ready-to-use oracle bridge.

    Args:
        checkpoint_dir: Path to the checkpoints/ directory (contains numeric subdirs).
        env_name:       Environment name as registered in mujoco_playground.

    Returns:
        JaxOracleBridge wrapping the jit-compiled teacher inference function.
    """
    # num_envs=1: the PPO setup compiles JAX env kernels for 1 environment instead
    # of the default 8192, reducing GPU memory usage from ~3 GB to ~50 MB.
    # The inference_fn itself is batch-size agnostic; JAX retraces on first real call.
    _env, inference_fn = load_policy(str(checkpoint_dir), env_name, num_envs=1)
    return JaxOracleBridge(inference_fn)
