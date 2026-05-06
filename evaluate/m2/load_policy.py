"""Shared utilities: checkpoint loading and environment creation for M2 eval."""

import functools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mujoco_playground"))

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

_KERNEL_INIT_FNS = {
    "lecun_uniform": jax.nn.initializers.lecun_uniform,
    "lecun_normal": jax.nn.initializers.lecun_normal,
    "glorot_uniform": jax.nn.initializers.glorot_uniform,
    "glorot_normal": jax.nn.initializers.glorot_normal,
    "he_uniform": jax.nn.initializers.he_uniform,
    "he_normal": jax.nn.initializers.he_normal,
    "orthogonal": jax.nn.initializers.orthogonal,
    "variance_scaling": jax.nn.initializers.variance_scaling,
}

_ACTIVATIONS = {
    "silu": jax.nn.silu,
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
    "elu": jax.nn.elu,
    "swish": jax.nn.swish,
    "gelu": jax.nn.gelu,
    "softplus": jax.nn.softplus,
    "sigmoid": jax.nn.sigmoid,
}

# Default checkpoint paths (relative to repo root unitree_go2_rl/)
DEFAULT_ROUGH_CKPT = (
    "mujoco_playground/logs/SpotJoystickRoughTerrain-20260428-123447/checkpoints"
)
DEFAULT_FLAT_CKPT = (
    "mujoco_playground/logs/SpotFlatTerrainJoystick-20260427-202132/checkpoints"
)


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path:
    """Return Path to the latest numeric subdirectory inside checkpoint_dir."""
    checkpoint_dir = Path(checkpoint_dir)
    ckpts = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not ckpts:
        raise FileNotFoundError(
            f"No numeric checkpoint directories found in {checkpoint_dir}"
        )
    return ckpts[-1]


def read_network_config(checkpoint_dir: str | Path) -> dict:
    """Read ppo_network_config.json from the latest checkpoint."""
    latest = find_latest_checkpoint(checkpoint_dir)
    config_path = latest / "ppo_network_config.json"
    with open(config_path) as f:
        return json.load(f)


def make_env(env_name: str, config_overrides: dict | None = None):
    """Create a MuJoCo Playground environment with optional config overrides."""
    cfg = registry.get_default_config(env_name)
    return registry.load(env_name, config=cfg, config_overrides=config_overrides or {})


def load_policy(checkpoint_dir: str | Path, env_name: str,
                config_overrides: dict | None = None,
                checkpoint_path: Path | None = None,
                num_envs: int | None = None,
                env=None):
    """Load a PPO policy from a checkpoint directory.

    Mirrors the --play_only path in train_jax_ppo.py:
      - Reads network architecture from ppo_network_config.json
      - Calls ppo.train(num_timesteps=0, restore_checkpoint_path=...)
      - Returns (env, jit_inference_fn)

    Args:
        checkpoint_dir:   Path to the checkpoints/ directory (contains numeric subdirs).
        env_name:         Environment name registered in mujoco_playground. Used to
                          look up PPO hyperparameters (locomotion_params.brax_ppo_config).
                          When `env` is also provided, this name is NOT used to construct
                          the env -- only for the params lookup.
        config_overrides: Optional env config overrides (e.g. episode_length). Only
                          applied when `env` is None.
        checkpoint_path:  Specific numeric checkpoint subdir to load. Defaults to latest.
        num_envs:         Override the default PPO num_envs for the JIT compilation.
                          Use a small value (e.g. 1) when loading for inference only to
                          avoid allocating GPU memory for thousands of dummy environments.
                          The returned inference_fn is batch-size agnostic regardless.
        env:              Pre-built environment to use instead of constructing one from
                          `env_name`. Required when the env is a custom subclass not in
                          the registry (e.g. SpotJoystickResidualEnv).
    """
    if env is not None and config_overrides:
        raise ValueError(
            "config_overrides is incompatible with a pre-built `env`; "
            "apply overrides when constructing the env instead."
        )

    checkpoint_dir = Path(checkpoint_dir)
    ckpt_to_load = Path(checkpoint_path) if checkpoint_path else find_latest_checkpoint(checkpoint_dir)
    net_cfg = read_network_config(checkpoint_dir)

    policy_hidden = tuple(net_cfg["network_factory_kwargs"]["policy_hidden_layer_sizes"])
    value_hidden = tuple(net_cfg["network_factory_kwargs"]["value_hidden_layer_sizes"])
    policy_obs_key = net_cfg["network_factory_kwargs"]["policy_obs_key"]
    value_obs_key = net_cfg["network_factory_kwargs"]["value_obs_key"]

    print(f"[load_policy] env={env_name}")
    print(f"[load_policy] checkpoint={ckpt_to_load}")
    print(f"[load_policy] policy_hidden={policy_hidden}, value_hidden={value_hidden}")
    print(f"[load_policy] policy_obs_key={policy_obs_key}, value_obs_key={value_obs_key}")

    if env is None:
        env = make_env(env_name, config_overrides=config_overrides)

    # PPO hyperparams from the registry defaults for this env
    ppo_params = locomotion_params.brax_ppo_config(env_name, "jax")
    training_params = dict(ppo_params)
    training_params.pop("network_factory", None)
    num_eval_envs = training_params.pop("num_eval_envs", 1)

    # Play-only: skip all training
    training_params["num_timesteps"] = 0
    training_params["num_evals"] = 0
    if num_envs is not None:
        training_params["num_envs"] = num_envs

    # Build network factory exactly matching the saved checkpoint
    net_kwargs = {k: v for k, v in net_cfg["network_factory_kwargs"].items() if v is not None}
    net_kwargs["policy_hidden_layer_sizes"] = policy_hidden
    net_kwargs["value_hidden_layer_sizes"] = value_hidden
    # JSON stores callables as strings; resolve them back to Python objects
    for key in ("policy_network_kernel_init_fn", "value_network_kernel_init_fn", "mean_kernel_init_fn"):
        if key in net_kwargs and isinstance(net_kwargs[key], str):
            name = net_kwargs[key]
            if name not in _KERNEL_INIT_FNS:
                raise ValueError(f"Unknown kernel init fn '{name}' in ppo_network_config.json")
            net_kwargs[key] = _KERNEL_INIT_FNS[name]
    if "activation" in net_kwargs and isinstance(net_kwargs["activation"], str):
        name = net_kwargs["activation"]
        if name not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{name}' in ppo_network_config.json")
        net_kwargs["activation"] = _ACTIVATIONS[name]
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_kwargs)

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        **training_params,
        network_factory=network_factory,
        seed=1,
        restore_checkpoint_path=epath.Path(str(ckpt_to_load)),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
        run_evals=False,
    )

    inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
    print("[load_policy] Policy loaded successfully.\n")
    return env, inference_fn
