"""M4 finetuning: inject the trained actuator residual into the MJX env and
resume PPO training from the M2 teacher checkpoint.

Steps:
  1. Load trained actuator_residual.pt and port weights to a Flax MLP.
  2. Build SpotJoystickResidualEnv -- a subclass that:
       - Converts position actuators to raw torque passthrough.
       - At each physics substep: computes tau_pd + tau_residual and uses it as ctrl.
       - Maintains a 4-step ring buffer of (qpos_err, qvel_err, tau_cmd) in state.info.
  3. Resume Brax PPO from the M2 checkpoint for 30M steps.
  4. Write the finetuned checkpoint to:
       mujoco_playground/logs/SpotJoystickRoughTerrain_residual-<timestamp>/

Run from repo root:
    python train/actuator_residual/finetune_teacher.py \\
        --teacher_ckpt mujoco_playground/logs/SpotJoystickRoughTerrain-20260428-123447/checkpoints \\
        --residual_ckpt checkpoints/actuator_residual.pt

Smoke test (fast):
    python train/actuator_residual/finetune_teacher.py \\
        --num_timesteps 200000 --num_evals 2 --smoke
"""

import argparse
import datetime
import functools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# JAX env vars must be set before any JAX / MuJoCo import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "mujoco_playground"))
sys.path.insert(0, str(_REPO_ROOT / "evaluate" / "m2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flax.linen as nn
import jax
import jax.numpy as jp
import numpy as np
import torch
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground import registry, wrapper
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.spot import joystick as spot_joystick
from mujoco_playground.config import locomotion_params

import load_policy  # evaluate/m2/load_policy.py
from features import FEAT_DIM, HISTORY_LEN, INPUT_DIM, KD, KP, OUTPUT_DIM

_DEFAULT_TEACHER_CKPT = str(
    _REPO_ROOT / "mujoco_playground/logs/SpotJoystickRoughTerrain-20260428-123447/checkpoints"
)
_DEFAULT_RESIDUAL_CKPT = str(_REPO_ROOT / "checkpoints/actuator_residual.pt")


# ---------------------------------------------------------------------------
# Flax residual MLP (must mirror residual_network.ResidualMLP exactly)
# ---------------------------------------------------------------------------

class FlaxResidualMLP(nn.Module):
    """Flax equivalent of residual_network.ResidualMLP.

    Architecture: 144 -> 128 (ELU) -> 128 (ELU) -> 64 (ELU) -> 12
    """
    hidden: tuple = (128, 128, 64)
    out_dim: int  = OUTPUT_DIM

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = jax.nn.elu(x)
        return nn.Dense(self.out_dim)(x)


def port_weights(pt_ckpt_path: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load PyTorch checkpoint and convert to Flax params dict.

    PyTorch nn.Linear.weight is (out, in); Flax Dense.kernel is (in, out).
    Always transpose kernel.  Biases copy directly.

    Returns:
        params:   Flax FrozenDict-compatible dict {'params': {'Dense_0': ..., ...}}
        in_mean:  (INPUT_DIM,) float32
        in_std:   (INPUT_DIM,) float32
        out_mean: (OUTPUT_DIM,) float32
        out_std:  (OUTPUT_DIM,) float32
    """
    ckpt = torch.load(pt_ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt["state_dict"]
    norm = ckpt["norm"]

    # net.0, net.2, net.4, net.6 are the Linear layers (net.1/3/5 are ELU).
    layer_map = [("net.0", "Dense_0"), ("net.2", "Dense_1"),
                 ("net.4", "Dense_2"), ("net.6", "Dense_3")]

    params_inner = {}
    for pt_prefix, flax_name in layer_map:
        w = sd[f"{pt_prefix}.weight"].numpy().T   # transpose (out,in) -> (in,out)
        b = sd[f"{pt_prefix}.bias"].numpy()
        params_inner[flax_name] = {
            "kernel": jax.device_put(jnp_cast(w)),
            "bias":   jax.device_put(jnp_cast(b)),
        }

    params = {"params": params_inner}

    # Sanity check: forward both nets on the same random input.
    _sanity_check_port(pt_ckpt_path, sd, params, norm)

    return (
        params,
        norm["in_mean"].astype(np.float32),
        norm["in_std"].astype(np.float32),
        norm["out_mean"].astype(np.float32),
        norm["out_std"].astype(np.float32),
    )


def jnp_cast(x: np.ndarray) -> jp.ndarray:
    return jp.array(x, dtype=jp.float32)


def _sanity_check_port(pt_ckpt_path: str, sd: dict, flax_params: dict, norm: dict) -> None:
    """Assert PyTorch and Flax produce the same output on a random input."""
    from residual_network import ResidualMLP

    rng_input = np.random.default_rng(42).standard_normal((1, INPUT_DIM)).astype(np.float32)

    # PyTorch forward.
    pt_model = ResidualMLP()
    pt_model.load_state_dict(sd)
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(rng_input)).numpy()

    # Flax forward.
    flax_model = FlaxResidualMLP()
    flax_out = np.array(flax_model.apply(flax_params, jp.array(rng_input[0])))

    max_diff = float(np.abs(pt_out[0] - flax_out).max())
    print(f"  Weight port sanity check: max|pt - flax| = {max_diff:.2e}")
    if max_diff > 1e-4:
        raise RuntimeError(
            f"PyTorch -> Flax weight port failed: max diff {max_diff:.2e} > 1e-4. "
            "Check layer ordering and transpose convention."
        )
    print("  Weight port OK.")


# ---------------------------------------------------------------------------
# Custom MJX environment with residual torque injection
# ---------------------------------------------------------------------------

class SpotJoystickResidualEnv(spot_joystick.Joystick):
    """Spot joystick env with an actuator residual injected at every physics step.

    Changes vs parent Joystick:
      - Position actuators converted to raw torque passthrough in __init__.
      - Each physics substep computes: ctrl = tau_pd + tau_residual.
      - state.info gains a 'residual_hist' ring buffer of shape (HISTORY_LEN, FEAT_DIM).
    """

    def __init__(
        self,
        task: str = "rough_terrain",
        config: config_dict.ConfigDict = spot_joystick.default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        residual_params: Optional[dict] = None,
        in_mean:  Optional[np.ndarray] = None,
        in_std:   Optional[np.ndarray] = None,
        out_mean: Optional[np.ndarray] = None,
        out_std:  Optional[np.ndarray] = None,
    ):
        super().__init__(task=task, config=config, config_overrides=config_overrides)

        # ----- Convert position actuators to raw torque passthrough -----
        # MuJoCo position actuator law: eff = gainprm[0]*ctrl + biasprm[1]*q + biasprm[2]*qd.
        # SpotEnv overrides gainprm[:,0]=Kp and biasprm[:,1]=-Kp but leaves biasprm[:,2]
        # at the XML's -kv (=-20). So the original env's actuator output is
        #     actuator_force = Kp*(target - q) - kv*qd
        # and dof_damping=1 contributes an additional -1*qd via qfrc_passive. Total
        # damping seen by the joint is kv+1 = 21.
        #
        # To preserve BOTH the dynamics AND the value of data.actuator_force (which
        # feeds privileged_state and therefore the value function), we:
        #   - apply tau_pd = -Kp*q_err - kv*qd as ctrl (matches original actuator)
        #   - KEEP dof_damping=1 so qfrc_passive contributes the remaining -1*qd
        # This way data.actuator_force in the residual env equals the original env's
        # actuator_force PLUS tau_res, and total dynamics are unchanged. Earlier
        # versions zeroed dof_damping and rolled it into ctrl, which silently shifted
        # data.actuator_force by -1*qd and biased the restored value function.
        actuator_kv = -self._mj_model.actuator_biasprm[:, 2].copy()  # (nu,) = +20

        # Setting gainprm[0]=1, biasprm[1]=0, biasprm[2]=0 makes eff=ctrl (raw torque).
        self._mj_model.actuator_gainprm[:, 0] = 1.0
        self._mj_model.actuator_biasprm[:, 1] = 0.0
        self._mj_model.actuator_biasprm[:, 2] = 0.0
        # Remove control range limits (position range is meaningless for torques).
        self._mj_model.actuator_ctrllimited[:] = 0
        # NOTE: dof_damping is intentionally NOT zeroed -- it contributes the -1*qd
        # passive damping that the original env had on top of the actuator's -kv*qd.
        # Rebuild MJX model -- MJX caches the model at put_model time.
        self._mjx_model = mjx.put_model(self._mj_model)

        # KD (= 1.0) is the conventional value used by features.py / collect_data.py
        # when computing the tau_cmd FEATURE for the residual network. Keep it for
        # feature computation so the network input distribution stays in-distribution.
        self._kp = KP
        self._kd = KD

        # Per-joint kv (=20) used by substep_fn to drive the env's actuator. Combined
        # with passive dof_damping=1 this reproduces the original env's total damping.
        self._actuator_kv = jp.array(actuator_kv, dtype=jp.float32)

        # Store residual as JAX arrays closed over at construction time.
        # Closing over is jit-safe since params are static pytrees.
        if residual_params is not None:
            self._residual_params = jax.tree_util.tree_map(jp.asarray, residual_params)
            self._in_mean  = jp.array(in_mean,  dtype=jp.float32)
            self._in_std   = jp.array(in_std,   dtype=jp.float32)
            self._out_mean = jp.array(out_mean, dtype=jp.float32)
            self._out_std  = jp.array(out_std,  dtype=jp.float32)
            self._flax_model = FlaxResidualMLP()
        else:
            # Allow instantiation without residual (for eval of baseline).
            self._residual_params = None

    def _apply_residual(self, flat_hist: jax.Array) -> jax.Array:
        """Normalize input, run Flax MLP, denormalize output."""
        flat_norm = (flat_hist - self._in_mean) / jp.maximum(self._in_std, 1e-8)
        tau_norm  = self._flax_model.apply(self._residual_params, flat_norm)
        return tau_norm * jp.maximum(self._out_std, 1e-8) + self._out_mean

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        state.info["residual_hist"] = jp.zeros((HISTORY_LEN, FEAT_DIM))
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng, cmd_rng, noise_rng, pert_rng = jax.random.split(state.info["rng"], 4)

        state = self._pert_func(state, pert_rng)

        motor_targets = self._default_pose + action * self._config.action_scale
        motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)

        # The Brax AutoResetWrapper (full_reset=False, the default) resets only data
        # and obs on episode end -- state.info is preserved verbatim. So residual_hist
        # carries over across episode boundaries, leaving the network with stale
        # features for the first ~4 substeps of every new episode (when the robot is
        # back at the home pose with zero velocity, the most fragile moment). The
        # parent env resets info["step"] to 0 at the end of any step that ended an
        # episode (or refreshed the command), so step==0 here means "new episode or
        # command refresh just happened" -- the exact moments residual_hist is stale.
        residual_hist = jp.where(
            state.info["step"] == 0,
            jp.zeros((HISTORY_LEN, FEAT_DIM)),
            state.info["residual_hist"],
        )

        # Run n_substeps with per-substep residual injection.
        def substep_fn(carry, _):
            data, hist = carry

            q_err = data.qpos[7:] - motor_targets    # (12,)
            qd    = data.qvel[6:]                    # (12,)

            # Two tau values:
            #   tau_cmd_feat: feature for the residual network. Uses self._kd (=KD=1.0)
            #     to match the convention in collect_data.py / features.py. Changing
            #     this would shift the network input distribution OOD.
            #   tau_pd_env:   what we apply as ctrl. Uses _actuator_kv (=20), matching
            #     the original env's actuator force. The remaining -1*qd damping comes
            #     from passive dof_damping (which we kept at 1), so total joint damping
            #     is 21 as in the original env, AND data.actuator_force matches it.
            tau_cmd_feat = -self._kp * q_err - self._kd * qd            # (12,)  feature
            tau_pd_env   = -self._kp * q_err - self._actuator_kv * qd   # (12,)  physics

            feat = jp.concatenate([q_err, qd, tau_cmd_feat])     # (36,)
            hist = jp.roll(hist, -1, axis=0).at[-1].set(feat)    # oldest dropped, newest at [-1]

            if self._residual_params is not None:
                tau_res = self._apply_residual(hist.reshape(-1))
            else:
                tau_res = jp.zeros(OUTPUT_DIM)

            ctrl = tau_pd_env + tau_res
            data = data.replace(ctrl=ctrl)
            data = mjx.step(self._mjx_model, data)
            return (data, hist), None

        (data, hist), _ = jax.lax.scan(
            substep_fn,
            (state.data, residual_hist),
            None,
            self.n_substeps,
        )

        state.info["residual_hist"] = hist
        state.info["motor_targets"] = motor_targets

        # --- Everything below is identical to parent Joystick.step ---
        contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self._feet_floor_found_sensor
        ])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f  = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        obs  = self._get_obs(data, state.info, noise_rng)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"]      = action
        state.info["step"]         += 1
        state.info["rng"]           = rng

        should_refresh = state.info["step"] > 200
        state.info["step"] = jp.where(
            done | (state.info["step"] > 200), 0, state.info["step"]
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"]   = contact
        state.info["swing_peak"]    *= ~contact

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"]      = jp.mean(state.info["swing_peak"])

        vel_ok = jp.abs(self.get_local_linvel(data)[0] - state.info["command"][0]) < 0.25
        state.info["vel_ok_steps"] = state.info["vel_ok_steps"] + vel_ok.astype(jp.int32)

        ok_fraction = state.info["vel_ok_steps"] / self._config.episode_length
        advance = done & (ok_fraction > 0.8)
        state.info["curriculum_level"] = jp.minimum(
            state.info["curriculum_level"] + jp.where(advance, 1, 0), 9
        )
        state.metrics["curriculum_level"] = state.info["curriculum_level"].astype(jp.float32)

        max_speeds = jp.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5])
        new_cmd  = self.sample_command(cmd_rng)
        max_vx   = max_speeds[state.info["curriculum_level"]]
        new_cmd  = new_cmd.at[0].set(jp.clip(new_cmd[0], -max_vx, max_vx))
        state.info["command"] = jp.where(should_refresh, new_cmd, state.info["command"])

        done  = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state


# ---------------------------------------------------------------------------
# PPO training launcher
# ---------------------------------------------------------------------------

_KERNEL_INIT_FNS = {
    "lecun_uniform":     jax.nn.initializers.lecun_uniform,
    "lecun_normal":      jax.nn.initializers.lecun_normal,
    "glorot_uniform":    jax.nn.initializers.glorot_uniform,
    "glorot_normal":     jax.nn.initializers.glorot_normal,
    "he_uniform":        jax.nn.initializers.he_uniform,
    "he_normal":         jax.nn.initializers.he_normal,
    "orthogonal":        jax.nn.initializers.orthogonal,
    "variance_scaling":  jax.nn.initializers.variance_scaling,
}
_ACTIVATIONS = {
    "silu": jax.nn.silu, "relu": jax.nn.relu, "tanh": jax.nn.tanh,
    "elu":  jax.nn.elu,  "swish": jax.nn.swish, "gelu": jax.nn.gelu,
}


def _resolve_network_factory(teacher_ckpt_dir: str):
    """Read ppo_network_config.json and return a network_factory callable."""
    ckpt_path = load_policy.find_latest_checkpoint(teacher_ckpt_dir)
    with open(ckpt_path / "ppo_network_config.json") as f:
        net_cfg = json.load(f)

    nkw = {k: v for k, v in net_cfg["network_factory_kwargs"].items() if v is not None}
    nkw["policy_hidden_layer_sizes"] = tuple(nkw["policy_hidden_layer_sizes"])
    nkw["value_hidden_layer_sizes"]  = tuple(nkw["value_hidden_layer_sizes"])

    for key in ("policy_network_kernel_init_fn", "value_network_kernel_init_fn",
                "mean_kernel_init_fn"):
        if key in nkw and isinstance(nkw[key], str):
            nkw[key] = _KERNEL_INIT_FNS[nkw[key]]

    if "activation" in nkw and isinstance(nkw["activation"], str):
        nkw["activation"] = _ACTIVATIONS[nkw["activation"]]

    return functools.partial(ppo_networks.make_ppo_networks, **nkw)


def run_finetune(
    teacher_ckpt:      str,
    residual_ckpt:     str,
    num_timesteps:     int,
    num_evals:         int,
    out_base:          str,
    learning_rate:     float,
    entropy_cost:      Optional[float] = None,
    smoke:             bool = False,
    no_residual:       bool = False,
    use_original_env:  bool = False,
) -> None:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_base) / f"SpotJoystickRoughTerrain_residual-{ts}"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # --- Load and port residual weights (skip if --no_residual) ---
    if no_residual:
        print("\n--no_residual: skipping residual checkpoint load. The env will run "
              "with the M4 modifications (torque passthrough + manual PD) but no "
              "residual injection. This isolates whether PPO finetune collapse is "
              "caused by the env modifications themselves or by the residual.")
        params = in_mean = in_std = out_mean = out_std = None
    else:
        print(f"\nLoading residual checkpoint from {residual_ckpt} ...")
        params, in_mean, in_std, out_mean, out_std = port_weights(residual_ckpt)

    # --- Build env ---
    cfg = registry.get_default_config("SpotJoystickRoughTerrain")

    if use_original_env:
        print("\n--use_original_env: using the registered SpotJoystickRoughTerrain env "
              "(no M4 modifications). Diagnostic: if PPO collapses here too, the "
              "issue is PPO finetuning the M2 checkpoint, not the residual env subclass.")
        if not no_residual:
            raise ValueError("--use_original_env requires --no_residual (the original "
                             "env has no place to inject the residual).")

        def _make_env(cfg=cfg):
            return registry.load("SpotJoystickRoughTerrain", config=cfg)
    else:
        print("\nBuilding SpotJoystickResidualEnv ...")

        def _make_env(cfg=cfg):
            return SpotJoystickResidualEnv(
                config=cfg,
                residual_params=params,
                in_mean=in_mean, in_std=in_std,
                out_mean=out_mean, out_std=out_std,
            )

    env      = _make_env()
    eval_env = _make_env()

    # --- PPO hyperparameters (mirror M2 training) ---
    ppo_params = locomotion_params.brax_ppo_config("SpotJoystickRoughTerrain", "jax")
    training_params = dict(ppo_params)

    # Pop fields we will pass explicitly to avoid duplicate keyword arg errors.
    num_eval_envs = training_params.pop("num_eval_envs", 128)
    training_params.pop("network_factory", None)   # replaced by resolved callable

    # Override training budget.
    training_params["num_timesteps"] = num_timesteps
    training_params["num_evals"]     = num_evals

    # Finetune-from-checkpoint overrides. The defaults in locomotion_params are
    # tuned for from-scratch training (LR=3e-4, entropy=1e-2); resuming a
    # converged policy with those values destroys it.
    training_params["learning_rate"] = learning_rate
    if entropy_cost is not None:
        training_params["entropy_cost"] = entropy_cost
    print(f"  learning_rate = {training_params['learning_rate']:.1e}  (from-scratch default 3e-4)")
    print(f"  entropy_cost  = {training_params['entropy_cost']:.1e}  (from-scratch default 1e-2)")

    # Smoke mode: fast sanity check only.
    if smoke:
        training_params["num_envs"]   = 64
        training_params["batch_size"] = 32
        num_eval_envs = 4

    network_factory = _resolve_network_factory(teacher_ckpt)
    restore_path = load_policy.find_latest_checkpoint(teacher_ckpt)

    print(f"\nRestoring from {restore_path}")
    print(f"Training for {num_timesteps:,} steps  ({num_evals} evals)")

    def progress(num_steps, metrics):
        reward = metrics.get("eval/episode_reward", float("nan"))
        print(f"  step={num_steps:10,}  eval_reward={reward:.3f}")

    ppo.train(
        environment=env,
        eval_env=eval_env,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_checkpoint_path=str(restore_path),
        save_checkpoint_path=str(ckpt_dir),
        num_eval_envs=num_eval_envs,
        progress_fn=progress,
        **training_params,
    )

    print(f"\nFinetuning complete.  Checkpoints at {ckpt_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Finetune M2 teacher with M4 actuator residual.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--teacher_ckpt",  default=_DEFAULT_TEACHER_CKPT)
    p.add_argument("--residual_ckpt", default=_DEFAULT_RESIDUAL_CKPT)
    p.add_argument("--num_timesteps", type=int, default=30_000_000)
    p.add_argument("--num_evals",     type=int, default=40)
    p.add_argument("--out_base",      default=str(_REPO_ROOT / "mujoco_playground/logs"))
    p.add_argument("--learning_rate", type=float, default=3e-5,
                   help="PPO LR. Default 3e-5 = 10x lower than from-scratch (3e-4) "
                        "to keep the converged M2 policy from drifting.")
    p.add_argument("--entropy_cost",  type=float, default=None,
                   help="PPO entropy coef. None = use config default (1e-2). "
                        "Try 1e-3 if the LR cut alone doesn't stop policy drift.")
    p.add_argument("--smoke",         action="store_true",
                   help="Run a minimal smoke test (200k steps, 2 evals, 64 envs).")
    p.add_argument("--no_residual",   action="store_true",
                   help="Skip residual injection. The env still uses the M4 modifications "
                        "(torque passthrough + manual PD), but no residual is added to ctrl. "
                        "Diagnostic: if PPO collapses here too, the residual isn't the cause.")
    p.add_argument("--use_original_env", action="store_true",
                   help="Use the registered SpotJoystickRoughTerrain env (no M4 modifications). "
                        "Requires --no_residual. Diagnostic: if PPO collapses here too, the "
                        "issue is intrinsic PPO finetuning of the M2 checkpoint, not our env "
                        "subclass.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_finetune(
        # Orbax requires absolute paths for checkpoint restoration.
        teacher_ckpt=str(Path(args.teacher_ckpt).resolve()),
        residual_ckpt=str(Path(args.residual_ckpt).resolve()),
        num_timesteps=args.num_timesteps if not args.smoke else 200_000,
        num_evals=args.num_evals if not args.smoke else 2,
        out_base=args.out_base,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        smoke=args.smoke,
        no_residual=args.no_residual,
        use_original_env=args.use_original_env,
    )
