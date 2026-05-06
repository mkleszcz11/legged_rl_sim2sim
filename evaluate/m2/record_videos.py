"""Record 10 predefined-movement videos for the M2 Spot policy.

Run from repo root (unitree_go2_rl/):
    python evaluate/m2/record_videos.py \
        --checkpoint_dir mujoco_playground/logs/SpotJoystickRoughTerrain-20260428-123447/checkpoints \
        --output_dir evaluate/m2/results/videos

Produces 10 MP4 files demonstrating the range of the trained policy.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "mujoco_playground"))
sys.path.insert(0, str(_REPO_ROOT / "train" / "actuator_residual"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_playground import registry

from load_policy import DEFAULT_ROUGH_CKPT, load_policy

# Used only when --env residual; importing lazily would also be fine, but the cost
# is negligible and a top-level import keeps the module's dependencies explicit.
from finetune_teacher import SpotJoystickResidualEnv, port_weights

# Environment choices for --env.
#   "original":     registered MJX env (position actuators, sim_dt=4ms, iters=1).
#   "residual":     M4 SpotJoystickResidualEnv (torque passthrough + residual injection).
#   "cpu_accurate": registered env with sim_dt=0.5ms and 4 solver iterations -- the
#                   "accurate physics" reference M4's residual was trying to bridge to.
#                   Uses the MJX backend (not literal mujoco.mj_step) but with the same
#                   solver iterations and timestep that collect_data.py used as the CPU
#                   ground truth, so it's a faithful proxy for sim2sim transfer.
_ENV_ORIGINAL     = "original"
_ENV_RESIDUAL     = "residual"
_ENV_CPU_ACCURATE = "cpu_accurate"
_ENV_CHOICES      = (_ENV_ORIGINAL, _ENV_RESIDUAL, _ENV_CPU_ACCURATE)

_DEFAULT_RESIDUAL_CKPT = _REPO_ROOT / "checkpoints" / "actuator_residual.pt"

# Match collect_data.py's CPU replay settings exactly so cpu_accurate is the same
# physics the M4 residual was trained to bridge to.
_CPU_TIMESTEP   = 5e-4   # 0.5 ms (vs default 4 ms)
_CPU_ITERATIONS = 4      # vs default 1


@dataclass
class Scenario:
    name: str
    filename: str
    command: tuple[float, float, float]  # (vx, vy, w_yaw) m/s, m/s, rad/s
    n_steps: int
    fixed_command: bool  # False → let env sample commands normally (stress test)
    description: str


SCENARIOS: list[Scenario] = [
    Scenario("Forward slow",     "forward_slow.mp4",   (0.5,  0.0,  0.0), 1000,  True,  "0.5 m/s forward"),
    Scenario("Forward nominal",  "forward_nom.mp4",    (1.0,  0.0,  0.0), 1000,  True,  "1.0 m/s forward (benchmark)"),
    Scenario("Forward fast",     "forward_fast.mp4",   (1.5,  0.0,  0.0), 1000,  True,  "1.5 m/s forward (max curriculum speed)"),
    Scenario("Backward",         "backward.mp4",       (-0.5, 0.0,  0.0), 1000,  True,  "-0.5 m/s backward"),
    Scenario("Lateral left",     "lateral_left.mp4",   (0.0,  0.5,  0.0), 1000,  True,  "0.5 m/s lateral left"),
    Scenario("Lateral right",    "lateral_right.mp4",  (0.0, -0.5,  0.0), 1000,  True,  "-0.5 m/s lateral right"),
    Scenario("Rotate left",      "rotate_left.mp4",    (0.0,  0.0,  1.0), 1000,  True,  "in-place CCW rotation"),
    Scenario("Rotate right",     "rotate_right.mp4",   (0.0,  0.0, -1.0), 1000,  True,  "in-place CW rotation"),
    Scenario("Curved path",      "curve.mp4",          (0.8,  0.0,  0.5), 1000,  True,  "0.8 m/s forward + 0.5 rad/s yaw"),
    Scenario("Random stress",    "stress.mp4",         (0.0,  0.0,  0.0), 1000, False, "30 s random commands (curriculum-sampled)"),
]

RENDER_EVERY = 2
VIDEO_HEIGHT = 480
VIDEO_WIDTH  = 640


def _make_render_state(data_snapshot: dict):
    """Wrap a dict of numpy arrays into a render-compatible state-like object."""
    import types
    data = types.SimpleNamespace(**data_snapshot)
    return types.SimpleNamespace(data=data)


def record_scenario(env, inference_fn, scenario: Scenario, seed: int, output_path: Path):
    """Record a single scenario video.

    For fixed-command scenarios: overrides state.info['command'] every step.
    For free-command (stress): lets the env sample commands normally.
    """
    rng = jax.random.PRNGKey(seed)
    init_state = jax.jit(env.reset)(rng)

    fixed_cmd = jp.array(list(scenario.command))
    if scenario.fixed_command:
        init_state = init_state.replace(info={**init_state.info, "command": fixed_cmd})

    # Build empty trajectory template (same pattern as train_jax_ppo.py)
    empty_data = init_state.data.__class__(
        **{k: None for k in init_state.data.__annotations__}
    )
    empty_traj = init_state.__class__(
        **{k: None for k in init_state.__annotations__}
    )
    empty_traj = empty_traj.replace(data=empty_data)

    def step_fn(carry, _):
        state, rng = carry
        rng, key = jax.random.split(rng)
        act, _ = inference_fn(state.obs, key)
        state = env.step(state, act)
        if scenario.fixed_command:
            state = state.replace(info={**state.info, "command": fixed_cmd})
        traj_data = empty_traj.tree_replace({
            "data.qpos":        state.data.qpos,
            "data.qvel":        state.data.qvel,
            "data.time":        state.data.time,
            "data.ctrl":        state.data.ctrl,
            "data.mocap_pos":   state.data.mocap_pos,
            "data.mocap_quat":  state.data.mocap_quat,
            "data.xfrc_applied": state.data.xfrc_applied,
        })
        return (state, rng), (traj_data, state.done)

    print(f"  Running {scenario.n_steps} steps …")
    (final_state, _), (traj_stacked, dones) = jax.jit(
        lambda s, r: jax.lax.scan(step_fn, (s, r), None, length=scenario.n_steps)
    )(init_state, rng)

    fell = bool(np.array(dones).any())

    # Build list of render states (every render_every-th step)
    indices = range(0, scenario.n_steps, RENDER_EVERY)
    rollout = [
        jax.tree.map(lambda x, i=i: x[i], traj_stacked)
        for i in indices
    ]

    # Render
    fps = 1.0 / env.dt / RENDER_EVERY
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE]   = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    frames = env.render(
        rollout,
        camera="track",
        height=VIDEO_HEIGHT,
        width=VIDEO_WIDTH,
        scene_option=scene_option,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(output_path), frames, fps=fps)

    duration_s = scenario.n_steps * env.dt
    print(f"  Saved: {output_path}  ({len(frames)} frames, {duration_s:.0f} s, fell={fell})")
    return {"filename": output_path.name, "fell": fell, "frames": len(frames)}


def _build_env(env_choice: str, residual_ckpt: Path, inject_residual: bool):
    """Construct the env to record in.

    "original":     registered SpotJoystickRoughTerrain (position actuators, MJX physics
                    at the training settings: sim_dt=4ms, iters=1).
    "residual":     SpotJoystickResidualEnv with the M4 env modifications applied
                    (torque-passthrough actuators, dof_damping zeroed, explicit PD via
                    substep_fn). When inject_residual is True the trained residual
                    network is loaded and added to ctrl every substep -- this is the
                    env an M4-finetuned policy was actually trained against. When
                    inject_residual is False, residual_params stays None (no network
                    injection) -- useful as an apples-to-apples comparison against
                    the original env, which isolates whether failures are caused by
                    the env modifications or by the residual itself.
    "cpu_accurate": registered SpotJoystickRoughTerrain with the physics overridden to
                    sim_dt=0.5ms and 4 solver iterations. Same backend (MJX) as the
                    other modes but with the high-accuracy settings that collect_data.py
                    treated as ground truth. Use this to test whether M2 transfers to
                    accurate physics WITHOUT going through the residual at all.
    """
    if env_choice == _ENV_ORIGINAL:
        return None  # signal: let load_policy build it from the registry

    if env_choice == _ENV_RESIDUAL:
        cfg = registry.get_default_config("SpotJoystickRoughTerrain")

        if not inject_residual:
            return SpotJoystickResidualEnv(config=cfg, residual_params=None)

        if not residual_ckpt.is_file():
            raise FileNotFoundError(
                f"Residual checkpoint not found: {residual_ckpt}\n"
                "Train one with train/actuator_residual/residual_network.py, "
                "pass --residual_ckpt, or pass --no_residual to skip injection."
            )
        params, in_mean, in_std, out_mean, out_std = port_weights(str(residual_ckpt))
        return SpotJoystickResidualEnv(
            config=cfg,
            residual_params=params,
            in_mean=in_mean, in_std=in_std,
            out_mean=out_mean, out_std=out_std,
        )

    if env_choice == _ENV_CPU_ACCURATE:
        # Build the registered env with sim_dt overridden via the config (this is
        # the supported override path; it propagates through SpotEnv.__init__).
        # Solver iterations are NOT a config field, so we mutate the loaded mj_model
        # afterward and re-put it through MJX. This is a one-time construction-time
        # mutation -- the env is never re-used with different physics settings.
        cfg = registry.get_default_config("SpotJoystickRoughTerrain")
        env = registry.load(
            "SpotJoystickRoughTerrain",
            config=cfg,
            config_overrides={"sim_dt": _CPU_TIMESTEP},
        )
        env._mj_model.opt.iterations = _CPU_ITERATIONS
        env._mjx_model = mjx.put_model(env._mj_model, impl=cfg.impl)
        return env

    raise ValueError(f"Unknown --env value '{env_choice}'. Choose from {_ENV_CHOICES}.")


def parse_args():
    p = argparse.ArgumentParser(description="Record 10 Spot M2 demonstration videos.")
    p.add_argument(
        "--checkpoint_dir", default=str(_REPO_ROOT / DEFAULT_ROUGH_CKPT),
        help="Path to checkpoints/ directory.",
    )
    p.add_argument(
        "--output_dir", default=str(Path(__file__).parent / "results" / "videos"),
        help="Directory to write MP4 files.",
    )
    p.add_argument(
        "--env", choices=_ENV_CHOICES, default=_ENV_ORIGINAL,
        help="Environment to record in. Use 'residual' to evaluate an M4-finetuned "
             "policy in the env it was trained against.",
    )
    p.add_argument(
        "--residual_ckpt", default=str(_DEFAULT_RESIDUAL_CKPT),
        help="Path to the trained actuator residual .pt checkpoint "
             "(only used when --env residual without --no_residual).",
    )
    p.add_argument(
        "--no_residual", action="store_true",
        help="When --env residual: build the residual env with the actuator/damping "
             "modifications applied but skip injecting the trained network. Use this "
             "to test whether failures come from the env modifications themselves vs "
             "from the residual injection.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--scenarios", nargs="*", default=None,
        help="Subset of scenario filenames to record (default: all 10).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = SCENARIOS
    if args.scenarios:
        names = set(args.scenarios)
        scenarios = [s for s in SCENARIOS if s.filename in names]
        if not scenarios:
            print(f"No matching scenarios found for {args.scenarios}")
            sys.exit(1)

    print("=" * 64)
    inject = (args.env == _ENV_RESIDUAL) and not args.no_residual
    env_label = (
        f"{args.env} (residual injected)" if inject
        else f"{args.env} (no residual)" if args.env == _ENV_RESIDUAL
        else args.env
    )
    print(f"Loading policy from {args.checkpoint_dir}  (env={env_label}) …")
    prebuilt_env = _build_env(args.env, Path(args.residual_ckpt), inject_residual=inject)
    env, inference_fn = load_policy(
        args.checkpoint_dir,
        env_name="SpotJoystickRoughTerrain",  # PPO hyperparam lookup key (same for both envs)
        env=prebuilt_env,
    )

    env.mj_model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    env.mj_model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    env.mj_model.vis.headlight.active = 1

    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {scenario.name}: {scenario.description}")
        out_path = output_dir / scenario.filename
        info = record_scenario(env, inference_fn, scenario, seed=args.seed + i, output_path=out_path)
        results.append(info)

    print("\n" + "=" * 64)
    print("Video summary:")
    for r in results:
        fell_str = "FELL" if r["fell"] else "ok"
        print(f"  {r['filename']:<25}  {r['frames']:>4} frames  [{fell_str}]")
    print(f"\nAll videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
