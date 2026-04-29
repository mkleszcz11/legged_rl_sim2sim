"""Record 10 predefined-movement videos for the M2 Spot policy.

Run from repo root (unitree_go2_rl/):
    python evaluate/m2/record_videos.py \\
        --checkpoint_dir mujoco_playground/logs/SpotJoystickRoughTerrain-20260428-123447/checkpoints \\
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
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np

from load_policy import DEFAULT_ROUGH_CKPT, load_policy


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
    print(f"Loading policy from {args.checkpoint_dir} …")
    env, inference_fn = load_policy(args.checkpoint_dir, "SpotJoystickRoughTerrain")

    env.mj_model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
    env.mj_model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
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
