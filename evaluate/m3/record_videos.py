"""Record demonstration videos for the M3 student policy.

The student (PyTorch GRU) drives the raw Brax env via a Python step loop with
zero-copy JAX↔torch bridging.  Same 10 scenarios as the M2 teacher recording.

Run from repo root (legged_rl_sim2sim/):
    python evaluate/m3/record_videos.py
    python evaluate/m3/record_videos.py --student_checkpoint checkpoints/.../student_spot_proprio_m3.pt
    python evaluate/m3/record_videos.py --scenarios forward_nom.mp4 curve.mp4

Outputs 10 MP4 files in evaluate/m3/results/videos/.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "mujoco_playground"))
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np
import torch

from mujoco_playground._src.wrapper_torch import _jax_to_torch, _torch_to_jax
from load_student import (
    DEFAULT_STUDENT_CKPT,
    ENV_NAME,
    build_env_raw,
    extract_proprio,
    load_student,
)

RENDER_EVERY = 1
VIDEO_HEIGHT = 480
VIDEO_WIDTH  = 640


@dataclass
class Scenario:
    name:          str
    filename:      str
    command:       tuple[float, float, float]  # (vx m/s, vy m/s, wz rad/s)
    n_steps:       int
    fixed_command: bool   # False → env samples commands naturally (stress test)
    description:   str


SCENARIOS: list[Scenario] = [
    Scenario("Forward slow",    "forward_slow.mp4",   (0.5,  0.0,  0.0), 1000, True,  "0.5 m/s forward"),
    Scenario("Forward nominal", "forward_nom.mp4",    (1.0,  0.0,  0.0), 1000, True,  "1.0 m/s forward (benchmark)"),
    Scenario("Forward fast",    "forward_fast.mp4",   (1.5,  0.0,  0.0), 1000, True,  "1.5 m/s forward (max speed)"),
    Scenario("Backward",        "backward.mp4",       (-0.5, 0.0,  0.0), 1000, True,  "-0.5 m/s backward"),
    Scenario("Lateral left",    "lateral_left.mp4",   (0.0,  0.5,  0.0), 1000, True,  "0.5 m/s lateral left"),
    Scenario("Lateral right",   "lateral_right.mp4",  (0.0, -0.5,  0.0), 1000, True,  "-0.5 m/s lateral right"),
    Scenario("Rotate left",     "rotate_left.mp4",    (0.0,  0.0,  1.0), 1000, True,  "in-place CCW rotation"),
    Scenario("Rotate right",    "rotate_right.mp4",   (0.0,  0.0, -1.0), 1000, True,  "in-place CW rotation"),
    Scenario("Curved path",     "curve.mp4",          (0.8,  0.0,  0.5), 1000, True,  "0.8 m/s forward + 0.5 rad/s yaw"),
    Scenario("Random stress",   "stress.mp4",         (0.0,  0.0,  0.0), 1000, False, "30 s random commands (env-sampled)"),
]


def record_scenario(
    env,
    jit_step,
    student,
    scenario: Scenario,
    seed: int,
    output_path: Path,
    device: str,
) -> dict:
    """Record a single scenario video using the student policy.

    Uses a Python step loop with DLPack zero-copy bridging between the JAX env
    and the PyTorch student.  Compact states are collected for memory-efficient
    rendering (only the fields env.render() needs are kept).
    """
    rng   = jax.random.PRNGKey(seed)
    state = jax.jit(env.reset)(rng)

    fixed_cmd = jp.array(list(scenario.command))
    if scenario.fixed_command:
        state = state.replace(info={**state.info, "command": fixed_cmd})

    # Compact state template — mirrors the M2 record_videos pattern.
    # env.render() only needs the data sub-fields below; all others are None.
    empty_data = state.data.__class__(**{k: None for k in state.data.__annotations__})
    empty_tmpl = state.__class__(**{k: None for k in state.__annotations__})
    empty_tmpl = empty_tmpl.replace(data=empty_data)

    hidden  = student.init_hidden(1, device)
    rollout: list = []
    fell    = False

    student.eval()
    with torch.no_grad():
        for t in range(scenario.n_steps):
            if t % RENDER_EVERY == 0:
                rollout.append(empty_tmpl.tree_replace({
                    "data.qpos":          state.data.qpos,
                    "data.qvel":          state.data.qvel,
                    "data.time":          state.data.time,
                    "data.ctrl":          state.data.ctrl,
                    "data.mocap_pos":     state.data.mocap_pos,
                    "data.mocap_quat":    state.data.mocap_quat,
                    "data.xfrc_applied":  state.data.xfrc_applied,
                }))

            obs_jax   = state.obs["state"]                                # (81,) JAX
            obs_torch = _jax_to_torch(obs_jax).unsqueeze(0)               # (1, 81) torch
            proprio   = extract_proprio(obs_torch)                        # (1, 69) torch
            act_torch, hidden = student(proprio, hidden)

            act_jax = jp.clip(_torch_to_jax(act_torch.squeeze(0)), -1.0, 1.0)  # (12,) JAX
            state   = jit_step(state, act_jax)

            if scenario.fixed_command:
                state = state.replace(info={**state.info, "command": fixed_cmd})

            if not fell and bool(state.done):
                fell = True

    # Render collected rollout.
    fps         = 1.0 / env.dt / RENDER_EVERY
    scene_opt   = mujoco.MjvOption()
    scene_opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]   = False
    scene_opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE]     = False
    scene_opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]  = False

    frames = env.render(
        rollout,
        camera       = "track",
        height       = VIDEO_HEIGHT,
        width        = VIDEO_WIDTH,
        scene_option = scene_opt,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(output_path), frames, fps=fps)

    duration_s = len(rollout) * RENDER_EVERY * env.dt
    print(f"  Saved: {output_path}  ({len(frames)} frames, {duration_s:.0f} s, fell={fell})")
    return {"filename": output_path.name, "fell": fell, "frames": len(frames)}


def parse_args():
    p = argparse.ArgumentParser(description="Record M3 student policy demonstration videos.")
    p.add_argument(
        "--student_checkpoint",
        default=str(_REPO_ROOT / DEFAULT_STUDENT_CKPT),
        help="Path to student_spot_proprio_m3.pt checkpoint file.",
    )
    p.add_argument(
        "--output_dir",
        default=str(Path(__file__).parent / "results" / "videos"),
        help="Directory to write MP4 files.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument(
        "--scenarios", nargs="*", default=None,
        help="Subset of scenario filenames to record (default: all 10).  "
             "Example: --scenarios forward_nom.mp4 curve.mp4",
    )
    p.add_argument(
        "--residual_ckpt", default=None,
        help="If set, drive the student through SpotJoystickResidualEnv with this "
             "M4 residual checkpoint injected at every substep. Inference-only test "
             "(no PPO finetune) of whether the residual closes the sim2sim gap.",
    )
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = SCENARIOS
    if args.scenarios:
        names     = set(args.scenarios)
        scenarios = [s for s in SCENARIOS if s.filename in names]
        if not scenarios:
            print(f"No matching scenarios found for {args.scenarios}")
            sys.exit(1)

    print("=" * 64)
    # Build env before loading student so JAX/Warp warm up before PyTorch claims GPU.
    print(f"Building raw env ({ENV_NAME}) …")
    env = build_env_raw(residual_ckpt=args.residual_ckpt)

    env.mj_model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    env.mj_model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    env.mj_model.vis.headlight.active = 1
    # Pre-compile env.step so Warp allocates its collision context once via XLA
    # rather than re-allocating in eager mode on every Python-loop iteration.
    jit_step = jax.jit(env.step)

    print(f"Loading student from {args.student_checkpoint} …")
    student = load_student(args.student_checkpoint, device=args.device)

    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {scenario.name}: {scenario.description}")
        out_path = output_dir / scenario.filename
        info = record_scenario(
            env, jit_step, student, scenario,
            seed        = args.seed + i,
            output_path = out_path,
            device      = args.device,
        )
        results.append(info)

    print("\n" + "=" * 64)
    print("Video summary:")
    for r in results:
        fell_str = "FELL" if r["fell"] else "ok"
        print(f"  {r['filename']:<25}  {r['frames']:>4} frames  [{fell_str}]")
    print(f"\nAll videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
