"""Record milestone videos for an M2 training run.

Picks every N-th checkpoint (1-indexed position) plus always the last,
records selected scenarios for each, and overlays a timestep watermark.

Run from repo root (unitree_go2_rl/):
    python evaluate/m2/record_milestones.py \\
        --checkpoint_dir /home/marcin/projects/robot_learning/mujoco_playground/logs/SpotJoystickRoughTerrain-20260428-123447/checkpoints \\
        --every_n 4

Output: evaluate/m2/results/milestones/milestone_<timestep>/<scenario>.mp4
"""

import argparse
import os
import sys
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

from load_policy import load_policy
from record_videos import RENDER_EVERY, SCENARIOS, VIDEO_HEIGHT, VIDEO_WIDTH, Scenario

_DEFAULT_SCENARIOS = ["forward_nom.mp4"]
_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_SIZE = 22


def _make_font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except (IOError, OSError):
        return ImageFont.load_default()


def _apply_watermark(frames: list[np.ndarray], text: str) -> list[np.ndarray]:
    from PIL import Image, ImageDraw
    font = _make_font(_FONT_SIZE)
    out = []
    for frame in frames:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.text((11, 11), text, fill=(0, 0, 0), font=font)
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        out.append(np.array(img))
    return out


def _select_milestones(checkpoint_dir: Path, every_n: int) -> tuple[list[Path], int]:
    """Return (selected checkpoint dirs, total checkpoint count)."""
    all_ckpts = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not all_ckpts:
        raise FileNotFoundError(f"No numeric checkpoint dirs in {checkpoint_dir}")
    indices = set(range(every_n - 1, len(all_ckpts), every_n))
    indices.add(len(all_ckpts) - 1)
    return [all_ckpts[i] for i in sorted(indices)], len(all_ckpts)


def _record_scenario(env, inference_fn, scenario: Scenario, seed: int,
                     output_path: Path, watermark: str) -> dict:
    rng = jax.random.PRNGKey(seed)
    init_state = jax.jit(env.reset)(rng)

    fixed_cmd = jp.array(list(scenario.command))
    if scenario.fixed_command:
        init_state = init_state.replace(info={**init_state.info, "command": fixed_cmd})

    empty_data = init_state.data.__class__(
        **{k: None for k in init_state.data.__annotations__}
    )
    empty_traj = init_state.__class__(**{k: None for k in init_state.__annotations__})
    empty_traj = empty_traj.replace(data=empty_data)

    def step_fn(carry, _):
        state, rng = carry
        rng, key = jax.random.split(rng)
        act, _ = inference_fn(state.obs, key)
        state = env.step(state, act)
        if scenario.fixed_command:
            state = state.replace(info={**state.info, "command": fixed_cmd})
        traj_data = empty_traj.tree_replace({
            "data.qpos":         state.data.qpos,
            "data.qvel":         state.data.qvel,
            "data.time":         state.data.time,
            "data.ctrl":         state.data.ctrl,
            "data.mocap_pos":    state.data.mocap_pos,
            "data.mocap_quat":   state.data.mocap_quat,
            "data.xfrc_applied": state.data.xfrc_applied,
        })
        return (state, rng), (traj_data, state.done)

    print(f"    Running {scenario.n_steps} steps …")
    (_, _), (traj_stacked, dones) = jax.jit(
        lambda s, r: jax.lax.scan(step_fn, (s, r), None, length=scenario.n_steps)
    )(init_state, rng)

    fell = bool(np.array(dones).any())

    rollout = [
        jax.tree.map(lambda x, i=i: x[i], traj_stacked)
        for i in range(0, scenario.n_steps, RENDER_EVERY)
    ]

    fps = 1.0 / env.dt / RENDER_EVERY
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]  = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE]    = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    frames = env.render(
        rollout, camera="track",
        height=VIDEO_HEIGHT, width=VIDEO_WIDTH,
        scene_option=scene_option,
    )
    frames = _apply_watermark(list(frames), watermark)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(output_path), frames, fps=fps)

    duration_s = scenario.n_steps * env.dt
    print(f"    Saved: {output_path}  ({len(frames)} frames, {duration_s:.0f} s, fell={fell})")
    return {"filename": output_path.name, "fell": fell, "frames": len(frames)}


def parse_args():
    p = argparse.ArgumentParser(description="Record M2 milestone videos at every N-th checkpoint.")
    p.add_argument(
        "--checkpoint_dir", required=True,
        help="Path to checkpoints/ directory (contains numeric subdirs).",
    )
    p.add_argument(
        "--output_dir",
        default=str(Path(__file__).parent / "results" / "milestones"),
        help="Root output directory for milestone subdirs.",
    )
    p.add_argument(
        "--every_n", type=int, default=4,
        help="Record every N-th checkpoint (1-indexed). Last checkpoint is always included.",
    )
    p.add_argument(
        "--scenarios", nargs="*", default=_DEFAULT_SCENARIOS,
        metavar="FILENAME",
        help=(
            f"Scenario mp4 filenames to record per milestone (default: {_DEFAULT_SCENARIOS}). "
            f"Available: {[s.filename for s in SCENARIOS]}"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    milestones, total_ckpts = _select_milestones(checkpoint_dir, args.every_n)

    scenario_names = set(args.scenarios)
    scenarios = [s for s in SCENARIOS if s.filename in scenario_names]
    if not scenarios:
        print(f"No matching scenarios for {args.scenarios}. "
              f"Available: {[s.filename for s in SCENARIOS]}")
        sys.exit(1)

    print("=" * 64)
    print(f"Checkpoint dir  : {checkpoint_dir}")
    print(f"Total checkpoints: {total_ckpts}")
    print(f"Milestones       : {len(milestones)} (every {args.every_n}, last always included)")
    print(f"Scenarios        : {[s.name for s in scenarios]}")
    print(f"Output dir       : {output_dir}")
    print("=" * 64)

    all_results = []
    for m_idx, ckpt_path in enumerate(milestones, 1):
        timestep = int(ckpt_path.name)
        watermark = f"Step: {timestep:,}"
        milestone_dir = output_dir / f"milestone_{ckpt_path.name}"

        print(f"\n[{m_idx}/{len(milestones)}] {watermark}")
        print(f"  Loading policy …")
        env, inference_fn = load_policy(
            checkpoint_dir=checkpoint_dir,
            env_name="SpotJoystickRoughTerrain",
            checkpoint_path=ckpt_path,
        )

        for s_idx, scenario in enumerate(scenarios, 1):
            print(f"  [{s_idx}/{len(scenarios)}] {scenario.name}: {scenario.description}")
            out_path = milestone_dir / scenario.filename
            info = _record_scenario(
                env, inference_fn, scenario,
                seed=args.seed + m_idx,
                output_path=out_path,
                watermark=watermark,
            )
            all_results.append({"milestone": ckpt_path.name, "timestep": timestep, **info})

    print("\n" + "=" * 64)
    print("Milestone summary:")
    for r in all_results:
        fell_str = "FELL" if r["fell"] else "ok"
        print(f"  {r['milestone']}  {r['filename']:<25}  {r['frames']:>4} frames  [{fell_str}]")
    print(f"\nAll videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
