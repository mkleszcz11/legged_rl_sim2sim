"""Sim-to-sim evaluation for the M3 student policy (M5 milestone).

Deploys the trained Spot student zero-shot on three simulators and reports a
quantitative gap table.

Simulators:
  mjx        — MJX-JAX (training env; 100 envs vmap'd in parallel)
  mujoco_cpu — Vanilla MuJoCo CPU (same MJCF; finer timestep + more solver iters)
  genesis    — Genesis (optional; skipped if not installed or Spot MJCF fails)

Metrics per episode (30 s, fixed 1.0 m/s forward command):
  tracking_error_mean   — mean |linvel_x - 1.0| over alive steps (m/s)
  survived              — no fall within 30 s
  base_height_dev_mean  — mean |base_z - 0.50| over alive steps (m)
  feet_force_rms        — sqrt(mean ||f_foot||^2) over alive steps × 4 feet (N)

Re-run safety:
  Results are keyed on (sim, seed).  Re-running adds only the missing rows;
  already-collected rows are never overwritten.

Run from repo root (unitree_go2_rl/):
    python evaluate/m5/sim2sim_eval.py
    python evaluate/m5/sim2sim_eval.py --sims mjx mujoco_cpu --num_episodes 5

Outputs:
    evaluate/m5/results/sim2sim_baseline.csv
    evaluate/m5/results/videos/{sim}.mp4
    stdout — progress + summary table
"""

import argparse
import gc
import os
import sys
from pathlib import Path

# ── Environment variables must be set before any JAX / CUDA import ───────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_M5_DIR = Path(__file__).resolve().parent

# Path setup (same style as evaluate/m3 scripts):
#   mujoco_playground — for registry, wrapper_torch, etc.
#   evaluate/m3       — for load_student (flat import)
#   evaluate/m5       — for metrics, adapters.* (flat imports)
sys.path.insert(0, str(_REPO_ROOT / "mujoco_playground"))
sys.path.insert(0, str(_REPO_ROOT / "evaluate" / "m3"))
sys.path.insert(0, str(_M5_DIR))

import torch

from load_student import DEFAULT_STUDENT_CKPT, load_student
from adapters.base import BackendUnavailable
from metrics import EpisodeRow, append_rows, print_summary, read_existing


_ALL_SIMS = ["mjx", "mujoco_cpu", "genesis"]

_DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"
_CSV_NAME = "sim2sim_baseline.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sim-to-sim evaluation for the M3 student policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--student_checkpoint",
        default=str(_REPO_ROOT / DEFAULT_STUDENT_CKPT),
        help="Path to student_spot_proprio.pt",
    )
    p.add_argument(
        "--num_episodes", type=int, default=100,
        help="Number of episodes per simulator",
    )
    p.add_argument(
        "--command", nargs=3, type=float, default=[1.0, 0.0, 0.0],
        metavar=("VX", "VY", "WZ"),
        help="Fixed velocity command: vx (m/s), vy (m/s), wz (rad/s)",
    )
    p.add_argument(
        "--sims", nargs="+", default=_ALL_SIMS, choices=_ALL_SIMS,
        help="Which simulator backends to run",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--results_dir", default=str(_DEFAULT_RESULTS_DIR),
        help="Directory for CSV and video output",
    )
    p.add_argument(
        "--no_video", action="store_true",
        help="Skip video recording",
    )
    return p.parse_args()


def _load_adapter(
    sim_name: str,
    student: torch.nn.Module,
    command: tuple[float, float, float],
    device: str,
):
    """Import and instantiate the adapter for sim_name; return None if unavailable."""
    if sim_name == "mjx":
        from adapters.mjx_adapter import MJXAdapter
        return MJXAdapter(student, command, device)

    if sim_name == "mujoco_cpu":
        try:
            from adapters.mujoco_cpu_adapter import MuJoCoCPUAdapter
            return MuJoCoCPUAdapter(student, command, device)
        except ImportError as e:
            print(f"  [skip] mujoco_cpu unavailable: {e}")
            return None

    if sim_name == "genesis":
        try:
            from adapters.genesis_adapter import GenesisAdapter
            return GenesisAdapter(student, command, device)
        except BackendUnavailable as e:
            print(f"  [skip] genesis unavailable: {e}")
            return None
        except ImportError as e:
            print(f"  [skip] genesis unavailable: {e}")
            return None

    raise ValueError(f"Unknown sim: {sim_name}")


def _seeds_for(num_episodes: int) -> list[int]:
    return list(range(num_episodes))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    csv_path = results_dir / _CSV_NAME
    video_dir = results_dir / "videos"
    command = tuple(args.command)

    print("=" * 70)
    print(f"M5 Sim-to-Sim Evaluation  —  {args.num_episodes} episodes × 30 s")
    print(f"Command: vx={command[0]:.2f} m/s  vy={command[1]:.2f}  wz={command[2]:.2f}")
    print(f"Sims: {', '.join(args.sims)}")
    print(f"Results: {csv_path}")
    print("=" * 70)

    student = load_student(args.student_checkpoint, device=args.device)
    all_seeds = _seeds_for(args.num_episodes)

    existing = read_existing(csv_path)

    for sim_name in args.sims:
        print(f"\n{'─'*70}")
        print(f"Backend: {sim_name}")

        # Determine which seeds are still missing for this sim.
        done_seeds = {seed for (sim, seed) in existing if sim == sim_name}
        missing = [s for s in all_seeds if s not in done_seeds]

        if not missing:
            print(f"  All {args.num_episodes} episodes already collected — skipping.")
            continue

        print(f"  Collecting {len(missing)} episodes (already done: {len(done_seeds)}) …")

        adapter = _load_adapter(sim_name, student, command, args.device)
        if adapter is None:
            continue

        video_seed = missing[0] if not args.no_video else None
        video_path = video_dir / f"{sim_name}.mp4" if not args.no_video else None

        rows = adapter.run_episodes(
            seeds=missing,
            video_seed=video_seed,
            video_path=video_path,
        )

        append_rows(csv_path, rows)
        print(f"  Appended {len(rows)} rows to {csv_path.name}")

        # Release GPU/CPU memory before loading the next backend.
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass

    print(f"\n{'='*70}")
    print_summary(csv_path)


if __name__ == "__main__":
    main()
