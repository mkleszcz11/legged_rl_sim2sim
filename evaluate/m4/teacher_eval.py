"""M4 Teacher Evaluation — quantify the actuator residual's sim-to-sim impact.

Evaluates the M2 teacher (JAX MLP) zero-shot across four configurations:
  mjx_baseline  — MJX env (training physics), no residual
  mjx_residual  — MJX env with per-substep residual injection (SpotJoystickResidualEnv)
  cpu_baseline  — MuJoCo CPU (dt=0.5ms, 4 iters), no residual
  cpu_residual  — MuJoCo CPU + per-ctrl-step residual correction (ctrl offset = tau/Kp)

The two baselines reproduce the M3.1 gap (teacher instead of student).
The two residual configs test whether the actuator network closes that gap.

Metrics per episode (30 s, fixed 1.0 m/s forward command):
  tracking_error_mean   — mean |linvel_x - 1.0| over alive steps (m/s)
  survived              — no fall within 30 s
  base_height_dev_mean  — mean |base_z - 0.50| over alive steps (m)
  feet_force_rms        — sqrt(mean ||f_foot||^2) over alive steps (N)

Results are keyed on (sim, seed).  Re-running adds only missing rows.

Run from repo root:
    python evaluate/m4/teacher_eval.py
    python evaluate/m4/teacher_eval.py --configs mjx_baseline cpu_baseline --num_episodes 10
    python evaluate/m4/teacher_eval.py --configs cpu_baseline cpu_residual --num_videos 5

Outputs:
    evaluate/m4/results/teacher_eval.csv
    evaluate/m4/results/videos/{config}_{seed}.mp4
    stdout — progress + summary table
"""

import argparse
import gc
import os
import sys
from pathlib import Path

# JAX / CUDA env vars must come before any import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_M4_DIR    = Path(__file__).resolve().parent

sys.path.insert(0, str(_REPO_ROOT / "mujoco_playground"))
sys.path.insert(0, str(_REPO_ROOT / "evaluate" / "m2"))
sys.path.insert(0, str(_REPO_ROOT / "train" / "actuator_residual"))
sys.path.insert(0, str(_REPO_ROOT / "evaluate" / "m5"))   # for metrics.py
sys.path.insert(0, str(_M4_DIR))

import numpy as np
import torch

import load_policy   # evaluate/m2/load_policy.py
from metrics import EpisodeRow, append_rows, read_existing

from adapters.mjx_adapter import MJXTeacherAdapter
from adapters.cpu_adapter  import CPUTeacherAdapter

_ALL_CONFIGS = ["mjx_baseline", "mjx_residual", "cpu_baseline", "cpu_residual", "mjx_cpu_faithful"]

_DEFAULT_TEACHER_CKPT  = str(
    _REPO_ROOT / "mujoco_playground/logs/SpotJoystickRoughTerrain-20260428-123447/checkpoints"
)
_DEFAULT_RESIDUAL_CKPT = str(_REPO_ROOT / "checkpoints/actuator_residual.pt")
_DEFAULT_RESULTS_DIR   = _M4_DIR / "results"
_CSV_NAME = "teacher_eval.csv"

# Configs that need the residual env / residual correction loaded.
_RESIDUAL_CONFIGS = {"mjx_residual", "cpu_residual"}
# Configs that run inside MJX (vmap batch).
_MJX_CONFIGS      = {"mjx_baseline", "mjx_residual", "mjx_cpu_faithful"}
# Configs that run on CPU (sequential).
_CPU_CONFIGS      = {"cpu_baseline", "cpu_residual"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="M4 teacher evaluation: baseline vs residual in MJX and MuJoCo CPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--teacher_ckpt", default=_DEFAULT_TEACHER_CKPT,
        help="Path to M2 teacher checkpoints/ directory.",
    )
    p.add_argument(
        "--residual_ckpt", default=_DEFAULT_RESIDUAL_CKPT,
        help="Path to actuator_residual.pt checkpoint.",
    )
    p.add_argument(
        "--configs", nargs="+", default=_ALL_CONFIGS, choices=_ALL_CONFIGS,
        help="Which configurations to evaluate.",
    )
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument(
        "--command", nargs=3, type=float, default=[1.0, 0.0, 0.0],
        metavar=("VX", "VY", "WZ"),
    )
    p.add_argument("--device", default="cuda:0",
                   help="PyTorch device for residual model (CPU adapter only).")
    p.add_argument(
        "--results_dir", default=str(_DEFAULT_RESULTS_DIR),
        help="Directory for CSV and video output.",
    )
    p.add_argument("--no_video",   action="store_true")
    p.add_argument("--num_videos", type=int, default=1,
                   help="Videos to record per config (first N seeds).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def _load_teacher(teacher_ckpt: str):
    """Return (baseline_env, inference_fn) for the M2 teacher.

    The policy is loaded via a minimal single-env (impl=jax, small naconmax)
    to avoid allocating a full Warp pool during weight restoration on 4GB VRAM.
    A separate env is then created for the vmap batch eval (100 envs in parallel).
    """
    from mujoco_playground import registry

    print(f"Loading M2 teacher from {teacher_ckpt} …")
    _, inference_fn = load_policy.load_policy(
        teacher_ckpt,
        env_name="SpotJoystickRoughTerrain",
        num_envs=1,
        config_overrides={"impl": "jax", "naconmax": 1024},
    )

    # Fresh env for vmap eval — same XML/config, no custom naconmax constraint.
    baseline_env = registry.load("SpotJoystickRoughTerrain")
    print("[mjx_baseline] SpotJoystickRoughTerrain env loaded for vmap eval.")
    return baseline_env, inference_fn


def _load_residual_env(residual_ckpt: str):
    """Build a SpotJoystickResidualEnv with the trained residual injected."""
    from finetune_teacher import SpotJoystickResidualEnv, port_weights
    from mujoco_playground import registry

    params, in_mean, in_std, out_mean, out_std = port_weights(residual_ckpt)
    cfg = registry.get_default_config("SpotJoystickRoughTerrain")
    env = SpotJoystickResidualEnv(
        config=cfg,
        residual_params=params,
        in_mean=in_mean, in_std=in_std,
        out_mean=out_mean, out_std=out_std,
    )
    print(f"[mjx_residual] SpotJoystickResidualEnv ready.")
    return env


def _load_cpu_faithful_env():
    """MJX env with CPU-faithful physics: dt=0.5ms, 4 solver iterations.

    Mutates the env's MjModel after loading so the policy sees the same
    contact-solver configuration as the CPU baseline.  This lets Phase 0 tell
    us whether the teacher itself is brittle (Case B) or only the student is
    (Case A).

    naconmax is capped at 1024 (vs the default 32768) to avoid CUDA OOM on
    4 GB GPUs: the default produces a 256 KB Warp collision-pair buffer that
    can't be satisfied after two prior MJX runs have grown JAX's BFC pool.
    1024 contacts is still ~10–50× more than a walking robot generates.
    """
    from mujoco import mjx as _mjx
    from mujoco_playground import registry

    env = registry.load(
        "SpotJoystickRoughTerrain",
        config_overrides={"naconmax": 1024},
    )
    env._mj_model.opt.timestep  = 5e-4
    env._mj_model.opt.iterations = 4
    env._sim_dt = 5e-4   # n_substeps → ctrl_dt / sim_dt = 0.02 / 0.0005 = 40
    env._mjx_model = _mjx.put_model(env._mj_model, impl=env._config.impl)
    print(f"[mjx_cpu_faithful] dt=0.5ms  iters=4  n_substeps={env.n_substeps}  naconmax={env._config.naconmax}")
    return env


def _load_residual_model(residual_ckpt: str, device: str):
    """Load PyTorch residual checkpoint; return (model, norm_dict)."""
    from residual_network import ResidualMLP

    ckpt = torch.load(residual_ckpt, map_location=device, weights_only=False)
    model = ResidualMLP().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    norm = {
        "in_mean":  ckpt["norm"]["in_mean"].astype(np.float32),
        "in_std":   ckpt["norm"]["in_std"].astype(np.float32),
        "out_mean": ckpt["norm"]["out_mean"].astype(np.float32),
        "out_std":  ckpt["norm"]["out_std"].astype(np.float32),
    }
    print(f"[cpu_residual] Residual MLP loaded from {Path(residual_ckpt).name}")
    return model, norm


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

def _build_adapter(
    config_name: str,
    inference_fn,
    baseline_env,
    residual_env,
    cpu_faithful_env,
    residual_model,
    residual_norm,
    command: tuple[float, float, float],
    device: str,
):
    if config_name == "mjx_baseline":
        return MJXTeacherAdapter(inference_fn, baseline_env, "mjx_baseline", command)

    if config_name == "mjx_residual":
        if residual_env is None:
            raise RuntimeError("residual_env not loaded — include mjx_residual in --configs.")
        return MJXTeacherAdapter(inference_fn, residual_env, "mjx_residual", command)

    if config_name == "mjx_cpu_faithful":
        if cpu_faithful_env is None:
            raise RuntimeError("cpu_faithful_env not loaded — include mjx_cpu_faithful in --configs.")
        return MJXTeacherAdapter(inference_fn, cpu_faithful_env, "mjx_cpu_faithful", command)

    if config_name == "cpu_baseline":
        return CPUTeacherAdapter(inference_fn, "cpu_baseline", command, device=device)

    if config_name == "cpu_residual":
        if residual_model is None:
            raise RuntimeError("residual_model not loaded — include cpu_residual in --configs.")
        return CPUTeacherAdapter(
            inference_fn, "cpu_residual", command,
            residual_model=residual_model,
            residual_norm=residual_norm,
            device=device,
        )

    raise ValueError(f"Unknown config: {config_name}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_m4_summary(csv_path: Path) -> None:
    """Extended summary: 4-config table + sim-to-sim gap before and after residual."""
    if not csv_path.exists():
        print("No results file found.")
        return

    import csv
    import statistics

    rows: list[EpisodeRow] = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(EpisodeRow(
                sim=r["sim"],
                seed=int(r["seed"]),
                survived=r["survived"].lower() == "true",
                tracking_error_mean=float(r["tracking_error_mean"]),
                base_height_dev_mean=float(r["base_height_dev_mean"]),
                feet_force_rms=float(r["feet_force_rms"]),
                episode_seconds=float(r["episode_seconds"]),
                fall_timestep=int(r.get("fall_timestep", -1)),
            ))

    by_sim: dict[str, list[EpisodeRow]] = {}
    for r in rows:
        by_sim.setdefault(r.sim, []).append(r)

    sep = "─" * 84

    def _fmt(vals):
        if len(vals) < 2:
            return f"{vals[0]:.3f}"
        return f"{statistics.mean(vals):.3f}±{statistics.stdev(vals):.3f}"

    # --- Main table ---
    print(f"\n{'M4 TEACHER EVALUATION SUMMARY':^84}")
    print(sep)
    fmt = "{:<22} {:>8} {:>17} {:>17} {:>14}"
    print(fmt.format("Config", "Surv%", "TrackErr(m/s)", "HeightDev(m)", "FeetForce(N)"))
    print(sep)

    stats: dict[str, dict] = {}
    for cname in _ALL_CONFIGS:
        if cname not in by_sim:
            continue
        sim_rows = by_sim[cname]
        n = len(sim_rows)
        surv  = 100.0 * sum(r.survived for r in sim_rows) / n
        track = [r.tracking_error_mean for r in sim_rows]
        ht    = [r.base_height_dev_mean for r in sim_rows]
        force = [r.feet_force_rms for r in sim_rows]
        print(fmt.format(
            f"{cname} (n={n})", f"{surv:.1f}%",
            _fmt(track), _fmt(ht), _fmt(force),
        ))
        stats[cname] = {
            "surv": surv,
            "track_mean": statistics.mean(track),
        }
    print(sep)

    # --- Gap analysis ---
    pairs = [
        ("mjx_baseline", "cpu_baseline", "Baseline gap (CPU − MJX)"),
        ("mjx_residual", "cpu_residual", "Residual gap (CPU − MJX)"),
    ]

    has_both = all(a in stats and b in stats for a, b, _ in pairs)
    if has_both:
        print(f"\n{'SIM-TO-SIM GAP COMPARISON':^84}")
        print(sep)
        fmt2 = "  {:<38}  Surv drop: {:>6.1f} pp   Track drift: {:>6.3f} m/s"
        for mjx_k, cpu_k, label in pairs:
            if mjx_k not in stats or cpu_k not in stats:
                continue
            surv_drop    = stats[mjx_k]["surv"] - stats[cpu_k]["surv"]
            track_drift  = stats[cpu_k]["track_mean"] - stats[mjx_k]["track_mean"]
            print(fmt2.format(label, surv_drop, track_drift))
        print(sep)

        # Gap reduction from residual.
        bl_drop  = stats["mjx_baseline"]["surv"]  - stats["cpu_baseline"]["surv"]
        res_drop = stats["mjx_residual"]["surv"]   - stats["cpu_residual"]["surv"]
        bl_drift  = stats["cpu_baseline"]["track_mean"] - stats["mjx_baseline"]["track_mean"]
        res_drift = stats["cpu_residual"]["track_mean"] - stats["mjx_residual"]["track_mean"]

        if abs(bl_drop) > 0.1:
            reduction_surv = (bl_drop - res_drop) / bl_drop * 100
            print(f"  Survival gap reduction:  {reduction_surv:+.1f}%  "
                  f"({bl_drop:.1f} pp → {res_drop:.1f} pp)")
        if abs(bl_drift) > 1e-4:
            reduction_track = (bl_drift - res_drift) / bl_drift * 100
            print(f"  Tracking gap reduction:  {reduction_track:+.1f}%  "
                  f"({bl_drift:.3f} → {res_drift:.3f} m/s)")

        # M4 criterion: gap shrinks by ≥ 15%.
        def _verdict(pct):
            return "\033[92mPASS\033[0m" if pct >= 15 else "\033[91mFAIL\033[0m"

        if abs(bl_drop) > 0.1 and abs(bl_drift) > 1e-4:
            print()
            print(f"  M4-C1  Survival gap reduction ≥ 15% :  {reduction_surv:+.1f}%  "
                  f"{_verdict(reduction_surv)}")
            print(f"  M4-C2  Tracking gap reduction ≥ 15% :  {reduction_track:+.1f}%  "
                  f"{_verdict(reduction_track)}")
        print(sep)

    # --- Phase 0 gate verdict (mjx_cpu_faithful) ---
    if "mjx_cpu_faithful" in stats:
        surv_cf = stats["mjx_cpu_faithful"]["surv"]
        print(f"\n{'PHASE 0 GATE — mjx_cpu_faithful':^84}")
        print(sep)
        print(f"  Teacher survival on CPU-faithful MJX: {surv_cf:.1f}%")
        print(f"  Decision thresholds:  ≥ 90% → Case A (distill only)  |  < 90% → Case B (fine-tune teacher)")
        if surv_cf >= 90.0:
            verdict = "\033[92mCase A\033[0m"
            action  = "teacher is robust; proceed with physics-DR distillation (~10 M steps)"
        else:
            verdict = "\033[93mCase B\033[0m"
            action  = f"teacher overfit to training MJX ({surv_cf:.1f}% < 90%); PPO fine-tune first (~30 M steps)"
        print(f"  Verdict: {verdict}  —  {action}")
        print(sep)

    # --- Fall timing ---
    _print_fall_timing(rows, by_sim)


def _print_fall_timing(rows: list[EpisodeRow], by_sim: dict) -> None:
    fall_rows = [r for r in rows if not r.survived and r.fall_timestep >= 0]
    if not fall_rows:
        return

    import statistics

    sep = "─" * 84
    buckets = [(0, 50), (51, 200), (201, 500), (501, 10**9)]
    labels  = ["[0–50]", "[51–200]", "[201–500]", "[500+]"]

    print(f"\n{'FALL TIMING ANALYSIS':^84}")
    print(sep)
    hdr = "{:<22} {:>6} " + " ".join(f"{lb:>10}" for lb in labels) + "  {:>10}"
    print(hdr.format("Config", "Falls", *labels, "MeanStep"))
    print(sep)

    for cname in _ALL_CONFIGS:
        if cname not in by_sim:
            continue
        fallen = [r for r in by_sim[cname] if not r.survived and r.fall_timestep >= 0]
        counts = [sum(1 for r in fallen if lo <= r.fall_timestep <= hi)
                  for lo, hi in buckets]
        if not fallen:
            row_str = "{:<22} {:>6} ".format(cname, 0)
            row_str += " ".join(f"{'0':>10}" for _ in buckets)
            row_str += f"  {'—':>10}"
        else:
            mean_step = statistics.mean(r.fall_timestep for r in fallen)
            row_str = "{:<22} {:>6} ".format(cname, len(fallen))
            row_str += " ".join(f"{c:>10}" for c in counts)
            row_str += f"  {mean_step:>10.1f}"
        print(row_str)
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    csv_path    = results_dir / _CSV_NAME
    video_dir   = results_dir / "videos"
    command     = tuple(args.command)
    configs     = args.configs

    print("=" * 70)
    print(f"M4 Teacher Evaluation  —  {args.num_episodes} episodes × 30 s")
    print(f"Command: vx={command[0]:.2f}  vy={command[1]:.2f}  wz={command[2]:.2f}")
    print(f"Configs: {', '.join(configs)}")
    print(f"Results: {csv_path}")
    print("=" * 70)

    # --- Load teacher (always required) ---
    baseline_env, inference_fn = _load_teacher(
        str(Path(args.teacher_ckpt).resolve())
    )

    # --- Load residual components only when needed ---
    needs_residual = any(c in _RESIDUAL_CONFIGS for c in configs)
    residual_env   = None
    residual_model = None
    residual_norm  = None

    if needs_residual:
        print(f"\nLoading residual checkpoint from {args.residual_ckpt} …")
        if any(c in _MJX_CONFIGS & _RESIDUAL_CONFIGS for c in configs):
            residual_env = _load_residual_env(args.residual_ckpt)
        if any(c in _CPU_CONFIGS & _RESIDUAL_CONFIGS for c in configs):
            residual_model, residual_norm = _load_residual_model(args.residual_ckpt, args.device)

    # --- Load CPU-faithful MJX env for Phase 0 gate ---
    cpu_faithful_env = None
    if "mjx_cpu_faithful" in configs:
        print("\nBuilding mjx_cpu_faithful env (dt=0.5ms, iters=4) …")
        cpu_faithful_env = _load_cpu_faithful_env()

    existing = read_existing(csv_path)
    all_seeds = list(range(args.num_episodes))

    for config_name in configs:
        print(f"\n{'─'*70}")
        print(f"Config: {config_name}")

        done_seeds = {seed for (sim, seed) in existing if sim == config_name}
        missing    = [s for s in all_seeds if s not in done_seeds]

        if not missing:
            print(f"  All {args.num_episodes} episodes already collected — skipping.")
            continue

        print(f"  Collecting {len(missing)} episodes (already done: {len(done_seeds)}) …")

        adapter = _build_adapter(
            config_name, inference_fn, baseline_env, residual_env,
            cpu_faithful_env, residual_model, residual_norm, command, args.device,
        )

        n_vid       = args.num_videos if not args.no_video else 0
        video_seeds = missing[:n_vid]
        vid_dir     = video_dir if video_seeds else None

        rows = adapter.run_episodes(
            seeds=missing,
            video_seeds=video_seeds,
            video_dir=vid_dir,
        )

        append_rows(csv_path, rows)
        print(f"  Appended {len(rows)} rows to {csv_path.name}")

        # Release memory before loading the next backend.
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass
        try:
            import warp as wp
            wp.synchronize()
        except Exception:
            pass

    print(f"\n{'='*70}")
    _print_m4_summary(csv_path)


if __name__ == "__main__":
    main()
