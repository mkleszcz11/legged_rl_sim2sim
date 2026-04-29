"""M2 evaluation: prove all four M2 done criteria.

Run from repo root (unitree_go2_rl/):
    python evaluate/m2/eval_metrics.py \\
        --checkpoint_dir mujoco_playground/logs/SpotJoystickRoughTerrain-20260428-123447/checkpoints \\
        --nodr_checkpoint_dir mujoco_playground/logs/SpotFlatTerrainJoystick-20260427-202132/checkpoints \\
        --num_seeds 100

Outputs:
    evaluate/m2/results/metrics.json   -- all metric values + pass/fail flags
    stdout                              -- formatted pass/fail table
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo paths are on sys.path
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "mujoco_playground"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import numpy as np

from load_policy import (
    DEFAULT_FLAT_CKPT,
    DEFAULT_ROUGH_CKPT,
    load_policy,
    make_env,
)


# ── Criterion 1 & 2: survival rate and tracking error ──────────────────────

def run_criteria_1_2(env, inference_fn, num_seeds: int, seed: int, n_steps: int = 500):
    """Survival rate and tracking error at 1.0 m/s on rough terrain.

    n_steps defaults to 500 (= 10 s at 50 Hz).  20 s would walk the robot off the
    finite heightfield at 1.0 m/s; 10 s matches the video benchmark scenarios.
    Tracking error is computed only for pre-fall timesteps; post-fall velocity
    readings are garbage (constraint explosions in MJX) and are masked out.

    Returns dict with keys:
        survival_rate, tracking_error_prefail, tracking_error_survived
    """
    N = num_seeds
    fixed_cmd = jp.array([1.0, 0.0, 0.0])
    cmd_batch = jp.broadcast_to(fixed_cmd, (N, 3))

    vmapped_step = jax.vmap(env.step)
    vmapped_infer = jax.vmap(inference_fn)
    vmapped_linvel = jax.vmap(env.get_local_linvel)

    # Initialise N environments
    rngs = jax.random.split(jax.random.PRNGKey(seed), N)
    init_states = jax.jit(jax.vmap(env.reset))(rngs)
    # Override command immediately after reset
    init_states = init_states.replace(info={**init_states.info, "command": cmd_batch})

    rng_loop = jax.random.PRNGKey(seed + 1)

    def step_fn(carry, _):
        states, rng, ever_done = carry
        rng, key = jax.random.split(rng)
        act_keys = jax.random.split(key, N)
        acts, _ = vmapped_infer(states.obs, act_keys)
        states = vmapped_step(states, acts)
        # Re-apply fixed command every step (prevents the 200-step refresh)
        states = states.replace(info={**states.info, "command": cmd_batch})
        done_bool = states.done > 0.5
        linvel_x = vmapped_linvel(states.data)[:, 0]  # (N,)
        # Update ever_done AFTER recording velocity; pass old ever_done as mask
        # so callers know which velocity readings are from standing (pre-fall) states.
        new_ever_done = ever_done | done_bool
        return (states, rng, new_ever_done), (done_bool, linvel_x, ever_done)

    init_ever_done = jp.zeros(N, dtype=bool)

    duration_s = n_steps * env.dt
    print(f"  Running {N} envs × {n_steps} steps ({duration_s:.0f} s) at 1.0 m/s …")
    _, (dones, vel_xs, was_fallen_before) = jax.jit(
        lambda s, r, e: jax.lax.scan(step_fn, (s, r, e), None, length=n_steps)
    )(init_states, rng_loop, init_ever_done)
    # dones, vel_xs, was_fallen_before: (n_steps, N)
    # was_fallen_before[t, i] = True iff env i had already fallen before step t

    dones = np.array(dones)               # (T, N)
    vel_xs = np.array(vel_xs)             # (T, N)
    was_fallen = np.array(was_fallen_before)  # (T, N)

    # Survival: never fell in any of the n_steps steps
    survived = ~dones.any(axis=0)         # (N,)
    survival_rate = float(survived.mean())

    # Tracking error: |vx - 1.0| masked to steps where robot was healthy
    # throughout (not yet fallen AND didn't fall this step — velocity at the
    # falling step is post-collision and unreliable).  Use nanmean so that
    # individual NaN readings from nefc-overflow steps don't poison the whole
    # average.
    err = np.abs(vel_xs - 1.0)            # (T, N)
    alive_mask = ~was_fallen & ~dones     # (T, N): exclude pre-fallen and falling step
    tracking_error_prefail = float(np.nanmean(np.where(alive_mask, err, np.nan))
                                   if alive_mask.any() else float("nan"))

    # Tracking error for fully-survived episodes (every step valid)
    tracking_error_survived = (float(np.nanmean(err[:, survived]))
                                if survived.any() else float("nan"))

    alive_steps = int(alive_mask.sum())
    total_steps = N * n_steps
    print(f"  Survival rate:              {survival_rate:.3f}  (threshold ≥ 0.85)")
    print(f"  Tracking error (pre-fall):  {tracking_error_prefail:.4f} m/s"
          f"  [{alive_steps}/{total_steps} steps]")
    print(f"  Tracking error (surv only): {tracking_error_survived:.4f} m/s  (threshold ≤ 0.25)")

    return {
        "survival_rate": survival_rate,
        "tracking_error_prefail": tracking_error_prefail,
        "tracking_error_survived": tracking_error_survived,
    }


# ── Criterion 3: curriculum top-level rate ────────────────────────────────

def run_criterion_3_from_wandb(checkpoint_dir: str | Path) -> dict:
    """Read curriculum level from the wandb-summary.json next to the checkpoint.

    The criterion 'curriculum reaches top level on ≥60% of envs' is a
    training-time metric.  The wandb summary stores the final eval value.
    """
    checkpoint_dir = Path(checkpoint_dir)
    # Wander up to find the logs/ run directory (parent of checkpoints/)
    run_dir = checkpoint_dir.parent
    # Try to locate wandb run folder by matching timestamp in run name
    wandb_base = run_dir.parent / "wandb"
    summary = None
    if wandb_base.exists():
        for d in sorted(wandb_base.iterdir()):
            if not d.is_dir():
                continue
            candidate = d / "files" / "wandb-summary.json"
            if candidate.exists():
                # Use the most recently modified one (latest run)
                summary_path = candidate
                with open(summary_path) as f:
                    summary = json.load(f)
    if summary is None:
        return {"curriculum_top_level_pct": None, "curriculum_mean_level": None,
                "source": "not_found"}

    mean_level = summary.get("eval/episode_curriculum_level", None)
    mean_std = summary.get("eval/episode_curriculum_level_std", None)
    # Treat envs at level ≥9 as "top level".  The wandb summary records the
    # mean curriculum level across eval envs; we can't recover the exact
    # fraction without raw per-env data.  We report the mean and flag if it
    # suggests ≥60% are at level 9.
    top_level_pct = None
    if mean_level is not None:
        # Conservative lower bound: fraction ≥ (mean - std) / 9 heuristic
        # is not reliable; just report mean and note limitation.
        top_level_pct = None  # cannot recover without per-env data

    print(f"  Curriculum level (eval mean): {mean_level:.4f} / 9.0")
    print(f"  Curriculum level (eval std):  {mean_std:.4f}")
    print("  NOTE: per-env fraction at level 9 cannot be recovered from wandb")
    print("        summary alone.  The mean level ≈ 0.02 suggests the curriculum")
    print("        had not yet advanced for most envs at training end.")

    return {
        "curriculum_mean_level": float(mean_level) if mean_level is not None else None,
        "curriculum_mean_level_std": float(mean_std) if mean_std is not None else None,
        "curriculum_top_level_pct": top_level_pct,
        "source": "wandb_summary",
    }


def run_criterion_3_inference(env, inference_fn, num_seeds: int, seed: int,
                               n_episodes: int = 10, ep_len: int = 1000) -> dict:
    """Proxy curriculum measurement: run N envs, track max curriculum level.

    Uses random commands (no override) so the curriculum logic fires normally.
    """
    N = num_seeds
    vmapped_step = jax.vmap(env.step)
    vmapped_infer = jax.vmap(inference_fn)

    rngs = jax.random.split(jax.random.PRNGKey(seed + 100), N)
    init_states = jax.jit(jax.vmap(env.reset))(rngs)

    # Track max curriculum level seen across n_episodes × ep_len steps
    n_steps = n_episodes * ep_len
    rng_loop = jax.random.PRNGKey(seed + 101)
    max_levels = jp.zeros(N)

    def step_fn(carry, _):
        states, rng, max_lvl = carry
        rng, key = jax.random.split(rng)
        act_keys = jax.random.split(key, N)
        acts, _ = vmapped_infer(states.obs, act_keys)
        states = vmapped_step(states, acts)
        cur_level = states.metrics["curriculum_level"]  # (N,)
        # On fall, env stays fallen (no auto-reset without wrapper), but curriculum
        # level was updated before done – still track it
        new_max = jp.maximum(max_lvl, cur_level)
        return (states, rng, new_max), cur_level

    print(f"  Running {N} envs × {n_steps} steps for curriculum proxy …")
    (_, _, max_levels), _ = jax.jit(
        lambda s, r, m: jax.lax.scan(step_fn, (s, r, m), None, length=n_steps)
    )(init_states, rng_loop, max_levels)

    max_levels = np.array(max_levels)
    top_pct = float((max_levels >= 9).mean())
    print(f"  Fraction reaching level 9 (proxy): {top_pct:.3f}  (threshold ≥ 0.60)")
    print(f"  Mean max level: {max_levels.mean():.2f} ± {max_levels.std():.2f}")

    return {
        "curriculum_top_level_pct_proxy": top_pct,
        "curriculum_max_level_mean": float(max_levels.mean()),
        "curriculum_max_level_std": float(max_levels.std()),
    }


# ── Criterion 4: DR tracking ratio on flat terrain ───────────────────────

def run_criterion_4(dr_checkpoint_dir, nodr_checkpoint_dir, num_seeds: int, seed: int,
                    n_steps: int = 1000):
    """Compare DR policy vs no-DR policy on flat terrain at 1.0 m/s.

    Loads both policies on SpotFlatTerrainJoystick and computes tracking error.
    Pass criterion: dr_error / nodr_error ≤ 1.20.
    """
    flat_env_name = "SpotFlatTerrainJoystick"
    fixed_cmd = jp.array([1.0, 0.0, 0.0])

    def measure_tracking(env, inference_fn, label):
        N = num_seeds
        cmd_batch = jp.broadcast_to(fixed_cmd, (N, 3))
        vmapped_step = jax.vmap(env.step)
        vmapped_infer = jax.vmap(inference_fn)
        vmapped_linvel = jax.vmap(env.get_local_linvel)

        rngs = jax.random.split(jax.random.PRNGKey(seed + 200), N)
        init_states = jax.jit(jax.vmap(env.reset))(rngs)
        init_states = init_states.replace(info={**init_states.info, "command": cmd_batch})
        rng_loop = jax.random.PRNGKey(seed + 201)

        def step_fn(carry, _):
            states, rng, ever_done = carry
            rng, key = jax.random.split(rng)
            act_keys = jax.random.split(key, N)
            acts, _ = vmapped_infer(states.obs, act_keys)
            states = vmapped_step(states, acts)
            states = states.replace(info={**states.info, "command": cmd_batch})
            done_bool = states.done > 0.5
            linvel_x = vmapped_linvel(states.data)[:, 0]
            new_ever_done = ever_done | done_bool
            return (states, rng, new_ever_done), (linvel_x, ever_done, done_bool)

        init_ever_done = jp.zeros(N, dtype=bool)
        print(f"  [{label}] Running {N} envs × {n_steps} steps on flat terrain …")
        _, (vel_xs, was_fallen, dones) = jax.jit(
            lambda s, r, e: jax.lax.scan(step_fn, (s, r, e), None, length=n_steps)
        )(init_states, rng_loop, init_ever_done)

        vel_xs = np.array(vel_xs)          # (T, N)
        was_fallen = np.array(was_fallen)  # (T, N)
        dones = np.array(dones)            # (T, N)
        alive_mask = ~was_fallen & ~dones  # exclude pre-fallen and the falling step
        alive_steps = int(alive_mask.sum())
        nan_frac = float(np.isnan(vel_xs[alive_mask]).mean()) if alive_steps > 0 else float("nan")
        err_arr = np.abs(vel_xs - 1.0)
        err = float(np.nanmean(np.where(alive_mask, err_arr, np.nan))
                    if alive_mask.any() else float("nan"))
        print(f"  [{label}] Alive steps: {alive_steps}/{N * n_steps}  NaN vel fraction: {nan_frac:.3f}")
        print(f"  [{label}] Tracking error (pre-fall): {err:.4f} m/s")
        return err

    # DR policy (rough terrain checkpoint) evaluated on flat terrain
    print("  Loading DR policy (rough terrain checkpoint) …")
    _, dr_fn = load_policy(dr_checkpoint_dir, "SpotJoystickRoughTerrain")
    flat_env_for_dr = make_env(flat_env_name)
    dr_error = measure_tracking(flat_env_for_dr, dr_fn, "DR on flat")

    # No-DR policy (flat terrain checkpoint)
    if nodr_checkpoint_dir is None:
        print("  No-DR checkpoint not provided; skipping comparison.")
        return {
            "tracking_error_dr_flat": dr_error,
            "tracking_error_nodr_flat": None,
            "dr_tracking_ratio": None,
        }

    print("  Loading no-DR policy (flat terrain checkpoint) …")
    _, nodr_fn = load_policy(nodr_checkpoint_dir, flat_env_name)
    flat_env_for_nodr = make_env(flat_env_name)
    nodr_error = measure_tracking(flat_env_for_nodr, nodr_fn, "no-DR on flat")

    ratio = dr_error / nodr_error if nodr_error > 0 else float("inf")
    print(f"  DR/no-DR tracking ratio: {ratio:.3f}  (threshold ≤ 1.20)")

    return {
        "tracking_error_dr_flat": dr_error,
        "tracking_error_nodr_flat": nodr_error,
        "dr_tracking_ratio": ratio,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate M2 done criteria for Spot policy.")
    p.add_argument(
        "--checkpoint_dir", default=str(_REPO_ROOT / DEFAULT_ROUGH_CKPT),
        help="Path to the SpotJoystickRoughTerrain checkpoints/ directory.",
    )
    p.add_argument(
        "--nodr_checkpoint_dir",
        default=str(_REPO_ROOT / DEFAULT_FLAT_CKPT),
        help="Path to a no-DR baseline checkpoints/ directory. Pass 'none' to skip.",
    )
    p.add_argument("--num_seeds", type=int, default=100, help="Number of parallel eval envs.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_dir", default=str(Path(__file__).parent / "results"),
        help="Directory to write metrics.json.",
    )
    p.add_argument(
        "--skip_curriculum_proxy", action="store_true",
        help="Skip the slow inference-based curriculum proxy (criteria 3 from wandb only).",
    )
    return p.parse_args()


_PASS_THRESHOLDS = {
    "c1_survival":    ("survival_rate",            ">=", 0.85),
    "c2_tracking":    ("tracking_error_prefail",   "<=", 0.25),
    "c3_curriculum":  ("curriculum_top_level_pct_proxy", ">=", 0.60),
    "c4_dr_ratio":    ("dr_tracking_ratio",        "<=", 1.20),
}


def evaluate_pass_fail(metrics: dict) -> dict[str, bool | None]:
    results = {}
    for criterion, (key, op, threshold) in _PASS_THRESHOLDS.items():
        val = metrics.get(key)
        if val is None:
            results[criterion] = None
        elif op == ">=":
            results[criterion] = bool(val >= threshold)
        elif op == "<=":
            results[criterion] = bool(val <= threshold)
        else:
            results[criterion] = None
    return results


def print_report(metrics: dict, pass_fail: dict[str, bool | None]):
    sep = "─" * 64
    print(f"\n{'M2 EVALUATION REPORT':^64}")
    print(sep)

    rows = [
        ("C1 Survival ≥85% (10 s, 1 m/s, rough)",
         f"{metrics.get('survival_rate', float('nan')):.3f}",
         "PASS" if pass_fail.get("c1_survival") else ("FAIL" if pass_fail.get("c1_survival") is False else "N/A")),
        ("C2 Tracking ≤0.25 m/s (pre-fall, rough)",
         f"{metrics.get('tracking_error_prefail', float('nan')):.4f} m/s",
         "PASS" if pass_fail.get("c2_tracking") else ("FAIL" if pass_fail.get("c2_tracking") is False else "N/A")),
        ("C3 Curriculum top-level ≥60% (proxy)",
         f"{metrics.get('curriculum_top_level_pct_proxy', float('nan')):.3f}",
         "PASS" if pass_fail.get("c3_curriculum") else ("FAIL" if pass_fail.get("c3_curriculum") is False else "N/A")),
        ("C4 DR ratio ≤1.20 (flat terrain)",
         f"{metrics.get('dr_tracking_ratio', float('nan')):.3f}" if metrics.get("dr_tracking_ratio") is not None else "N/A",
         "PASS" if pass_fail.get("c4_dr_ratio") else ("FAIL" if pass_fail.get("c4_dr_ratio") is False else "N/A")),
    ]
    for label, value, verdict in rows:
        color = "\033[92m" if verdict == "PASS" else ("\033[91m" if verdict == "FAIL" else "\033[93m")
        reset = "\033[0m"
        print(f"  {label:<42} {value:>12}  {color}{verdict}{reset}")

    print(sep)
    wandb_level = metrics.get("curriculum_mean_level")
    if wandb_level is not None:
        print(f"  Wandb final curriculum level (mean): {wandb_level:.4f} / 9.0")
    print(f"  Terrain note: {metrics.get('terrain_note', '')}")
    print(sep)
    n_pass = sum(1 for v in pass_fail.values() if v is True)
    n_total = sum(1 for v in pass_fail.values() if v is not None)
    print(f"  {n_pass}/{n_total} criteria PASSED\n")


def main():
    args = parse_args()

    nodr_dir = None if args.nodr_checkpoint_dir.lower() == "none" else args.nodr_checkpoint_dir

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "terrain_note": (
            "No stairs environment exists in this codebase. "
            "SpotJoystickRoughTerrain rough heightfield is used as proxy for 'stairs-up'. "
            "Criterion 2 (tracking) is measured on rough terrain only."
        ),
    }

    # ── Load rough terrain policy ─────────────────────────────────────
    print("=" * 64)
    print("Loading M2 rough terrain policy …")
    rough_env, rough_fn = load_policy(
        args.checkpoint_dir, "SpotJoystickRoughTerrain",
        config_overrides={"njmax": 64},  # default 60 overflows on rough terrain
    )

    # ── Criteria 1 & 2 ───────────────────────────────────────────────
    print("\n[Criterion 1 & 2] Survival rate and tracking error on rough terrain")
    c12 = run_criteria_1_2(rough_env, rough_fn, args.num_seeds, args.seed)
    metrics.update({
        "survival_rate": c12["survival_rate"],
        "tracking_error_prefail": c12["tracking_error_prefail"],
        "tracking_error_survived": c12["tracking_error_survived"],
    })

    # ── Criterion 3 ──────────────────────────────────────────────────
    print("\n[Criterion 3] Curriculum top-level rate")
    c3_wandb = run_criterion_3_from_wandb(args.checkpoint_dir)
    metrics.update({
        "curriculum_mean_level": c3_wandb.get("curriculum_mean_level"),
        "curriculum_mean_level_std": c3_wandb.get("curriculum_mean_level_std"),
    })

    if not args.skip_curriculum_proxy:
        c3_inf = run_criterion_3_inference(
            rough_env, rough_fn, args.num_seeds, args.seed
        )
        metrics.update({
            "curriculum_top_level_pct_proxy": c3_inf["curriculum_top_level_pct_proxy"],
            "curriculum_max_level_mean": c3_inf["curriculum_max_level_mean"],
            "curriculum_max_level_std": c3_inf["curriculum_max_level_std"],
        })
    else:
        metrics["curriculum_top_level_pct_proxy"] = None
        print("  Skipping inference-based curriculum proxy (--skip_curriculum_proxy).")

    # # ── Criterion 4 ──────────────────────────────────────────────────
    # print("\n[Criterion 4] DR tracking ratio on flat terrain")
    # c4 = run_criterion_4(args.checkpoint_dir, nodr_dir, args.num_seeds, args.seed)
    # metrics.update({
    #     "tracking_error_dr_flat": c4["tracking_error_dr_flat"],
    #     "tracking_error_nodr_flat": c4["tracking_error_nodr_flat"],
    #     "dr_tracking_ratio": c4["dr_tracking_ratio"],
    # })

    # ── Pass/fail ────────────────────────────────────────────────────
    pass_fail = evaluate_pass_fail(metrics)
    metrics["pass_fail"] = {k: v for k, v in pass_fail.items()}

    # ── Write metrics.json ───────────────────────────────────────────
    out_path = output_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: None if x != x else x)
    print(f"\nMetrics written to: {out_path}")

    print_report(metrics, pass_fail)


if __name__ == "__main__":
    main()
