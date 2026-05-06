"""Test 2 — M2 teacher under M5 DR distribution.

Samples N envs from the M5 physics-DR sampling distribution (the same
`domain_randomize` function used during student distillation), runs the M2
teacher zero-shot on each for 30 s, and reports survival.

This is the gate that decides Case A vs Case B for M5:

    Survival ≥ 90%   → Case A: teacher is robust across the DR distribution;
                        distill student under physics-DR, no teacher fine-tune.
    Survival 60–89%  → Case A marginal: narrow DR ranges or fall through to B.
    Survival < 60%   → Case B: PPO fine-tune teacher under physics-DR (~30 M
                        steps) before distillation.

The mjx_cpu_faithful test (one corner of the DR box, 100% survival) does NOT
imply Case A across the full distribution — that was the methodology error in
the first M5 attempt. This test samples the *actual* training distribution.

Run from repo root:
    python evaluate/m4/teacher_under_dr.py --num_episodes 100

Outputs:
    evaluate/m4/results/teacher_under_dr.csv
    stdout — per-env results + Case A/B verdict
"""

import argparse
import functools
import importlib
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
sys.path.insert(0, str(_REPO_ROOT / "evaluate" / "m5"))   # for metrics.py
sys.path.insert(0, str(_M4_DIR))

import jax
import jax.numpy as jp
import numpy as np

import load_policy   # evaluate/m2/load_policy.py
from metrics import EpisodeRow, append_rows, read_existing
from adapters.base import CTRL_DT, BASE_HEIGHT_TARGET

_DEFAULT_TEACHER_CKPT = str(
    _REPO_ROOT / "mujoco_playground/logs/SpotJoystickRoughTerrain-20260428-123447/checkpoints"
)
_DEFAULT_RAND_SPEC = "mujoco_playground._src.locomotion.spot.randomize:domain_randomize"
_DEFAULT_RESULTS_DIR = _M4_DIR / "results"
_CSV_NAME = "teacher_under_dr.csv"
_SIM_NAME = "mjx_dr"

_N_STEPS = round(30.0 / CTRL_DT)   # 1500 ctrl steps = 30 s
_FOOT_BODIES = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]
_RNG_INFER = jax.random.PRNGKey(0)   # deterministic policy ignores this


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test 2: M2 teacher under M5 DR distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--teacher_ckpt", default=_DEFAULT_TEACHER_CKPT)
    p.add_argument(
        "--randomize_fn", default=_DEFAULT_RAND_SPEC,
        help="module:function spec of the DR sampling function.",
    )
    p.add_argument("--num_episodes", type=int, default=100,
                   help="Number of envs sampled from the DR distribution.")
    p.add_argument("--seed", type=int, default=0,
                   help="Master RNG seed for both DR sampling and reset keys.")
    p.add_argument(
        "--command", nargs=3, type=float, default=[1.0, 0.0, 0.0],
        metavar=("VX", "VY", "WZ"),
    )
    p.add_argument("--results_dir", default=str(_DEFAULT_RESULTS_DIR))
    p.add_argument("--naconmax", type=int, default=1024,
                   help="Cap on Warp collision-pair buffer (VRAM safety).")
    return p.parse_args()


def _resolve_randomize_fn(spec: str):
    """'pkg.mod:func' → callable."""
    if ":" not in spec:
        raise ValueError(f"randomize_fn spec must be 'module:function', got {spec!r}")
    module_path, func_name = spec.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _load_teacher_and_env(teacher_ckpt: str, naconmax: int):
    """Load M2 teacher inference_fn and a fresh SpotJoystickRoughTerrain env."""
    from mujoco_playground import registry

    print(f"Loading M2 teacher from {teacher_ckpt} …")
    _, inference_fn = load_policy.load_policy(
        teacher_ckpt,
        env_name="SpotJoystickRoughTerrain",
        num_envs=1,
        config_overrides={"impl": "jax", "naconmax": 1024},
    )

    env = registry.load(
        "SpotJoystickRoughTerrain",
        config_overrides={"naconmax": naconmax},
    )
    print(f"[{_SIM_NAME}] Env loaded.  n_substeps={env.n_substeps}  "
          f"sim_dt={env.sim_dt}  ctrl_dt={env.dt}  naconmax={naconmax}")
    return env, inference_fn


def _build_dr_wrapper(env, randomize_fn, num_envs: int, seed: int):
    """Wrap the env with BraxDomainRandomizationVmapWrapper bound to N envs."""
    from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper

    master = jax.random.PRNGKey(seed)
    _, key_rand = jax.random.split(master)
    rand_keys = jax.random.split(key_rand, num_envs)
    v_rand_fn = functools.partial(randomize_fn, rng=rand_keys)
    wrapped = BraxDomainRandomizationVmapWrapper(env, v_rand_fn)
    print(f"[{_SIM_NAME}] DR wrapper ready over {num_envs} envs.")
    return wrapped


def _run_batch(wrapped_env, inference_fn, base_env, command, num_envs: int,
               seed: int) -> list[EpisodeRow]:
    cmd_batch = jp.tile(jp.array(command, dtype=jp.float32), (num_envs, 1))

    master = jax.random.PRNGKey(seed)
    key_reset, _ = jax.random.split(master)
    reset_keys = jax.random.split(key_reset, num_envs)

    jit_reset = jax.jit(wrapped_env.reset)
    jit_step  = jax.jit(wrapped_env.step)

    print(f"[{_SIM_NAME}] Resetting {num_envs} envs (JIT compile + reset) …")
    states = jit_reset(reset_keys)
    states = states.replace(info={**states.info, "command": cmd_batch})

    mj_model = base_env.mj_model
    foot_body_ids = np.array([mj_model.body(name).id for name in _FOOT_BODIES])

    n = num_envs
    alive         = np.ones(n, dtype=bool)
    ever_fallen   = np.zeros(n, dtype=bool)
    fall_step     = np.full(n, _N_STEPS, dtype=int)
    track_sum     = np.zeros(n, dtype=np.float64)
    height_sum    = np.zeros(n, dtype=np.float64)
    force_rms_sum = np.zeros(n, dtype=np.float64)
    alive_cnt     = np.zeros(n, dtype=np.int64)

    print(f"[{_SIM_NAME}] Stepping {_N_STEPS} ctrl steps …")
    for t in range(_N_STEPS):
        actions, _ = inference_fn(states.obs, _RNG_INFER)
        actions    = jp.clip(actions, -1.0, 1.0)
        states     = jit_step(states, actions)
        states     = states.replace(info={**states.info, "command": cmd_batch})

        done_t    = np.array(states.done)
        just_fell = done_t > 0.5
        fall_step[just_fell & ~ever_fallen] = t + 1
        ever_fallen |= just_fell

        if alive.any():
            linvel_x = np.array(states.obs["privileged_state"][:, 90])
            base_z   = np.array(states.data.qpos[:, 2])
            foot_f   = np.array(states.data.cfrc_ext[:, foot_body_ids, 3:6])
            foot_rms = np.sqrt(np.mean(np.sum(foot_f ** 2, axis=-1), axis=-1))

            track_sum[alive]     += np.abs(linvel_x[alive] - 1.0)
            height_sum[alive]    += np.abs(base_z[alive] - BASE_HEIGHT_TARGET)
            force_rms_sum[alive] += foot_rms[alive]
            alive_cnt[alive]     += 1

        alive = alive & ~just_fell

        if (t + 1) % 250 == 0:
            n_alive = int(alive.sum())
            print(f"  step {t+1:>4d}/{_N_STEPS}  alive={n_alive}/{n}")

    rows = []
    for i in range(n):
        cnt = max(alive_cnt[i], 1)
        rows.append(EpisodeRow(
            sim=_SIM_NAME,
            seed=i,
            survived=not ever_fallen[i],
            tracking_error_mean=float(track_sum[i] / cnt),
            base_height_dev_mean=float(height_sum[i] / cnt),
            feet_force_rms=float(force_rms_sum[i] / cnt),
            episode_seconds=float(fall_step[i] * CTRL_DT),
            fall_timestep=int(fall_step[i] - 1) if ever_fallen[i] else -1,
        ))
    return rows


def _print_summary(rows: list[EpisodeRow]) -> None:
    import statistics
    sep = "─" * 84
    n = len(rows)
    surv = 100.0 * sum(r.survived for r in rows) / n
    track = [r.tracking_error_mean for r in rows]
    ht    = [r.base_height_dev_mean for r in rows]
    force = [r.feet_force_rms for r in rows]

    def _fmt(v):
        if len(v) < 2:
            return f"{v[0]:.3f}"
        return f"{statistics.mean(v):.3f}±{statistics.stdev(v):.3f}"

    print(f"\n{'TEST 2 — M2 TEACHER UNDER M5 DR DISTRIBUTION':^84}")
    print(sep)
    print(f"{'Config':<22} {'Surv%':>8} {'TrackErr(m/s)':>17} "
          f"{'HeightDev(m)':>17} {'FeetForce(N)':>14}")
    print(sep)
    print(f"{_SIM_NAME + f' (n={n})':<22} {surv:>7.1f}% "
          f"{_fmt(track):>17} {_fmt(ht):>17} {_fmt(force):>14}")
    print(sep)

    # Decision rule per README §M5 Test 2.
    if surv >= 90.0:
        verdict = "\033[92mCase A\033[0m"
        action  = "Distill student under physics-DR; no teacher fine-tune."
    elif surv >= 60.0:
        verdict = "\033[93mCase A marginal\033[0m"
        action  = ("Narrow DR ranges to where teacher survives, or fall through "
                   "to Case B (PPO fine-tune).")
    else:
        verdict = "\033[91mCase B\033[0m"
        action  = ("PPO fine-tune teacher under physics-DR (~30 M steps from "
                   "M2 ckpt, LR=3e-5) before distillation.")

    print(f"\n  Decision: {verdict}  (survival = {surv:.1f}%)")
    print(f"  → {action}")
    print(sep)

    # Fall timing histogram.
    fallen = [r for r in rows if not r.survived and r.fall_timestep >= 0]
    if fallen:
        buckets = [(0, 50), (51, 200), (201, 500), (501, 10**9)]
        labels  = ["[0–50]", "[51–200]", "[201–500]", "[500+]"]
        counts  = [sum(1 for r in fallen if lo <= r.fall_timestep <= hi)
                   for lo, hi in buckets]
        mean_step = statistics.mean(r.fall_timestep for r in fallen)
        print(f"\n  Fall timing  (total falls = {len(fallen)})")
        for lb, c in zip(labels, counts):
            print(f"    {lb:>10} : {c}")
        print(f"    mean fall step = {mean_step:.1f}")
        print(sep)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / _CSV_NAME
    command  = tuple(args.command)

    print("=" * 84)
    print(f"Test 2 — M2 teacher under M5 DR distribution")
    print(f"  num_episodes : {args.num_episodes}")
    print(f"  command      : vx={command[0]:.2f}  vy={command[1]:.2f}  wz={command[2]:.2f}")
    print(f"  randomize_fn : {args.randomize_fn}")
    print(f"  teacher_ckpt : {args.teacher_ckpt}")
    print(f"  results      : {csv_path}")
    print("=" * 84)

    randomize_fn = _resolve_randomize_fn(args.randomize_fn)
    base_env, inference_fn = _load_teacher_and_env(
        str(Path(args.teacher_ckpt).resolve()), args.naconmax,
    )
    wrapped = _build_dr_wrapper(
        base_env, randomize_fn, args.num_episodes, args.seed,
    )

    rows = _run_batch(
        wrapped, inference_fn, base_env, command,
        args.num_episodes, args.seed,
    )

    # Overwrite CSV (Test 2 is a one-shot diagnostic; no resume semantics).
    if csv_path.exists():
        csv_path.unlink()
    append_rows(csv_path, rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    _print_summary(rows)


if __name__ == "__main__":
    main()
