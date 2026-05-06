"""M3 evaluation: prove all four M3 done criteria.

Criteria:
  C1  Imitation MSE     ≤ 0.05   student vs teacher actions (free-cmd rollout)
  C2  Survival rate     ≥ 0.75   rough terrain, random commands
  C3  Gait frequency  in [1.5, 4.0] Hz  healthy trot
  C4  Tracking error   ≤ 0.40 m/s  forward tracking at fixed 1.0 m/s command

Run from repo root (legged_rl_sim2sim/):
    python evaluate/m3/eval_metrics.py
    python evaluate/m3/eval_metrics.py --student_checkpoint checkpoints/.../student_spot_proprio_m3.pt

Outputs:
    evaluate/m3/results/metrics.json   -- all metric values + pass/fail flags
    stdout                              -- formatted pass/fail table
"""

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "mujoco_playground"))
sys.path.insert(0, str(_REPO_ROOT / "train" / "utils"))
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax.numpy as jp
import torch
import torch.nn.functional as F

from mujoco_playground._src.wrapper_torch import _jax_to_torch, _torch_to_jax
from jax_oracle import load_teacher
from load_student import (
    DEFAULT_STUDENT_CKPT,
    ENV_NAME,
    TEACHER_CKPT_DIR,
    build_env_wrapper,
    extract_proprio,
    load_student,
)


# Check 1: free-command rollout via RSLRLBraxWrapper

def run_check1(env, student, oracle, num_steps: int, device: str) -> dict:
    """Run student + teacher for num_steps steps with env-sampled commands.

    Returns C1 (imitation MSE), C2 (survival rate), C3 (gait Hz).
    """
    num_envs      = env.num_envs
    ctrl_dt: float = 0.02  # seconds per control step

    prev_feet_air  = torch.zeros(num_envs, 4, device=device)
    swing_counts   = torch.zeros(num_envs,    device=device)
    ep_steps       = torch.zeros(num_envs,    device=device)
    survival_count = torch.zeros(num_envs,    device=device)
    total_episodes = torch.zeros(num_envs,    device=device)
    total_mse      = 0.0

    obs    = env.reset()
    hidden = student.init_hidden(num_envs, device)

    student.eval()
    with torch.no_grad():
        for _ in range(num_steps):
            state_obs              = obs["state"]                    # (N, 81)
            proprio                = extract_proprio(state_obs)      # (N, 69)
            teacher_acts           = oracle.query(state_obs)         # (N, 12)
            student_acts, hidden   = student(proprio, hidden)        # (N, 12)

            total_mse += F.mse_loss(student_acts, teacher_acts).item()

            obs, _, done, _ = env.step(student_acts)
            hidden = hidden * (1.0 - done).view(1, -1, 1)

            feet_air      = _jax_to_torch(env.env_state.info["feet_air_time"])  # (N, 4)
            liftoffs      = (feet_air > 0) & (prev_feet_air == 0)
            swing_counts += liftoffs.float().sum(dim=1)
            prev_feet_air = feet_air.clone()

            ep_steps += 1.0
            terminated = done > 0.5
            if terminated.any():
                survived = (ep_steps[terminated] >= env.max_episode_length * 0.9).float()
                survival_count[terminated] += survived
                total_episodes[terminated] += 1.0
                ep_steps[terminated]        = 0.0

    mean_mse  = total_mse / num_steps
    gait_hz   = swing_counts.mean().item() / (num_steps * ctrl_dt) / 4
    surv_rate = (
        (survival_count.sum() / total_episodes.sum()).item()
        if total_episodes.sum() > 0 else 0.0
    )

    print(f"  Imitation MSE:    {mean_mse:.5f}  (threshold ≤ 0.05)")
    print(f"  Survival rate:    {surv_rate:.3f}  (threshold ≥ 0.75)")
    print(f"  Gait frequency:   {gait_hz:.2f} Hz  (threshold [1.5, 4.0])")
    return {
        "imitation_mse": mean_mse,
        "survival_rate": surv_rate,
        "gait_hz":       gait_hz,
    }


# Check 2: fixed 1.0 m/s command, reusing the wrapped env from Check 1

def run_check2(env_wrapped, student, device: str, num_steps: int = 500) -> dict:
    """Fixed 1.0 m/s forward command, measures tracking error.

    Reuses the already-built RSLRLBraxWrapper so no second env is allocated
    (a second env would OOM the 4 GB GPU while check 1's 128-env wrapper is live).

    Linvel is read from privileged_state[90:93].  The privileged_state layout is:
        [0:81]  state obs  (same as student obs + feet_pos)
        [81:84] gyro
        [84:87] accelerometer
        [87:90] gravity
        [90:93] get_local_linvel(data)  ← forward velocity is index 90
    """
    num_envs  = env_wrapped.num_envs
    fixed_cmd = jp.broadcast_to(jp.array([1.0, 0.0, 0.0]), (num_envs, 3))

    obs    = env_wrapped.reset()
    hidden = student.init_hidden(num_envs, device)

    # Pin the command to 1.0 m/s forward before the first step.
    env_wrapped.env_state = env_wrapped.env_state.replace(
        info={**env_wrapped.env_state.info, "command": fixed_cmd}
    )

    ever_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    err_sum   = torch.zeros(num_envs, device=device)
    alive_cnt = torch.zeros(num_envs, device=device)

    print(f"  Running {num_envs} envs × {num_steps} steps at 1.0 m/s …")
    student.eval()
    with torch.no_grad():
        for _ in range(num_steps):
            proprio          = extract_proprio(obs["state"])     # (N, 69)
            act, hidden      = student(proprio, hidden)

            obs, _, done, _  = env_wrapped.step(act)
            hidden           = hidden * (1.0 - done).view(1, -1, 1)

            # Re-pin command; env refreshes it every 200 steps otherwise.
            env_wrapped.env_state = env_wrapped.env_state.replace(
                info={**env_wrapped.env_state.info, "command": fixed_cmd}
            )

            # privileged_state[90] = local_linvel_x (verified from obs layout above).
            linvel_x   = obs["privileged_state"][:, 90]          # (N,) torch
            alive_mask = ~ever_done & (done < 0.5)
            err_sum[alive_mask]   += (linvel_x[alive_mask] - 1.0).abs()
            alive_cnt[alive_mask] += 1.0
            ever_done = ever_done | (done > 0.5)

    total_alive    = int(alive_cnt.sum().item())
    tracking_error = (err_sum.sum() / alive_cnt.sum()).item() if total_alive > 0 else float("nan")
    survival_1ms   = float((~ever_done).float().mean().item())

    print(f"  Tracking error (pre-fall): {tracking_error:.4f} m/s  (threshold ≤ 0.40)")
    print(f"  Survival @1.0 m/s:         {survival_1ms:.3f}  "
          f"[{total_alive} alive steps / {num_envs * num_steps} total]")
    return {
        "tracking_error_1ms": tracking_error,
        "survival_rate_1ms":  survival_1ms,
        "alive_steps_1ms":    total_alive,
    }


_PASS_THRESHOLDS = {
    "c1_mse":      ("imitation_mse",      "<=", 0.05),
    "c2_survival": ("survival_rate",      ">=", 0.75),
    "c3_gait":     ("gait_hz",            "in", (1.5, 4.0)),
    "c4_tracking": ("tracking_error_1ms", "<=", 0.40),
}


def evaluate_pass_fail(metrics: dict) -> dict[str, bool | None]:
    results = {}
    for crit, (key, op, threshold) in _PASS_THRESHOLDS.items():
        val = metrics.get(key)
        if val is None or (isinstance(val, float) and val != val):
            results[crit] = None
        elif op == "<=":
            results[crit] = bool(val <= threshold)
        elif op == ">=":
            results[crit] = bool(val >= threshold)
        elif op == "in":
            lo, hi = threshold
            results[crit] = bool(lo <= val <= hi)
        else:
            results[crit] = None
    return results


def print_report(metrics: dict, pass_fail: dict):
    sep = "─" * 68
    print(f"\n{'M3 EVALUATION REPORT':^68}")
    print(sep)
    rows = [
        ("C1.1 Imitation MSE ≤ 0.05 (free cmd, student vs teacher)",
         f"{metrics.get('imitation_mse', float('nan')):.5f}",
         pass_fail.get("c1_mse")),
        ("C1.2 Survival rate ≥ 0.75 (rough terrain, random cmd)",
         f"{metrics.get('survival_rate', float('nan')):.3f}",
         pass_fail.get("c2_survival")),
        ("C1.3 Gait frequency in [1.5, 4.0] Hz (trot health)",
         f"{metrics.get('gait_hz', float('nan')):.2f} Hz",
         pass_fail.get("c3_gait")),
        ("C2.1 Tracking error ≤ 0.40 m/s (1.0 m/s cmd, pre-fall)",
         f"{metrics.get('tracking_error_1ms', float('nan')):.4f} m/s",
         pass_fail.get("c4_tracking")),
    ]
    for label, value, verdict in rows:
        v     = "PASS" if verdict else ("FAIL" if verdict is False else "N/A")
        color = "\033[92m" if v == "PASS" else ("\033[91m" if v == "FAIL" else "\033[93m")
        print(f"  {label:<52} {value:>10}  {color}{v}\033[0m")
    print(sep)
    n_pass  = sum(1 for v in pass_fail.values() if v is True)
    n_total = sum(1 for v in pass_fail.values() if v is not None)
    print(f"  {n_pass}/{n_total} criteria PASSED\n")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate M3 done criteria for the student policy.")
    p.add_argument(
        "--student_checkpoint",
        default=str(_REPO_ROOT / DEFAULT_STUDENT_CKPT),
        help="Path to student_spot_proprio_m3.pt checkpoint file.",
    )
    p.add_argument(
        "--teacher_checkpoint_dir",
        default=str(_REPO_ROOT / TEACHER_CKPT_DIR),
        help="Path to the teacher checkpoints/ directory.",
    )
    p.add_argument("--num_envs",      type=int, default=128, help="Envs for check-1 rollout.")
    p.add_argument("--num_steps",     type=int, default=1024, help="Steps per env in check 1.")
    p.add_argument("--num_steps_c4",  type=int, default=500, help="Steps per env in C2.1 (reuses Check 1 env).")
    p.add_argument("--device",        default="cuda:0")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument(
        "--output_dir",
        default=str(Path(__file__).parent / "results"),
        help="Directory to write metrics.json.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("M3 Evaluation: Student Distillation (Proprio-only GRU → Spot)")
    print("=" * 68)

    print(f"\nLoading student …")
    student = load_student(args.student_checkpoint, device=args.device)
    print(f"Loaded student from {args.student_checkpoint}")

    print(f"\nLoading teacher oracle …")
    oracle = load_teacher(args.teacher_checkpoint_dir, ENV_NAME)
    print(f"Loaded teacher from {args.teacher_checkpoint_dir}")

    print(f"\nBuilding wrapped env ({args.num_envs} envs) …")
    env_wrapped = build_env_wrapper(num_envs=args.num_envs, seed=args.seed)

    print(f"\n[Check 1] Free-command rollout - C1.1 MSE, C1.2 Survival, C1.3 Gait")
    p1 = run_check1(env_wrapped, student, oracle,
                    num_steps=args.num_steps, device=args.device)

    print(f"\n[Check 2] Fixed 1.0 m/s command - C2.1 Tracking error")
    p2 = run_check2(env_wrapped, student, device=args.device,
                    num_steps=args.num_steps_c4)

    metrics   = {**p1, **p2}
    pass_fail = evaluate_pass_fail(metrics)
    metrics["pass_fail"] = {k: v for k, v in pass_fail.items()}

    out_path = output_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=lambda x: None if x != x else x)
    print(f"\nMetrics written to: {out_path}")

    print_report(metrics, pass_fail)


if __name__ == "__main__":
    main()
