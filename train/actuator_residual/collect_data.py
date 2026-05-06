"""M4 data collection: roll out M2 teacher and capture the physics accuracy gap.

For each physics substep in the rollout:
  1. Record (q_before, qd_before, motor_targets, q_mjx_after) from MJX.
     q_before is the state BEFORE mjx.step; q_mjx_after is AFTER.
  2. Starting from the same q_before, advance 8×0.5ms on accurate CPU MuJoCo
     (4 solver iters) to get q_cpu_after.  8×0.5ms = 4ms = one MJX sim_dt.
  3. Label = KP * (q_cpu_after[7:] − q_mjx_after[7:]): the position gap converted
     to an equivalent spring-stiffness torque correction (N·m).

Label design rationale:
  For position actuators with identical Kp, tau_cpu == tau_mjx at the same
  starting state -- there is no instantaneous force gap.  The accuracy gap
  manifests in the resulting trajectory.  Converting the 4ms position difference
  to torque (KP * Δq) gives a label the network can learn to correct in-situ.

States are injected fresh for every sample -- no drift across substeps.

Run from repo root:
    python train/actuator_residual/collect_data.py \
        --teacher_ckpt mujoco_playground/logs/SpotJoystickRoughTerrain-20260428-123447/checkpoints \
        --n_episodes 50 --ctrl_steps 1000 \
        --out train/actuator_residual/data/residual.npz

Smoke test (fast):
    python train/actuator_residual/collect_data.py --n_episodes 2 --ctrl_steps 50 --sanity_check
"""

import argparse
import os
import sys
import time
from pathlib import Path

# JAX env vars must be set before any JAX / MuJoCo import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "mujoco_playground"))
sys.path.insert(0, str(_REPO_ROOT / "evaluate" / "m2"))

import functools

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from features import (
    KD,
    KP,
    HISTORY_LEN,
    build_history_features,
    compute_step_feature,
)
from mujoco_playground._src.locomotion.spot.base import get_assets as _get_spot_assets

# Loaded lazily after sys.path setup.
import load_policy  # evaluate/m2/load_policy.py

_DEFAULT_TEACHER_CKPT = str(
    _REPO_ROOT / "mujoco_playground/logs/SpotJoystickRoughTerrain-20260428-123447/checkpoints"
)
_DEFAULT_OUT = str(_REPO_ROOT / "train/actuator_residual/data/residual.npz")
_CPU_TIMESTEP = 5e-4
_CPU_ITERATIONS = 4


# ---------------------------------------------------------------------------
# MJX per-substep collection
# ---------------------------------------------------------------------------

def _make_substep_collector(mjx_model, n_substeps: int):
    """Return a jit'd function that steps n_substeps and returns all intermediates."""

    @functools.partial(jax.jit)
    def collect(data: mjx.Data, motor_targets: jax.Array):
        """Step n_substeps from `data` with fixed motor_targets.

        Returns:
            final_data:  MJX data after all substeps.
            qbefore_seq: (n_substeps, 19) generalized positions BEFORE each step.
            qvel_seq:    (n_substeps, 18) generalized velocities BEFORE each step.
            qafter_seq:  (n_substeps, 19) generalized positions AFTER each step.

        Capturing q_before (not q_after) is critical: the CPU replay must start
        from the same pre-step state that MJX uses, so both cover the same 4ms
        window and their resulting q_after values are directly comparable.
        """
        def single_step(data, _):
            q_before  = data.qpos   # capture state BEFORE physics step
            qd_before = data.qvel
            data = data.replace(ctrl=motor_targets)
            data = mjx.step(mjx_model, data)
            return data, (q_before, qd_before, data.qpos)

        final_data, (qbefore_seq, qvel_seq, qafter_seq) = jax.lax.scan(
            single_step, data, None, n_substeps
        )
        return final_data, qbefore_seq, qvel_seq, qafter_seq

    return collect


# ---------------------------------------------------------------------------
# CPU single-step replay
# ---------------------------------------------------------------------------

def _build_cpu_model(xml_path: str, timestep: float, iterations: int):
    """Load MJCF on CPU and apply the same parameter overrides that SpotEnv
    applies to the MJX model -- otherwise the (CPU - MJX) gap is dominated by
    a kp/damping mismatch, not by integration accuracy.

    Specifically, SpotEnv.__init__ in spot/base.py overrides:
      dof_damping[6:]        = config.Kd            (XML 2 -> 1)
      actuator_gainprm[:, 0] = config.Kp            (XML 400 -> 300)
      actuator_biasprm[:, 1] = -config.Kp           (XML -400 -> -300)
    Without these overrides, CPU runs ~33% stiffer position feedback and 2x
    joint damping vs MJX, so the residual labels capture the wrong gap.
    """
    assets = _get_spot_assets()
    xml_string = Path(xml_path).read_text()
    mj_model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
    mj_model.opt.timestep = timestep
    mj_model.opt.iterations = iterations
    # Match SpotEnv overrides on the CPU model.
    mj_model.dof_damping[6:]        = KD
    mj_model.actuator_gainprm[:, 0] = KP
    mj_model.actuator_biasprm[:, 1] = -KP
    return mj_model


_N_CPU_SUBSTEPS = 8  # 8 × 0.5ms = 4ms = one MJX sim_dt


def _replay_full_step(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    qpos_before: np.ndarray,   # (nq,) full generalized position BEFORE MJX step
    qvel_before: np.ndarray,   # (nv,) full generalized velocity BEFORE MJX step
    motor_targets: np.ndarray, # (12,) desired joint positions
) -> np.ndarray:
    """Inject pre-step state, advance 8×0.5ms CPU steps, return resulting qpos.

    8 × 0.5ms = 4ms = one MJX sim_dt, so CPU and MJX cover identical timespan.
    The position gap (q_cpu_after − q_mjx_after) captures the accuracy difference
    between the two physics backends starting from the same initial state.
    """
    mj_data.qpos[:] = qpos_before
    mj_data.qvel[:] = qvel_before
    mj_data.ctrl[:] = motor_targets
    for _ in range(_N_CPU_SUBSTEPS):
        mujoco.mj_step(mj_model, mj_data)
    return mj_data.qpos.copy()


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_and_collect(
    env,
    infer_fn,
    n_episodes: int,
    ctrl_steps: int,
    seed: int,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Roll out the teacher and collect per-substep physics data.

    Returns:
        qbefore_all:  (N_total, nq) float32  -- qpos BEFORE each MJX substep
        qvel_all:     (N_total, nv) float32  -- qvel BEFORE each MJX substep
        targets_all:  (N_total, 12) float32  -- motor_targets (constant per ctrl step)
        qafter_mjx_all: (N_total, nq) float32  -- qpos AFTER each MJX substep
    """
    n_substeps = env.n_substeps
    mjx_model = env.mjx_model

    collect_substeps = _make_substep_collector(mjx_model, n_substeps)
    jit_reset = jax.jit(env.reset)
    jit_step  = jax.jit(env.step)

    default_pose = np.array(env._default_pose, dtype=np.float32)
    lowers       = np.array(env._lowers, dtype=np.float32)
    uppers       = np.array(env._uppers, dtype=np.float32)
    action_scale = env._config.action_scale

    rng = jax.random.PRNGKey(seed)
    all_qbefore, all_qvel, all_targets, all_qafter_mjx = [], [], [], []

    for ep in range(n_episodes):
        rng, ep_rng, act_rng = jax.random.split(rng, 3)
        state = jit_reset(ep_rng)

        for cs in range(ctrl_steps):
            act_rng, key = jax.random.split(act_rng)
            action, _ = infer_fn(state.obs, key)
            motor_targets = np.clip(
                default_pose + np.array(action) * action_scale, lowers, uppers
            )
            motor_targets_jax = jp.array(motor_targets)

            # Collect per-substep states from MJX (q_before, qd_before, q_after).
            _, qbefore_seq, qvel_seq, qafter_seq = collect_substeps(
                state.data, motor_targets_jax
            )
            # Convert to numpy: (n_substeps, dim)
            qbefore_np = np.array(qbefore_seq, dtype=np.float32)
            qvel_np    = np.array(qvel_seq,    dtype=np.float32)
            qafter_np  = np.array(qafter_seq,  dtype=np.float32)

            for sub in range(n_substeps):
                all_qbefore.append(qbefore_np[sub])
                all_qvel.append(qvel_np[sub])
                all_targets.append(motor_targets)
                all_qafter_mjx.append(qafter_np[sub])

            # Advance env state for policy observation.
            state = jit_step(state, action)

            if bool(state.done):
                break

        if verbose:
            pct = (ep + 1) / n_episodes * 100
            print(f"  Episode {ep+1}/{n_episodes} ({pct:.0f}%)  "
                  f"samples so far: {len(all_qbefore)}", end="\r")

    if verbose:
        print()

    return (
        np.stack(all_qbefore),
        np.stack(all_qvel),
        np.stack(all_targets),
        np.stack(all_qafter_mjx),
    )


# ---------------------------------------------------------------------------
# CPU replay loop
# ---------------------------------------------------------------------------

def compute_qpos_cpu(
    mj_model: mujoco.MjModel,
    qbefore_all: np.ndarray,  # (N, nq)
    qvel_all: np.ndarray,     # (N, nv)
    targets_all: np.ndarray,  # (N, 12)
    verbose: bool = True,
) -> np.ndarray:
    """Full-step CPU replay for every sample.  Returns (N, nq) q_cpu_after.

    Each call injects q_before and advances 8×0.5ms (= 4ms = one MJX sim_dt)
    under accurate CPU physics (4 solver iterations vs MJX's 1).
    """
    n = len(qbefore_all)
    nq = mj_model.nq
    q_cpu_after = np.zeros((n, nq), dtype=np.float32)
    mj_data = mujoco.MjData(mj_model)

    t0 = time.time()
    for i in range(n):
        q_cpu_after[i] = _replay_full_step(
            mj_model, mj_data, qbefore_all[i], qvel_all[i], targets_all[i]
        )
        if verbose and (i + 1) % 10_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (n - i - 1) / rate
            print(f"  CPU replay {i+1}/{n}  ({rate:.0f} steps/s, ETA {eta:.0f}s)", end="\r")

    if verbose:
        print()
    return q_cpu_after


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def assemble_dataset(
    qbefore_all: np.ndarray,    # (N, nq)  -- qpos BEFORE each MJX substep
    qvel_all: np.ndarray,       # (N, nv)  -- qvel BEFORE each MJX substep
    targets_all: np.ndarray,    # (N, 12)
    qafter_mjx_all: np.ndarray, # (N, nq)  -- qpos AFTER MJX substep
    q_cpu_after_all: np.ndarray, # (N, nq) -- qpos after 8×0.5ms CPU steps
    n_substeps: int,
) -> dict:
    """Build per-step feature vectors and labels.

    Label = KP * (q_cpu_after[7:] − q_mjx_after[7:])
    This converts the position gap between accurate-CPU and fast-MJX into an
    equivalent spring-stiffness torque correction (units: N·m).
    Features are computed from the pre-step state to match collect and finetune.
    """
    n = len(qbefore_all)
    feats_raw = np.zeros((n, 36), dtype=np.float32)
    for i in range(n):
        feats_raw[i] = compute_step_feature(qbefore_all[i], qvel_all[i], targets_all[i])

    # Build 4-step history.  Reset history at episode boundaries.
    # Episodes are n_substeps * ctrl_steps long; we don't have exact boundaries here,
    # so we build history treating the full array as one sequence.
    # (A small amount of cross-episode leakage at episode boundaries is benign;
    #  those samples are padded with the previous episode's last features.)
    features = build_history_features(feats_raw, history_len=HISTORY_LEN)

    # Position gap → equivalent spring-stiffness torque label.
    delta_q = q_cpu_after_all[:, 7:] - qafter_mjx_all[:, 7:]  # (N, 12) rad
    labels  = (KP * delta_q).astype(np.float32)                # (N, 12) N·m

    return {"features": features, "labels": labels}


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_check_delta(qafter_mjx: np.ndarray, q_cpu_after: np.ndarray) -> None:
    """Print per-joint position gap and equivalent torque magnitude.

    Expected with accurate CPU (dt=0.5ms, iters=4):  RMS ~0.001–0.01 rad per joint,
                                                       KP*RMS ~0.3–3 N·m.
    With MJX-matching CPU config (dt=4ms, iters=1):   RMS should be near 0.
    """
    delta_q = q_cpu_after[:, 7:] - qafter_mjx[:, 7:]   # (N, 12) rad
    rms_per_joint = np.sqrt(np.mean(delta_q**2, axis=0))
    label_rms     = KP * rms_per_joint                  # (12,) N·m
    mean_rms_q    = rms_per_joint.mean()
    mean_label    = label_rms.mean()
    print("\nSanity check -- per-joint RMS(q_cpu_after − q_mjx_after) [rad] → label [N·m]:")
    for j, (dq, tau) in enumerate(zip(rms_per_joint, label_rms)):
        print(f"  joint {j:2d}: Δq={dq:.5f} rad   label≈{tau:.3f} N·m")
    print(f"  mean: Δq={mean_rms_q:.5f} rad   label≈{mean_label:.3f} N·m")
    if mean_label < 0.05:
        print("  WARNING: labels near-zero. CPU config may match MJX exactly, "
              "or q_before was used incorrectly.")
    else:
        print("  OK: non-trivial position gap detected (accurate-CPU vs fast-MJX).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Collect M4 actuator residual training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--teacher_ckpt", default=_DEFAULT_TEACHER_CKPT,
                   help="Path to M2 teacher checkpoints/ directory.")
    p.add_argument("--n_episodes",  type=int, default=50)
    p.add_argument("--ctrl_steps",  type=int, default=1000,
                   help="Max control steps per episode.")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--out",         default=_DEFAULT_OUT,
                   help="Output .npz path.")
    p.add_argument("--sanity_check", action="store_true",
                   help="Print RMS(delta) and exit without saving.")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Orbax requires an absolute path for checkpoint restoration.
    teacher_ckpt = str(Path(args.teacher_ckpt).resolve())
    print(f"Loading M2 teacher from {teacher_ckpt} ...")
    # impl="jax" uses pure JAX physics (no Warp pool), avoiding the Warp/XLA
    # CUDA memory conflict on 4 GB GPU.  naconmax reduced for single-env rollout
    # (training used 32768 for 8192 parallel envs; single env needs ~100–300).
    env, infer_fn = load_policy.load_policy(
        teacher_ckpt,
        env_name="SpotJoystickRoughTerrain",
        num_envs=1,
        config_overrides={"impl": "jax", "naconmax": 1024},
    )
    print(f"n_substeps={env.n_substeps}  ctrl_dt={env._config.ctrl_dt}s  "
          f"sim_dt={env._config.sim_dt}s")

    # Find MJCF path from env.
    xml_path = env.xml_path
    print(f"MJCF: {xml_path}")
    print(f"CPU physics: dt={_CPU_TIMESTEP*1000:.1f}ms  iters={_CPU_ITERATIONS}")

    print(f"\nRolling out {args.n_episodes} episodes × {args.ctrl_steps} ctrl steps ...")
    t0 = time.time()
    qbefore_all, qvel_all, targets_all, qafter_mjx_all = rollout_and_collect(
        env, infer_fn, args.n_episodes, args.ctrl_steps, args.seed
    )
    print(f"MJX rollout done in {time.time()-t0:.1f}s  ({len(qbefore_all)} substeps)")

    print(f"\nRunning CPU full-step replay ({_N_CPU_SUBSTEPS}×{_CPU_TIMESTEP*1000:.1f}ms each) ...")
    mj_model = _build_cpu_model(xml_path, _CPU_TIMESTEP, _CPU_ITERATIONS)
    t0 = time.time()
    q_cpu_after_all = compute_qpos_cpu(mj_model, qbefore_all, qvel_all, targets_all)
    print(f"CPU replay done in {time.time()-t0:.1f}s")

    sanity_check_delta(qafter_mjx_all, q_cpu_after_all)

    if args.sanity_check:
        print("\n--sanity_check: exiting without saving.")
        return

    print("\nAssembling dataset ...")
    dataset = assemble_dataset(
        qbefore_all, qvel_all, targets_all, qafter_mjx_all, q_cpu_after_all, env.n_substeps
    )

    meta = {
        "kp": KP, "kd": KD,
        "cpu_dt": _CPU_TIMESTEP, "cpu_iters": _CPU_ITERATIONS,
        "n_episodes": args.n_episodes, "ctrl_steps": args.ctrl_steps,
        "n_substeps": env.n_substeps,
        "history_len": HISTORY_LEN,
    }

    np.savez(
        out_path,
        features=dataset["features"],
        labels=dataset["labels"],
        meta=np.array([meta]),   # object array wrapper for dict
    )
    n = len(dataset["features"])
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {n} samples to {out_path}  ({size_mb:.1f} MB)")
    print(f"  features: {dataset['features'].shape}  labels: {dataset['labels'].shape}")


if __name__ == "__main__":
    main()
