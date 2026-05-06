"""Shared feature computation for M4 actuator residual.

All three scripts (collect_data, residual_network, finetune_teacher) import
from here to guarantee identical conventions.

Convention (matches env's qpos_error_history in joystick._get_obs):
    qpos_err  = qpos[7:]  - motor_targets   (positive when joint is past target)
    qvel_err  = qvel[6:]                     (raw joint velocity; no commanded vel)
    tau_cmd   = -Kp * qpos_err - Kd * qvel_err   (= Kp*(q_des - q) - Kd*qd)

Feature per physics step: concat(qpos_err, qvel_err, tau_cmd) -> R^36
Network input: 4-step ring buffer flattened -> R^144, ordered oldest-to-newest.
Network output: tau_delta = tau_cpu - tau_mjx -> R^12
"""

import numpy as np

KP: float = 300.0
KD: float = 1.0
HISTORY_LEN: int = 4
FEAT_DIM: int = 36   # per physics step: 12 + 12 + 12
INPUT_DIM: int = HISTORY_LEN * FEAT_DIM   # 144
OUTPUT_DIM: int = 12


def compute_step_feature(
    qpos: np.ndarray,        # (19,) full generalized position
    qvel: np.ndarray,        # (18,) full generalized velocity
    motor_targets: np.ndarray,  # (12,) desired joint positions
    kp: float = KP,
    kd: float = KD,
) -> np.ndarray:
    """Return the 36-dim feature vector for one physics step."""
    qpos_err = qpos[7:] - motor_targets          # (12,)
    qvel_err = qvel[6:]                           # (12,)
    tau_cmd  = -kp * qpos_err - kd * qvel_err    # (12,)
    return np.concatenate([qpos_err, qvel_err, tau_cmd]).astype(np.float32)


def build_history_features(
    per_step_feats: np.ndarray,   # (N, 36) in episode order (substeps contiguous)
    history_len: int = HISTORY_LEN,
) -> np.ndarray:
    """Slide a ring buffer over per_step_feats -> (N, 144) input array.

    The first (history_len - 1) samples are padded with zeros so every sample
    has a well-defined feature vector.  History is ordered oldest -> newest in
    the flat output (first 36 = t-(history_len-1), last 36 = t).
    """
    n = len(per_step_feats)
    feat_dim = per_step_feats.shape[1]
    out = np.zeros((n, history_len * feat_dim), dtype=np.float32)
    for i in range(n):
        for h in range(history_len):
            src = i - (history_len - 1 - h)
            if src >= 0:
                out[i, h * feat_dim:(h + 1) * feat_dim] = per_step_feats[src]
    return out
