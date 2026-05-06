"""Shared constants and observation buffers for M4 teacher evaluation.

TeacherObsBuffer builds the 81-dim state observation the M2 teacher expects.
Layout (mirrors joystick._get_obs without noise):
  [0:3]    gyro
  [3:6]    gravity (upvector sensor)
  [6:18]   joint_angles - default_pose
  [18:54]  qpos_error_history  (3 frames × 12 joints)
  [54:66]  feet_pos            (4 feet × 3 coords, body-relative, from *_pos sensors)
  [66:78]  last_act
  [78:81]  command

ResidualFeatureBuffer maintains the 4-step ring buffer consumed by the actuator
residual network.  Feature per step: concat(qpos_err, qvel_err, tau_cmd) ∈ R^36.
History: 4 steps × 36 → 144-dim flat vector, ordered oldest → newest.

Convention matches features.py (train/actuator_residual/):
  qpos_err = joint_angles - motor_targets
  qvel_err = joint_velocities          (no reference subtraction)
  tau_cmd  = -KP * qpos_err - KD * qvel_err   (KP=300, KD=1.0)
"""

from __future__ import annotations

import numpy as np

# Must match features.py values so the residual network input is in-distribution.
KP: float = 300.0
KD: float = 1.0
_HIST_LEN: int = 4
_FEAT_DIM: int = 36   # 12 qpos_err + 12 qvel_err + 12 tau_cmd

DEFAULT_POSE: np.ndarray = np.array([
    0.0,  1.04, -1.8,
    0.0,  1.04, -1.8,
    0.0,  1.04, -1.8,
    0.0,  1.04, -1.8,
], dtype=np.float32)

ACTION_SCALE: float = 0.3
BASE_HEIGHT_TARGET: float = 0.50
CTRL_DT: float = 0.02

# Body-relative foot position sensors (from spot_constants.FEET_POS_SENSOR).
FEET_POS_SENSORS: list[str] = ["FL_pos", "FR_pos", "HL_pos", "HR_pos"]


class TeacherObsBuffer:
    """Tracks motor_targets, qpos_error_history, and last_act for 81-dim teacher obs.

    Identical in logic to m5/adapters/base.ObsBuffer except it also receives
    feet_pos as an argument and places it at indices [54:66].
    """

    def __init__(self, ctrlrange: np.ndarray):
        """ctrlrange: (12, 2) array of [lower, upper] per actuator."""
        self._ctrlrange = ctrlrange
        self._motor_targets = np.zeros(12, dtype=np.float32)
        self._last_act = np.zeros(12, dtype=np.float32)
        self._qpos_error_history = np.zeros(36, dtype=np.float32)

    def reset(self) -> None:
        self._motor_targets = np.zeros(12, dtype=np.float32)
        self._last_act = np.zeros(12, dtype=np.float32)
        self._qpos_error_history = np.zeros(36, dtype=np.float32)

    def build_obs81(
        self,
        gyro: np.ndarray,           # (3,) angular velocity in body frame
        gravity: np.ndarray,        # (3,) upvector sensor reading
        joint_angles: np.ndarray,   # (12,) from qpos[7:19]
        feet_pos: np.ndarray,       # (12,) raveled 4×3 body-relative positions
        command: np.ndarray,        # (3,) vx vy wz
    ) -> np.ndarray:
        qpos_error = joint_angles - self._motor_targets

        self._qpos_error_history = np.roll(self._qpos_error_history, 12)
        self._qpos_error_history[:12] = qpos_error

        return np.concatenate([
            gyro,
            gravity,
            joint_angles - DEFAULT_POSE,
            self._qpos_error_history,
            feet_pos,
            self._last_act,
            command,
        ]).astype(np.float32)  # (81,)

    def update_after_action(self, action: np.ndarray) -> np.ndarray:
        """Clip action to target position and store; return motor_targets."""
        motor_targets = np.clip(
            DEFAULT_POSE + action * ACTION_SCALE,
            self._ctrlrange[:, 0],
            self._ctrlrange[:, 1],
        )
        self._motor_targets = motor_targets
        self._last_act = action.copy()
        return motor_targets


class ResidualFeatureBuffer:
    """4-step ring buffer of per-step features for the actuator residual MLP.

    Call update_and_get() at each step (before applying ctrl) to advance the
    buffer and retrieve the 144-dim input expected by the residual model.
    """

    def __init__(self):
        self._hist = np.zeros((_HIST_LEN, _FEAT_DIM), dtype=np.float32)

    def reset(self) -> None:
        self._hist[:] = 0.0

    def update_and_get(
        self,
        joint_angles: np.ndarray,   # (12,) qpos[7:19]
        joint_vels: np.ndarray,     # (12,) qvel[6:18]
        motor_targets: np.ndarray,  # (12,)
    ) -> np.ndarray:
        """Append new step feature and return 144-dim flat history (oldest→newest)."""
        qpos_err = joint_angles - motor_targets
        tau_cmd  = -KP * qpos_err - KD * joint_vels
        feat = np.concatenate([qpos_err, joint_vels, tau_cmd]).astype(np.float32)

        self._hist = np.roll(self._hist, -1, axis=0)
        self._hist[-1] = feat
        return self._hist.reshape(-1).copy()  # (144,)
