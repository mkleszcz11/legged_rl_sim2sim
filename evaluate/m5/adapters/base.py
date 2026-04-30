"""Shared types and observation construction for non-MJX simulator backends.

ObsBuffer replicates the obs construction from
mujoco_playground/_src/locomotion/spot/joystick.py _get_obs() without noise,
building the 69-dim proprio that the student expects.

81-dim state layout (from joystick._get_obs):
  [0:3]    gyro         (angular velocity, body frame)
  [3:6]    gravity      (upvector sensor — z-axis of IMU in world frame)
  [6:18]   joint_angles - default_pose    (12 joints)
  [18:54]  qpos_error_history             (3 frames × 12 joints)
  [54:66]  feet_pos                       (dropped by student — not built here)
  [66:78]  last_act                       (previous action)
  [78:81]  command                        (vx, vy, wz)

Student input = 81-dim with [54:66] dropped = 69 dims.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from metrics import EpisodeRow

# Matches the "home" keyframe in scene_mjx_feetonly_rough_terrain.xml.
DEFAULT_POSE: np.ndarray = np.array([
    0.0,  1.04, -1.8,   # fl
    0.0,  1.04, -1.8,   # fr
    0.0,  1.04, -1.8,   # hl
    0.0,  1.04, -1.8,   # hr
], dtype=np.float32)

ACTION_SCALE: float = 0.3   # from joystick.py config default
BASE_HEIGHT_TARGET: float = 0.50  # m, reward posture target in joystick.py
CTRL_DT: float = 0.02       # seconds per control step (50 Hz)


class BackendUnavailable(Exception):
    """Raised when a simulator backend cannot be imported or initialised."""


class ObsBuffer:
    """Tracks qpos_error_history and last_act to build 69-dim proprio.

    Must be reset before each episode.  Call build_obs69() to get the current
    observation, then update_after_action() once the student has chosen an action
    and physics has been stepped.
    """

    def __init__(self, ctrlrange: np.ndarray):
        """ctrlrange: (12, 2) array of [lower, upper] per actuator."""
        self._ctrlrange = ctrlrange
        self._motor_targets = np.zeros(12, dtype=np.float32)  # init = zeros (matches reset info)
        self._last_act = np.zeros(12, dtype=np.float32)
        self._qpos_error_history = np.zeros(36, dtype=np.float32)

    def reset(self) -> None:
        self._motor_targets = np.zeros(12, dtype=np.float32)
        self._last_act = np.zeros(12, dtype=np.float32)
        self._qpos_error_history = np.zeros(36, dtype=np.float32)

    def build_obs_69dim(
        self,
        gyro: np.ndarray,           # (3,) angular velocity in body frame
        gravity: np.ndarray,        # (3,) upvector sensor reading
        joint_angles: np.ndarray,   # (12,) from qpos[7:]
        command: np.ndarray,        # (3,) vx vy wz
    ) -> np.ndarray:
        """Build the 69-dim student input and advance the history buffer.

        Mirrors joystick._get_obs() without observation noise (clean obs for eval).
        """
        # qpos_error = current joints minus the motor targets that caused them.
        qpos_error = joint_angles - self._motor_targets

        # Roll history forward by one frame (12 joints) and insert the new error.
        self._qpos_error_history = np.roll(self._qpos_error_history, 12)
        self._qpos_error_history[:12] = qpos_error

        return np.concatenate([
            gyro,
            gravity,
            joint_angles - DEFAULT_POSE,
            self._qpos_error_history,
            self._last_act,
            command,
        ]).astype(np.float32)  # (69,)

    def update_after_action(self, action: np.ndarray) -> np.ndarray:
        """Store the action and compute clipped motor targets.

        Returns the motor_targets array that should be sent to the physics engine.
        """
        motor_targets = np.clip(
            DEFAULT_POSE + action * ACTION_SCALE,
            self._ctrlrange[:, 0],
            self._ctrlrange[:, 1],
        )
        self._motor_targets = motor_targets
        self._last_act = action.copy()
        return motor_targets


class SimAdapter(Protocol):
    """Common interface for each simulator backend."""

    name: str

    def run_episodes(
        self,
        seeds: list[int],
        video_seed: int | None,
        video_path: Path | None,
    ) -> list[EpisodeRow]:
        """Run episodes for the given seeds; optionally write one video.

        video_seed: which seed's trajectory to record (None = no video).
        video_path: where to write the MP4 (None = no video).
        """
        ...
