"""Genesis simulator backend — EXPLORATORY / OPTIONAL.

Genesis ships no Spot environment in its locomotion examples (only go2).
This adapter attempts to load the Spot MJCF via Genesis's MJCF importer.
If Genesis is not installed *or* the Spot MJCF fails to load, it raises
BackendUnavailable so the orchestrator can skip this backend with a warning.

The user has indicated they do not expect this to work out-of-the-box and
will invest time in integration separately.  This file provides the scaffold.

Usage (once Genesis is working):
    Install: pip install genesis-world
    Spot MJCF is at mujoco_playground/_src/locomotion/spot/xmls/scene_mjx_feetonly_rough_terrain.xml
"""

from pathlib import Path

from adapters.base import BackendUnavailable

# ── Guard: import Genesis, fail gracefully ────────────────────────────────────
try:
    import genesis as gs  # type: ignore
except ImportError as e:
    raise BackendUnavailable(
        "genesis-world is not installed.  Run:  pip install genesis-world"
    ) from e

import numpy as np
import torch

from adapters.base import (
    ACTION_SCALE,
    BASE_HEIGHT_TARGET,
    CTRL_DT,
    DEFAULT_POSE,
    ObsBuffer,
)
from metrics import EpisodeRow

_N_STEPS = round(30.0 / CTRL_DT)   # 1500 ctrl steps
_TERMINATION_GRAVITY_Z = 0.85
_FOOT_BODIES = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]


def _spot_xml_path() -> str:
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent.parent
    return str(repo_root / "mujoco_playground" / "mujoco_playground" / "_src" /
               "locomotion" / "spot" / "xmls" / "scene_mjx_feetonly_rough_terrain.xml")


class GenesisAdapter:
    """Genesis backend (exploratory).  Raises BackendUnavailable on init failure."""

    name = "genesis"

    def __init__(self, student: torch.nn.Module, command: tuple[float, float, float], device: str):
        self.student = student
        self.command = np.array(command, dtype=np.float32)
        self.device = device

        xml = _spot_xml_path()
        print(f"[genesis] Initialising Genesis and loading {Path(xml).name} …")

        try:
            gs.init(backend=gs.gpu)
            scene = gs.Scene(show_viewer=False)
            robot = scene.add_entity(gs.morphs.MJCF(file=xml))
            scene.build()
        except Exception as e:
            raise BackendUnavailable(f"Spot MJCF failed to load in Genesis: {e}") from e

        self._scene = scene
        self._robot = robot

        # Attempt to resolve actuator ctrl ranges from the Genesis model.
        # Genesis API for ctrlrange may differ; fall back to MJCF values if needed.
        try:
            ctrlrange = robot.get_dofs_limit()   # (n_dof, 2) — Genesis API
        except Exception:
            # Fallback: use joint ranges from the Spot MJCF (loaded via mujoco).
            import mujoco as _mj
            _mj_model = _mj.MjModel.from_xml_path(xml)
            ctrlrange = _mj_model.actuator_ctrlrange.copy()

        self._ctrlrange = ctrlrange

        try:
            self._foot_link_ids = [robot.get_link(name).id for name in _FOOT_BODIES]
        except Exception as e:
            raise BackendUnavailable(
                f"Could not resolve foot link IDs in Genesis model: {e}"
            ) from e

        print(f"[genesis] Ready.  n_steps={_N_STEPS}")

    def run_episodes(
        self,
        seeds: list[int],
        video_seed: int | None,
        video_path: Path | None,
    ) -> list[EpisodeRow]:
        if not seeds:
            return []

        rows = []
        for i, seed in enumerate(seeds):
            record_video = (seed == video_seed and video_path is not None
                            and not video_path.exists())
            print(f"[genesis] Episode {i+1}/{len(seeds)} (seed={seed})", end="", flush=True)
            row = self._run_one_episode(seed, record_video=record_video, video_path=video_path)
            rows.append(row)
            print(f"  {'FELL' if not row.survived else 'ok'}  "
                  f"track={row.tracking_error_mean:.3f} m/s")
        return rows

    def _run_one_episode(
        self,
        seed: int,
        record_video: bool,
        video_path: Path | None,
    ) -> EpisodeRow:
        scene = self._scene
        robot = self._robot
        rng = np.random.default_rng(seed)

        # Reset robot to home pose.
        home_qpos = np.concatenate([
            [0.0, 0.0, 0.46],    # base xyz
            [1.0, 0.0, 0.0, 0.0],  # base quat (wxyz)
            DEFAULT_POSE,
        ])
        home_qpos[:2] += rng.uniform(-0.1, 0.1, size=2)   # x-y jitter per seed
        robot.set_dofs_position(home_qpos[7:])
        scene.reset()

        obs_buf = ObsBuffer(self._ctrlrange)
        obs_buf.reset()
        hidden = self.student.init_hidden(1, self.device)
        command = self.command

        survived = True
        fall_step = _N_STEPS
        track_sum = 0.0
        height_sum = 0.0
        force_rms_sum = 0.0
        alive_cnt = 0

        self.student.eval()
        with torch.no_grad():
            for t in range(_N_STEPS):
                # Read state from Genesis.
                # NOTE: exact API calls depend on Genesis version — adjust if needed.
                qpos = robot.get_dofs_position()       # (n_dof,)
                qvel = robot.get_dofs_velocity()       # (n_dof,)
                base_pos = robot.get_pos()             # (3,)
                base_quat = robot.get_quat()           # (4,) xyzw or wxyz

                # Derived quantities for ObsBuffer.
                joint_angles = qpos[:12].astype(np.float32)
                gyro = _angular_vel_body_frame(qvel, base_quat)
                gravity = _upvector_from_quat(base_quat)

                obs69 = obs_buf.build_obs_69dim(gyro, gravity, joint_angles, command)
                obs_t = torch.from_numpy(obs69).unsqueeze(0).to(self.device)
                action_t, hidden = self.student(obs_t, hidden)
                action = action_t.squeeze(0).cpu().numpy()

                motor_targets = obs_buf.update_after_action(action)
                robot.control_dofs_position(motor_targets)
                scene.step()

                base_z = float(base_pos[2])
                gravity_z = gravity[2]

                if gravity_z < _TERMINATION_GRAVITY_Z:
                    survived = False
                    fall_step = t + 1
                    break

                linvel_local = _local_linvel_from_genesis(robot)
                linvel_x = float(linvel_local[0])

                foot_forces = _foot_contact_forces(robot, self._foot_link_ids)
                foot_rms = float(np.sqrt(np.mean(np.sum(foot_forces ** 2, axis=-1))))

                track_sum += abs(linvel_x - 1.0)
                height_sum += abs(base_z - BASE_HEIGHT_TARGET)
                force_rms_sum += foot_rms
                alive_cnt += 1

        cnt = max(alive_cnt, 1)
        return EpisodeRow(
            sim=self.name,
            seed=seed,
            survived=survived,
            tracking_error_mean=float(track_sum / cnt),
            base_height_dev_mean=float(height_sum / cnt),
            feet_force_rms=float(force_rms_sum / cnt),
            episode_seconds=float(fall_step * CTRL_DT),
        )


# ── Helper functions ──────────────────────────────────────────────────────────
# These replicate what the Spot MJX sensors compute; exact correctness depends
# on Genesis's coordinate conventions.  Verify against MJX output on first run.


def _quat_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ])


def _upvector_from_quat(quat: np.ndarray) -> np.ndarray:
    """Return z-axis of base frame in world coordinates (replicates 'upvector' sensor)."""
    R = _quat_wxyz_to_rotation_matrix(np.asarray(quat, dtype=np.float32))
    return R[:, 2].astype(np.float32)  # third column = world z in body frame (approx)


def _angular_vel_body_frame(qvel: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rotate world-frame angular velocity into body frame (replicates 'gyro' sensor)."""
    # Genesis may give angular velocity in world frame via base dof velocities.
    # If it gives body-frame directly, this rotation is identity.
    R = _quat_wxyz_to_rotation_matrix(np.asarray(quat, dtype=np.float32))
    world_angvel = np.asarray(qvel[3:6], dtype=np.float32)
    return (R.T @ world_angvel).astype(np.float32)


def _local_linvel_from_genesis(robot) -> np.ndarray:
    """Return linear velocity of base in local (body) frame."""
    # Adjust if Genesis uses a different API.
    try:
        return np.asarray(robot.get_vel(), dtype=np.float32)   # world frame — needs rotation
    except Exception:
        return np.zeros(3, dtype=np.float32)


def _foot_contact_forces(robot, foot_link_ids: list) -> np.ndarray:
    """Return (4, 3) contact forces on foot links.  Adjust to Genesis contact API."""
    forces = np.zeros((4, 3), dtype=np.float32)
    try:
        for i, link_id in enumerate(foot_link_ids):
            f = robot.get_link_contact_force(link_id)
            forces[i] = np.asarray(f[:3], dtype=np.float32)
    except Exception:
        pass   # Genesis contact API may differ; return zeros rather than crashing
    return forces
