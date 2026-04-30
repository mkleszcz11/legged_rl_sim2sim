"""MuJoCo CPU backend: single-env sequential rollout using vanilla mujoco bindings.

Loads the same MJCF as MJX but overrides physics settings per the M5 plan:
  - timestep  = 0.5 ms  (finer than MJX's 4 ms; intentional CPU-quality setting)
  - iterations = 4       (vs MJX's 1)

This makes contact dynamics more accurate but deliberately diverges from MJX,
which is the point: we want to measure the sim-to-sim gap.

Observation construction uses ObsBuffer to replicate joystick._get_obs()
without noise, building the same 69-dim proprio as the MJX env exposes.
"""

from pathlib import Path

try:
    import mujoco
except ImportError as e:
    raise ImportError(
        "mujoco CPU bindings not found.  Install with:  pip install mujoco"
    ) from e

import mediapy
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

# ── Physics parameters ────────────────────────────────────────────────────────
_CPU_TIMESTEP = 5e-4          # 0.5 ms per physics step
_CPU_ITERATIONS = 4
_N_SUBSTEPS = round(CTRL_DT / _CPU_TIMESTEP)   # 40 substeps per ctrl step
_N_STEPS = round(30.0 / CTRL_DT)               # 1500 ctrl steps = 30 s

_TERMINATION_GRAVITY_Z = 0.85  # matches joystick._get_termination()

# Body names containing the sphere foot geoms.
_FOOT_BODIES = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]


def _xml_path() -> str:
    """Absolute path to the rough-terrain scene MJCF (same file as MJX uses)."""
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent.parent
    return str(repo_root / "mujoco_playground" / "mujoco_playground" / "_src" /
               "locomotion" / "spot" / "xmls" / "scene_mjx_feetonly_rough_terrain.xml")


class MuJoCoCPUAdapter:
    name = "mujoco_cpu"

    def __init__(self, student: torch.nn.Module, command: tuple[float, float, float], device: str):
        self.student = student
        self.command = np.array(command, dtype=np.float32)
        self.device = device

        xml = _xml_path()
        print(f"[mujoco_cpu] Loading model from {Path(xml).name} …")
        self._mj_model = mujoco.MjModel.from_xml_path(xml)

        # Override physics settings (intentional divergence from MJX defaults).
        self._mj_model.opt.timestep = _CPU_TIMESTEP
        self._mj_model.opt.iterations = _CPU_ITERATIONS

        # Cache sensor addresses (read once, reuse per step).
        # Pattern: model.sensor(name).id → model.sensor_adr[id] for sensordata offset.
        self._gyro_adr = self._mj_model.sensor_adr[self._mj_model.sensor("gyro").id]
        self._upvector_adr = self._mj_model.sensor_adr[self._mj_model.sensor("upvector").id]
        self._local_linvel_adr = self._mj_model.sensor_adr[self._mj_model.sensor("local_linvel").id]

        # Foot body IDs for cfrc_ext contact force extraction.
        self._foot_body_ids = np.array(
            [self._mj_model.body(name).id for name in _FOOT_BODIES]
        )

        # Keyframe index for "home" reset pose.
        self._home_key_id = self._mj_model.keyframe("home").id

        print(
            f"[mujoco_cpu] Ready.  n_steps={_N_STEPS}  n_substeps={_N_SUBSTEPS}"
            f"  dt={_CPU_TIMESTEP*1000:.1f} ms  iters={_CPU_ITERATIONS}"
        )

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

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
            print(f"[mujoco_cpu] Episode {i+1}/{len(seeds)} (seed={seed})", end="", flush=True)
            row, frames = self._run_one_episode(seed, capture_frames=record_video)
            rows.append(row)
            print(f"  {'FELL' if not row.survived else 'ok'}  "
                  f"track={row.tracking_error_mean:.3f} m/s")

            if record_video and frames:
                _write_video(frames, video_path, fps=round(1.0 / CTRL_DT))

        return rows

    def _run_one_episode(
        self,
        seed: int,
        capture_frames: bool,
    ) -> tuple[EpisodeRow, list]:
        mj_model = self._mj_model
        data = mujoco.MjData(mj_model)

        # Reset to "home" keyframe with a seed-perturbed base position (adds variety
        # across seeds without breaking the policy; same spirit as random RNG reset).
        mujoco.mj_resetDataKeyframe(mj_model, data, self._home_key_id)
        rng = np.random.default_rng(seed)
        data.qpos[:2] += rng.uniform(-0.1, 0.1, size=2)    # small x-y jitter
        mujoco.mj_forward(mj_model, data)

        obs_buf = ObsBuffer(mj_model.actuator_ctrlrange.copy())
        obs_buf.reset()
        hidden = self.student.init_hidden(1, self.device)
        command = self.command

        # Accumulators.
        survived = True
        fall_step = _N_STEPS
        track_sum = 0.0
        height_sum = 0.0
        force_rms_sum = 0.0
        alive_cnt = 0

        frames = []
        renderer = None
        if capture_frames:
            renderer = mujoco.Renderer(mj_model, height=480, width=640)

        self.student.eval()
        with torch.no_grad():
            for t in range(_N_STEPS):
                gyro = data.sensordata[self._gyro_adr: self._gyro_adr + 3].copy()
                gravity = data.sensordata[self._upvector_adr: self._upvector_adr + 3].copy()
                joint_angles = data.qpos[7:19].copy()

                obs69 = obs_buf.build_obs_69dim(gyro, gravity, joint_angles, command)
                obs_t = torch.from_numpy(obs69).unsqueeze(0).to(self.device)  # (1, 69)
                action_t, hidden = self.student(obs_t, hidden)
                action = action_t.squeeze(0).cpu().numpy()

                motor_targets = obs_buf.update_after_action(action)
                data.ctrl[:] = motor_targets
                mujoco.mj_step(mj_model, data, nstep=_N_SUBSTEPS)

                # Capture frame before checking termination.
                if capture_frames and renderer is not None:
                    renderer.update_scene(data, camera="track")
                    frames.append(renderer.render().copy())

                # Termination: base tilt beyond threshold.
                gravity_z = data.sensordata[self._upvector_adr + 2]
                if gravity_z < _TERMINATION_GRAVITY_Z:
                    survived = False
                    fall_step = t + 1
                    break

                # Metrics.
                linvel_x = data.sensordata[self._local_linvel_adr]
                base_z = data.qpos[2]
                foot_forces = data.cfrc_ext[self._foot_body_ids, :3]   # (4, 3)
                foot_rms = float(np.sqrt(np.mean(np.sum(foot_forces ** 2, axis=-1))))

                track_sum += abs(linvel_x - 1.0)
                height_sum += abs(base_z - BASE_HEIGHT_TARGET)
                force_rms_sum += foot_rms
                alive_cnt += 1

        if renderer is not None:
            renderer.close()

        cnt = max(alive_cnt, 1)
        row = EpisodeRow(
            sim=self.name,
            seed=seed,
            survived=survived,
            tracking_error_mean=float(track_sum / cnt),
            base_height_dev_mean=float(height_sum / cnt),
            feet_force_rms=float(force_rms_sum / cnt),
            episode_seconds=float(fall_step * CTRL_DT),
        )
        return row, frames


def _write_video(frames: list, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=fps)
    print(f"[mujoco_cpu] Video saved: {path}  ({len(frames)} frames)")
