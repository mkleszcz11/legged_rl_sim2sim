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
from etils import epath

from mujoco_playground._src.locomotion.spot.base import get_assets
from mujoco_playground._src.locomotion.spot import spot_constants as consts

from adapters.base import BASE_HEIGHT_TARGET, CTRL_DT, ObsBuffer
from metrics import EpisodeRow

# ── Physics parameters ────────────────────────────────────────────────────────
_CPU_TIMESTEP = 5e-4          # 0.5 ms per physics step
_CPU_ITERATIONS = 4
_N_SUBSTEPS = round(CTRL_DT / _CPU_TIMESTEP)   # 40 substeps per ctrl step
_N_STEPS = round(30.0 / CTRL_DT)               # 1500 ctrl steps = 30 s

_TERMINATION_GRAVITY_Z = 0.85  # matches joystick._get_termination()

# Body names containing the sphere foot geoms.
_FOOT_BODIES = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]

# Default PD gains from joystick.py default_config (Kp=300, Kd=1.0).
_KP = 300.0
_KD = 1.0


class MuJoCoCPUAdapter:
    name = "mujoco_cpu"

    def __init__(self, student: torch.nn.Module, command: tuple[float, float, float], device: str):
        self.student = student
        self.command = np.array(command, dtype=np.float32)
        self.device = device

        # Load via from_xml_string + assets dict, the same way SpotEnv.__init__ does.
        # from_xml_path() fails because the MJCF meshdir is a relative path that
        # points into mujoco_menagerie which mujoco_playground manages separately.
        xml_path = consts.FEET_ONLY_ROUGH_TERRAIN_XML
        xml_str = epath.Path(xml_path).read_text()
        assets = get_assets()

        print(f"[mujoco_cpu] Loading model from {Path(str(xml_path)).name} …")
        self._mj_model = mujoco.MjModel.from_xml_string(xml_str, assets=assets)

        # Match the PD controller gains used during training.
        self._mj_model.actuator_gainprm[:, 0] = _KP
        self._mj_model.actuator_biasprm[:, 1] = -_KP
        self._mj_model.dof_damping[6:] = _KD

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
        video_seeds: list[int],
        video_dir: Path | None,
    ) -> list[EpisodeRow]:
        if not seeds:
            return []

        video_seeds_set = set(video_seeds)
        rows = []
        for i, seed in enumerate(seeds):
            video_path = None
            if seed in video_seeds_set and video_dir is not None:
                video_path = video_dir / f"{self.name}_{seed}.mp4"
                if video_path.exists():
                    video_path = None  # already recorded

            print(f"[mujoco_cpu] Episode {i+1}/{len(seeds)} (seed={seed})", end="", flush=True)
            row, frames = self._run_one_episode(seed, capture_frames=video_path is not None)
            rows.append(row)
            if row.survived:
                print(f"  ok  track={row.tracking_error_mean:.3f} m/s")
            else:
                print(f"  FELL at step {row.fall_timestep}/{_N_STEPS}  "
                      f"track={row.tracking_error_mean:.3f} m/s")

            if video_path is not None and frames:
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
        fall_timestep = -1
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

                obs69 = obs_buf.build_obs69(gyro, gravity, joint_angles, command)
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
                    fall_timestep = t
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
            fall_timestep=fall_timestep,
        )
        return row, frames


    def record_scenario_video(
        self,
        command: np.ndarray,
        video_path: Path,
        seed: int,
    ) -> None:
        """Run one episode with `command` and save a video to `video_path`."""
        if video_path.exists():
            print(f"[mujoco_cpu] Perspective video already exists: {video_path.name}")
            return
        orig_command = self.command
        self.command = np.asarray(command, dtype=np.float32)
        try:
            _, frames = self._run_one_episode(seed, capture_frames=True)
        finally:
            self.command = orig_command
        if frames:
            _write_video(frames, video_path, fps=round(1.0 / CTRL_DT))


def _write_video(frames: list, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=fps)
    print(f"[mujoco_cpu] Video saved: {path}  ({len(frames)} frames)")
