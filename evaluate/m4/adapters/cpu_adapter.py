"""MuJoCo CPU backend for M4 teacher evaluation.

Runs episodes sequentially with vanilla mujoco bindings.  Physics settings
match m5/mujoco_cpu_adapter (dt=0.5ms, 4 solver iterations) — this is the
intentionally divergent config that quantifies the sim-to-sim gap.

Teacher observation (81-dim) is reconstructed from CPU state:
  gyro, gravity, joint_angles, qpos_error_history, feet_pos, last_act, command.
feet_pos comes from the body-relative position sensors (FL_pos … HR_pos).

Residual correction (optional):
  tau_delta (N·m) ← ResidualFeatureBuffer → normalise → PyTorch MLP → denormalise.
  Applied as a position target offset:  delta_q = tau_delta / KP.
  The correction is computed once per control step (50 Hz) from the state at
  the start of the step — a simplification vs. per-substep MJX injection, but
  appropriate for this evaluation since R²≈0.62 already limits precision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import mujoco
except ImportError as e:
    raise ImportError("mujoco CPU bindings not found.  pip install mujoco") from e

import jax
import jax.numpy as jp
import mediapy
import numpy as np
import torch
from etils import epath

from mujoco_playground._src.locomotion.spot.base import get_assets
from mujoco_playground._src.locomotion.spot import spot_constants as consts

from adapters.base import (
    BASE_HEIGHT_TARGET, CTRL_DT, KP,
    FEET_POS_SENSORS,
    TeacherObsBuffer, ResidualFeatureBuffer,
)
from metrics import EpisodeRow

_CPU_TIMESTEP   = 5e-4
_CPU_ITERATIONS = 4
_N_SUBSTEPS = round(CTRL_DT / _CPU_TIMESTEP)   # 40 substeps per ctrl step
_N_STEPS    = round(30.0 / CTRL_DT)             # 1500 ctrl steps = 30 s
_TERMINATION_GRAVITY_Z = 0.85

_KP = 300.0
_KD = 1.0
_FOOT_BODIES = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]

_RNG_KEY = jax.random.PRNGKey(0)   # deterministic policy — key is ignored


class CPUTeacherAdapter:
    """Evaluates the JAX teacher policy in vanilla MuJoCo CPU, episode by episode."""

    def __init__(
        self,
        inference_fn,
        sim_name: str,
        command: tuple[float, float, float],
        residual_model: Optional[torch.nn.Module] = None,
        residual_norm: Optional[dict] = None,   # keys: in_mean, in_std, out_mean, out_std
        device: str = "cpu",
    ):
        self.name = sim_name
        self._inference_fn = inference_fn
        self.command = np.array(command, dtype=np.float32)
        self._device = device
        self._residual_model = residual_model
        self._residual_norm  = residual_norm

        xml_path = consts.FEET_ONLY_ROUGH_TERRAIN_XML
        xml_str  = epath.Path(xml_path).read_text()
        assets   = get_assets()

        print(f"[{sim_name}] Loading model from {Path(str(xml_path)).name} …")
        self._mj_model = mujoco.MjModel.from_xml_string(xml_str, assets=assets)

        # Match PD gains and physics divergence (same as m5 CPU adapter).
        self._mj_model.actuator_gainprm[:, 0] = _KP
        self._mj_model.actuator_biasprm[:, 1] = -_KP
        self._mj_model.dof_damping[6:]         = _KD
        self._mj_model.opt.timestep            = _CPU_TIMESTEP
        self._mj_model.opt.iterations          = _CPU_ITERATIONS

        # Cache sensor addresses for fast per-step lookup.
        def _adr(name: str) -> int:
            return int(self._mj_model.sensor_adr[self._mj_model.sensor(name).id])

        self._gyro_adr         = _adr("gyro")
        self._upvector_adr     = _adr("upvector")
        self._local_linvel_adr = _adr("local_linvel")
        self._feet_pos_adrs    = [_adr(s) for s in FEET_POS_SENSORS]

        self._foot_body_ids = np.array(
            [self._mj_model.body(name).id for name in _FOOT_BODIES]
        )
        self._home_key_id = self._mj_model.keyframe("home").id

        print(
            f"[{sim_name}] Ready.  n_steps={_N_STEPS}  n_substeps={_N_SUBSTEPS}"
            f"  dt={_CPU_TIMESTEP*1000:.1f}ms  iters={_CPU_ITERATIONS}"
            + ("  +residual" if residual_model is not None else "")
        )

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
                p = video_dir / f"{self.name}_{seed}.mp4"
                video_path = None if p.exists() else p

            print(f"[{self.name}] Episode {i+1}/{len(seeds)} (seed={seed})", end="", flush=True)
            row, frames = self._run_one_episode(seed, capture_frames=video_path is not None)
            rows.append(row)

            if row.survived:
                print(f"  ok  track={row.tracking_error_mean:.3f} m/s")
            else:
                print(f"  FELL at step {row.fall_timestep}/{_N_STEPS}  "
                      f"track={row.tracking_error_mean:.3f} m/s")

            if video_path is not None and frames:
                _write_video(frames, video_path)

        return rows

    # ------------------------------------------------------------------

    def _run_one_episode(self, seed: int, capture_frames: bool) -> tuple[EpisodeRow, list]:
        mj_model = self._mj_model
        data = mujoco.MjData(mj_model)

        mujoco.mj_resetDataKeyframe(mj_model, data, self._home_key_id)
        rng = np.random.default_rng(seed)
        data.qpos[:2] += rng.uniform(-0.1, 0.1, size=2)
        mujoco.mj_forward(mj_model, data)

        obs_buf = TeacherObsBuffer(mj_model.actuator_ctrlrange.copy())
        obs_buf.reset()
        res_buf = ResidualFeatureBuffer() if self._residual_model is not None else None

        survived      = True
        fall_step     = _N_STEPS
        fall_timestep = -1
        track_sum     = 0.0
        height_sum    = 0.0
        force_rms_sum = 0.0
        alive_cnt     = 0

        renderer = mujoco.Renderer(mj_model, height=480, width=640) if capture_frames else None
        frames: list = []

        for t in range(_N_STEPS):
            # --- Build 81-dim teacher observation ---
            gyro    = data.sensordata[self._gyro_adr     : self._gyro_adr + 3].copy()
            gravity = data.sensordata[self._upvector_adr : self._upvector_adr + 3].copy()
            joints  = data.qpos[7:19].copy()
            feet_pos = np.concatenate([
                data.sensordata[adr : adr + 3] for adr in self._feet_pos_adrs
            ])  # (12,) raveled 4×3

            obs81 = obs_buf.build_obs81(gyro, gravity, joints, feet_pos, self.command)

            # --- Teacher inference (JAX, deterministic) ---
            obs_dict = {"state": jp.array(obs81)[None]}   # (1, 81)
            actions, _ = self._inference_fn(obs_dict, _RNG_KEY)
            action = np.array(actions)[0]                 # (12,)

            motor_targets = obs_buf.update_after_action(action)

            # --- Residual correction (optional) ---
            if res_buf is not None:
                flat_hist = res_buf.update_and_get(joints, data.qvel[6:18].copy(), motor_targets)
                tau_delta = self._eval_residual(flat_hist)   # (12,) N·m
                delta_q   = tau_delta / KP
                motor_targets = np.clip(
                    motor_targets + delta_q,
                    mj_model.actuator_ctrlrange[:, 0],
                    mj_model.actuator_ctrlrange[:, 1],
                )

            data.ctrl[:] = motor_targets
            mujoco.mj_step(mj_model, data, nstep=_N_SUBSTEPS)

            if capture_frames and renderer is not None:
                renderer.update_scene(data, camera="track")
                frames.append(renderer.render().copy())

            # --- Termination check ---
            gravity_z = data.sensordata[self._upvector_adr + 2]
            if gravity_z < _TERMINATION_GRAVITY_Z:
                survived      = False
                fall_timestep = t
                fall_step     = t + 1
                break

            # --- Metric accumulation ---
            linvel_x   = float(data.sensordata[self._local_linvel_adr])
            base_z     = float(data.qpos[2])
            foot_forces = data.cfrc_ext[self._foot_body_ids, :3]
            foot_rms   = float(np.sqrt(np.mean(np.sum(foot_forces ** 2, axis=-1))))

            track_sum     += abs(linvel_x - 1.0)
            height_sum    += abs(base_z - BASE_HEIGHT_TARGET)
            force_rms_sum += foot_rms
            alive_cnt     += 1

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

    @torch.no_grad()
    def _eval_residual(self, flat_hist: np.ndarray) -> np.ndarray:
        norm = self._residual_norm
        flat_norm = (flat_hist - norm["in_mean"]) / np.maximum(norm["in_std"], 1e-8)
        x = torch.from_numpy(flat_norm[None]).to(self._device)   # (1, 144)
        tau_norm = self._residual_model(x).cpu().numpy()[0]       # (12,) normalised
        return tau_norm * np.maximum(norm["out_std"], 1e-8) + norm["out_mean"]


def _write_video(frames: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = round(1.0 / CTRL_DT)
    mediapy.write_video(str(path), frames, fps=fps)
    print(f"  Video saved: {path}  ({len(frames)} frames)")
