"""MJX-JAX backend: runs all episodes in one vmap'd batch (100 envs at once).

Uses the Brax/MJX raw env from mujoco_playground, which is identical to the
training environment.  Obs construction is handled by the env; we only need to
extract the 69-dim proprio from the 81-dim state obs.

Video recording is done inline during the batch loop via mujoco.Renderer (CPU
offscreen), avoiding a second jax.jit(env.step) compilation.  Two separate
compiled Warp step functions on 4 GB VRAM would OOM.
"""

from pathlib import Path

import jax
import jax.numpy as jp
import mediapy
import mujoco
import numpy as np
import torch

from mujoco_playground import registry
from mujoco_playground._src.wrapper_torch import _jax_to_torch, _torch_to_jax

from load_student import ENV_NAME, extract_proprio
from adapters.base import CTRL_DT, BASE_HEIGHT_TARGET
from metrics import EpisodeRow


_N_STEPS = round(30.0 / CTRL_DT)   # 1500 ctrl steps = 30 s
_VIDEO_FPS = round(1.0 / CTRL_DT)  # 50 fps

# Lower-leg body names that hold the sphere foot geoms.
_FOOT_BODIES = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]


class MJXAdapter:
    name = "mjx"

    def __init__(self, student: torch.nn.Module, command: tuple[float, float, float], device: str):
        self.student = student
        self.command = np.array(command, dtype=np.float32)
        self.device = device

        print(f"[mjx] Loading env {ENV_NAME} …")
        self.env = registry.load(ENV_NAME)
        self.vmap_reset = jax.jit(jax.vmap(self.env.reset))
        self.vmap_step = jax.jit(jax.vmap(self.env.step))

        mj_model = self.env.mj_model
        self._foot_body_ids = np.array(
            [mj_model.body(name).id for name in _FOOT_BODIES]
        )
        print(f"[mjx] Ready.  n_steps={_N_STEPS}  foot_body_ids={self._foot_body_ids}")

    def run_episodes(
        self,
        seeds: list[int],
        video_seeds: list[int],
        video_dir: Path | None,
    ) -> list[EpisodeRow]:
        if not seeds:
            return []

        print(f"[mjx] Running {len(seeds)} episodes (vmap batch) …")
        return self._run_batch(seeds, video_seeds, video_dir)

    # ------------------------------------------------------------------

    def _run_batch(
        self,
        seeds: list[int],
        video_seeds: list[int],
        video_dir: Path | None,
    ) -> list[EpisodeRow]:
        n = len(seeds)
        cmd_jax = jp.array(self.command)
        cmd_batch = jp.tile(cmd_jax, (n, 1))

        keys = jp.stack([jax.random.PRNGKey(s) for s in seeds])
        states = self.vmap_reset(keys)
        states = states.replace(info={**states.info, "command": cmd_batch})

        hidden = self.student.init_hidden(n, self.device)

        # Per-episode accumulators (CPU side).
        alive = np.ones(n, dtype=bool)
        ever_fallen = np.zeros(n, dtype=bool)
        fall_step = np.full(n, _N_STEPS, dtype=int)
        track_sum = np.zeros(n, dtype=np.float64)
        height_sum = np.zeros(n, dtype=np.float64)
        force_rms_sum = np.zeros(n, dtype=np.float64)
        alive_cnt = np.zeros(n, dtype=np.int64)

        # Buffer raw state per video seed during the loop; render after to avoid
        # holding EGL/OpenGL GPU contexts alongside the vmap batch (4 GB VRAM).
        recorders = []
        if video_dir is not None:
            for vs in video_seeds:
                if vs not in seeds:
                    continue
                path = video_dir / f"{self.name}_{vs}.mp4"
                if path.exists():
                    continue
                recorders.append({
                    "idx": seeds.index(vs),
                    "path": path,
                    "qpos_buf": [],
                    "qvel_buf": [],
                    "ctrl_buf": [],
                })

        self.student.eval()
        with torch.no_grad():
            for t in range(_N_STEPS):
                obs81 = _jax_to_torch(states.obs["state"])        # (n, 81)
                proprio = extract_proprio(obs81)                   # (n, 69)
                actions, hidden = self.student(proprio, hidden)
                actions_jax = jp.clip(_torch_to_jax(actions.detach()), -1.0, 1.0)

                states = self.vmap_step(states, actions_jax)
                states = states.replace(info={**states.info, "command": cmd_batch})

                # Buffer state snapshots for deferred video rendering.
                for rec in recorders:
                    idx = rec["idx"]
                    rec["qpos_buf"].append(np.array(states.data.qpos[idx]))
                    rec["qvel_buf"].append(np.array(states.data.qvel[idx]))
                    rec["ctrl_buf"].append(np.array(states.data.ctrl[idx]))

                # Reset hidden for fallen envs.
                done_t = np.array(_jax_to_torch(states.done).cpu())  # (n,)
                hidden = hidden * (1.0 - torch.tensor(done_t, device=self.device).view(1, -1, 1))

                just_fell = done_t > 0.5
                fall_step[just_fell & ~ever_fallen] = t + 1
                ever_fallen |= just_fell

                # Accumulate metrics only while alive.
                mask = alive
                if mask.any():
                    linvel_x = np.array(_jax_to_torch(states.obs["privileged_state"][:, 90]).cpu())
                    base_z = np.array(_jax_to_torch(states.data.qpos[:, 2]).cpu())
                    foot_forces = np.array(
                        _jax_to_torch(states.data.cfrc_ext[:, self._foot_body_ids, :3]).cpu()
                    )  # (n, 4, 3)
                    foot_rms_per_env = np.sqrt(
                        np.mean(np.sum(foot_forces ** 2, axis=-1), axis=-1)
                    )  # (n,)

                    track_sum[mask] += np.abs(linvel_x[mask] - 1.0)
                    height_sum[mask] += np.abs(base_z[mask] - BASE_HEIGHT_TARGET)
                    force_rms_sum[mask] += foot_rms_per_env[mask]
                    alive_cnt[mask] += 1

                alive = alive & ~just_fell

        # Render videos now that the vmap batch (JAX GPU state) is done.
        # Opening EGL renderers after freeing the batch avoids VRAM exhaustion.
        if recorders:
            del states
            mj_model = self.env.mj_model
            for rec in recorders:
                renderer = mujoco.Renderer(mj_model, height=480, width=640)
                md = mujoco.MjData(mj_model)
                frames = []
                for qpos, qvel, ctrl in zip(rec["qpos_buf"], rec["qvel_buf"], rec["ctrl_buf"]):
                    md.qpos[:] = qpos
                    md.qvel[:] = qvel
                    md.ctrl[:] = ctrl
                    mujoco.mj_forward(mj_model, md)
                    renderer.update_scene(md, camera="track")
                    frames.append(renderer.render().copy())
                renderer.close()
                rec["path"].parent.mkdir(parents=True, exist_ok=True)
                mediapy.write_video(str(rec["path"]), frames, fps=_VIDEO_FPS)
                print(f"[mjx] Video saved: {rec['path']}  ({len(frames)} frames)")

        rows = []
        for i, seed in enumerate(seeds):
            cnt = max(alive_cnt[i], 1)
            rows.append(EpisodeRow(
                sim=self.name,
                seed=seed,
                survived=not ever_fallen[i],
                tracking_error_mean=float(track_sum[i] / cnt),
                base_height_dev_mean=float(height_sum[i] / cnt),
                feet_force_rms=float(force_rms_sum[i] / cnt),
                episode_seconds=float(fall_step[i] * CTRL_DT),
                fall_timestep=int(fall_step[i] - 1) if ever_fallen[i] else -1,
            ))
        return rows

    # ------------------------------------------------------------------

    def record_scenario_video(
        self,
        command: np.ndarray,
        video_path: Path,
        seed: int,
    ) -> None:
        """Run one episode with `command` and save a video to `video_path`."""
        if video_path.exists():
            print(f"[mjx] Perspective video already exists: {video_path.name}")
            return

        cmd_jax = jp.array(command)
        cmd_batch = jp.tile(cmd_jax, (1, 1))   # (1, 3)

        key = jp.stack([jax.random.PRNGKey(seed)])
        states = self.vmap_reset(key)
        states = states.replace(info={**states.info, "command": cmd_batch})

        hidden = self.student.init_hidden(1, self.device)
        qpos_buf: list = []
        qvel_buf: list = []
        ctrl_buf: list = []

        self.student.eval()
        with torch.no_grad():
            for _ in range(_N_STEPS):
                obs81 = _jax_to_torch(states.obs["state"])
                proprio = extract_proprio(obs81)
                actions, hidden = self.student(proprio, hidden)
                actions_jax = jp.clip(_torch_to_jax(actions.detach()), -1.0, 1.0)
                states = self.vmap_step(states, actions_jax)
                states = states.replace(info={**states.info, "command": cmd_batch})
                qpos_buf.append(np.array(states.data.qpos[0]))
                qvel_buf.append(np.array(states.data.qvel[0]))
                ctrl_buf.append(np.array(states.data.ctrl[0]))

        del states
        mj_model = self.env.mj_model
        renderer = mujoco.Renderer(mj_model, height=480, width=640)
        md = mujoco.MjData(mj_model)
        frames = []
        for qpos, qvel, ctrl in zip(qpos_buf, qvel_buf, ctrl_buf):
            md.qpos[:] = qpos
            md.qvel[:] = qvel
            md.ctrl[:] = ctrl
            mujoco.mj_forward(mj_model, md)
            renderer.update_scene(md, camera="track")
            frames.append(renderer.render().copy())
        renderer.close()

        video_path.parent.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(str(video_path), frames, fps=_VIDEO_FPS)
        print(f"[mjx] Perspective video: {video_path.name}  ({len(frames)} frames)")
