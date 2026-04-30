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
        video_seed: int | None,
        video_path: Path | None,
    ) -> list[EpisodeRow]:
        if not seeds:
            return []

        want_video = (video_seed is not None and video_path is not None
                      and not video_path.exists() and video_seed in seeds)
        print(f"[mjx] Running {len(seeds)} episodes (vmap batch) …")
        return self._run_batch(seeds, video_seed if want_video else None, video_path)

    # ------------------------------------------------------------------

    def _run_batch(
        self,
        seeds: list[int],
        video_seed: int | None,
        video_path: Path | None,
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

        # Optional inline video capture — uses mujoco.Renderer (CPU/OpenGL) so
        # we never need a second JAX/Warp step compilation.
        renderer = video_idx = mj_data = frames = None
        if video_seed is not None:
            video_idx = seeds.index(video_seed)
            mj_model = self.env.mj_model
            mj_data = mujoco.MjData(mj_model)
            renderer = mujoco.Renderer(mj_model, height=480, width=640)
            frames = []

        self.student.eval()
        with torch.no_grad():
            for t in range(_N_STEPS):
                obs81 = _jax_to_torch(states.obs["state"])        # (n, 81)
                proprio = extract_proprio(obs81)                   # (n, 69)
                actions, hidden = self.student(proprio, hidden)
                actions_jax = jp.clip(_torch_to_jax(actions.detach()), -1.0, 1.0)

                states = self.vmap_step(states, actions_jax)
                states = states.replace(info={**states.info, "command": cmd_batch})

                # Capture video frame before metrics (so we get the post-step pose).
                if renderer is not None:
                    mj_data.qpos[:] = np.array(states.data.qpos[video_idx])
                    mj_data.qvel[:] = np.array(states.data.qvel[video_idx])
                    mj_data.ctrl[:] = np.array(states.data.ctrl[video_idx])
                    mujoco.mj_forward(self.env.mj_model, mj_data)
                    renderer.update_scene(mj_data, camera="track")
                    frames.append(renderer.render().copy())

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

        if renderer is not None:
            renderer.close()
            video_path.parent.mkdir(parents=True, exist_ok=True)
            mediapy.write_video(str(video_path), frames, fps=_VIDEO_FPS)
            print(f"[mjx] Video saved: {video_path}  ({len(frames)} frames)")

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
