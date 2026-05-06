"""MJX-JAX backend for M4 teacher evaluation.

Runs all episodes in one vmap'd batch.  The env is passed in at construction
time, which decouples this adapter from the choice of environment:
  - Pass a plain SpotJoystickRoughTerrain env for the baseline.
  - Pass a SpotJoystickResidualEnv for the residual-injected case.

Video recording is deferred until after the batch loop (same approach as m5)
to avoid VRAM contention between the Warp/XLA compiled step and EGL renderers.

Teacher inference is stateless (no GRU), so there is no hidden state to carry:
    actions, _ = inference_fn(states.obs, rng_key)
The policy is loaded with deterministic=True, so rng_key is ignored.
"""

from pathlib import Path

import jax
import jax.numpy as jp
import mediapy
import mujoco
import numpy as np

from adapters.base import CTRL_DT, BASE_HEIGHT_TARGET
from metrics import EpisodeRow

_N_STEPS = round(30.0 / CTRL_DT)   # 1500 ctrl steps = 30 s
_VIDEO_FPS = round(1.0 / CTRL_DT)  # 50 fps

_FOOT_BODIES = ["fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"]
_RNG_KEY = jax.random.PRNGKey(0)   # deterministic policy — key is ignored


class MJXTeacherAdapter:
    """Evaluates the JAX teacher policy in an MJX environment via vmap."""

    def __init__(
        self,
        inference_fn,
        env,
        sim_name: str,
        command: tuple[float, float, float],
    ):
        self.name = sim_name
        self._inference_fn = inference_fn
        self._env = env
        self.command = np.array(command, dtype=np.float32)

        self._vmap_reset = jax.jit(jax.vmap(env.reset))
        self._vmap_step  = jax.jit(jax.vmap(env.step))

        mj_model = env.mj_model
        self._foot_body_ids = np.array(
            [mj_model.body(name).id for name in _FOOT_BODIES]
        )
        print(f"[{sim_name}] Ready.  n_steps={_N_STEPS}  foot_body_ids={self._foot_body_ids}")

    def run_episodes(
        self,
        seeds: list[int],
        video_seeds: list[int],
        video_dir: Path | None,
    ) -> list[EpisodeRow]:
        if not seeds:
            return []
        print(f"[{self.name}] Running {len(seeds)} episodes (vmap batch) …")
        return self._run_batch(seeds, video_seeds, video_dir)

    # ------------------------------------------------------------------

    def _run_batch(
        self,
        seeds: list[int],
        video_seeds: list[int],
        video_dir: Path | None,
    ) -> list[EpisodeRow]:
        n = len(seeds)
        cmd_batch = jp.tile(jp.array(self.command), (n, 1))
        keys      = jp.stack([jax.random.PRNGKey(s) for s in seeds])

        states = self._vmap_reset(keys)
        states = states.replace(info={**states.info, "command": cmd_batch})

        # Per-episode accumulators (CPU).
        alive        = np.ones(n, dtype=bool)
        ever_fallen  = np.zeros(n, dtype=bool)
        fall_step    = np.full(n, _N_STEPS, dtype=int)
        track_sum    = np.zeros(n, dtype=np.float64)
        height_sum   = np.zeros(n, dtype=np.float64)
        force_rms_sum = np.zeros(n, dtype=np.float64)
        alive_cnt    = np.zeros(n, dtype=np.int64)

        # Build recorder entries for any video seeds that haven't been recorded yet.
        recorders = []
        if video_dir is not None:
            for vs in video_seeds:
                if vs not in seeds:
                    continue
                path = video_dir / f"{self.name}_{vs}.mp4"
                if path.exists():
                    continue
                recorders.append({
                    "idx":      seeds.index(vs),
                    "path":     path,
                    "qpos_buf": [],
                    "qvel_buf": [],
                    "ctrl_buf": [],
                })

        for t in range(_N_STEPS):
            actions, _ = self._inference_fn(states.obs, _RNG_KEY)
            actions     = jp.clip(actions, -1.0, 1.0)

            states = self._vmap_step(states, actions)
            states = states.replace(info={**states.info, "command": cmd_batch})

            # Buffer state snapshots for deferred video rendering.
            for rec in recorders:
                idx = rec["idx"]
                rec["qpos_buf"].append(np.array(states.data.qpos[idx]))
                rec["qvel_buf"].append(np.array(states.data.qvel[idx]))
                rec["ctrl_buf"].append(np.array(states.data.ctrl[idx]))

            done_t    = np.array(states.done)   # (n,)
            just_fell = done_t > 0.5
            fall_step[just_fell & ~ever_fallen] = t + 1
            ever_fallen |= just_fell

            if alive.any():
                # privileged_state[90] = local_linvel[0] = forward velocity.
                linvel_x  = np.array(states.obs["privileged_state"][:, 90])
                base_z    = np.array(states.data.qpos[:, 2])
                foot_f    = np.array(states.data.cfrc_ext[:, self._foot_body_ids, :3])
                foot_rms  = np.sqrt(np.mean(np.sum(foot_f ** 2, axis=-1), axis=-1))

                track_sum[alive]     += np.abs(linvel_x[alive] - 1.0)
                height_sum[alive]    += np.abs(base_z[alive] - BASE_HEIGHT_TARGET)
                force_rms_sum[alive] += foot_rms[alive]
                alive_cnt[alive]     += 1

            alive = alive & ~just_fell

        # Render videos after the batch to avoid holding VRAM alongside EGL.
        if recorders:
            del states
            mj_model = self._env.mj_model
            for rec in recorders:
                _render_video(mj_model, rec)
                print(f"[{self.name}] Video saved: {rec['path']}  "
                      f"({len(rec['qpos_buf'])} frames)")

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


def _render_video(mj_model, rec: dict) -> None:
    """Replay buffered qpos/qvel/ctrl through a CPU renderer and write MP4."""
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
