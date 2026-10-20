# SPDX-FileCopyrightText: derived from flash_chord's replay.py (Apache-2.0)
"""Headless replay + object-pose recording for the V2D Track 2 submission.

Runs the same scene/replay stack as scripts/debug/replay.py but with
voc_scale=0 (NO virtual object control: the object evolves under physics and
real hand contact only), stepping every reference frame and recording the
simulated object pose per frame. Saves an npz with achieved + reference poses
for pack-time conversion back to tier_1 coordinates.

Usage (inside the flash_chord container):
    python /tmp/record_replay.py \
        "task.parquet='/v2d/assets/human_motion_data/whole_body/tier1_r2/.../data.parquet'" \
        out=/workspace/outputs/replay_ep0000_r2.npz
"""

from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene


@hydra.main(version_base="1.3", config_path="/workspace/src/flash_chord/configs", config_name="replay")
def main(cfg: DictConfig) -> None:
    voc_scale = float(cfg.replay.voc_scale)
    assert voc_scale == 0.0, (
        f"refusing to record with voc_scale={voc_scale}: nonzero VOC drags the object "
        "along the reference kinematically, which reproduces the input trajectory "
        "(the thing the challenge explicitly disqualifies)"
    )
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    replay_cfg = instantiate_typed(cfg.replay, ReplayConfig)
    setup = setup_scene(
        parquet=cfg.task.parquet,
        control_fps=replay_cfg.sim.fps,
        motion_speed=float(cfg.task.get("motion_speed", 1.0)),
        embodiment=embodiment,
        collision=collision,
        world_count=int(cfg.scene.world_count),
        include_support=bool(cfg.scene.support),
        decompose_objects=bool(cfg.scene.decompose_objects),
        object_scale_min=float(cfg.scene.object_scale_min),
        object_scale_max=float(cfg.scene.object_scale_max),
        object_scale_seed=int(cfg.scene.object_scale_seed),
        contact_friction=float(cfg.scene.get("contact_friction", 1.0)),
        object_free_joint_damping=float(cfg.scene.get("object_free_joint_damping", 0.0)),
    )
    reference = setup.reference
    scene = setup.scene
    runner = ReplayRunner(scene, reference, replay_cfg, device=cfg.device)
    runner.reset(0)

    ids = np.asarray(runner.command.body_ids_w.numpy())
    world_ids = ids.reshape(1, -1)[0] if ids.ndim == 1 else ids[0]  # world 0
    T = int(reference.num_frames)
    n_bodies = len(world_ids)
    achieved = np.zeros((T, n_bodies, 7), dtype=np.float64)
    ref_pos = np.zeros((T, n_bodies, 3), dtype=np.float64)
    ref_quat = np.zeros((T, n_bodies, 4), dtype=np.float64)
    print(f"recording {T} frames, {n_bodies} object bodies, voc_scale=0", flush=True)

    for frame in range(T):
        runner.step(frame)
        body_q = runner.state_0.body_q.numpy()
        achieved[frame] = body_q[world_ids]
        ref_pos[frame] = runner.command.body_target_pos_w.numpy()[0]
        ref_quat[frame] = runner.command.body_target_quat_w.numpy()[0]
        if frame % 100 == 0:
            err = np.linalg.norm(achieved[frame, :, :3] - ref_pos[frame], axis=-1)
            print(f"frame {frame}: obj pos err vs reference = {err.tolist()}", flush=True)

    out = str(cfg.get("out", "/workspace/outputs/replay_record.npz"))
    np.savez_compressed(
        out,
        achieved_pose_w=achieved,
        reference_pos_w=ref_pos,
        reference_quat_w=ref_quat,
        body_ids=world_ids,
        voc_scale=np.float64(voc_scale),
    )
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
