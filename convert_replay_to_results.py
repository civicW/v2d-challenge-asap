#!/usr/bin/env python3
"""Convert a record_replay.py npz into a kit results parquet for pack_submission.py.

- downsamples sim-rate (100 Hz) frames onto the tier_1 reference frame grid
- removes the rebase offset applied before retargeting:
      P_tier1 = P_sim - R_total @ F @ delta_source
  (translation only; orientations pass through)
- determines the recorded quaternion order empirically against the reference
  at frame 0 (the sim snaps the object onto the reference before stepping)
- writes the kit's legacy trajectory format with episode_index metadata
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NPZ = Path(sys.argv[1])
T_TOTAL_NPY = Path(sys.argv[2])
DELTA_SRC_NPY = Path(sys.argv[3])
TIER_PARQUET = Path(sys.argv[4])
OUT = Path(sys.argv[5])
EPISODE = int(sys.argv[6])

import hashlib


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


F = np.diag([1.0, -1.0, -1.0, 1.0])

z = np.load(NPZ)
achieved = z["achieved_pose_w"][:, 0, :]  # (T_sim, 7)
ref_pos = z["reference_pos_w"][:, 0, :]
ref_quat = z["reference_quat_w"][:, 0, :]  # wxyz (from motion parquet convention)

T_sim = achieved.shape[0]

tier = pq.read_table(TIER_PARQUET).to_pandas()
frame_index = tier["frame_index"].to_numpy(dtype=np.int64)
T_ref = len(frame_index)

# --- quaternion order of the recorded body_q -------------------------------
q_rec = achieved[0, 3:]
q_ref = ref_quat[0]
def ang(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))
# hypothesis A: recorded is wxyz like the reference
err_wxyz = ang(q_rec, q_ref)
# hypothesis B: recorded is xyzw
err_xyzw = ang(q_rec[[3, 0, 1, 2]], q_ref)
if err_wxyz <= err_xyzw:
    order = "wxyz"
    quat_xyzw = np.concatenate([achieved[:, 4:], achieved[:, 3:4]], axis=1)
else:
    order = "xyzw"
    quat_xyzw = achieved[:, 3:]
print(f"recorded quat order = {order} (frame0 err wxyz={err_wxyz:.4f} xyzw={err_xyzw:.4f})")

# --- downsample onto the reference grid ------------------------------------
idx = np.clip(np.round(frame_index * (T_sim / T_ref)).astype(np.int64), 0, T_sim - 1)
pos_sim = achieved[idx, :3]
quat_sim_d = quat_xyzw[idx]

# --- remove the rebase offset ----------------------------------------------
T_total = np.load(T_TOTAL_NPY)
delta_src = np.load(DELTA_SRC_NPY)
delta_cv = F[:3, :3] @ delta_src
offset_robot = T_total[:3, :3] @ delta_cv
pos_tier1 = pos_sim - offset_robot
print(f"rebase offset (robot world) = {offset_robot.round(3)}")

# --- sanity: frame 0 should land near the tier_1 reference pose ------------
obj7 = np.stack(tier["observation.object.pose"].to_numpy())
d0 = np.linalg.norm(pos_tier1[0] - obj7[0, :3])
print(f"frame0 mapped-vs-tier1 distance = {d0:.3f} m (lift/physics delta expected)")

# --- write flash_chord_object_trajectories_v1 results parquet -------------
n = T_ref
step = np.arange(n, dtype=np.int64)
time_s = frame_index / 30.0
zeros = np.zeros(n, dtype=np.int64)
ones = np.ones(n, dtype=bool)
table = pa.table(
    {
        "environment_id": pa.array(zeros, pa.int64()),
        "step": pa.array(step, pa.int64()),
        "reference_frame": pa.array(frame_index, pa.int64()),
        "time_s": pa.array(time_s, pa.float64()),
        "object_id": pa.array(zeros, pa.int64()),
        "object_body_id": pa.array(zeros, pa.int64()),
        "object_body_name": pa.array(["tall_bar_stool"] * n),
        "environment_valid": pa.array(ones),
        "achieved_position_x": pa.array(pos_tier1[:, 0], pa.float64()),
        "achieved_position_y": pa.array(pos_tier1[:, 1], pa.float64()),
        "achieved_position_z": pa.array(pos_tier1[:, 2], pa.float64()),
        "achieved_quaternion_x": pa.array(quat_sim_d[:, 0], pa.float64()),
        "achieved_quaternion_y": pa.array(quat_sim_d[:, 1], pa.float64()),
        "achieved_quaternion_z": pa.array(quat_sim_d[:, 2], pa.float64()),
        "achieved_quaternion_w": pa.array(quat_sim_d[:, 3], pa.float64()),
        "reference_position_x": pa.array(obj7[:, 0], pa.float64()),
        "reference_position_y": pa.array(obj7[:, 1], pa.float64()),
        "reference_position_z": pa.array(obj7[:, 2], pa.float64()),
        "reference_quaternion_x": pa.array(obj7[:, 4], pa.float64()),
        "reference_quaternion_y": pa.array(obj7[:, 5], pa.float64()),
        "reference_quaternion_z": pa.array(obj7[:, 6], pa.float64()),
        "reference_quaternion_w": pa.array(obj7[:, 3], pa.float64()),
    },
    metadata={
        b"flash_chord.schema": b"flash_chord_object_trajectories_v1",
        b"flash_chord.episode_index": str(EPISODE).encode(),
        b"flash_chord.world_count": b"1",
        b"flash_chord.quaternion_order": b"xyzw",
        b"flash_chord.position_unit": b"meter",
        b"v2d.baseline": b"flash-chord replay voc_scale=0 (physics)",
        b"flash_chord.eval_script_sha256": _sha256(Path(sys.argv[7])),
        b"flash_chord.eval_code_sha256": _sha256(Path(sys.argv[7])),
        b"flash_chord.checkpoint_sha256": _sha256(NPZ),
    },
)
OUT.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(table, OUT)
print(f"wrote {OUT} ({n} rows, xyzw)")
