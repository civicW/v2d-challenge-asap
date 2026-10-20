import json, sys
from pathlib import Path
sys.path.insert(0, "/big/v2d/v2d_submission_kit")
import numpy as np
import pandas as pd
from v2dlb import object_pose as op
from v2dlb import report_schema as rs
from v2dlb.trajectory import read_episode, episode_parquet
from v2dlb.submission_parquet import read_trajectory
from v2dlb.vertices import mesh_path_for, sample_vertices

root = Path("/big/v2d/v2d_challenge_data/track_2/tier_1_multiview_caption")
spec = json.load(open("/big/v2d/v2d_submission_kit/config/tracks.json"))["tracks"]["track_2_tier1"]
EP = 0
ep = read_episode(episode_parquet(root, EP), EP)

def rid(world, frame, slot):
    return f"track_2/e{EP:06d}/w{world:03d}/f{frame:06d}/o{slot}"

rows = []
for step, frame in enumerate(ep.frame_index):
    rows.append((rid(0, int(frame), 0), EP, 0, int(frame), 0, ep.object_names[0], *ep.pose_xyzw[step, 0]))
solution = pd.DataFrame(rows, columns=[
    rs.ROW_ID_COLUMN, "episode_index", "world_index", "frame_index",
    "object_slot", "object_name", *op._REF_COLUMNS])

vertices = sample_vertices(mesh_path_for(root / "mesh", ep.object_names[0]))
V = np.stack([vertices])

def score(parquet):
    traj = read_trajectory(Path(parquet))
    sub = pd.DataFrame({
        rs.ROW_ID_COLUMN: [rid(0, int(f), 0) for f in traj.frame_index],
        **{c: traj.pose_xyzw[:, i] for i, c in enumerate(op._POSE_COLUMNS)},
    })
    merged = solution.merge(sub, on=rs.ROW_ID_COLUMN, how="left")
    achieved, ref_tensor, ids, names = op._episode_tensors(merged)
    achieved = op._normalise_quaternions(achieved, "submission", ValueError)
    ref_tensor = op._normalise_quaternions(ref_tensor, "reference", ValueError)
    out = {}
    for m in spec["metrics"]:
        v = op.episode_metric(m, achieved, ref_tensor, ids, V,
                              align_initial_pose=bool(spec.get("align_initial_pose", False)))
        out[m] = round(float(v), 6) if not np.isnan(v) else None
    return out

print("SIM    ep0:", json.dumps(score("/big/v2d/results_ep0/episode_000000.parquet")))
print("STATIC ep0:", json.dumps(score("/big/v2d/results/episode_000000.parquet")))
