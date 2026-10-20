#!/usr/bin/env python3
"""V2D Challenge Track 2 Tier 1 — static-object baseline (no policy).

Reads the released Tier 1 multiview-caption parquets and writes one
flash_chord_object_trajectories_v1 trajectory parquet per episode, predicting
the object stays at its first visible released pose for the whole episode.

This is an honest baseline entry: there is no robot policy and no simulator
rollout. The `checkpoint` provenance per episode is the SHA-256 of a JSON
manifest that fully describes the constant-pose predictor for that episode.
The `eval_script_sha256`/`eval_code_sha256` provenance is the SHA-256 of this
file. See the companion README.md in this directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wxyz_to_xyzw(pose: np.ndarray) -> np.ndarray:
    """[x,y,z,qw,qx,qy,qz] -> [x,y,z,qx,qy,qz,qw]."""
    out = np.empty_like(pose)
    out[..., :3] = pose[..., :3]
    out[..., 3:6] = pose[..., 4:7]
    out[..., 6] = pose[..., 3]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="directory containing track_2/tier_1_multiview_caption")
    parser.add_argument("--tier-folder", default="tier_1_multiview_caption",
                        help="folder name under track_2/ holding the released input")
    parser.add_argument("--out", required=True, type=Path, help="results directory")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    tier_root = args.dataset_root / "track_2" / args.tier_folder
    self_sha = sha256_bytes(Path(__file__).read_bytes())
    args.out.mkdir(parents=True, exist_ok=True)

    parquets = sorted((tier_root / "data").rglob("episode_*.parquet"))
    if not parquets:
        raise SystemExit(f"no episode parquets under {tier_root / 'data'}")

    summary = []
    for parquet_path in parquets:
        episode = int(parquet_path.stem.split("_")[1])
        table = pq.read_table(parquet_path)
        rows = table.num_rows
        frame_index = np.asarray(
            table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
        flat = table["observation.object.pose"].combine_chunks().flatten().to_numpy()
        pose = wxyz_to_xyzw(np.asarray(flat, dtype=np.float64).reshape(rows, 7))
        visible = np.asarray(
            table["observation.object.visible"].to_numpy(zero_copy_only=False),
            dtype=bool).reshape(rows)
        keep = visible
        frames = frame_index[keep]
        reference = pose[keep]
        if frames.size == 0:
            raise SystemExit(f"episode {episode}: no visible object frames")

        static_pose = reference[0].copy()
        manifest = {
            "method": "static-object baseline (first visible released pose held constant)",
            "track": "track_2_tier1",
            "episode_index": episode,
            "source_parquet": str(parquet_path.relative_to(args.dataset_root)),
            "first_visible_frame": int(frames[0]),
            "predicted_pose_xyzw": [float(v) for v in static_pose],
            "generator": Path(__file__).name,
            "generator_sha256": self_sha,
        }
        checkpoint_sha = sha256_bytes(
            json.dumps(manifest, sort_keys=True).encode("utf-8"))

        steps = np.arange(frames.size, dtype=np.int32)
        n = frames.size
        data = {
            "environment_id": np.zeros(n, dtype=np.int32),
            "step": steps,
            "reference_frame": frames.astype(np.int32),
            "time_s": steps.astype(np.float64) / args.fps,
            "object_id": np.zeros(n, dtype=np.int32),
            "object_body_id": np.zeros(n, dtype=np.int32),
            "object_body_name": ["object"] * n,
            "environment_valid": np.ones(n, dtype=bool),
        }
        for axis, name in enumerate(("x", "y", "z")):
            data[f"achieved_position_{name}"] = np.full(n, static_pose[axis], dtype=np.float32)
        for axis, name in enumerate(("x", "y", "z", "w")):
            data[f"achieved_quaternion_{name}"] = np.full(n, static_pose[3 + axis], dtype=np.float32)
        for axis, name in enumerate(("x", "y", "z")):
            data[f"reference_position_{name}"] = reference[:, axis].astype(np.float32)
        for axis, name in enumerate(("x", "y", "z", "w")):
            data[f"reference_quaternion_{name}"] = reference[:, 3 + axis].astype(np.float32)

        out_table = pa.table(data)
        out_table = out_table.replace_schema_metadata({
            "flash_chord.schema": "flash_chord_object_trajectories_v1",
            "flash_chord.episode_index": str(episode),
            "flash_chord.world_count": "1",
            "flash_chord.quaternion_order": "xyzw",
            "flash_chord.position_unit": "meter",
            "flash_chord.eval_script_sha256": self_sha,
            "flash_chord.eval_code_sha256": self_sha,
            "flash_chord.checkpoint_sha256": checkpoint_sha,
            "v2d.baseline": "static-object (no policy, no simulator)",
        })
        out_path = args.out / f"episode_{episode:06d}.parquet"
        pq.write_table(out_table, out_path)
        summary.append((episode, int(frames.size), int(rows - frames.size)))

    for episode, kept, dropped in summary:
        print(f"episode {episode:06d}: {kept} visible frames "
              f"({dropped} dropped as invisible)")
    print(f"wrote {len(summary)} trajectory parquets to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
