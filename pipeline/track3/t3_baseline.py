#!/usr/bin/env python3
"""Team ASAP Track 3 static baseline generator.

Produces, on the official sample's exact frame/object grid:
  results_t3/episode_XXXXXX.parquet  static object poses (pos=0, quat=identity)
  meshes_t3/episode_XXXXXX/<object>.glb  public meshes reused; coarse spoon fabricated

Provenance columns carry this script's real sha256 (the static generator IS the
producing policy). Improvement iterations replace the poses with tracked object
trajectories; meshes stay body-coordinate-compatible with the public URDFs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KIT = Path("/big/v2d/v2d_submission_kit")
OUT = Path("/big/v2d/t3_baseline")
SCRIPT = Path(__file__).resolve()

PROV_COLS = [
    "eval_script_sha256", "eval_code_sha256", "checkpoint_sha256",
    "packer_script_sha256", "uploader_script_sha256", "code_commit_url",
]

# object name -> public mesh file (visual variant where the asset is split)
MESH_MAP = {
    "white_pot": "white_pot/white_pot.glb",
    "white_pot_lid": "white_pot_lid/white_pot_lid.glb",
    "beige_cup": "beige_cup/beige_cup.glb",
    "blue_cup": "blue_cup/blue_cup.glb",
    "plastic_dish_rack": "plastic_dish_rack/plastic_dish_rack.glb",
    "mini_dust_pan": "mini_dust_pan/mini_dust_pan.glb",
    "mini_sweeper": "mini_sweeper/mini_sweeper_visual.glb",
    "water_pitcher": "water_pitcher/water_pitcher.glb",
    "wooden_piece_1": "wooden_piece_1/wooden_piece_1_visual.glb",
    "wooden_piece_2": "wooden_piece_2/wooden_piece_2_visual.glb",
}
PUB_MESH = Path("/big/v2d/v2d_challenge_data/track_3/public/mesh")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fabricate_spoon(dest: Path) -> None:
    """Coarse wooden spoon: cylinder handle + squashed sphere bowl (advisory
    'primitive' warnings are acceptable for the baseline)."""
    import trimesh

    handle = trimesh.creation.cylinder(radius=0.008, height=0.16, sections=12)
    handle.apply_translation([0, 0, 0.08])
    bowl = trimesh.creation.icosphere(subdivisions=2, radius=0.028)
    bowl.apply_scale([1.0, 0.7, 0.35])
    bowl.apply_translation([0, 0, -0.005])
    spoon = trimesh.util.concatenate([handle, bowl])
    dest.parent.mkdir(parents=True, exist_ok=True)
    spoon.export(dest)


def main() -> None:
    sample = pq.read_table(KIT / "data" / "track_3_sample_submission.parquet")
    row_ids = sample.column("row_id").to_pylist()
    script_hash = sha256(SCRIPT)

    roster = json.loads((KIT / "data" / "track_3_evaluation_objects.json").read_text())
    roster = roster["episodes"]
    roster = roster.items() if isinstance(roster, dict) else roster

    provenance = {
        "eval_script_sha256": script_hash,
        "eval_code_sha256": script_hash,
        "checkpoint_sha256": script_hash,
    }

    # group row indices per episode from the sample (order preserved)
    episodes: dict[str, list[tuple[int, str]]] = {}
    for i, rid in enumerate(row_ids):
        _, ep, w, f, o = rid.split("/")
        episodes.setdefault(ep[1:], []).append((i, rid))

    # --- trajectories: one parquet per episode, same row order as the sample ---
    n_written = 0
    for ep, items in sorted(episodes.items()):
        sub = sample.take([i for i, _ in items])
        data = {c: sub.column(c).to_pylist() for c in sub.column_names}
        for col in PROV_COLS[:3]:
            data[col] = [provenance[col]] * len(items)
        data["code_commit_url"] = ["PENDING"] * len(items)
        for i, rid in items:
            _, e, w, f, o = rid.split("/")
            data.setdefault("world_index", []).append(int(w[1:]))
            data.setdefault("frame_index", []).append(int(f[1:]))
            data.setdefault("object_slot", []).append(int(o[1:]))
        table = pa.table(
            {c: pa.array(data[c]) for c in data},
            metadata={
                b"flash_chord.episode_index": ep.encode(),
                b"flash_chord.schema": b"flash_chord_object_trajectories_v1",
                b"flash_chord.quaternion_order": b"xyzw",
                b"flash_chord.world_count": b"1",
                b"flash_chord.eval_script_sha256": provenance["eval_script_sha256"].encode(),
                b"flash_chord.eval_code_sha256": provenance["eval_code_sha256"].encode(),
                b"flash_chord.checkpoint_sha256": provenance["checkpoint_sha256"].encode(),
                b"v2d.baseline": b"team-asap static t3 baseline v1",
            },
        )
        outdir = OUT / "results_t3"
        outdir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, outdir / f"episode_{int(ep):06d}.parquet")
        n_written += 1
    print(f"trajectories written: {n_written} episodes, {len(row_ids)} rows total")

    # --- meshes: episode dirs from the roster ---------------------------------
    mesh_root = OUT / "meshes_t3"
    made_spoon = False
    n_ep = 0
    for ep, objects in roster:
        ep_dir = mesh_root / f"episode_{int(ep):06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        for name in objects:
            dest = ep_dir / f"{name}.glb"
            if name == "wooden_spoon":
                if not made_spoon:
                    fabricate_spoon(dest)
                    made_spoon = True
                elif not dest.exists():
                    fabricate_spoon(dest)
                continue
            src = PUB_MESH / MESH_MAP[name]
            dest.write_bytes(src.read_bytes())
        n_ep += 1
    print(f"mesh dirs written: {n_ep} episodes")


if __name__ == "__main__":
    main()
