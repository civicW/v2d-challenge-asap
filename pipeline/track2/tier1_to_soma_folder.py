#!/usr/bin/env python3
"""Materialize a soma_to_g1.py data_folder from a tier_1 episode parquet.

soma_to_g1.py (official NVIDIA SOMA->G1 retargeter) consumes a folder:
    soma_params.npz                    SOMA-X params (poses/transl/identity/scale/joint_orient/unit/keep_root)
    poses.npy                          (T,4,4) object poses, OpenCV world (the tool flips Y,Z into SOMA axes)
    reconstructed_mesh/output_aligned.glb   object mesh (copied from the tier_1 release)

This script rebuilds exactly that layout from the challenge's tier_1
multiview-caption release, so the official retargeter can run unmodified.

Usage (inside gr00t_env on the A100, after setup_soma_assets.py):
    python tier1_to_soma_folder.py \
        --tier1-root /big/v2d/v2d_challenge_data/track_2/tier_1_multiview_caption \
        --episode 0 --out /big/v2d/v2d_work/soma_folders/ep0000

Frame notes:
- tier_1 object quaternions are xyzw (kit v2dlb convention).
- The tool hard-applies a fixed OpenCV->SOMA flip diag(1,-1,-1) to poses.npy,
  so we PRE-apply the same flip here (it is its own inverse) and the tool's
  result equals the raw tier_1 trajectory.
- The retargeter re-anchors body+object at frame 0; the anchor transform is
  recoverable from the motion parquet provenance for pack-time inversion.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


_R_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0, 1.0])


def load_episode(tier1_root: Path, episode: int):
    parquet = tier1_root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(parquet).to_pandas()
    human_pose = np.stack(table["observation.human.pose"].to_numpy()).astype(np.float32)
    T = human_pose.shape[0]
    human_pose = human_pose.reshape(T, 77, 3)
    transl = np.stack(table["observation.human.translation"].to_numpy()).astype(np.float32)
    identity = np.stack(table["observation.human.identity_coeffs"].to_numpy()).astype(np.float32)[:1]
    scale = np.stack(table["observation.human.scale_params"].to_numpy()).astype(np.float32)[:1]
    flex = np.stack(table["observation.human.bone_length_flexibles"].to_numpy()).astype(np.float32)[:1]
    obj = np.stack(table["observation.object.pose"].to_numpy()).astype(np.float64)  # (T,7) xyzw
    visible = table["observation.object.visible"].to_numpy().astype(bool)
    return human_pose, transl, identity, scale, flex, obj, visible


def object_poses_to_mat4(obj7: np.ndarray) -> np.ndarray:
    """(T,7) xyz+quat_xyzw -> (T,4,4), pre-flipped so the tool's CV->SOMA flip restores truth."""
    from scipy.spatial.transform import Rotation as R

    mats = np.zeros((obj7.shape[0], 4, 4))
    mats[:, 3, 3] = 1.0
    mats[:, :3, 3] = obj7[:, :3]
    mats[:, :3, :3] = R.from_quat(obj7[:, 3:7]).as_matrix()
    # pre-apply the flip (self-inverse): loader applies it again -> identity
    return np.einsum("ij,tjk->tik", _R_CV_TO_SOMA, mats)


def resolve_joint_orient(joint_orient_arg: str | None):
    """joint_orient (J+1,3,3) from the staged SOMA rig, or None for --joint-orient none."""
    if joint_orient_arg == "none":
        return None, None
    from robotic_grounding.retarget.read_soma import SOMA  # noqa: PLC0415
    from robotic_grounding.retarget.params import SOMA_JOINTS_ORDER  # noqa: PLC0415

    s = SOMA()
    joint_names = list(s.rig_joint_names)
    if len(SOMA_JOINTS_ORDER) == len(joint_names):
        joint_names = [str(n) for n in SOMA_JOINTS_ORDER]
    rig = s.layer.rig_data
    rig_keys = set(rig.files) if hasattr(rig, "files") else set(rig.keys())
    if joint_orient_arg == "auto":
        candidates = {}
        for key in ("joint_orient", "bind_pose_local", "t_pose_local"):
            if key in rig_keys:
                arr = np.asarray(rig[key])
                if arr.ndim == 3 and arr.shape[1:] == (4, 4):
                    arr = np.ascontiguousarray(arr[:, :3, :3])
                if arr.ndim == 3 and arr.shape[1:] == (3, 3):
                    candidates[key] = arr
        if not candidates:
            raise SystemExit(f"no (N,3,3) rig array found; rig keys: {sorted(rig_keys)}")
        for name, arr in candidates.items():
            print(f"[joint_orient] candidate {name}: shape {arr.shape}")
        # 'auto' picks joint_orient if present, else t_pose_local, else bind_pose_local
        pick = ("joint_orient" in candidates and "joint_orient") or (
            "t_pose_local" in candidates and "t_pose_local"
        ) or "bind_pose_local"
        arr = candidates[pick]
        if arr.shape[0] == len(joint_names):  # rig minus root -> prepend identity for root
            arr = np.concatenate([np.eye(3, dtype=arr.dtype)[None], arr], axis=0)
        return arr.astype(np.float64), joint_names
    raise SystemExit(f"unknown --joint-orient {joint_orient_arg!r}")


def align_object_to_source(tier1_root: Path, meta: dict, out: Path, npz_path: Path) -> None:
    """Map tier_1 object poses (g1_retarget_world) back into the CV/source frame
    soma_to_g1 expects, by inverting its exact forward chain:
        P_robot = Rw @ L_src @ N @ F @ P_cv (+ constant ground lift)
    The lift is re-derived by the tool from the mesh, so inverting only the
    rigid part keeps everything self-consistent.
    """
    from robotic_grounding.retarget.read_soma import SOMA, _first_frame_transform
    from robotic_grounding.retarget.robot_config import load_robot_config
    from robotic_grounding.retarget.ground_alignment import (
        compute_plane_leveling_transform,
        load_ground_plane_robot_frame,
    )

    gp_src = tier1_root / meta["ground_plane"]
    gp_dst = out / "ground_plane.json"
    shutil.copy(gp_src, gp_dst)

    s = SOMA()
    m = s.load_motion(str(npz_path), normalize=False)
    R_inv = np.asarray(m["first_frame_R_inv"], dtype=np.float64)
    t_first = np.asarray(m["first_frame_transl"], dtype=np.float64)
    N = np.eye(4)
    N[:3, :3] = R_inv
    N[:3, 3] = -R_inv @ t_first
    F = np.diag([1.0, -1.0, -1.0, 1.0])
    cfg = load_robot_config("g1")
    Rw = np.eye(4)
    Rw[:3, :3] = np.asarray(cfg.r_world, dtype=np.float64)
    plane = load_ground_plane_robot_frame(
        gp_dst, cv_to_source=F, first_frame_anchor=N, source_to_robot=Rw[:3, :3]
    )
    R_lvl, t_lvl = compute_plane_leveling_transform(plane)
    L4 = np.eye(4)
    L4[:3, :3] = Rw[:3, :3].T @ R_lvl @ Rw[:3, :3]
    L4[:3, 3] = Rw[:3, :3].T @ t_lvl
    T_total = Rw @ L4 @ N @ F
    T_inv = np.linalg.inv(T_total)

    obj7 = load_episode(tier1_root, meta["episode_index"])[-2]
    P_robot = object_poses_to_mat4(obj7)  # already pre-flipped; undo that first
    P_robot = np.einsum("ij,tjk->tik", F, P_robot)
    P_cv = np.einsum("ij,tjk->tik", T_inv, P_robot)
    np.save(out / "poses.npy", P_cv)
    np.save(out / "T_total.npy", T_total)
    print(f"[align] object poses mapped robot_world -> CV source; wrote {out / 'poses.npy'}")


def rebase_from_robot_palms(out: Path, retarget_parquet: Path) -> None:
    """Shift the object so its closest mesh vertex touches the RETARGETED
    robot's palm at the frame of closest approach.

    Basing on the robot's actual palms (not the SOMA hand joints) bakes the
    IK reach residual into the placement, which is what makes contact
    physically possible — the official template shows 283/800 contact frames
    this way. The robot motion is object-independent, so one retarget run
    suffices for calibration; rerun the retarget afterwards.
    """
    import pyarrow.parquet as pq
    import trimesh

    tb = pq.read_table(retarget_parquet)
    ee = np.asarray(tb.column("ee_pose_w")[0].as_py(), dtype=np.float64)  # (T,2,7)
    obj = np.asarray(tb.column("object_body_position")[0].as_py(), dtype=np.float64)
    if obj.ndim == 3:
        obj = obj[:, 0, :]
    quat = np.asarray(tb.column("object_body_wxyz")[0].as_py(), dtype=np.float64)  # (T,1,4)
    if quat.ndim == 3:
        quat = quat[:, 0, :]

    mesh = trimesh.load(out / "reconstructed_mesh" / "output_aligned.glb", force="scene")
    verts_local = np.concatenate([g.vertices for g in mesh.geometry.values()]).astype(np.float64)

    from scipy.spatial.transform import Rotation as R

    best = (np.inf, None, None)
    for t in range(0, ee.shape[0], 5):
        Rm = R.from_quat(quat[t][[1, 2, 3, 0]]).as_matrix()  # wxyz -> xyzw
        vw = verts_local @ Rm.T + obj[t]
        d = np.linalg.norm(ee[t][:, None, :3] - vw[None, :, :], axis=-1)
        mi = np.unravel_index(np.argmin(d), d.shape)
        if d[mi] < best[0]:
            best = (float(d[mi]), t, mi)
    d_min, t_star, (pi, vi) = best
    palm = ee[t_star, pi, :3]
    Rm = R.from_quat(quat[t_star][[1, 2, 3, 0]]).as_matrix()
    vert_w = verts_local[vi] @ Rm.T + obj[t_star]
    delta_robot = palm - vert_w
    print(f"[rebase2] closest robot-palm approach: frame {t_star} dist {d_min:.3f} m")

    T_total = np.load(out / "T_total.npy")
    delta_cv = T_total[:3, :3].T @ delta_robot  # robot-frame shift -> CV/source frame

    poses = np.load(out / "poses.npy")
    poses[:, :3, 3] += delta_cv
    np.save(out / "poses.npy", poses)
    prev = out / "rebase_delta_source.npy"
    if prev.exists():
        F = np.diag([1.0, -1.0, -1.0, 1.0])
        prev_delta_src = np.load(prev)
        total_src = prev_delta_src + F[:3, :3] @ delta_cv
        np.save(prev, total_src)
        print(f"[rebase2] total source-frame shift now: {total_src.round(3)}")
    print(f"[rebase2] applied robot-palm shift (CV frame): {delta_cv.round(3)}")


def rebase_object_to_hands(out: Path, npz_path: Path) -> np.ndarray:
    """Translate the object trajectory so its closest mesh vertex touches the
    nearest SOMA hand joint at the frame of closest approach.

    tier_1's shipped human params and object trajectory are not scene-registered
    (palms end up meters from the object), so training/sim needs this re-base.
    The offset is a constant in the source frame; at pack time the simulated
    track is shifted back by the same constant mapped through the tool's
    forward chain, keeping the submission in tier_1 coordinates.
    Returns the source-frame translation delta (3,).
    """
    from robotic_grounding.retarget.read_soma import SOMA
    import trimesh

    s = SOMA()
    m = s.load_motion(str(npz_path), normalize=False)
    joints = np.asarray(m["joints"], dtype=np.float64)  # (T, 77+, 3)
    names = list(m["joint_names"])
    hand_idx = [i for i, n in enumerate(names) if "Hand" in n and "End" not in n and "Thumb" not in n and "Index" not in n and "Middle" not in n and "Ring" not in n and "Pinky" not in n]
    print(f"[rebase] hand joints: {[(names[i], i) for i in hand_idx]}")
    hands = joints[:, hand_idx, :]  # (T, H, 3)

    mesh = trimesh.load(out / "reconstructed_mesh" / "output_aligned.glb", force="scene")
    verts_local = np.concatenate([g.vertices for g in mesh.geometry.values()]).astype(np.float64)

    poses = np.load(out / "poses.npy")  # (T,4,4) in CV frame
    F = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
    P_src = np.einsum("ij,tjk->tik", F, poses)  # to source frame

    best = (np.inf, None, None)
    for t in range(0, P_src.shape[0], 5):
        Rt, pt = P_src[t, :3, :3], P_src[t, :3, 3]
        vw = verts_local @ Rt.T + pt
        d = np.linalg.norm(hands[t][:, None, :] - vw[None, :, :], axis=-1)
        mi = np.unravel_index(np.argmin(d), d.shape)
        if d[mi] < best[0]:
            best = (float(d[mi]), t, mi)
    d_min, t_star, (hi, vi) = best
    print(f"[rebase] closest approach: frame {t_star} dist {d_min:.3f} m")

    Rt, pt = P_src[t_star, :3, :3], P_src[t_star, :3, 3]
    target_vert = verts_local[vi] @ Rt.T + pt
    delta_src = hands[t_star, hi] - target_vert

    delta_cv = F[:3, :3] @ delta_src  # express the same shift in the CV frame
    poses[:, :3, 3] += delta_cv
    np.save(out / "poses.npy", poses)
    np.save(out / "rebase_delta_source.npy", delta_src)
    print(f"[rebase] applied constant shift (source frame): {delta_src.round(3)}")
    return delta_src


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier1-root", type=Path, required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--joint-orient", default="auto", choices=("auto", "none"))
    ap.add_argument(
        "--align-robot-world",
        action="store_true",
        help="tier_1 object poses are in g1_retarget_world: invert the tool's "
        "forward chain (flip/anchor/leveling/r_world) and copy ground_plane.json",
    )
    ap.add_argument(
        "--rebase-to-hands",
        action="store_true",
        help="shift the object trajectory so its mesh touches the nearest hand "
        "joint at closest approach (constant offset, recorded for pack-time "
        "inversion as rebase_delta_source.npy)",
    )
    ap.add_argument(
        "--rebase-from-parquet",
        type=Path,
        default=None,
        help="calibrate the object shift from a retarget run's output parquet: "
        "place the mesh vertex at the RETARGETED robot palm at closest approach",
    )
    ap.add_argument("--skip-npz", action="store_true", help="only write poses.npy + mesh")
    args = ap.parse_args()

    meta_lines = (args.tier1_root / "meta" / "episodes_metadata.jsonl").read_text().splitlines()
    meta = json.loads(next(l for l in meta_lines if f'"episode_index": {args.episode},' in l))
    object_name = meta["object"]

    out = args.out
    (out / "reconstructed_mesh").mkdir(parents=True, exist_ok=True)

    human_pose, transl, identity, scale, flex, obj7, visible = load_episode(args.tier1_root, args.episode)
    print(f"episode {args.episode}: T={human_pose.shape[0]} object={object_name} visible={visible.mean():.3f}")

    poses_path = out / "poses.npy"
    np.save(poses_path, object_poses_to_mat4(obj7))

    mesh_src = args.tier1_root / meta["mesh"]
    mesh_dst = out / "reconstructed_mesh" / "output_aligned.glb"
    shutil.copy(mesh_src, mesh_dst)
    print(f"mesh: {mesh_src} -> {mesh_dst}")

    if args.skip_npz:
        return

    joint_orient, joint_names = resolve_joint_orient(args.joint_orient)
    if joint_orient is None and args.joint_orient != "none":
        return
    from soma.io import save_soma_npz  # noqa: PLC0415

    # save_soma_npz strips index 0 unless keep_root=True: feed the virtual
    # Root joint (dummy rotvec, discarded) so the stored npz is Hips-first 77.
    poses78 = np.concatenate(
        [np.zeros((human_pose.shape[0], 1, 3), np.float32), human_pose], axis=1
    )
    names78 = ["Root"] + list(joint_names)
    npz_kwargs = dict(
        poses=poses78,
        transl=transl,
        joint_names=names78,
        identity_model_type="mhr",
        identity_coeffs=identity,
        scale_params=scale,
        joint_orient=joint_orient,
        unit="meters",
        keep_root=False,
        extra_arrays={"bone_length_flexibles": flex},
    )
    try:
        save_soma_npz(out / "soma_params.npz", **npz_kwargs)
    except TypeError:
        # older signature without extra_arrays
        npz_kwargs.pop("extra_arrays")
        save_soma_npz(out / "soma_params.npz", **npz_kwargs)
    print(f"soma_params.npz written to {out / 'soma_params.npz'}")

    if args.align_robot_world:
        align_object_to_source(args.tier1_root, meta, out, out / "soma_params.npz")
    if args.rebase_to_hands:
        rebase_object_to_hands(out, out / "soma_params.npz")
    if args.rebase_from_parquet:
        rebase_from_robot_palms(out, args.rebase_from_parquet)


if __name__ == "__main__":
    main()
