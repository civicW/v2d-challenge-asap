#!/usr/bin/env python3
"""validate_mano_sharpa — 按 flash_chord load_mano_sharpa 的校验规则检查单行 parquet."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_SIDES = ("left", "right")


def main() -> int:
    path = Path(sys.argv[1])
    table = pq.read_table(path)
    assert table.num_rows == 1, f"必须是单行 parquet, got {table.num_rows}"
    cells = {name: table.column(name)[0].as_py() for name in table.column_names}

    def arr(name):
        a = np.asarray(cells[name], dtype=np.float64)
        return None if a.size == 0 else a

    errors: list[str] = []
    T = None

    obj_pos, obj_quat = arr("object_body_position"), arr("object_body_wxyz")
    if obj_pos is None or obj_quat is None:
        errors.append("缺 object_body_position/wxyz")
    else:
        if obj_pos.ndim != 3 or obj_pos.shape[-1] != 3 or not np.isfinite(obj_pos).all():
            errors.append(f"object_body_position 形状/有限性: {obj_pos.shape}")
        T, B = obj_pos.shape[0], obj_pos.shape[1]
        if obj_quat.shape != (T, B, 4) or not np.isfinite(obj_quat).all():
            errors.append(f"object_body_wxyz 形状/有限性: {obj_quat.shape}")
        norms = np.linalg.norm(obj_quat, axis=-1)
        if np.any(norms <= np.finfo(np.float64).eps) or not np.allclose(norms, 1.0, atol=1e-4):
            errors.append(f"物体四元数非单位/含零: norm∈[{norms.min():.4f},{norms.max():.4f}]")
        names = cells.get("object_body_names") or []
        if not names or len(names) != B or len(set(names)) != B or any(not n for n in names):
            errors.append(f"object_body_names 与 B={B} 不符: {names}")
        mesh_paths = cells.get("object_mesh_paths") or []
        if len(mesh_paths) != B or any(not p for p in mesh_paths):
            errors.append(f"object_mesh_paths 与 B 不符: {mesh_paths}")
        urdf_paths = cells.get("object_urdf_paths") or []
        radius = arr("object_mesh_radius")
        if radius is None or radius.shape != (B,) or np.any(radius <= 0):
            errors.append(f"object_mesh_radius 应为 ({B},) 正值: {radius}")
        art = arr("object_articulation")
        art = np.zeros((T, 0)) if art is None else art.reshape(T, -1)
        if urdf_paths and art.shape[1] and not (art.shape[1] == 1 and np.all(art == 0)):
            errors.append("带独立 URDF 时 articulation 只能是单列全零")

    for side in _SIDES:
        wp, wq = arr(f"robot_{side}_wrist_position"), arr(f"robot_{side}_wrist_wxyz")
        if wp is None or wp.shape != (T, 3) or not np.isfinite(wp).all():
            errors.append(f"{side} wrist_position: {None if wp is None else wp.shape}")
        if wq is None or wq.shape != (T, 4) or not np.isfinite(wq).all():
            errors.append(f"{side} wrist_wxyz: {None if wq is None else wq.shape}")
        else:
            n = np.linalg.norm(wq, axis=-1)
            if np.any(n <= np.finfo(np.float64).eps) or not np.allclose(n, 1.0, atol=1e-4):
                errors.append(f"{side} 腕四元数非单位: norm∈[{n.min():.4f},{n.max():.4f}]")
        fj = arr(f"robot_{side}_finger_joints")
        fnames = cells.get(f"{side}_robot_finger_joint_names") or []
        if fj is None or fj.ndim != 2 or fj.shape[0] != T or not np.isfinite(fj).all():
            errors.append(f"{side} finger_joints: {None if fj is None else fj.shape}")
        elif len(fnames) != fj.shape[1] or len(set(fnames)) != len(fnames):
            errors.append(f"{side} finger_joint_names 数量/唯一性与 N={fj.shape[1]} 不符")
        cp, cn, cid = (arr(f"mano_{side}_object_contact_positions"),
                       arr(f"mano_{side}_object_contact_normals"),
                       arr(f"mano_{side}_object_contact_part_ids"))
        if cp is None or cn is None or cid is None:
            errors.append(f"{side} contacts 三件套缺失")
        else:
            if cp.ndim != 3 or cp.shape[0] != T or cp.shape[-1] != 3 or cn.shape != cp.shape:
                errors.append(f"{side} contacts 形状: {cp.shape} vs {cn.shape}")
            if cid.shape != cp.shape[:-1] or not np.isfinite(cid).all():
                errors.append(f"{side} part_ids 形状/有限性")
            else:
                if np.any(cid != np.floor(cid)) or np.any(cid < 0) or np.any(cid > B):
                    errors.append(f"{side} part_ids 越界")
                active = np.linalg.norm(cn, axis=-1)[cid > 0]
                if active.size and not np.allclose(active, 1.0, atol=1e-4):
                    errors.append(f"{side} 活跃接触法向非单位")

    if errors:
        print("FAIL:")
        for e in errors:
            print("  -", e)
        return 1
    fnames = cells.get("left_robot_finger_joint_names") or []
    print(f"PASS  T={T} B={B} fingers/hand={len(fnames)} fps={cells['fps']}")
    print(f"  objects: {cells['object_body_names']}")
    print(f"  wrist L 范例: pos={np.asarray(cells['robot_left_wrist_position'])[0]} quat={np.asarray(cells['robot_left_wrist_wxyz'])[0]}")
    fj = np.asarray(cells['robot_left_finger_joints'])
    print(f"  finger L 范围: min={fj.min():.3f} max={fj.max():.3f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
