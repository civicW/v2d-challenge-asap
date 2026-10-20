#!/usr/bin/env python3
"""mint_to_sharpa — MINT 世界系 MANO 双手 → ManoSharpaData 单行 parquet (V2D Track 3).

输入:
- MINT 服务 decode=1 产出的 ``decoded.npz`` (left/right_joints_world (T,21,3)、presence、camera_c2w)
- 物体轨迹来源(二选一):
  * ``--object-parquet``: track_3/public 的 data parquet(GT, w-first quat, 不可见帧填零) — 验证/回放模式
  * ``--object-static x y z qw qx qy qz``: 静止物体占位
- Sharpa 双手 URDF(取名与限位): left_sharpawave.urdf / right_sharpawave.urdf

输出: 单行 parquet, 逐帧嵌套数组, 列名严格对齐 flash_chord ``load_mano_sharpa``:
fps / object_body_position / object_body_wxyz / object_name / object_body_names /
object_mesh_paths / object_urdf_paths / object_mesh_radius / object_articulation /
robot_{side}_wrist_position / robot_{side}_wrist_wxyz / robot_{side}_finger_joints /
{side}_robot_finger_joint_names / mano_{side}_object_contact_{positions,normals,part_ids}

手指映射(MANO 21 关节 → Sharpa 22 revolute): 每指 curl = 归一化(腕↔指尖距/掌长)线性
映射到该指各 FE/PIP/DIP 关节(AA/CMC 置限位中点), 全部按 URDF limit 钳制。腕姿态 =
掌构架(x=腕→食指根, y=叉乘法向, z 补全), 与 retarget_dm 同约定但输出 wxyz。
手不在场帧 hold 上一有效值。contacts 输出 K=1 全无效槽(part_id=0)。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# MANO 21 关节索引(与 decoded.npz / smplx joints[:, :21] 约定一致)
WRIST = 0
THUMB = (1, 2, 3)
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# curl(0=伸直,1=屈曲) → 各关节角(rad)的一次映射系数; 粗但有界, 后续可用优化器精修
_CURL_GAINS = {"MCP_FE": 1.1, "PIP": 1.3, "DIP": 1.0, "IP": 1.1, "CMC_FE": 0.8, "CMC": 0.8}

_DIGITS = {
    "thumb": ("CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "IP"),
    "index": ("MCP_FE", "MCP_AA", "PIP", "DIP"),
    "middle": ("MCP_FE", "MCP_AA", "PIP", "DIP"),
    "ring": ("MCP_FE", "MCP_AA", "PIP", "DIP"),
    "pinky": ("CMC", "MCP_FE", "MCP_AA", "PIP", "DIP"),
}
_DIGIT_TIP = {"thumb": 3, "index": INDEX_TIP, "middle": MIDDLE_TIP, "ring": RING_TIP, "pinky": PINKY_TIP}


def parse_urdf_fingers(path: Path) -> dict[str, dict]:
    """{joint_name: {'lower':…, 'upper':…, 'effort':…}} for revolute non-wrist joints."""
    xml = path.read_text()
    joints = re.findall(r'<joint name="([^"]+)" type="revolute">(.*?)</joint>', xml, re.S)
    out = {}
    for name, body in joints:
        lm = re.search(r'<limit lower="([^"]*)" upper="([^"]*)"(?: effort="([^"]*)")?', body)
        if not lm:
            continue
        out[name] = {
            "lower": float(lm.group(1)) if lm.group(1) else -np.pi,
            "upper": float(lm.group(2)) if lm.group(2) else np.pi,
        }
    return out


def quat_wxyz_from_frame(x_axis: np.ndarray, y_raw: np.ndarray) -> np.ndarray:
    """右手系: x 给定, y 正交化, z=x×y → wxyz 四元数."""
    x = x_axis / (np.linalg.norm(x_axis) + 1e-9)
    y = y_raw - np.dot(y_raw, x) * x
    yn = np.linalg.norm(y)
    if yn < 1e-6:
        y = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], x) * x
        yn = np.linalg.norm(y)
    y = y / yn
    z = np.cross(x, y)
    R = np.stack([x, y, z], axis=1)  # 列 = 基向量
    # Roth convention: R → quaternion (wxyz)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, a, b, c = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, a, b, c = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, a, b, c = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, a, b, c = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, a, b, c], dtype=np.float64)
    return q / np.linalg.norm(q)


def hold_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """presence=false 的帧 hold 上一有效帧(帧0无效则取首个有效值)。"""
    out = values.copy()
    idx = np.where(valid)[0]
    if len(idx) == 0:
        raise ValueError("没有任何有效帧")
    last = values[idx[0]]
    for i in range(len(out)):
        if valid[i]:
            last = values[i]
        else:
            out[i] = last
    return out


def finger_track(joints_w: np.ndarray, limits: dict, side: str) -> tuple[np.ndarray, list[str]]:
    """(T,Nf) 关节角 + 名单(URDF 顺序)。curl 按掌长归一。"""
    T = joints_w.shape[0]
    wrist = joints_w[:, WRIST]
    palm = np.linalg.norm(joints_w[:, MIDDLE_MCP] - wrist, axis=1) + 1e-9

    def curl(tip_idx: int) -> np.ndarray:
        d = np.linalg.norm(joints_w[:, tip_idx] - wrist, axis=1) / palm
        # 张开(d≈1.6+)→0, 握拳(d≈0.8)→1
        c = np.clip((1.55 - d) / (1.55 - 0.85), 0.0, 1.0)
        return c

    names: list[str] = []
    cols: list[np.ndarray] = []
    for digit, suffixes in _DIGITS.items():
        c = curl(_DIGIT_TIP[digit])
        for suffix in suffixes:
            name = f"{side}_{digit}_{suffix}"
            if name not in limits:
                raise ValueError(f"URDF 缺关节 {name}; 检查 URDF 与脚本约定")
            lim = limits[name]
            if suffix in ("MCP_AA", "CMC_AA"):
                ang = np.full(T, 0.5 * (lim["lower"] + lim["upper"]))
            else:
                ang = np.clip(c * _CURL_GAINS.get(suffix, 1.0), lim["lower"], lim["upper"])
            names.append(name)
            cols.append(ang.astype(np.float64))
    return np.stack(cols, axis=1), names


def load_object_track(parquet_path: Path | None, static: list[float] | None, T: int):
    """返回 (T,B,3) pos, (T,B,4) wxyz quat, 物体名列表。不可见帧 hold。"""
    if static is not None:
        x, y, z, qw, qx, qy, qz = static
        q = np.array([qw, qx, qy, qz], dtype=np.float64)
        q = q / np.linalg.norm(q)
        pos = np.tile(np.array([x, y, z], dtype=np.float64), (T, 1))
        quat = np.tile(q, (T, 1))
        return pos[:, None], quat[:, None], ["object"]
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    rows = table.num_rows
    nested = table["observation.objects"].to_pylist()
    names = [e["name"] for e in nested[0]]
    B = len(names)
    pose = np.zeros((rows, B, 7), dtype=np.float64)
    visible = np.ones((rows, B), dtype=bool)
    for r, entries in enumerate(nested):
        for b, e in enumerate(entries):
            pose[r, b] = e["pose"]  # x y z qw qx qy qz
            visible[r, b] = bool(e["visible"])
    pos = pose[:, :, :3]
    quat = pose[:, :, 3:]  # wxyz
    for b in range(B):
        quat[:, b] /= np.maximum(np.linalg.norm(quat[:, b], axis=1, keepdims=True), 1e-9)
        pos[:, b] = hold_invalid(pos[:, b], visible[:, b])
        quat[:, b] = hold_invalid(quat[:, b], visible[:, b])
    return pos, quat, names


def mesh_radius(glb_path: Path) -> float:
    import trimesh

    mesh = trimesh.load(glb_path, force="mesh", process=False)
    center = mesh.bounding_sphere.center
    return float(np.linalg.norm(mesh.vertices - center, axis=1).max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decoded", required=True, type=Path)
    ap.add_argument("--summary", type=Path, help="MINT summary.json(取 fps; 缺省用 --fps)")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--left-urdf", required=True, type=Path)
    ap.add_argument("--right-urdf", required=True, type=Path)
    ap.add_argument("--object-parquet", type=Path)
    ap.add_argument("--object-static", nargs=7, type=float,
                    metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"))
    ap.add_argument("--object-mesh", required=True,
                    help="B=1 时给 GLB 文件; B>1 时给 mesh 根目录(mesh/<name>/<name>.glb)")
    ap.add_argument("--object-urdf", type=Path,
                    help="B=1 时给 URDF 文件; B>1 时给 urdf 根目录(<name>_rigid.urdf); 可缺省")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.object_parquet and not args.object_static:
        ap.error("需要 --object-parquet 或 --object-static")

    d = np.load(args.decoded, allow_pickle=False)
    left = d["left_joints_world"].astype(np.float64)
    right = d["right_joints_world"].astype(np.float64)
    lv = d["left_presence"].astype(bool)
    rv = d["right_presence"].astype(bool)
    T = left.shape[0]
    if right.shape[0] != T:
        sys.exit(f"左右手帧数不一致 {left.shape[0]} vs {right.shape[0]}")
    fps = args.fps
    if args.summary and args.summary.is_file():
        fps = float(json.loads(args.summary.read_text()).get("fps", fps))

    limits = {s: parse_urdf_fingers(p) for s, p in (("left", args.left_urdf), ("right", args.right_urdf))}
    for side in ("left", "right"):
        if not limits[side]:
            sys.exit(f"{side} URDF 未解析到任何 revolute 关节: {getattr(args, side + '_urdf')}")

    cells: dict[str, object] = {"fps": np.float64(fps)}

    obj_pos, obj_quat, obj_names = load_object_track(args.object_parquet, args.object_static, T)
    if obj_pos.shape[0] != T:
        sys.exit(f"物体轨迹帧数 {obj_pos.shape[0]} != 手部帧数 {T}(需要对齐/重采样)")
    B = len(obj_names)

    mesh_arg, urdf_arg = Path(args.object_mesh), (Path(args.object_urdf) if args.object_urdf else None)
    if B == 1 and mesh_arg.is_file():
        mesh_paths = [str(mesh_arg.resolve())]
    else:
        mesh_paths = [str((mesh_arg / n / f"{n}.glb").resolve()) for n in obj_names]
    for p in mesh_paths:
        if not Path(p).is_file():
            sys.exit(f"物体网格缺失: {p}")
    urdf_paths: list[str] = []
    if urdf_arg is not None:
        if B == 1 and urdf_arg.is_file():
            urdf_paths = [str(urdf_arg.resolve())]
        else:
            urdf_paths = [str((urdf_arg / f"{n}_rigid.urdf").resolve()) for n in obj_names]
        for p in urdf_paths:
            if not Path(p).is_file():
                sys.exit(f"物体 URDF 缺失: {p}")

    cells["object_body_position"] = obj_pos
    cells["object_body_wxyz"] = obj_quat
    cells["object_name"] = obj_names[0] if B == 1 else obj_names[0]
    cells["object_body_names"] = list(obj_names)
    cells["object_mesh_paths"] = list(mesh_paths)
    cells["object_urdf_paths"] = list(urdf_paths)
    cells["object_mesh_radius"] = np.array([mesh_radius(Path(p)) for p in mesh_paths], dtype=np.float64)
    cells["object_articulation"] = np.zeros((T, 1), dtype=np.float64)

    for side, joints, presence in (("left", left, lv), ("right", right, rv)):
        wrist = hold_invalid(joints[:, WRIST].copy(), presence)
        # 掌构架: x=腕→食指根, y=腕→中指根与腕→小指根叉乘的稳定法向, z=x×y
        x_raw = joints[:, INDEX_MCP] - joints[:, WRIST]
        y_raw = np.cross(joints[:, MIDDLE_MCP] - joints[:, WRIST], joints[:, PINKY_MCP] - joints[:, WRIST])
        quats = np.stack([quat_wxyz_from_frame(x_raw[t], y_raw[t]) for t in range(T)])
        quats = hold_invalid(quats, presence)
        fingers, names = finger_track(joints, limits[side], side)
        fingers = hold_invalid(fingers, presence)
        cells[f"robot_{side}_wrist_position"] = wrist
        cells[f"robot_{side}_wrist_wxyz"] = quats
        cells[f"robot_{side}_finger_joints"] = fingers
        cells[f"{side}_robot_finger_joint_names"] = list(names)
        cells[f"mano_{side}_object_contact_positions"] = np.zeros((T, 1, 3), dtype=np.float64)
        cells[f"mano_{side}_object_contact_normals"] = np.zeros((T, 1, 3), dtype=np.float64)
        cells[f"mano_{side}_object_contact_part_ids"] = np.zeros((T, 1), dtype=np.float64)

    import pyarrow as pa
    import pyarrow.parquet as pq

    def nested_type(dims: tuple[int, ...]):
        """cell 类型: 变长 T 外层 × 嵌套 fixed_size_list(整个 cell 是单行)."""
        t = pa.float64()
        for d in reversed(dims):
            t = pa.list_(t, d)
        return pa.list_(t)

    arrays, fields = [], []
    for k, v in cells.items():
        if isinstance(v, str):
            arrays.append(pa.array([v], type=pa.string()))
            fields.append(pa.field(k, pa.string()))
        elif isinstance(v, list) and v and isinstance(v[0], str):
            arrays.append(pa.array([v], type=pa.list_(pa.string())))
            fields.append(pa.field(k, pa.list_(pa.string())))
        else:
            a = np.asarray(v)
            if a.ndim == 0:
                arrays.append(pa.array([a.item()], type=pa.float64()))
                fields.append(pa.field(k, pa.float64()))
            else:
                typ = nested_type(a.shape[1:])
                arrays.append(pa.array([a.tolist()], type=typ))
                fields.append(pa.field(k, typ))
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    table = table.replace_schema_metadata({"flash_chord.reference_format": "ManoSharpaData"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)

    print(f"wrote {args.out}")
    print(f"  frames={T} fps={fps} fingers={len(cells['left_robot_finger_joint_names'])}")
    print(f"  objects={obj_names} meshes={mesh_paths}")
    print(f"  presence: L={lv.mean():.2f} R={rv.mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
