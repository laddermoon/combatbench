"""10 步态姿态可视化：GaitPosture → P5 → Pose.infer_from_p5 → PNG。

几何参数（拍定第一版，s42i 实测依据）：
    width_normal = 0.475 s42i 0.76 过宽，取 5/8
    stagger_unit = 0.22  一档前后错位（双支撑 |stagger| 中位~0.25）
    width_wide   = 0.625 同缩保持 1.3×，待仿真实测校
    ankle_z      = 0.107 踝锚（ankle_y xanchor，站姿态实测）

头/手 landmark 取 s42i 5 clips 标准化均值（guard 抱架）；
宽距时头高分档下探直到可解（腿够不着就低头=屈膝降重心）。

    PYTHONPATH=. python3 -m baseline.humanoid21.v1.render_gait_poses \
        [--out baseline/humanoid21/v1/gait_poses]
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
from PIL import Image

from baseline.humanoid21.v1.step_gait import POSTURES, GaitPosture
from point5 import P5
from point5.humanoid21.robot_state import Pose

# ---- 拍定几何参数 v1 -------------------------------------------------
WIDTH_N, WIDTH_W = 0.475, 0.625
STAGGER_UNIT = 0.22
ANKLE_Z = 0.107

# s42i 标准化均值模板（踝中点原点、躯干+x、yaw=0）
HEAD = np.array([0.07, 0.0, 1.38])
HAND_L = np.array([0.22, 0.23, 0.91])
HAND_R = np.array([0.24, -0.28, 0.94])
HEAD_Z_CANDIDATES = (1.38, 1.32, 1.26, 1.20, 1.14)   # 宽距降档序列


def posture_p5(p: GaitPosture, head_z: float) -> P5:
    """姿态 → P5：踝中点原点，s<0 左前；横向 ±W/2。"""
    w = WIDTH_W if p.width else WIDTH_N
    u = STAGGER_UNIT
    ank_l = np.array([-p.stagger * u / 2, +w / 2, ANKLE_Z])
    ank_r = np.array([+p.stagger * u / 2, -w / 2, ANKLE_Z])
    return P5(head=np.array([HEAD[0], HEAD[1], head_z]),
              left_hand=HAND_L, right_hand=HAND_R,
              left_ankle=ank_l, right_ankle=ank_r,
              torso_yaw=0.0)


# ---- 脚掌贴平后处理 --------------------------------------------------
# 阶段三踝先验是弹簧中立位，脚掌会翘起；把 ankle_y/x 迭代调到脚
# 底面水平（踝锚在关节转轴上，调整不动 P5 landmark）。
# qpos21 索引（CONTROLLED_JOINTS 序）
_ANK = {  # side -> (foot_body, jy, jx, 局部轴 y/x)
    "left":  ("foot_left_a", 13, 14,
              np.array([0, 1, 0]), np.array([-0.8944, 0, -0.4472])),
    "right": ("foot_right_a", 7, 8,
              np.array([0, 1, 0]), np.array([0.8944, 0, 0.4472])),
}


def _rotvec(R: np.ndarray) -> np.ndarray:
    """R 的轴角向量（log map）。"""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def flatten_feet(pose: Pose, iters: int = 8) -> Pose:
    """迭代把双脚掌贴到水平（root/其余关节不动）。"""
    import mujoco
    from point5.humanoid21.robot_state import _zyx_to_quat_wxyz
    model, data, r, _ = Pose._kinematics()
    adr = int(r["root_qpos_adr"])
    data.qpos[adr:adr + 3] = pose.root_position
    data.qpos[adr + 3:adr + 7] = _zyx_to_quat_wxyz(
        pose.root_orientation)
    qi = np.asarray(r["qpos_indices"])
    q = pose.qpos.copy()
    for _ in range(iters):
        data.qpos[qi] = q
        mujoco.mj_kinematics(model, data)
        done = True
        for name, (bname, jy, jx, ay, ax) in _ANK.items():
            bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, bname)
            Rf = data.xmat[bid].reshape(3, 3)
            v = -_rotvec(Rf)                      # 世界系修正量
            if np.linalg.norm(v) < 1e-3:
                continue
            A = np.column_stack([Rf @ ay, Rf @ ax])
            dq, *_ = np.linalg.lstsq(A, v, rcond=None)
            q[jy] += dq[0]
            q[jx] += dq[1]
            done = False
        if done:
            break
    return Pose(qpos=q, root_position=pose.root_position,
                root_orientation=pose.root_orientation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "gait_poses"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tiles = []
    for p in POSTURES:
        pose, hz = None, None
        for cand in HEAD_Z_CANDIDATES:
            try:
                pose = Pose.infer_from_p5(posture_p5(p, cand))
                hz = cand
                break
            except ValueError:
                continue
        if pose is None:
            print(f"[{p.code:>4s}] {p.name}: 不可解（所有头高）")
            continue
        pose = flatten_feet(pose)
        front = pose.render(azimuth=0, elevation=-5, distance=2.9)
        side = pose.render(azimuth=90, elevation=-5, distance=2.9)
        front.save(os.path.join(args.out, f"{p.code}_front.png"))
        side.save(os.path.join(args.out, f"{p.code}_side.png"))
        # p5 复核：反解后 FK 回读的踝点 vs 目标
        got = pose.to_p5()
        err = np.linalg.norm(
            got.left_ankle[:2] - posture_p5(p, hz).left_ankle[:2]) \
            + np.linalg.norm(
            got.right_ankle[:2] - posture_p5(p, hz).right_ankle[:2])
        print(f"[{p.code:>4s}] {p.name}: head_z={hz:.2f} "
              f"踝点回读误差={err:.3f}m")
        tiles.append((p, front, side))

    if tiles:
        W, H = tiles[0][1].size
        sheet = Image.new("RGB", (W * 2, H * len(tiles)), "white")
        for i, (p, f, s) in enumerate(tiles):
            sheet.paste(f, (0, i * H))
            sheet.paste(s, (W, i * H))
        sheet.save(os.path.join(args.out, "_contact.png"))
        print(f"contact sheet: {args.out}/_contact.png")


if __name__ == "__main__":
    main()
