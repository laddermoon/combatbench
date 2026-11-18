"""手臂 IK 锚点渲染验证：描述符→关节角→MuJoCo 离屏渲染。

内容：
  1. FK 对拍——numpy FK vs mujoco xpos（随机角，验证轴/序约定）
  2. 五锚点渲染——front/side 双视角 PNG，输出到 --out
  3. 圆角往返——solve→describe 一致性 + 已知好姿态(guard/jab norm)反查

    MUJOCO_GL=egl PYTHONPATH=. python3 -m \
        baseline.humanoid21.v1.render_arm_poses --out /tmp/v1_anchors
"""
from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
from PIL import Image

from baseline.humanoid21.v1.arm_ik import ArmIK, ArmPose

XML = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                   "envs", "humanoid21", "battle_circular_v2.xml")
XML = os.path.abspath(XML)

W, H = 640, 720
LOOKAT = np.array([-1.5, 0.0, 1.15])
DIST = 1.9

# 锚点描述符（按臂约定，左右臂同值=镜像动作）。azimuth θ:
# atan2(z,x) 矢状扫掠 0前/+90天/−90地/±180后；polar φ:
# 与外展轴夹角 0纯外展/90前平面/180穿中线；extension: |手−肩|
# /LMax；hint=None=自然肘。拳击区 φ≈[25°,115°]。
# 预置姿态集（v1 草案，待用户看图调参）：
#   ready       抱架中枢——颏旁微前送，去其它姿态的共同起点
#   guard_front 防直——拳护脸前中线（φ 大=更靠中线，θ 抬到面高）
#   guard_side  防摆——拳贴太阳穴侧前（θ 更高、ρ 更贴身）
#   jab_pre     直拳蓄力——从 ready 沿拳路后收（θ 降、ρ 收近）
#   hook_pre    摆拳蓄力——侧向水平后拉（θ 入后半区、φ 半开、较伸）
#   jab_*       出拳参照——打击点不固定，仅示意方向（φ 递减=更侧向）
ANCHORS = {
    "ready":       ArmPose.deg(+12, 105, 0.36),
    "guard_front": ArmPose.deg(+18, 115, 0.40),
    "guard_side":  ArmPose.deg(+30, 95, 0.27),
    "jab_pre":     ArmPose.deg(-90, 105, 0.32),   # 0.2m 下后拉→垂收胸侧
    "hook_pre":    ArmPose.deg(170, 35, 0.64),    # 0.4m 侧后拉→肩后20cm
    "jab_front":   ArmPose.deg(+2, 95, 1.0),
    "jab_45":      ArmPose.deg(+4, 45, 1.0),
    "jab_side":    ArmPose.deg(+4, 25, 1.0),
}

JOINTS = {
    "right": ["shoulder1_right_a", "shoulder2_right_a", "elbow_right_a"],
    "left": ["shoulder1_left_a", "shoulder2_left_a", "elbow_left_a"],
}


def _load():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    # 沉掉 robot_b
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root_b")
    d.qpos[m.jnt_qposadr[jid] + 2] = -10.0
    return m, d


def _set_arms(m, d, ang_r, ang_l):
    for side, ang in (("right", ang_r), ("left", ang_l)):
        for name, q in zip(JOINTS[side], ang):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[m.jnt_qposadr[jid]] = q
    mujoco.mj_forward(m, d)


def _hand_torso(m, d, side):
    """拳心 torso 系坐标（肩心原点）。torso euler=0 ⇒ 系轴=世界轴。"""
    suf = "right_a" if side == "right" else "left_a"
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"hand_{suf}")
    sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                           f"upper_arm_{suf}")
    return d.xpos[hb] - d.xpos[sb]


def _elbow_torso(m, d, side):
    suf = "right_a" if side == "right" else "left_a"
    eb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"lower_arm_{suf}")
    sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                           f"upper_arm_{suf}")
    return d.xpos[eb] - d.xpos[sb]


def _render(m, d, azimuth):
    r = mujoco.Renderer(m, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = LOOKAT.tolist()
    cam.distance = DIST
    cam.azimuth = azimuth
    cam.elevation = -5.0
    r.update_scene(d, camera=cam)
    img = r.render()
    r.close()
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/v1_anchors")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ik = ArmIK()
    m, d = _load()
    print(f"LMax={ik.LMax:.4f} d_min={ik.d_min:.4f}")

    # ---- 1. FK 对拍：numpy vs mujoco（随机角） ----
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        ang = np.array([
            rng.uniform(*(-1.48, 1.05)), rng.uniform(*(-1.48, 1.05)),
            rng.uniform(*(-1.74, 0.87))])
        for side in ("right", "left"):
            zero = np.zeros(3)
            _set_arms(m, d, ang if side == "right" else zero,
                      ang if side == "left" else zero)
            hand, elbow = ik.fk(side, *ang)
            max_err = max(max_err,
                          np.abs(hand - _hand_torso(m, d, side)).max(),
                          np.abs(elbow - _elbow_torso(m, d, side)).max())
    print(f"[fk-check] max|err|={max_err:.2e} "
          f"{'OK' if max_err < 1e-9 else 'FAIL — 轴序/偏移不对'}")

    # ---- 2. 往返校验：已知好姿态（norm 直控版）回描述符 ----
    for name, norm in (("guard_v3", (0.4, -0.3, 0.8)),
                       ("jab_v3", (0.4, 0.8, -0.8))):
        ang = np.array([norm[0] * 1.2654 - 0.2182,
                        norm[1] * 1.2654 - 0.2182,
                        norm[2] * 1.3090 - 0.4363])
        p = ik.describe("right", ang)
        print(f"[roundtrip] {name}: az={math.degrees(p.azimuth):.0f}° "
              f"pol={math.degrees(p.polar):.0f}° ext={p.extension:.2f}")

    # ---- 3. 锚点求解 + 渲染 ----
    print("\n[anchors]")
    for name, pose in ANCHORS.items():
        sol_r = ik.solve("right", pose)
        sol_l = ik.solve("left", pose)      # 按臂约定：同值=镜像
        print(f"  {name:9s} R norm=({sol_r.norm[0]:+.2f},{sol_r.norm[1]:+.2f},"
              f"{sol_r.norm[2]:+.2f}) rad=({sol_r.angles[0]:+.2f},"
              f"{sol_r.angles[1]:+.2f},{sol_r.angles[2]:+.2f}) "
              f"cands={sol_r.candidates} clip={sol_r.clipped} "
              f"dir_err={sol_r.dir_err_deg:.1f}° pos_err={sol_r.pos_err:.3f}m "
              f"hand={np.round(sol_r.hand, 2).tolist()}")
        print(f"  {'':9s} L norm=({sol_l.norm[0]:+.2f},{sol_l.norm[1]:+.2f},"
              f"{sol_l.norm[2]:+.2f}) cands={sol_l.candidates} "
              f"clip={sol_l.clipped} dir_err={sol_l.dir_err_deg:.1f}°")
        _set_arms(m, d, sol_r.angles, sol_l.angles)
        for view, az in (("front", 0), ("side", 90), ("qtr", 45)):
            img = _render(m, d, az)
            Image.fromarray(img).save(
                os.path.join(args.out, f"{name}_{view}.png"))
    print(f"renders → {args.out}")


if __name__ == "__main__":
    main()
