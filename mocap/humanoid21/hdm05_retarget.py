"""HDM05 位置序列 -> retarget_frame 输入适配 + 片段重定向。

与 CMU 路径的差异（hdm05 .data 是 normPOS 位置流，无旋转角）：
  - joint/distal 直接取 31 点（语义按 ASF 骨序，实为标记点位置）
  - F["thorax"]/F["root"] 由位置估计：lat=肩线(左-右)，up=脊柱方向，
    fwd=lat×up；与 CMU 约定一致（直立时 F≈I，x 左 y 上 z 前）
  - 根为原点锚定（in-place）：joint["root"] 合成为
    (0, -min(脚底y), 0) —— 骨盆高度取当前帧最低足的落差；
    跳跃/双脚离地时会退化为"下蹲式"骨盆高度（已知近似）
  - 坐标单位重标定：按每段腿长（骨盆到最低足）折算到 CMU 等效单位，
    之后复用 calibration SCALE 与全套 IK 目标，零改动

已知近似（写进 index adaptation 字段）：
  - 无世界位移/偏航（normPOS 已剥离）→ 走路/转身类只剩原地姿势
  - 骨盆朝向由髋线+脊柱代理估计（真实骨盆 roll/yaw 不可观测）
  - 标记点 ≠ 关节中心（lfemur 点是大腿中段标记），相对位置目标
    对标记语义鲁棒，但绝对骨长比例与 ASF 骨架不同
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mocap.humanoid21.parsers.hdm05_data import (  # noqa: E402
    HdmSegment, JOINT_NAMES, iter_segments)
from mocap.humanoid21.retarget import Retargeter, mirror_frame  # noqa: E402
from mocap.humanoid21.robot_kin import CONTROLLED_JOINTS  # noqa: E402

FPS = 120.0

# CMU 标定骨架的腿伸展长（units）：机器人腿 0.79m / 0.117 = 6.75，
# 用髋->脚底（而非骨盆落差）标定，保证伸直腿映射到机器人全伸展
CMU_LEG_UNITS = 6.75

# lfoot/rfoot 是贴地脚跟标记；踝目标抬到标记上方（踝高 ~10cm =
# 0.85 cmu 等效单位——此处 pts 已被 k 缩放，单位与 cmu 等效）
ANKLE_UP_UNITS = 0.85

# distal 端 = 子关节位置（标记语义）；叶子骨向前延长
_DISTAL_OF = {
    "lfemur": "ltibia", "ltibia": "lfoot", "lfoot": "ltoes",
    "rfemur": "rtibia", "rtibia": "rfoot", "rfoot": "rtoes",
    "lhumerus": "lradius", "lradius": "lwrist", "lwrist": "lhand",
    "rhumerus": "rradius", "rradius": "rwrist", "rwrist": "rhand",
    "lowerback": "upperback", "upperback": "thorax", "thorax": "lowerneck",
    "lowerneck": "upperneck", "upperneck": "head",
    "lclavicle": "lhumerus", "rclavicle": "rhumerus",
}
_LEAF_EXTEND = {  # 叶子：沿自身方向再延一小段作 distal
    "ltoes": ("lfoot", 0.3), "rtoes": ("rfoot", 0.3),
    "lhand": ("lwrist", 0.15), "rhand": ("rwrist", 0.15),
    "head": ("upperneck", 0.2),
    "lfingers": ("lhand", 0.1), "lthumb": ("lhand", 0.1),
    "rfingers": ("rhand", 0.1), "rthumb": ("rhand", 0.1),
}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _frame_from_positions(j: Dict[str, np.ndarray]) -> np.ndarray:
    """由位置估计正交骨架朝向 F（列=骨局部轴，CMU 约定 x左/y上/z前）。"""
    def orth(lat_raw, up_raw):
        lat = _unit(lat_raw); up = _unit(up_raw)
        fwd = _unit(np.cross(lat, up))
        lat = np.cross(up, fwd)
        return np.stack([lat, up, fwd], axis=1)  # columns

    F = {}
    # 躯干：肩线 + 脊柱
    F["thorax"] = orth(j["lhumerus"] - j["rhumerus"],
                       j["thorax"] - j["lowerback"])
    # 骨盆：髋线 + 脊柱代理（骨盆 up 不可观测，取同脊柱方向）
    mid_hips = 0.5 * (j["lhipjoint"] + j["rhipjoint"])
    F["root"] = orth(j["lhipjoint"] - j["rhipjoint"],
                     j["lowerback"] - mid_hips)
    # 其余骨给恒等（retarget 只用 thorax/root）
    for n in JOINT_NAMES:
        F.setdefault(n, np.eye(3))
    return F


def hdm05_frame_inputs(pts: np.ndarray) -> Tuple[dict, dict, dict]:
    """单帧 31x3 HDM 坐标 -> (joint, distal, F)，坐标已按段缩放。"""
    joint = {n: pts[i].copy() for i, n in enumerate(JOINT_NAMES)}
    # distal 先按原始标记构建（避免被下面的语义重映射污染方向）
    distal = {}
    for b, child in _DISTAL_OF.items():
        distal[b] = joint[child].copy()
    for b, (base, ext) in _LEAF_EXTEND.items():
        distal[b] = joint[b] + _unit(joint[b] - joint[base]) * ext * max(
            1.0, np.linalg.norm(joint[b] - joint[base]))
    for n in JOINT_NAMES:
        distal.setdefault(n, joint[n].copy())
    F = _frame_from_positions(joint)
    # HDM 点是标记非关节中心：lfemur/rfemur 是大腿中段标记。
    # IK 的髋锚点必须是关节位置 -> 用 hipjoint 覆盖 femur 键。
    # 大腿方向项用：保留中点标记为 lthigh_marker（joint['lfemur'] 被覆盖为
    # 髋锚点，retarget leg_dir='thigh' 读这个键）
    joint["lthigh_marker"] = pts[JOINT_NAMES.index("lfemur")].copy()
    joint["rthigh_marker"] = pts[JOINT_NAMES.index("rfemur")].copy()
    joint["lfemur"] = joint["lhipjoint"].copy()
    joint["rfemur"] = joint["rhipjoint"].copy()
    # lfoot/rfoot 是贴地脚跟标记：踝位置目标抬高 ANKLE_UP_UNITS，
    # 否则机器人脚踝被压到地面、腿持续压缩
    joint["lfoot"] = joint["lfoot"] + np.array([0.0, ANKLE_UP_UNITS, 0.0])
    joint["rfoot"] = joint["rfoot"] + np.array([0.0, ANKLE_UP_UNITS, 0.0])
    return joint, distal, F


def segment_scale(seg: HdmSegment) -> float:
    """cmu_equivalent_units = hdm_units * k。

    k = CMU 等效腿长 / 本段中位（髋线中心到最低足底的竖直落差）。
    站立帧髋在 ~-2 单位、足底在 -16 附近 -> hip2sole ≈ 14-16 单位。"""
    j = seg.joints()
    hip_y = 0.5 * (j["lhipjoint"][:, 1] + j["rhipjoint"][:, 1])
    sole_y = np.minimum.reduce([j["ltoes"][:, 1], j["rtoes"][:, 1]])
    leg_drop = float(np.median(hip_y - sole_y))
    if leg_drop < 1e-3:
        leg_drop = 1.0
    return CMU_LEG_UNITS / leg_drop


def retarget_hdm05_segment(seg: HdmSegment, mirror: bool = False,
                           verbose: bool = False) -> dict:
    """对齐 retarget.retarget_clip 的输出契约。"""
    k = segment_scale(seg)
    pts = seg.points * k
    T = len(pts)

    rt = Retargeter(leg_dir="thigh")
    qang = np.zeros((T, len(CONTROLLED_JOINTS)))
    rpos = np.zeros((T, 3)); rquat = np.zeros((T, 4))
    res_keys = ["leg_left", "leg_right", "ankle_left", "ankle_right",
                "arm_left", "arm_right"]
    resid = {kk: np.zeros(T) for kk in res_keys}

    def _inputs(t):
        j, d, F = hdm05_frame_inputs(pts[t])
        # 合成骨盆世界位置：原地 x/z=0，高度=最低足落差
        feet_y = min(j["ltoes"][1], j["rtoes"][1],
                     j["lfoot"][1], j["rfoot"][1])
        j["root"] = np.array([0.0, -feet_y, 0.0])
        if mirror:
            j, d, F = mirror_frame(j, d, F)
        return j, d, F

    for t in range(T):
        j, d, F = _inputs(t)
        r = rt.retarget_frame(j, d, F)
        rpos[t] = r.root_pos; rquat[t] = r.root_quat_wxyz
        for i, n in enumerate(CONTROLLED_JOINTS):
            qang[t, i] = r.q.get(n, 0.0)
        for kk in res_keys:
            resid[kk][t] = r.residuals[kk]

    # 前缘修复（与 retarget_clip 相同的 backward re-solve）
    qmax = np.abs(np.diff(qang, axis=0)).max(axis=1)
    t_end = 0
    for t in range(1, min(240, T - 4)):
        if np.all(qmax[t:t + 4] < 0.08):
            t_end = t
            break
    if t_end >= 1:
        bwd = Retargeter(leg_dir="thigh")
        bwd.prev_q = dict(zip(CONTROLLED_JOINTS, qang[t_end]))
        for t in range(t_end - 1, -1, -1):
            j, d, F = _inputs(t)
            r = bwd.retarget_frame(j, d, F)
            rpos[t] = r.root_pos
            for i, n in enumerate(CONTROLLED_JOINTS):
                qang[t, i] = r.q.get(n, 0.0)
            for kk in res_keys:
                resid[kk][t] = r.residuals[kk]

    # 落地：统一 z 平移使最低脚掌 ~0
    sole_min = np.inf
    for t in range(0, T, 4):
        qvec = rt.fk.qvec(rpos[t], rquat[t],
                          dict(zip(CONTROLLED_JOINTS, qang[t])))
        pos, _ = rt.fk.forward(qvec)
        sole = min(pos["foot_left"][2], pos["foot_right"][2]) - 0.027
        sole_min = min(sole_min, sole)
    rpos[:, 2] -= sole_min - 0.005
    return {"root_pos": rpos, "root_quat": rquat, "joint_angles": qang,
            "residuals": resid, "hdm05_scale_factor": k}
