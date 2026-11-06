"""算法1 功能点 f1-f4 —— alg1_infoflow.md / algo1_meta.py 的实现层。

    f1(obs)          -> I1   当前支撑状态
    f2(obs)          -> I2   当前质心动量
    f3(clip)         -> I3   逐帧目标支撑状态
    f4(i1, i2, i3)   -> R    支撑状态序列

物理可行+先验系统，非最优化。当前全部为空壳——仅落签名。
"""
from __future__ import annotations

import numpy as np

from point5.contracts import P5Clip
from .algo1_meta import I1, I2, I3, R
from .contracts import Observation


_RANGES = None


def _ranges():
    """(21,2) joint radian ranges for obs denormalization."""
    global _RANGES
    if _RANGES is None:
        from point5.humanoid21.pose import Pose
        _RANGES = Pose.joint_ranges("robot_a")
    return _RANGES


def f1(obs: Observation) -> I1:
    """f1(Observation) -> I1：obs 解码当前支撑状态。

    Pose.from_obs -> anchor('ankle_mid') 一次 FK 出：
    脚位 xy + 精确质心（质量加权）+ obs[52:54] 实测力度分配。
    """
    from .contracts import FootSupport, SupportLandMark
    from point5.humanoid21.pose import Pose

    a = Pose.from_obs(np.asarray(obs, np.float64),
                            _ranges()).anchor('ankle_mid')
    fp = a.foot_pos()
    fr = obs[52:54]
    feet = {s: FootSupport(force_ratio=float(fr[k]), pos=fp[s])
            for s, k in (("right", 0), ("left", 1))}
    return I1(lm=SupportLandMark(com=a.com(), feet=feet))


def f2(obs: Observation) -> I2:
    """f2(Observation) -> I2：obs 解码质心速度 + 角动量。"""
    raise NotImplementedError


def f3(clip: P5Clip) -> I3:
    """f3(Clip) -> I3：每帧 P5 → 对应支撑状态。

    逐帧几何变换，不做事件推断：脚位取 P5 踝点、质心经
    工具1 推荐并投影支撑域（工具6）。
    """
    raise NotImplementedError


def f4(i1: I1, i2: I2, i3: I3) -> R:
    """f4(I1, I2, I3) -> R：逐帧目标 → 稀疏锚点序列。

    把逐帧目标压缩成锚点：哪里必须换脚（支撑事件锚点）、
    哪里只需姿态锚点；com 转移须物理可行且有裕度（I2 动量
    决定转移方向——离地锚点要求 capture point 指向落脚区）。
    """
    raise NotImplementedError
