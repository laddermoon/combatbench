"""Tool 4: P5 -> COM-change capability expectation (empirical).

业务约束的经验函数：该姿态下质心能多快地被改变。粗糙、
90% 信息量即可——用于算法1 定段时长与速度上限。

规则 v1（局部系 P5 -> Capability）：

- 双脚着地：横向能力 ~ 脚间距 d 归一（相对标准站距 ~0.25m），
  越宽横向越稳但质心移动上限略降；单脚（一脚 z>0.05）能力减半。
- 垂向能力 ~ 腿部剩余伸展量：head_z 越接近站立 (~1.65m)
  向下储备越少；深蹲时向上储备大。
- v_xy / v_z 为该姿态下 COM 速度包络上限 (m/s)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_STAND_D = 0.25          # nominal stance foot separation (m)
_LIFT_Z = 0.12           # foot airborne threshold (m; ankle pts ~0.06)
_STAND_HEAD_Z = 1.65


@dataclass
class Capability:
    """COM-change envelope for a posture (local frame)."""

    v_xy: float          # max horizontal COM speed (m/s)
    v_z: float           # max vertical COM speed (m/s)
    single_support: bool
    stance_d: float      # foot separation used (m)


def capability(local_p5: np.ndarray) -> Capability:
    """Empirical capability for a P5 local pose."""
    p5 = np.asarray(local_p5, np.float64)
    fr, fl = p5[3], p5[4]
    d = float(np.linalg.norm(fr[:2] - fl[:2]))
    single = bool(max(fr[2], fl[2]) > _LIFT_Z)

    v_xy = 1.5 * np.clip(d / _STAND_D, 0.5, 1.3)
    if single:
        v_xy *= 0.5
    crouch = np.clip((_STAND_HEAD_Z - p5[0, 2]) / 0.5, 0.0, 1.0)
    v_z = 0.8 * (0.4 + 0.6 * crouch)
    return Capability(v_xy=float(v_xy), v_z=float(v_z),
                      single_support=single, stance_d=d)
