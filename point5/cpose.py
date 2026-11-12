"""CPose — 标准化姿态协议（身体局部系版）。

与 P5 的关系：P5 = landmark 的世界系坐标（意图协议）；
CPose = 肢体端点的身体局部系向量（姿态策略 IR）。
两者并存，各自发展。

姿态 = Pelvis(位置+朝向) + Torso(相对 Pelvis 的朝向) + 4 肢端向量：
  手向量在 torso 系（原点=肩旋转中心），踝向量在 pelvis 系（原点=髋
  旋转中心）。向量单位 = 该肢 LMax（拉直时两端旋转中心距离），
  故 |v| = 伸展度 ∈ (0,1]——最大为 1，最小不为 0。

轴约定（所有局部系一致）：X+ 身前、Y+ 向左、Z+ 向头——全部右手系。
朝向语义：pry = (pitch, roll, yaw) rad——三个意图通道，参考约定
  R = Rz(yaw)·Ry(pitch)·Rx(roll)：pitch>0 前倾、roll>0 右倾、yaw>0 左转。
  三元组只携带意图，机械解释（躯干可含任意复杂度铰链、可无明确
  torso；关节限位、映射顺序）由各机型适配器自定，不在协议内。

本协议只定义数据与语义，不含几何运算——旋转矩阵、插值、镜像等
由消费方按本约定实现。构造只做形状/合法性校验，不改写输入。

显式不表达（下游先验补全）：踝策略/脚朝向、髋自旋、头部姿态。
接地与 Location 语义由持有方约定；纯形状用法可省略 pelvis_* 字段。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

_PT = (3,)
_ZERO3 = np.zeros(3)
_EXT_MAX = 1.0 + 1e-6          # 伸展度上限（含数值容差）
_EXT_MIN = 1e-9                # 伸展度 >0——最小不为 0

# 规范肢序：索引稳定协议——[0,1]左/右手 [2,3]左/右踝
LIMB_NAMES = ("left_hand", "right_hand", "left_ankle", "right_ankle")
LIMB_INDEX = {n: i for i, n in enumerate(LIMB_NAMES)}
# 各肢向量所在的父刚体系：手→torso 系，踝→pelvis 系
LIMB_FRAME = {"left_hand": "torso", "right_hand": "torso",
              "left_ankle": "pelvis", "right_ankle": "pelvis"}


def _vec(v, name):
    v = np.array(v, np.float64)          # copy, never alias
    if v.shape != _PT:
        raise ValueError(f"{name}: expected {_PT}, got {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError(f"{name}: non-finite values")
    return v


@dataclass(frozen=True)
class CPose:
    """CPose 姿态 — 躯干二元结构 + 4 肢端局部向量。

    字段语义：
      left_hand/right_hand : 手球心−肩旋转中心，torso 系分量，单位=LMax
      left_ankle/right_ankle: 踝心−髋旋转中心，pelvis 系分量，单位=LMax
      torso_pry            : torso 相对 pelvis 朝向意图（p/r/y 三通道；
                            躯干实现与分解由机型适配器定）
      pelvis_pry/pelvis_pos: pelvis 在所在帧的朝向/位置——Location
                            语义由持有方约定；纯形状可省略（缺省恒零）
    """
    left_hand: np.ndarray
    right_hand: np.ndarray
    left_ankle: np.ndarray
    right_ankle: np.ndarray
    torso_pry: np.ndarray
    pelvis_pry: np.ndarray = field(default_factory=_ZERO3.copy)
    pelvis_pos: np.ndarray = field(default_factory=_ZERO3.copy)

    def __post_init__(self):
        for n in LIMB_NAMES:
            v = _vec(getattr(self, n), n)
            e = float(np.linalg.norm(v))
            if not (_EXT_MIN < e <= _EXT_MAX):
                raise ValueError(f"{n}: ext {e:.4g} out of (0,1]")
            object.__setattr__(self, n, v)
        for n in ("torso_pry", "pelvis_pry", "pelvis_pos"):
            object.__setattr__(self, n, _vec(getattr(self, n), n))

    # ---- 肢端 ----
    def limb(self, name: str) -> np.ndarray:
        """肢端向量（3,)——所在父刚体系分量，单位=LMax。"""
        return getattr(self, name)

    def limb_extension(self, name: str) -> float:
        """肢端伸展度 = |v| ∈ (0,1]。"""
        return float(np.linalg.norm(getattr(self, name)))

    def limb_direction(self, name: str) -> Tuple[float, float]:
        """肢端方向的极坐标分解 (azimuth, polar)。

        azimuth = atan2(y, x) ∈ (-π,π]：0=正前、+π/2=正左、±π=正后。
        polar   = arccos(z/|v|) ∈ [0,π]：0=向头(+z)、π/2=水平、π=下垂。
        """
        v = getattr(self, name)
        x, y, z = v / np.linalg.norm(v)
        return (float(np.arctan2(y, x)),
                float(np.arccos(np.clip(z, -1.0, 1.0))))


@dataclass(frozen=True)
class CPoseClip:
    """CPose 序列 — 全部帧共用同一帧语义。可索引/可迭代。"""
    frames: Tuple[CPose, ...]

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, i):
        return self.frames[i]
