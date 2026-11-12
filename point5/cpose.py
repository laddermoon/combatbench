"""CPose — 标准化姿态协议（身体局部系版）。

与 P5 的关系：P5 = landmark 的世界系坐标（意图协议）；
CPose = 肢体端点的身体局部系向量（姿态策略 IR）。
两者并存，各自发展。

姿态 = Pelvis(位置+朝向) + Torso(相对 Pelvis 的朝向) + 4 肢端向量：
  手向量在 torso 系（原点=肩旋转中心），踝向量在 pelvis 系（原点=髋
  旋转中心）。向量单位 = 该肢 LMax（拉直时两端旋转中心距离），
  故 |v| = 伸展度 ∈ (0,1]——最大为 1，最小不为 0。

轴约定（所有局部系一致）：X+ 身前、Y+ 向左、Z+ 向头——全部右手系。
朝向编码：pry = (pitch, roll, yaw) rad，R = Rz(yaw)·Ry(pitch)·Rx(roll)，
  标准分支 cos(pitch)>=0（与 Pose.root_orientation 同约定）；
  pitch>0 前倾（绕 +y 左轴）、roll>0 右倾（绕 +x 前轴）。

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


def _wrap(a: float) -> float:
    """角度归一到 (-pi, pi]。"""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _canonical_pry(pry: np.ndarray) -> np.ndarray:
    """[pitch,roll,yaw] -> 标准 ZYX 分支（cos(pitch)>=0）的等价角。

    非标准分支（cos(pitch)<0）的同一旋转写成
    [pi-pitch, roll+pi, yaw+pi]；奇异点 cos(pitch)=0 原样保留。
    """
    p, r, y = pry
    if np.cos(p) < 0.0:
        p, r, y = np.pi - p, r + np.pi, y + np.pi
    return np.array([_wrap(p), _wrap(r), _wrap(y)])


def pry_to_mat(pry) -> np.ndarray:
    """(pitch,roll,yaw) -> R = Rz(yaw)·Ry(pitch)·Rx(roll)。"""
    p, r, y = (float(v) for v in pry)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def mat_to_pry(R) -> np.ndarray:
    """R -> (pitch,roll,yaw) 标准分支，pry_to_mat 之逆。

    奇异 |pitch|=pi/2：roll 归 0，自由度全部折进 yaw。
    """
    R = np.asarray(R, np.float64)
    sp = float(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(sp) >= 1.0 - 1e-12:
        pitch = np.copysign(np.pi / 2.0, sp)
        roll, yaw = 0.0, float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        pitch = float(np.arcsin(sp))
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([pitch, roll, yaw])


@dataclass(frozen=True)
class CPose:
    """CPose 姿态 — 躯干二元结构 + 4 肢端局部向量。

    字段语义：
      left_hand/right_hand : 手球心−肩旋转中心，torso 系分量，单位=LMax
      left_ankle/right_ankle: 踝心−髋旋转中心，pelvis 系分量，单位=LMax
      torso_pry            : torso 相对 pelvis 朝向（ZYX 标准分支；
                            R_torso = R_pelvis @ pry_to_mat(torso_pry)）
      pelvis_pry/pelvis_pos: pelvis 在所在帧的朝向/位置——Location
                            语义由持有方约定；纯形状可省略（缺省恒零）
    pry 输入构造时规范化到标准分支（同 Pose.root_orientation）。
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
            v = _vec(getattr(self, n), n)
            if n != "pelvis_pos":
                v = _canonical_pry(v)
            object.__setattr__(self, n, v)

    # ---- 帧 ----
    @property
    def pelvis_mat(self) -> np.ndarray:
        """pelvis 系 -> 所在帧 的旋转。"""
        return pry_to_mat(self.pelvis_pry)

    @property
    def torso_rel_mat(self) -> np.ndarray:
        """torso 系 -> pelvis 系 的旋转。"""
        return pry_to_mat(self.torso_pry)

    @property
    def torso_mat(self) -> np.ndarray:
        """torso 系 -> 所在帧 的旋转。"""
        return self.pelvis_mat @ self.torso_rel_mat

    def frame_mat(self, limb: str) -> np.ndarray:
        """该肢向量所在父刚体系的旋转矩阵（-> 所在帧）。"""
        return (self.torso_mat if LIMB_FRAME[limb] == "torso"
                else self.pelvis_mat)

    # ---- 肢端 ----
    def limb(self, name: str) -> np.ndarray:
        return getattr(self, name)

    @property
    def vecs(self) -> np.ndarray:
        """(4,3) 肢端向量阵 — LIMB_NAMES 序。"""
        return np.array([getattr(self, n) for n in LIMB_NAMES])

    @property
    def exts(self) -> np.ndarray:
        """(4,) 伸展度 = 各肢向量模长 ∈(0,1]。"""
        return np.linalg.norm(self.vecs, axis=1)

    @property
    def dirs(self) -> np.ndarray:
        """(4,3) 肢端单位方向（各父刚体系内）。"""
        v = self.vecs
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def vec_host(self, name: str) -> np.ndarray:
        """肢端向量表达到所在帧——方向随体旋转、模长不变。"""
        return self.frame_mat(name) @ getattr(self, name)


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
