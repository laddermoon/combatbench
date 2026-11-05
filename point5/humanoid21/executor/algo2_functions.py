"""算法2 功能点 f1-f5 —— alg2_infoflow.md / algo2_meta.py 的实现层。

    f1(landmarks, clip) -> I2     锚点 P5 求取
    f2(i1, i2, i8)      -> I3     下一帧 P5 目标
    f3(i3)              -> I5     P5 → 关节姿态（工具2）
    f4(i6, i7)          -> I8     下一帧质心目标
    f5(i8, i5)          -> R      求解器本体

当前全部为空壳——仅落签名。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from point5.contracts import P5Clip
from .algo2_meta import I1, I2, I3, I4, I5, I6, I7, I8, R
from .contracts import LandMark


def f1(landmarks: List[LandMark], clip: P5Clip) -> I2:
    """f1(landmarks, clip) -> I2。

    姿态锚点：frame 链接到 clip 关键帧取 pos；
    接触锚点：由锚点质心态 + 先验/后续姿态 P5 补全。
    """
    raise NotImplementedError


def f2(i1: I1, i2: I2, i8: I8, i6: I6, i7: I7) -> I3:
    """f2(I1, I2, I8, I6, I7) -> I3。

    当前 P5 → 锚点 P5 的过渡，须与质心轨迹一致（I8 输入）；
    I8 的进度语义由 I6→I7 路径定义。
    """
    raise NotImplementedError


def f3(i3: I3, i4: I4, i8: I8) -> I5:
    """f3(I3, I4, I8) -> I5：P5 + 约束 → 关节姿态。

    I3 参考姿态：生成的姿态不能离它太远；
    I4 新姿态的 5 点约束，同时可经 pose_solver 给参考动作；
    I8 质心位置约束。
    多约束姿态求解——本链中较难的一个算法。
    """
    raise NotImplementedError


def f4(i6: I6, i7: I7) -> I8:
    """f4(I6, I7) -> I8：当前质心 → 锚点质心的下一帧目标。"""
    raise NotImplementedError


def f5(i8: I8, i5: I5) -> R:
    """f5(I8, I5) -> R：求解器本体——目标 → 动作。

    候选 = 跟踪控制器参数，在内部仿真器闭环 rollout 评估；
    反馈信号来自内部 sim 输出，非信息项。
    """
    raise NotImplementedError
