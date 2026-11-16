"""V1 技能契约基类（v0 最简版）。

``Skill`` = 无状态技能规约：描述"这个技能是什么"，不是控制器——
执行器是共享的技能条件策略。谓词（entry_ok/completed/failed）在训练、
评测、可行性查询中复用同一份实现。

本文件只固定已决策的部分：D1 的 6 维编码、"每技能自定义完成态"、
以及 locate() 锚定接口。步边界判据、本步上下文、运行态结构等
语义由具体技能实现倒逼后再长进来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from point5.cpose import CPose, CPoseClip
from point5.humanoid21.robot_state import RobotState


@dataclass(frozen=True)
class AnchorMatch:
    """locate() 结果：最接近的参考帧索引与匹配分。index=-1 表示无参考序列。"""
    index: int
    score: float


class Skill:
    """技能契约基类。

    类属性：
      name        技能名
      instruction 6 维技能码（DECISION_LOG D1）
      reference   关键帧参考序列（CPoseClip），可无
      step_budget 步时上限（控制步数；README §2 要求必须有界）

    谓词输入统一为 ``RobotState``（point5）：cpose / support_geometry /
    foot_height 都由它派生；obs 路径经 ``RobotState.from_obs`` 桥接。
    触地按 ``foot_height < 0.05`` 的现有惯例判。
    """

    name: str = ""
    instruction: np.ndarray = np.zeros(6, np.float64)
    reference: Optional[CPoseClip] = None
    step_budget: int = 40

    # ── 判定谓词（默认"未声明"，由具体技能定义）──
    def entry_ok(self, s: RobotState) -> bool:
        """静态入口必要条件（如"迈右脚要求左脚接地"）。默认无限制。"""
        return True

    def completed(self, s: RobotState) -> bool:
        """完成态判定。默认 False——未定义完成态的技能永不完成。"""
        return False

    def failed(self, s: RobotState) -> Optional[str]:
        """失败判定，返回原因或 None。默认无失败条件。"""
        return None

    # ── 锚定（bootstrip Q5：当前状态最接近参考序列哪一帧）──
    def frame_score(self, s: RobotState, index: int) -> float:
        """当前状态对参考帧 index 的匹配分（越大越像）。

        默认 = −cpose_distance(live, reference[index])；
        技能可覆写换用自己的度量（如打击类加权手部向量）。"""
        return -cpose_distance(s.to_cpose(), self.reference[index])

    def locate(self, s: RobotState, start: int = 0,
               end: Optional[int] = None) -> AnchorMatch:
        """当前状态最接近参考序列哪一帧。

        ``start=0, end=None`` 全局搜索（起动/改选时重锚定）；
        限定 ``[start, end)`` 窗口即进度跟踪。无 reference 返回
        AnchorMatch(-1)。
        """
        ref = self.reference
        if ref is None or len(ref) == 0:
            return AnchorMatch(-1, 0.0)
        lo = max(0, int(start))
        hi = len(ref) if end is None else min(len(ref), int(end))
        if lo >= hi:
            return AnchorMatch(-1, float("-inf"))
        best_i, best_s = lo, float("-inf")
        for i in range(lo, hi):
            sc = self.frame_score(s, i)
            if sc > best_s:
                best_i, best_s = i, sc
        return AnchorMatch(best_i, best_s)


def cpose_distance(a: CPose, b: CPose) -> float:
    """两 CPose 的距离（v0：四肢端向量差 + 躯干/骨盆朝向差，未加权）。"""
    d = sum(float(np.linalg.norm(a.limb(n) - b.limb(n)))
            for n in ("left_hand", "right_hand",
                      "left_ankle", "right_ankle"))
    d += float(np.linalg.norm(a.torso_pry - b.torso_pry)
               + np.linalg.norm(a.pelvis_pry - b.pelvis_pry))
    return d
