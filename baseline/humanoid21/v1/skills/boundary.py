"""共享步边界检测 —— 所有技能共用的"一步"边界服务（README §2）。

判定语义（v0，沿用 detect_step_cycles / stepping_state_machine 精神）：

    摆动脚有效离地（contact=False 连续 ≥ min_air_frames 且峰值高度
    ≥ min_peak_height）→ 重新接触（contact=True）并持续承重
    ≥ load_frames 帧 → 本步边界到达，产生 StepEvent。

擦地、瞬时抖动接触不算落地；"承重"v0 用接触持续帧数近似，
升级到足力阈值时只改本文件。每个被追踪的脚独立判定；
跟踪由运行时按 skill.boundary_foot 选择性启用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StepEvent:
    """一次有效脚步边界到达。"""
    foot: str                 # 落地脚 "left"/"right"
    air_frames: int           # 连续离地帧数
    peak_height: float        # 离地峰值高度（踝心高 − 基准）
    steps: int                # 本步总控制步数（离地起点→落地确认）


class _FootTracker:
    _IDLE, _AIRBORNE, _LOADING = 0, 1, 2

    def __init__(self, min_air_frames: int, min_peak_height: float,
                 load_frames: int):
        self._min_air, self._min_peak, self._load_frames = (
            min_air_frames, min_peak_height, load_frames)
        self.reset()

    def reset(self) -> None:
        self.phase = self._IDLE
        self.air_frames = 0
        self.peak = 0.0
        self.load = 0
        self.step_start = 0

    def update(self, contact: bool, height: float, step: int
               ) -> Optional[StepEvent]:
        if self.phase == self._IDLE:
            if not contact:
                self.phase = self._AIRBORNE
                self.air_frames, self.peak, self.step_start = 1, height, step
        elif self.phase == self._AIRBORNE:
            if not contact:
                self.air_frames += 1
                self.peak = max(self.peak, height)
            else:
                # 有效离地成立才进入承重确认；否则整段作废（擦地/抖触）
                if (self.air_frames >= self._min_air
                        and self.peak >= self._min_peak):
                    self.phase, self.load = self._LOADING, 1
                else:
                    self.reset()
        else:  # _LOADING
            if not contact:
                self.reset()          # 承重未坐实又抬起——不算落地
            else:
                self.load += 1
                if self.load >= self._load_frames:
                    ev = StepEvent("", self.air_frames, self.peak,
                                   step - self.step_start + 1)
                    self.reset()
                    return ev
        return None


class StepBoundaryTracker:
    """双脚步边界跟踪器。每控制步喂 (contact, foot_height) 更新。"""

    def __init__(self, min_air_frames: int = 3,
                 min_peak_height: float = 0.05,
                 load_frames: int = 2):
        self._t = {
            "left": _FootTracker(min_air_frames, min_peak_height,
                                 load_frames),
            "right": _FootTracker(min_air_frames, min_peak_height,
                                  load_frames),
        }
        self._step = 0

    def reset(self) -> None:
        for t in self._t.values():
            t.reset()
        self._step = 0

    def update(self, contact_l: bool, contact_r: bool,
               height_l: float, height_r: float) -> Optional[StepEvent]:
        """返回本步产生的 StepEvent 或 None（同帧双脚落地只报先确认的）。"""
        self._step += 1
        for foot, contact, height in (("left", contact_l, height_l),
                                      ("right", contact_r, height_r)):
            ev = self._t[foot].update(bool(contact), float(height),
                                      self._step)
            if ev is not None:
                return StepEvent(foot, ev.air_frames, ev.peak_height,
                                 ev.steps)
        return None
