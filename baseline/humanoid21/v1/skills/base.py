"""V1 技能契约基类 —— 无状态技能规约 + 锚定接口。

设计定位（DECISION_LOG D3 方向，README §2-§4、bootstrip Q4/Q5）：

1. ``Skill`` 是**无状态契约描述对象**：描述"这个技能是什么"，不是控制器。
   执行器是共享的技能条件策略；本类只提供编码、规约与判定谓词。
2. **同一份判定**服务三个消费方：训练（奖励/结算）、评测（成功率/质量）、
   可行性查询（entry_ok 粗筛 + rollout 完成判定）——训练逻辑=执行逻辑。
3. 每 episode 的运行态（锚定进度、事件位、已耗预算）在 ``SkillRuntime``
   中，由运行时（observer）按 env 持有；``Skill`` 单例不存 episode 状态。
4. 步边界检测是共享服务（``boundary.StepBoundaryTracker``），技能只声明
   受哪只脚的落地边界约束（``boundary_foot``）。
5. ``locate()`` = bootstrip Q5 的"重锚定"：查当前状态最接近参考序列
   哪一帧。入口重锚定（全局搜索）与进度跟踪（窗口搜索）共用同一实现。

谓词的统一输入是 ``SkillState``——由 obs 或 accessor 构造的视图，
保证仿真内（训练）与快照分支内（rollout 可行性查询）拿到同一份判定。
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from point5.cpose import CPose, CPoseClip
from point5.humanoid21.robot_state import RobotState
from point5.humanoid21.support_state import SupportState

FOOT_LEFT, FOOT_RIGHT = "left", "right"


# ---------------------------------------------------------------------------
# 状态视图
# ---------------------------------------------------------------------------

class SkillState:
    """谓词消费的统一状态视图。

    包装 ``RobotState``（point5 全系量：qpos/qvel/com/角动量）加接触信息，
    懒加载 ``support``/``cpose``。构造入口二选一：

    - ``from_obs(obs)``：策略观测路径（训练、评测）。obs 取本体段
      [0:52]（qpos/qvel/重力/root_z/速度），内部补零到 96 维走
      ``RobotState.from_obs``。
    - ``from_robot_state(rs, contacts)``：快照/规划路径（可行性查询
      从复制仿真态取 ``RobotState`` 直接构造，不经 obs 编解码）。
    """

    def __init__(self, robot: RobotState,
                 contacts: Tuple[bool, bool] = (True, True)):
        self.robot = robot
        # (left, right) 触地标志——优先用 sim 接触真值；无接触源时
        # 由调用方按 foot_height 阈值估计（与 stepping_state_machine 同源）。
        self.contacts = (bool(contacts[0]), bool(contacts[1]))
        self._support: Optional[SupportState] = None
        self._cpose: Optional[CPose] = None

    @classmethod
    def from_obs(cls, obs: np.ndarray,
                 contacts: Tuple[bool, bool] = (True, True)
                 ) -> "SkillState":
        obs = np.asarray(obs, np.float64)
        if obs.shape[0] < 52:
            raise ValueError(f"obs: need >=52 proprio dims, got {obs.shape}")
        full = np.zeros(96, np.float64)
        full[:52] = obs[:52]
        return cls(RobotState.from_obs(full), contacts)

    @classmethod
    def from_robot_state(cls, robot: RobotState,
                         contacts: Tuple[bool, bool] = (True, True)
                         ) -> "SkillState":
        return cls(robot, contacts)

    @property
    def support(self) -> SupportState:
        """SupportState（com/踝心/foot_height/yaw + 速度/角动量/捕获点）。"""
        if self._support is None:
            self._support = SupportState.from_robot_state(self.robot)
        return self._support

    @property
    def cpose(self) -> CPose:
        if self._cpose is None:
            self._cpose = self.robot.to_cpose()
        return self._cpose

    def foot_height(self, foot: str) -> float:
        g = self.support
        return (g.left_foot_height if foot == FOOT_LEFT
                else g.right_foot_height)

    def contact(self, foot: str) -> bool:
        return self.contacts[0] if foot == FOOT_LEFT else self.contacts[1]


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorMatch:
    """locate() 结果：最接近的参考帧索引与匹配分。

    index=-1 表示无参考序列可锚（skill.reference is None）。
    aligned 由技能自己的 align_threshold 判定；未设阈值时为 None，
    语义"只报告最近邻，不做对齐裁决"。
    """
    index: int
    score: float
    aligned: Optional[bool] = None


@dataclass(frozen=True)
class SkillScore:
    """完成质量：total 供奖励/排序，detail 供诊断与台账。"""
    total: float
    detail: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 完成态规约（bootstrip Q4：CPose 姿态 + SupportState 支撑）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TerminalSpec:
    """技能完成态要求。全部条件同时满足才算完成。

    - ``cpose`` + ``cpose_tol``：完成姿态与容差（cpose_distance 度量）；
      None 表示不限定姿态。
    - ``require_can_balance``：完成态必须 0 步可捕获（支撑核内）。
    - ``max_balance_margin``：balance_margin() 上限；负数表示要求核内
      深度（如 ≤-0.02 = com 在可控 COP 界内留 2cm 余量）。None 不查。
    - ``landed_foot``：要求该脚已完成"有效离地→再触地承重"事件
      （由 StepBoundaryTracker 提供，rt.events 中出现 "landed"）。
    """
    cpose: Optional[CPose] = None
    cpose_tol: float = 0.3
    require_can_balance: bool = True
    max_balance_margin: Optional[float] = None
    landed_foot: Optional[str] = None


# ---------------------------------------------------------------------------
# 每 episode 运行态（由运行时/observer 持有，Skill 单例不存）
# ---------------------------------------------------------------------------

@dataclass
class SkillRuntime:
    """技能运行态：锚定进度、事件位、预算消耗、结局。

    events 约定词汇：``"landed"``（边界脚落地）、``"extended"``（打击
    到位，"本步曾完成出拳"类事件——README §3 本步上下文的事件信息）、
    技能可扩展自定义事件。事件只增不减。
    """
    skill: str
    anchor: int = 0             # 锚定进度指针（reference 帧索引，单调不退）
    events: set = field(default_factory=set)
    steps: int = 0              # 本技能已耗控制步数
    started: bool = False       # entry_ok 曾满足（技能已实质启动）
    done: bool = False
    failed: bool = False
    fail_reason: Optional[str] = None

    def add_event(self, name: str) -> None:
        self.events.add(name)


# ---------------------------------------------------------------------------
# 本步上下文规约（README §3：策略除技能码外需要的最小任务语义）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextSpec:
    """本步上下文向量布局声明。v0 只开三个口子：

    - budget：剩余预算 / step_budget ∈ [0,1]
    - phase：anchor / (len(reference)-1) ∈ [0,1]；无 reference 恒 0
    - events：按 event_names 顺序的 0/1 事件位
    """
    include_budget: bool = True
    include_phase: bool = True
    event_names: Tuple[str, ...] = ()

    @property
    def dim(self) -> int:
        return (int(self.include_budget) + int(self.include_phase)
                + len(self.event_names))


# ---------------------------------------------------------------------------
# Skill 基类
# ---------------------------------------------------------------------------

class Skill(ABC):
    """技能契约基类。

    子类声明（类属性）：
      name / family / instruction / step_budget / terminal /
      reference / boundary_foot / context_spec / lookahead /
      align_threshold（可选）

    子类可覆写（判定语义）：
      entry_ok / completed / failed / score / frame_score /
      cpose_weights / locate_window
    """

    name: str = ""
    family: str = ""
    instruction: np.ndarray = np.zeros(6, np.float64)
    step_budget: int = 40
    terminal: TerminalSpec = TerminalSpec()
    reference: Optional[CPoseClip] = None
    # 该技能受哪只脚的落地边界约束；None=边界语义待技能契约明确
    # （第一阶段直拳技能必须在本契约中给出答案——见 bootstrip Q5）。
    boundary_foot: Optional[str] = None
    context_spec: ContextSpec = ContextSpec()
    # 进度跟踪窗口：每步最多向前看 lookahead 帧（strike_track 惯例=2）
    lookahead: int = 2
    # locate() 的 aligned 阈值；None=不做对齐裁决
    align_threshold: Optional[float] = None

    # ────────────────────────── 判定谓词 ──────────────────────────
    # 训练、评测、可行性查询共用同一实现；全部以 SkillState 为输入。

    def entry_ok(self, s: SkillState) -> bool:
        """静态入口必要条件（便宜判定，非可行性 oracle）。

        例："迈右脚要求左脚接地"。收紧程度注意 STRIKE_LAUNCH_GATE
        教训：过严 → 入口永不满足 → 学习信号断供。
        """
        return True

    def completed(self, s: SkillState, rt: SkillRuntime) -> bool:
        """完成判定：terminal 全部条件 + 边界事件（若声明 landed_foot）。"""
        t = self.terminal
        if t.landed_foot is not None and "landed" not in rt.events:
            return False
        if t.cpose is not None:
            if cpose_distance(s.cpose, t.cpose) > t.cpose_tol:
                return False
        if t.require_can_balance and not s.support.can_balance():
            return False
        if t.max_balance_margin is not None:
            if not (s.support.balance_margin() <= t.max_balance_margin):
                return False
        return True

    def failed(self, s: SkillState, rt: SkillRuntime) -> Optional[str]:
        """失败判定；返回原因字符串或 None。"""
        if rt.steps >= self.step_budget:
            return "budget_exceeded"
        return self.is_fallen(s, rt)

    def is_fallen(self, s: SkillState, rt: SkillRuntime) -> Optional[str]:
        """失控/倒地判定钩子。v0 默认不判——can_balance()=False 只是
        "不迈步无法捕获"，迈步中必然经过，不能当失败。子类按需要覆写
        （如 torso 高度跌出、margin 越界量超阈值持续 N 帧）。"""
        return None

    def score(self, s: SkillState, rt: SkillRuntime) -> SkillScore:
        """完成质量（v0：margin 深度 + 姿态误差 + 耗步占比）。"""
        detail: Dict[str, float] = {}
        margin = float(s.support.balance_margin())
        detail["balance_margin"] = margin
        if self.terminal.cpose is not None:
            detail["cpose_err"] = float(cpose_distance(
                s.cpose, self.terminal.cpose))
        detail["steps_frac"] = rt.steps / max(1, self.step_budget)
        return SkillScore(total=margin, detail=detail)

    # ────────────────────────── 锚定 ──────────────────────────

    def frame_score(self, s: SkillState, index: int) -> float:
        """当前状态对参考帧 index 的匹配分（越大越像）。

        默认 = −cpose_distance(live, reference[index])；
        打击类技能可覆写加权（如手部向量权重上调）。"""
        return -cpose_distance(s.cpose, self.reference[index])

    def locate(self, s: SkillState, start: int = 0,
               end: Optional[int] = None) -> AnchorMatch:
        """查当前状态最接近参考序列哪一帧（bootstrip Q5 重锚定）。

        - ``start=0, end=None``：全局搜索——技能起动/中途改选时的
          重锚定（"当前状态最像序列中哪一帧"）。
        - ``start=rt.anchor, end=rt.anchor+lookahead+1``：窗口搜索——
          进度跟踪（与 advance() 配合实现单调指针）。

        无 reference 时返回 AnchorMatch(-1, 0.0, None)。
        """
        ref = self.reference
        if ref is None or len(ref) == 0:
            return AnchorMatch(-1, 0.0, None)
        lo = max(0, int(start))
        hi = len(ref) if end is None else min(len(ref), int(end))
        if lo >= hi:
            return AnchorMatch(-1, float("-inf"), None)
        best_i, best_s = lo, float("-inf")
        for i in range(lo, hi):
            sc = self.frame_score(s, i)
            if sc > best_s:
                best_i, best_s = i, sc
        aligned = (None if self.align_threshold is None
                   else best_s >= self.align_threshold)
        return AnchorMatch(best_i, best_s, aligned)

    def advance(self, s: SkillState, rt: SkillRuntime) -> AnchorMatch:
        """每控制步推进运行态：窗口 locate → 单调指针 → 预算/结局。

        进度指针只进不退（strike_track 语义）：做得快指针跟得快，
        跟不上就停在原地等。返回本步 locate 结果供诊断。"""
        rt.steps += 1
        ref = self.reference
        if ref is not None and len(ref) > 0:
            m = self.locate(s, start=rt.anchor,
                            end=rt.anchor + self.lookahead + 1)
            if m.index >= 0:
                rt.anchor = max(rt.anchor, m.index)
        else:
            m = AnchorMatch(-1, 0.0, None)
        if not rt.started and self.entry_ok(s):
            rt.started = True
        reason = self.failed(s, rt)
        if reason is not None:
            rt.failed, rt.fail_reason = True, reason
        elif self.completed(s, rt):
            rt.done = True
        return m

    # ────────────────────────── 上下文编码 ──────────────────────────

    def encode_context(self, s: SkillState, rt: SkillRuntime) -> np.ndarray:
        """本步上下文向量（接在 instruction 之后进网络）。"""
        spec = self.context_spec
        parts = []
        if spec.include_budget:
            parts.append(1.0 - rt.steps / max(1, self.step_budget))
        if spec.include_phase:
            n = len(self.reference) - 1 if self.reference is not None else 0
            parts.append(rt.anchor / n if n > 0 else 0.0)
        parts.extend(1.0 if e in rt.events else 0.0
                     for e in spec.event_names)
        return np.asarray(parts, np.float64)

    # ────────────────────────── 序列化/查询 ──────────────────────────

    def describe(self) -> Dict[str, Any]:
        """契约摘要：台账、日志、上层查询接口用。"""
        return {
            "name": self.name, "family": self.family,
            "instruction": self.instruction.tolist(),
            "step_budget": self.step_budget,
            "boundary_foot": self.boundary_foot,
            "reference_frames": (len(self.reference)
                                 if self.reference is not None else 0),
            "context_dim": self.context_spec.dim,
        }


# ---------------------------------------------------------------------------
# 默认 CPose 距离（v0：四肢端向量差 + 躯干/骨盆朝向差加权）
# ---------------------------------------------------------------------------

def cpose_distance(a: CPose, b: CPose,
                   limb_w: float = 1.0, ori_w: float = 0.5) -> float:
    """两 CPose 的距离。单位混合（肢端向量长=LMax 比例、朝向=rad），
    权重为 v0 约定，技能可经 frame_score 换用自己的度量。"""
    d = sum(float(np.linalg.norm(a.limb(n) - b.limb(n)))
            for n in ("left_hand", "right_hand",
                      "left_ankle", "right_ankle"))
    d += float(np.linalg.norm(a.torso_pry - b.torso_pry)
               + np.linalg.norm(a.pelvis_pry - b.pelvis_pry)) * ori_w
    return d * limb_w
