"""STEPACE 任务令流脚本（STEPACE_V1_PLAN.md §3）。

每机器人一个 TaskScript 实例，由 V2StepAceSimulator 驱动：
每 obs 帧调 should_issue(ctx) 查门控；晋升时调 produce(ctx)
取 (StepCmd, dur_cap)；预览调 peek(ctx)。

门控（gate）：
    land   — 动脚离地后重新触地即换令（步态衔接门控）
    arrive — 首达(t_arrive≥0)即换令（ASAP 上限）
    fixed  — 不门控，dur_cap 到期（保持/转身类用）

纠偏（replan=True）：声明式 P 纠偏、有界——
    行走类：dyaw_cmd = −clip(kh·heading_err + kl·lat_err, ±0.26)
    转角类：dyaw_i = nominal + clip(kθ·(θ_sched − θ_done), ±0.26)
    corrections 记录每次施加的修正量（=策略系统偏差度量）。

ctx（sim 每帧构建）：
    fl,fr,yaw  双脚世界 xy + 躯干 yaw
    z_l,z_r    脚 z；c_l,c_r  脚触地
    age,dur,t_arrive,cur  当前令状态；cmd_i  已发令数
    task0      任务系锚点 {x,y,yaw}（reset 时中点+yaw）
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .step_cmd_gen import StepCmd


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _rot(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], float)


class TaskScript:
    """基类：交替脚序列 + 门控 + P 纠偏 + 修正日志。"""

    name = "base"
    # 子类参数（构造时生效）
    N_CMD = 10            # 任务令数（之后持续保持令）
    GATE = "land"         # land/arrive/fixed
    MIN_AGE = 2           # 门控最小令龄（防瞬时连转）
    DUR_CAP = 12          # 门控任务的兜底时长
    CORR_MAX = 0.26       # ≈15°

    def __init__(self, rng: np.random.Generator,
                 gate: Optional[str] = None,
                 replan: bool = True, **kw: Any) -> None:
        self.rng = rng
        self.gate = gate or self.GATE
        self.replan = bool(replan)
        self._init_params(kw)
        self.log: List[Dict[str, Any]] = []          # 已发令记录
        self.corrections: List[Dict[str, float]] = []  # 修正量记录
        self._task0: Optional[Dict[str, float]] = None
        self._seen_air = False
        self._pending: Optional[Tuple[StepCmd, int]] = None
        self._i = 0                                    # 已发令数
        self._yaw_done = 0.0                           # 累计已确认转角

    # ------------------------------------------------ 子类钩子 ----------
    def _init_params(self, kw: Dict[str, Any]) -> None:
        pass

    def _nominal(self, ctx: Dict[str, Any]) -> Tuple[StepCmd, int]:
        """第 i 条名义令 + dur_cap。"""
        raise NotImplementedError

    def _correct(self, cmd: StepCmd, ctx: Dict[str, Any],
                 preview: bool = False) -> StepCmd:
        """P 纠偏（默认：直线路径保向）。preview=预览，无副作用。"""
        return cmd

    # ------------------------------------------------ 主接口 ----------
    def reset(self, ctx: Dict[str, Any]) -> None:
        mx, my = 0.5 * (ctx["fl"][0] + ctx["fr"][0]), \
                 0.5 * (ctx["fl"][1] + ctx["fr"][1])
        self._task0 = {"x": float(mx), "y": float(my),
                       "yaw": float(ctx["yaw"])}
        self._i = 0
        self._yaw_done = 0.0
        self._pending = None
        self._seen_air = False

    @property
    def done(self) -> bool:
        return self._i >= self.N_CMD

    def should_issue(self, ctx: Dict[str, Any]) -> bool:
        if self.gate == "fixed":
            return False
        c = ctx["cur"]
        if self.gate == "age":
            return ctx["age"] >= self.MIN_AGE   # 固定令龄切换
        if self.gate == "arrive":
            return ctx["t_arrive"] >= 0
        if c is None or c.foot == 0 or ctx["age"] < self.MIN_AGE:
            return False
        c_mv = ctx["c_l"] if c.foot < 0 else ctx["c_r"]
        if self.gate == "air":
            return not c_mv                  # 腾空瞬间切换
        # land 门控：动脚离地→再触地
        if not c_mv:
            self._seen_air = True
        elif self._seen_air:
            return True
        return False

    def produce(self, ctx: Dict[str, Any]) -> Tuple[StepCmd, int]:
        """晋升下一条令：弹出 peek 缓存（peek 即承诺，
        与训练协议 cur←nxt 一致），否则现生成。"""
        self._seen_air = False            # 新令重武装门控
        if self._pending is None:
            self._pending = self._gen(ctx)
        cmd, dur = self._pending
        self._pending = None
        self._i += 1
        if cmd.foot != 0:
            self.log.append({"i": self._i, "foot": cmd.foot,
                             "sag": cmd.sag, "lat": cmd.lat,
                             "dyaw": cmd.dyaw, "dur": dur})
        return cmd, dur

    def peek(self, ctx: Dict[str, Any]) -> StepCmd:
        """预览下一条：生成并缓存 pending——produce 将原样
        弹出它。修正量此刻计算并记一次（决策点）。"""
        if self.done:
            return StepCmd()
        if self._pending is None:
            self._pending = self._gen(ctx)
        return self._pending[0]

    def _gen(self, ctx: Dict[str, Any]) -> Tuple[StepCmd, int]:
        if self.done:
            return StepCmd(), 15          # 任务尾=保持令
        cmd, dur = self._nominal(ctx)
        return self._correct(cmd, ctx), dur

    # ------------------------------------------------ 纠偏原语 ----------
    def _steer_corr(self, ctx: Dict[str, Any],
                    kh: float = 1.0, kl: float = 0.8,
                    preview: bool = False) -> float:
        """直线路径纠偏量（rad）：航向误差+横向偏移 → dyaw。"""
        if not self.replan or self._task0 is None:
            return 0.0
        t0 = self._task0
        head_err = _wrap_pi(ctx["yaw"] - t0["yaw"])
        mid = 0.5 * (ctx["fl"] + ctx["fr"])
        n = _rot(t0["yaw"] + math.pi / 2) @ np.array([1.0, 0.0])
        lat_err = float(n @ (mid - np.array([t0["x"], t0["y"]])))
        corr = -float(np.clip(kh * head_err + kl * lat_err,
                              -self.CORR_MAX, self.CORR_MAX))
        if not preview:
            self.corrections.append({"dyaw": corr, "head": head_err,
                                     "lat": lat_err})
        return corr

    def _yaw_sched_corr(self, ctx: Dict[str, Any],
                        nominal: float,
                        preview: bool = False) -> float:
        """转角计划纠偏：按累计达成补差。"""
        if not self.replan or self._task0 is None:
            return nominal
        yaw_done = _wrap_pi(ctx["yaw"] - self._task0["yaw"])
        sched = self._yaw_sched + nominal
        if not preview:
            self._yaw_sched = sched
        diff = _wrap_pi(sched - yaw_done)
        corr = float(np.clip(diff * 0.6, -self.CORR_MAX,
                             self.CORR_MAX))
        if not preview:
            self.corrections.append({"dyaw": corr, "sched": diff})
        return nominal + corr


class _WalkBase(TaskScript):
    """直线行走基类：sag_rng 正=前 负=后。"""
    N_CMD = 12
    GATE = "land"
    DUR_CAP = 12
    LAT = 0.50

    def _init_params(self, kw: Dict[str, Any]) -> None:
        self._sag_rng = tuple(kw.get("sag_rng", (0.35, 0.50)))
        self._yaw_sched = 0.0

    def _nominal(self, ctx: Dict[str, Any],
                 preview: bool = False) -> Tuple[StepCmd, int]:
        i = self._i + (1 if preview else 0)
        foot = -1 if i % 2 == 0 else +1
        sag = float(self.rng.uniform(*self._sag_rng))
        return StepCmd(foot=foot, sag=sag, lat=self.LAT, dyaw=0.0), \
            self.DUR_CAP

    def _correct(self, cmd: StepCmd, ctx: Dict[str, Any],
                 preview: bool = False) -> StepCmd:
        if cmd.foot == 0:
            return cmd
        return StepCmd(foot=cmd.foot, sag=cmd.sag, lat=cmd.lat,
                       dyaw=self._steer_corr(ctx, preview=preview))


class WalkFwd(_WalkBase):
    name = "walk_fwd"


class WalkBwd(_WalkBase):
    name = "walk_bwd"

    def _init_params(self, kw: Dict[str, Any]) -> None:
        super()._init_params(kw)
        self._sag_rng = tuple(kw.get("sag_rng", (-0.40, -0.25)))


class TurnInPlace(TaskScript):
    """原地转身：交替小步 + 同向 dyaw；转足 θ_total 后保持。"""
    name = "turn_in_place"
    N_CMD = 12
    GATE = "land"
    DUR_CAP = 12
    LAT = 0.55

    def _init_params(self, kw: Dict[str, Any]) -> None:
        self._dyaw_rng = tuple(kw.get("dyaw_rng", (0.14, 0.26)))
        self._dir = float(kw.get("direction", 1.0))
        self._yaw_sched = 0.0

    def _nominal(self, ctx: Dict[str, Any],
                 preview: bool = False) -> Tuple[StepCmd, int]:
        i = self._i + (1 if preview else 0)
        foot = -1 if i % 2 == 0 else +1
        sag = float(self.rng.uniform(-0.05, 0.05))
        dyaw = self._dir * float(self.rng.uniform(*self._dyaw_rng))
        return StepCmd(foot=foot, sag=sag, lat=self.LAT,
                       dyaw=dyaw), self.DUR_CAP

    def _correct(self, cmd: StepCmd, ctx: Dict[str, Any],
                 preview: bool = False) -> StepCmd:
        if cmd.foot == 0:
            return cmd
        return StepCmd(foot=cmd.foot, sag=cmd.sag, lat=cmd.lat,
                       dyaw=self._yaw_sched_corr(ctx, cmd.dyaw,
                                                 preview))


class TurnArc(_WalkBase):
    """行进中转向：前迈步 + 同向 dyaw（弧线）。"""
    name = "turn_arc"
    N_CMD = 10

    def _init_params(self, kw: Dict[str, Any]) -> None:
        super()._init_params(kw)
        self._sag_rng = tuple(kw.get("sag_rng", (0.30, 0.45)))
        self._dyaw_rng = tuple(kw.get("dyaw_rng", (0.14, 0.21)))
        self._dir = float(kw.get("direction", 1.0))

    def _nominal(self, ctx: Dict[str, Any],
                 preview: bool = False) -> Tuple[StepCmd, int]:
        cmd, dur = super()._nominal(ctx, preview)
        dyaw = self._dir * float(self.rng.uniform(*self._dyaw_rng))
        return StepCmd(foot=cmd.foot, sag=cmd.sag, lat=cmd.lat,
                       dyaw=dyaw), dur

    def _correct(self, cmd: StepCmd, ctx: Dict[str, Any],
                 preview: bool = False) -> StepCmd:
        if cmd.foot == 0:
            return cmd
        return StepCmd(foot=cmd.foot, sag=cmd.sag, lat=cmd.lat,
                       dyaw=self._yaw_sched_corr(ctx, cmd.dyaw,
                                                 preview))


class TorsoTwist(TaskScript):
    """脚不动转躯干：名义动脚目标=原位 + dyaw 同向。"""
    name = "torso_twist"
    N_CMD = 6
    GATE = "fixed"          # 必须持续到期——脚不许动
    DUR_CAP = 15

    def _init_params(self, kw: Dict[str, Any]) -> None:
        self._dyaw_rng = tuple(kw.get("dyaw_rng", (0.17, 0.44)))
        self._dir = float(kw.get("direction", 1.0))
        self._yaw_sched = 0.0

    def _nominal(self, ctx: Dict[str, Any],
                 preview: bool = False) -> Tuple[StepCmd, int]:
        i = self._i + (1 if preview else 0)
        foot = -1 if i % 2 == 0 else +1
        dyaw = self._dir * float(self.rng.uniform(*self._dyaw_rng))
        # 目标=动脚当前位置：解 (sag,lat) 使 tgt=p_foot
        yaw_t = ctx["yaw"] + dyaw
        stance = ctx["fr"] if foot < 0 else ctx["fl"]
        pfoot = ctx["fl"] if foot < 0 else ctx["fr"]
        rel = _rot(yaw_t).T @ (np.asarray(pfoot) - np.asarray(stance))
        sag, lat = float(rel[0]), float(rel[1])
        if foot > 0:
            lat = -lat
        return StepCmd(foot=foot, sag=sag, lat=lat, dyaw=dyaw), \
            self.DUR_CAP

    def _correct(self, cmd: StepCmd, ctx: Dict[str, Any],
                 preview: bool = False) -> StepCmd:
        if cmd.foot == 0:
            return cmd
        dyaw = self._yaw_sched_corr(ctx, cmd.dyaw, preview)
        if abs(dyaw - cmd.dyaw) > 1e-6:
            # 纠偏改了 dyaw → 重解 (sag,lat) 保持目标=原位
            yaw_t = ctx["yaw"] + dyaw
            f = cmd.foot
            stance = ctx["fr"] if f < 0 else ctx["fl"]
            pfoot = ctx["fl"] if f < 0 else ctx["fr"]
            rel = _rot(yaw_t).T @ (
                np.asarray(pfoot) - np.asarray(stance))
            lat = float(rel[1]) if f < 0 else -float(rel[1])
            return StepCmd(foot=f, sag=float(rel[0]), lat=lat,
                           dyaw=dyaw)
        return cmd


class Sprint(_WalkBase):
    """最快前进：固定步幅交替步 + 可扫切换门控。

    变量（构造 kw）：
        sag      固定步幅 m（默认 0.40）
        min_age  门控武装令龄：gate=age 时为切换帧
        gate     air=腾空即换 / age=令龄到即换 / land=落地换
    半空换令时新令锚在悬空轴心脚的当前位置（D20 语义）。
    """
    name = "sprint"
    N_CMD = 16
    GATE = "air"
    MIN_AGE = 2
    DUR_CAP = 14

    def _init_params(self, kw: Dict[str, Any]) -> None:
        super()._init_params(kw)
        self._sag = float(kw.get("sag", 0.40))
        self.MIN_AGE = int(kw.get("min_age", self.MIN_AGE))

    def _nominal(self, ctx: Dict[str, Any],
                 preview: bool = False) -> Tuple[StepCmd, int]:
        i = self._i + (1 if preview else 0)
        foot = -1 if i % 2 == 0 else +1
        return StepCmd(foot=foot, sag=self._sag, lat=self.LAT,
                       dyaw=0.0), self.DUR_CAP


_REGISTRY = {c.name: c for c in
             (WalkFwd, WalkBwd, TurnInPlace, TurnArc, TorsoTwist,
              Sprint)}


def make_script(name: str, rng: np.random.Generator,
                **kw: Any) -> TaskScript:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"unknown task script '{name}' "
                       f"(available={sorted(_REGISTRY)})")
    return cls(rng, **kw)


def list_scripts() -> List[str]:
    return sorted(_REGISTRY)


__all__ = ["TaskScript", "make_script", "list_scripts"]
