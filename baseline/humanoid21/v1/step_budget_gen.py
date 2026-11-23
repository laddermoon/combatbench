"""STEPBUDGET 步令生成器：Clip 随机组合 + 20% 逐步随机（STEPBUDGET_V1_PLAN §5）。

一条步令 = (动脚, 方向, 朝向目标增量, 时间预算)：
    foot : −1 左脚动 / +1 右脚动 / 0 保持（双脚不动）
    rel  : 迈步方向，相对**本令目标朝向**的角度 rad（+左）；
           None = 原地踏步（方向向量 (0,0)）
    dpsi : 目标朝向增量 rad（相对上一条令的目标朝向，累积成
           绝对目标 ψ*，由 sim 维护）
    budget: 帧数 [BUDGET_MIN, BUDGET_MAX]

方向锚在目标朝向系——与站姿包络同系（§3：轴心脚 + 目标终定
朝向），sim 在令成为 preview 时即把 ψ*/方向钉到世界系（预览
即承诺）。

本模块纯 numpy/math，无 sim 依赖：可单测、可脚印动画。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

BUDGET_MIN, BUDGET_MAX = 3, 20
P_RANDOM = 0.20                  # 逐步纯随机替换概率
DIR_NOISE = math.radians(15.0)   # 方向扰动 σ
BUDGET_JITTER = 2                # 预算抖动 ±帧
DPSI_STEP_MAX = math.radians(30.0)  # 单令朝向增量上限

# 站姿包络（D12/D14 10 态：L2W/R2W 终态，轴心脚 + 目标朝向系）
ENV_SAG = 0.528                  # |纵向错位| ≤ 2 档 × 0.264
ENV_LAT_MAX = 0.625              # 横向脚距 ≤ WIDTH_W
ENV_LAT_MIN = 0.25               # 横向脚距 ≥（防交叉腿）

_D = math.radians


@dataclass(frozen=True)
class BudgetCmd:
    foot: int = 0
    rel: Optional[float] = None
    dpsi: float = 0.0
    budget: int = 10
    clip: int = -1               # 来源 clip 索引；−1 = 纯随机
    rand: bool = False


# ---------------------------------------------------------------------------
# Clip 库：每个 clip 是 (rng, last_foot) → List[(foot, rel, dpsi, budget)]
# ---------------------------------------------------------------------------

Step = Tuple[int, Optional[float], float, int]


def _alt(first: int, n: int) -> List[int]:
    return [first if i % 2 == 0 else -first for i in range(n)]


def _first_alt(rng, last_foot: int) -> int:
    return -last_foot if last_foot != 0 else int(rng.choice([-1, 1]))


def _bud(rng, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _clip_straight(rng, lf) -> List[Step]:
    """直行/斜行：方向 ±30°（非正前），朝向锁定。"""
    n = _bud(rng, 4, 10)
    rel = float(rng.uniform(-_D(30), _D(30)))
    b = _bud(rng, 8, 14)
    return [(f, rel, 0.0, b) for f in _alt(_first_alt(rng, lf), n)]


def _clip_fast(rng, lf) -> List[Step]:
    """快走：短预算，方向 ±20°，朝向锁定。"""
    n = _bud(rng, 4, 10)
    rel = float(rng.uniform(-_D(20), _D(20)))
    b = _bud(rng, 3, 7)
    return [(f, rel, 0.0, b) for f in _alt(_first_alt(rng, lf), n)]


def _clip_arc(rng, lf) -> List[Step]:
    """弧线行走：朝向逐步旋转，方向跟随朝向（rel=0）。"""
    n = _bud(rng, 4, 10)
    s = float(rng.choice([-1, 1]))
    d = s * float(rng.uniform(_D(8), _D(20)))
    b = _bud(rng, 8, 14)
    return [(f, 0.0, d, b) for f in _alt(_first_alt(rng, lf), n)]


def _clip_strafe(rng, lf) -> List[Step]:
    """侧移：±90°±20°，运动侧脚先迈，朝向锁定。"""
    n = _bud(rng, 3, 6)
    s = float(rng.choice([-1, 1]))           # +1 左移
    rel = s * _D(90) + float(rng.uniform(-_D(20), _D(20)))
    lead = -1 if s > 0 else +1               # 左移 → 左脚先
    b = _bud(rng, 5, 10)
    return [(f, rel, 0.0, b) for f in _alt(lead, n)]


def _clip_back(rng, lf) -> List[Step]:
    """斜后撤：后方 ±45°，朝向锁定。"""
    n = _bud(rng, 2, 5)
    rel = math.pi + float(rng.uniform(-_D(45), _D(45)))
    b = _bud(rng, 8, 14)
    return [(f, rel, 0.0, b) for f in _alt(_first_alt(rng, lf), n)]


def _clip_circle(rng, lf) -> List[Step]:
    """绕点侧移：切向迈步 + 朝向持续转向圆心（前方）。

    左移（rel=+90°）绕前方圆心 → 身体右转（dpsi<0）。"""
    n = _bud(rng, 4, 8)
    s = float(rng.choice([-1, 1]))
    d = -s * float(rng.uniform(_D(8), _D(20)))
    lead = -1 if s > 0 else +1
    b = _bud(rng, 8, 14)
    return [(f, s * _D(90), d, b) for f in _alt(lead, n)]


def _clip_turn(rng, lf) -> List[Step]:
    """原地转身：原地踏步（方向 0），总转角 45°–120° 均摊。"""
    tot = float(rng.choice([-1, 1])) * float(rng.uniform(_D(45), _D(120)))
    n = max(2, min(5, int(math.ceil(abs(tot) / DPSI_STEP_MAX))))
    b = _bud(rng, 5, 10)
    return [(f, None, tot / n, b)
            for f in _alt(_first_alt(rng, lf), n)]


def _clip_shuffle(rng, lf) -> List[Step]:
    """碎步调整：随机方向、短预算、朝向微调。"""
    n = _bud(rng, 2, 6)
    out: List[Step] = []
    for f in _alt(_first_alt(rng, lf), n):
        out.append((f, float(rng.uniform(-math.pi, math.pi)),
                    float(rng.uniform(-_D(10), _D(10))),
                    _bud(rng, 3, 5)))
    return out


def _clip_lunge_front(rng, lf) -> List[Step]:
    """前弓 + 回收：左领/右领——同脚斜前长预算，再沿反向收回。"""
    f = int(rng.choice([-1, 1]))
    side = 1.0 if f < 0 else -1.0            # 左脚领 → 偏左前
    rel = side * float(rng.uniform(_D(0), _D(30)))
    return [(f, rel, 0.0, _bud(rng, 14, 20)),
            (f, rel + math.pi, 0.0, _bud(rng, 14, 20))]


def _clip_lunge_side(rng, lf) -> List[Step]:
    """侧弓 + 回收：同侧脚横向长预算外迈，再收回。"""
    f = int(rng.choice([-1, 1]))
    rel = (_D(90) if f < 0 else -_D(90)) + float(
        rng.uniform(-_D(15), _D(15)))
    return [(f, rel, 0.0, _bud(rng, 14, 20)),
            (f, rel + math.pi, 0.0, _bud(rng, 14, 20))]


def _clip_hold(rng, lf) -> List[Step]:
    """保持：双脚不动。"""
    return [(0, None, 0.0, _bud(rng, 5, 15))
            for _ in range(_bud(rng, 1, 3))]


CLIPS: List[Tuple[str, float, Callable]] = [
    ("straight", 0.15, _clip_straight),
    ("fast", 0.10, _clip_fast),
    ("arc", 0.10, _clip_arc),
    ("strafe", 0.10, _clip_strafe),
    ("back", 0.08, _clip_back),
    ("circle", 0.08, _clip_circle),
    ("turn", 0.08, _clip_turn),
    ("shuffle", 0.10, _clip_shuffle),
    ("lunge_front", 0.07, _clip_lunge_front),
    ("lunge_side", 0.07, _clip_lunge_side),
    ("hold", 0.07, _clip_hold),
]
CLIP_NAMES = [c[0] for c in CLIPS]
_W = np.array([c[1] for c in CLIPS], np.float64)
_W = _W / _W.sum()


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class BudgetCmdGen:
    """逐条产出 BudgetCmd 的有状态生成器（一机一个）。"""

    rng: np.random.Generator
    p_random: float = P_RANDOM
    _queue: List[Tuple[int, Step]] = field(default_factory=list)
    _last_foot: int = 0

    def next(self) -> BudgetCmd:
        if not self._queue:
            k = int(self.rng.choice(len(CLIPS), p=_W))
            self._queue = [(k, s) for s in
                           CLIPS[k][2](self.rng, self._last_foot)]
        k, (foot, rel, dpsi, b) = self._queue.pop(0)
        rand = bool(self.rng.uniform() < self.p_random)
        if rand:
            k = -1
            foot = int(self.rng.choice([-1, 1]))
            rel = (None if self.rng.uniform() < 0.1
                   else float(self.rng.uniform(-math.pi, math.pi)))
            dpsi = float(self.rng.uniform(-DPSI_STEP_MAX, DPSI_STEP_MAX))
            b = _bud(self.rng, BUDGET_MIN, BUDGET_MAX)
        else:
            if rel is not None:
                rel = _wrap(rel + float(self.rng.normal(0.0, DIR_NOISE)))
            b = b + int(self.rng.integers(-BUDGET_JITTER,
                                          BUDGET_JITTER + 1))
        b = int(np.clip(b, BUDGET_MIN, BUDGET_MAX))
        dpsi = float(np.clip(dpsi, -DPSI_STEP_MAX, DPSI_STEP_MAX))
        if foot != 0:
            self._last_foot = foot
        else:
            rel = None
        return BudgetCmd(foot=foot, rel=rel, dpsi=dpsi, budget=b,
                         clip=k, rand=rand)


# ---------------------------------------------------------------------------
# 包络几何（目标朝向系，原点 = 轴心脚）
# ---------------------------------------------------------------------------

def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def env_local(p_mv: np.ndarray, p_pv: np.ndarray, yaw_t: float,
              foot: int) -> Tuple[float, float]:
    """动脚相对轴心脚在目标朝向系的 (纵向 s, 外侧横距 l)。

    l > 0 = 动脚在其应在的一侧（左脚在左），l<0 = 交叉。"""
    d = _rot(yaw_t).T @ (np.asarray(p_mv) - np.asarray(p_pv))
    side = 1.0 if foot < 0 else -1.0
    return float(d[0]), float(side * d[1])


def env_violation(s: float, l: float) -> float:
    """到包络 [|s|≤SAG, LAT_MIN≤l≤LAT_MAX] 的距离（0=包络内）。"""
    ds = max(0.0, abs(s) - ENV_SAG)
    dl = max(0.0, ENV_LAT_MIN - l, l - ENV_LAT_MAX)
    return math.hypot(ds, dl)


def ray_exit(s0: float, l0: float, us: float, ul: float) -> float:
    """包络内沿 (us,ul) 方向的最远可达参数 t（起点在外/不进入 → 0）。

    slab 法：返回 t_exit = max t≥0 使 (s0,l0)+t·u 仍在包络内。"""
    t_lo, t_hi = 0.0, math.inf
    for p0, u, lo, hi in ((s0, us, -ENV_SAG, ENV_SAG),
                          (l0, ul, ENV_LAT_MIN, ENV_LAT_MAX)):
        if abs(u) < 1e-9:
            if p0 < lo or p0 > hi:
                return 0.0
            continue
        a, b = (lo - p0) / u, (hi - p0) / u
        if a > b:
            a, b = b, a
        t_lo, t_hi = max(t_lo, a), min(t_hi, b)
    if t_hi < t_lo or not math.isfinite(t_hi):
        return 0.0
    return max(0.0, t_hi)


__all__ = ["BudgetCmd", "BudgetCmdGen", "CLIPS", "CLIP_NAMES",
           "BUDGET_MIN", "BUDGET_MAX", "ENV_SAG", "ENV_LAT_MAX",
           "ENV_LAT_MIN", "env_local", "env_violation", "ray_exit"]
