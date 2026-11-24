"""STEPBUDGET_ATTACK_PLANNER：方向+预算协议的攻击走位规划器。

    obs96 ─→ ego57 + opp 段解析 → BudgetAttackPlanner.update() → cmd22
    obs79 = cat(ego57, cmd22) → 基座 policy（v2_stepbudget）→ action21

与训练时 V2StepBudgetSimulator 完全一致的指令语义：

- step_cur/nxt（cmd22[10:16]/[16:22]）：
    [foot, ux, uy, Δψ, budget/20, 0]
    u  = 世界锚定方向的机体系表示（每帧按当前 yaw 换算）
    Δψ = wrap(ψ* − yaw)：到绝对目标朝向的残差
    cur 预算 = 剩余帧；nxt 预算 = 总预算
- 预览即承诺：nxt 在入队时钉死世界系 dir_w 与 ψ*，到期原样晋升。
- ψ* 累积并夹 |ψ*−yaw_plan| ≤ 45°（与 _make 相同）。
- foot：−1 左动 / +1 右动 / 0 保持；侧移时运动侧脚先迈
  （strafe clip：rel>0 左移 → 左脚先）。

世界锚定需要 ego yaw 与双方根位的世界坐标——Harness 在进程内
直接读 env.simulator 真值（与训练 sim 记账同源）；部署版改为
陀螺仪/里程计积分即可，观测写法不变。

状态机（仅走位——本基座无臂令）：
    APPROACH（d > strike_range）：方向=对手方位，ψ*=对手方位，
        预算按距离分档（远→长）
    ORBIT（close < d ≤ strike_range）：方向=方位+orbit·90° 切向，
        ψ*=对手方位（面向对手绕圈）——对应 circle clip
    RETREAT（d ≤ close_range）：方向=方位+180°，ψ*=对手方位，
        短预算——对应 back clip
    GUARD（upright 崩）：双脚保持，恢复后重规划。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

try:
    from envs.framework.policy import Policy
except Exception:                      # pragma: no cover
    class Policy:                      # type: ignore[no-redef]
        def act(self, observation, *, want_extra=False):
            raise NotImplementedError


BUDGET_MAX = 20
_PSI_CLAMP = math.pi / 4


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


_DEFAULT_PARAMS: Dict[str, Any] = {
    "strike_range": 0.62,    # 进入此根距 → ORBIT
    "close_range": 0.42,     # 更近 → RETREAT（低于攻击距离下限）
    "orbit_sign": +1,        # +1 左侧切向（逆时针绕对手）
    "bud_far": 11,           # d > 1.2：长预算大步逼近
    "bud_mid": 8,            # 0.9 < d ≤ 1.2
    "bud_near": 5,           # 0.62 < d ≤ 0.9：短预算精调
    "bud_orbit": 8,          # 绕行预算
    "bud_retreat": 5,        # 后撤预算
    "guard_z": 0.45,         # upright < 此 → GUARD
    "guard_recover": 0.60,
    "guard_frames": 10,
    "settle_frames": 8,      # 开局长令前保持帧数
}


def _mk(foot: int, dir_w: Optional[float], psi: float,
        budget: int) -> Dict[str, Any]:
    return {"foot": int(foot), "dir": dir_w, "psi": float(psi),
            "budget": int(budget)}


def _vec(c: Dict[str, Any], yaw: float, bud: float) -> np.ndarray:
    """与 V2StepBudgetSimulator._vec 逐字一致。"""
    v = np.zeros(6, np.float32)
    v[0] = float(c["foot"])
    if c["dir"] is not None:
        a = c["dir"] - yaw
        v[1], v[2] = math.cos(a), math.sin(a)
    v[3] = _wrap(c["psi"] - yaw)
    v[4] = float(bud) / BUDGET_MAX
    return v


class BudgetAttackPlanner:
    """每帧调用 update(obs96, yaw, my_xy, opp_xy) → cmd22。"""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.p = dict(_DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.state = "APPROACH"
        self._guard_run = 0
        self._last_foot = 0
        self.cur: Optional[Dict[str, Any]] = None
        self.nxt: Optional[Dict[str, Any]] = None
        self._age = 0

    # ------------------------------------------------------------ 令 -----
    def reset(self, yaw: float) -> None:
        self.state = "APPROACH"
        self._guard_run = 0
        self._last_foot = 0
        self._age = 0
        self.cur = _mk(0, None, yaw, self.p["settle_frames"])
        self.nxt = None

    def _alt_foot(self) -> int:
        self._last_foot = -self._last_foot if self._last_foot else -1
        return self._last_foot

    def _side_foot(self, rel: float) -> int:
        """侧移运动侧脚先迈（sin(rel)>0 左移 → 左脚 −1）。"""
        f = -1 if math.sin(rel) > 0 else 1
        self._last_foot = f
        return f

    def _plan(self, yaw: float, d: float, bearing: float
              ) -> Dict[str, Any]:
        """规划一条新 nxt：钉死世界 dir_w 与 ψ*。"""
        p = self.p
        if d > p["strike_range"]:
            psi = yaw + float(np.clip(_wrap(bearing - yaw),
                                      -_PSI_CLAMP, _PSI_CLAMP))
            bud = (p["bud_far"] if d > 1.2 else
                   p["bud_mid"] if d > 0.9 else p["bud_near"])
            return _mk(self._alt_foot(), bearing, psi, bud)
        if d <= p["close_range"]:
            psi = yaw + float(np.clip(_wrap(bearing - yaw),
                                      -_PSI_CLAMP, _PSI_CLAMP))
            return _mk(self._alt_foot(), bearing + math.pi, psi,
                       p["bud_retreat"])
        # ORBIT：切向迈步 + 面向圆心（ψ*=对手方位）
        rel = self.p["orbit_sign"] * math.pi / 2
        dir_w = bearing + rel
        psi = yaw + float(np.clip(_wrap(bearing - yaw),
                                  -_PSI_CLAMP, _PSI_CLAMP))
        return _mk(self._side_foot(rel), dir_w, psi,
                   p["bud_orbit"])

    # ---------------------------------------------------------- 主入口 ---
    def update(self, obs: np.ndarray, yaw: float,
               my_xy: np.ndarray, opp_xy: np.ndarray) -> np.ndarray:
        p = self.p
        rel_opp = np.asarray(opp_xy, np.float64) - np.asarray(
            my_xy, np.float64)
        d = float(np.linalg.norm(rel_opp))
        bearing = float(math.atan2(rel_opp[1], rel_opp[0]))
        upright = -float(obs[44])

        # ---- 状态机（仅影响"下一条令"怎么规划）----
        if upright < p["guard_z"]:
            self.state = "GUARD"
        elif self.state == "GUARD":
            self._guard_run += 1
            if upright > p["guard_recover"] and \
                    self._guard_run >= p["guard_frames"]:
                self.state, self._guard_run = "APPROACH", 0
        elif d > p["strike_range"]:
            self.state = "APPROACH"
        elif d <= p["close_range"]:
            self.state = "RETREAT"
        else:
            self.state = "ORBIT"

        if self.cur is None:
            self.reset(yaw)

        # ---- 时钟：到期 → cur←nxt 原样晋升 → 规划新 nxt ----
        self._age += 1
        if self._age >= self.cur["budget"]:
            if self.nxt is not None:
                self.cur = self.nxt
            else:
                self.cur = _mk(0, None, yaw, p["bud_near"])
            self._age = 0
            self.nxt = None
        if self.nxt is None:
            if self.state == "GUARD":
                self.nxt = _mk(0, None, yaw, p["bud_near"])
            else:
                self.nxt = self._plan(yaw, d, bearing)

        c = np.zeros(22, np.float32)
        c[10:16] = _vec(self.cur, yaw,
                        self.cur["budget"] - self._age)
        c[16:22] = _vec(self.nxt, yaw, self.nxt["budget"])
        return c


class BudgetAttackPolicy(Policy):
    """obs96 + 世界真值 → planner → cmd22 → obs79 → 基座 → action21。"""

    def __init__(self, base: Policy,
                 planner: BudgetAttackPlanner) -> None:
        self.base = base
        self.planner = planner
        self._yaw_fn = None   # () -> (yaw, my_xy, opp_xy)

    def attach_world(self, fn) -> None:
        """fn() → (ego_yaw, ego_root_xy, opp_root_xy)。"""
        self._yaw_fn = fn

    def reset(self, seed: int = 0) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset(seed)
        if self._yaw_fn is not None:
            yaw, _, _ = self._yaw_fn()
            self.planner.reset(yaw)

    def act(self, observation: np.ndarray, *,
            want_extra: bool = False):
        yaw, my_xy, opp_xy = self._yaw_fn()
        cmd = self.planner.update(observation, yaw, my_xy, opp_xy)
        obs79 = np.concatenate([observation[:57], cmd])
        act, extra = self.base.act(obs79, want_extra=want_extra)
        if want_extra:
            extra = dict(extra or {})
            extra.update({"cmd22": cmd, "planner": self.planner.state})
            return act, extra
        return act, None


__all__ = ["BudgetAttackPlanner", "BudgetAttackPolicy"]
