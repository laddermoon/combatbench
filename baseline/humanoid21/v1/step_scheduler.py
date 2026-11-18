"""步令调度器（S4 Markov 链采样器，`STEP_GAIT.md` 调度侧代码化）。

独立可测模块：不依赖 MuJoCo/仿真器，注入 collision_fn 即接入
碰撞约束层。采样上下文 = 指令态（commanded posture + 上条步令），
不读实测脚位——链本身可复现，机器人没踩准由评分处理。

因子化采样（讨论定稿）：
    ① 脚   P(交替)=p_alternate(0.9)，同脚=调整/跳步覆盖
    ② 转   持续转向：t_prev=0 → {0:0.85, ±:0.075}
                    t_prev≠0 → {续:0.55, 停:0.40, 反:0.05}
    ③ 目标姿态 P'=(s',w') 双轴因子化：
           Δstagger 档位加权 {0:.30,±1:.44,±2:.16,±3:.06,±4:.04}
           × 方向偏置 e=sign(Δs)·foot {前进:1.4, 横/原:1.0, 后撤:0.8}
           Δwidth 独立 Bernoulli(0.12)
    ④ 约束层 collision_fn(order,state)->bool：拒绝重采 ≤N 次，
       失败退化为"踏"（交替脚→当前姿态·不转）。

用法：
    sch = StepScheduler(seed=0)
    st = SchedState.initial(FLAT)          # 首条步令无 last_foot
    o = sch.sample(st, collision_fn=...)   # 采一条步令
    st = sch.advance(st, o)                # 边界轮转后更新上下文
    seq = sch.simulate(st, n=200)          # 造测试序列
    S, M = transition_matrix(sch)          # 物化 20×20 状态矩阵
    pi = stationary(M)                     # 稳态分布（验遍历性）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from baseline.humanoid21.v1.step_gait import (
    FLAT,
    POSTURES,
    GaitPosture,
    StepOrder,
    order_name,
)

# ---------------------------------------------------------------------------
# 参数与上下文
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerParams:
    """全部采样旋钮（热路径=常规走步，稀有项非零保覆盖）。"""

    p_alternate: float = 0.9          # 脚：交替概率
    p_turn_start: float = 0.15        # 转：直行中发起转向总概率
    p_turn_hold: float = 0.55         # 转：转向中继续同向
    p_turn_reverse: float = 0.05      # 转：转向中突然反向
    stagger_w: Tuple[float, ...] = (0.30, 0.44, 0.16, 0.06, 0.04)
    #                                 |Δs| = 0/1/2/3/4 档权
    dir_fwd: float = 1.4              # 方向偏置：动脚前进（e>0）
    dir_back: float = 0.8             #            后撤（e<0）
    p_width_flip: float = 0.12        # 宽距：切换概率
    max_resample: int = 3             # 约束层：拒绝重采上限


@dataclass(frozen=True)
class SchedState:
    """采样上下文（指令态）。last_foot=0 表示无上条步令（ep 首步）。"""

    posture: GaitPosture
    last_foot: int = 0                # −1 左 / +1 右 / 0 无
    last_turn: int = 0                # −1/0/+1，供持续转向

    @staticmethod
    def initial(posture: GaitPosture = FLAT) -> "SchedState":
        return SchedState(posture, 0, 0)


CollisionFn = Callable[[StepOrder, SchedState], bool]
"""碰撞预测：True=该步令预期撞墙/对手，应重采。"""


# ---------------------------------------------------------------------------
# 调度器
# ---------------------------------------------------------------------------


class StepScheduler:
    """因子化 Markov 采样器。rng 注入保证可复现、可单测。"""

    def __init__(
        self,
        seed: int = 0,
        params: SchedulerParams = SchedulerParams(),
    ) -> None:
        self.p = params
        self.rng = np.random.default_rng(seed)

    # -- 因子① 脚 ------------------------------------------------------

    def _foot_probs(self, last_foot: int) -> Dict[int, float]:
        if last_foot == 0:
            return {-1: 0.5, +1: 0.5}
        return {-last_foot: self.p.p_alternate,
                last_foot: 1.0 - self.p.p_alternate}

    # -- 因子② 转向 ----------------------------------------------------

    def _turn_probs(self, last_turn: int) -> Dict[int, float]:
        if last_turn == 0:
            h = self.p.p_turn_start / 2.0
            return {-1: h, 0: 1.0 - self.p.p_turn_start, +1: h}
        return {
            last_turn: self.p.p_turn_hold,
            0: 1.0 - self.p.p_turn_hold - self.p.p_turn_reverse,
            -last_turn: self.p.p_turn_reverse,
        }

    # -- 因子③ 目标姿态 -------------------------------------------------

    def _stagger_probs(self, s: int, foot: int) -> Dict[int, float]:
        """P(s'|s,foot)：|Δs|档权 × 方向偏置 e=sign(Δs)·foot。"""
        w: Dict[int, float] = {}
        for s2 in range(-2, 3):
            ds = s2 - s
            e = (0 if ds == 0 else (1 if ds > 0 else -1)) * foot
            mult = self.p.dir_fwd if e > 0 else (
                self.p.dir_back if e < 0 else 1.0)
            w[s2] = self.p.stagger_w[abs(ds)] * mult
        z = sum(w.values())
        return {k: v / z for k, v in w.items()}

    def _posture_probs(self, posture: GaitPosture,
                       foot: int) -> Dict[GaitPosture, float]:
        """P(P'|posture,foot) = 交错分布 ⊗ 宽距翻转。"""
        sdist = self._stagger_probs(posture.stagger, foot)
        out: Dict[GaitPosture, float] = {}
        for s2, ps in sdist.items():
            for w2, pw in ((posture.width, 1.0 - self.p.p_width_flip),
                           (1 - posture.width, self.p.p_width_flip)):
                out[GaitPosture(s2, w2)] = ps * pw
        return out

    # -- 合成：全动作分布 / 采样 ----------------------------------------

    def order_probs(self, state: SchedState) -> Dict[StepOrder, float]:
        """60 动作全分布（采样与矩阵物化共用同一实现）。"""
        fp, tp = self._foot_probs(state.last_foot), \
            self._turn_probs(state.last_turn)
        out: Dict[StepOrder, float] = {}
        for f, pf in fp.items():
            for t, pt in tp.items():
                for dst, pp in self._posture_probs(state.posture, f).items():
                    o = StepOrder(f, t, dst.lead, dst.sag,
                                  -1 if dst.width == 0 else +1)
                    out[o] = pf * pt * pp
        return out

    def _fallback_tap(self, state: SchedState) -> StepOrder:
        """约束兜底：交替脚→当前姿态·不转 = 原地踏步。"""
        f = -state.last_foot if state.last_foot != 0 else -1
        p = state.posture
        return StepOrder(f, 0, p.lead, p.sag,
                         -1 if p.width == 0 else +1)

    def sample(self, state: SchedState,
               collision_fn: Optional[CollisionFn] = None) -> StepOrder:
        probs = self.order_probs(state)
        orders = list(probs.keys())
        w = np.array([probs[o] for o in orders])
        w /= w.sum()
        tries = self.p.max_resample if collision_fn else 1
        for _ in range(tries):
            o = orders[int(self.rng.choice(len(orders), p=w))]
            if collision_fn is None or not collision_fn(o, state):
                return o
        return self._fallback_tap(state)

    @staticmethod
    def advance(state: SchedState, order: StepOrder) -> SchedState:
        """边界轮转后的新上下文（指令态推进，不问执行结果）。"""
        return SchedState(order.target, order.foot, order.turn)

    # -- 造数据：序列仿真 / 统计 -----------------------------------------

    def simulate(
        self,
        state: SchedState,
        n: int,
        collision_fn: Optional[CollisionFn] = None,
    ) -> List[Tuple[SchedState, StepOrder]]:
        """从 state 出发连采 n 条，返回 [(ctx, order)] 供分析/测试。"""
        seq, st = [], state
        for _ in range(n):
            o = self.sample(st, collision_fn)
            seq.append((st, o))
            st = self.advance(st, o)
        return seq

    @staticmethod
    def stats(seq: List[Tuple[SchedState, StepOrder]]) -> Dict[str, float]:
        """序列边际统计：对照参数旋钮验证采样器。"""
        n = len(seq)
        alt = turn = fwd = back = tap = wflip = 0
        ds_hist: Dict[int, int] = {}
        for st, o in seq:
            ds = o.target.stagger - st.posture.stagger
            ds_hist[abs(ds)] = ds_hist.get(abs(ds), 0) + 1
            if st.last_foot != 0 and o.foot != st.last_foot:
                alt += 1
            if o.turn != 0:
                turn += 1
            e = (0 if ds == 0 else (1 if ds > 0 else -1)) * o.foot
            fwd += e > 0
            back += e < 0
            if o.target == st.posture and o.turn == 0:
                tap += 1
            if o.target.width != st.posture.width:
                wflip += 1
        eff = sum(1 for st, _ in seq if st.last_foot != 0)
        return {
            "n": n,
            "alternation": alt / max(eff, 1),
            "turn_rate": turn / n,
            "fwd": fwd / n,
            "back": back / n,
            "tap_rate": tap / n,
            "width_flip": wflip / n,
            **{f"|ds|={k}": v / n for k, v in sorted(ds_hist.items())},
        }


# ---------------------------------------------------------------------------
# 状态矩阵物化（验证用，非采样驱动）
# ---------------------------------------------------------------------------


def transition_matrix(
    sch: StepScheduler,
) -> Tuple[List[Tuple[GaitPosture, int]], np.ndarray]:
    """物化 20×20 状态矩阵（态=姿态×上轮动脚；不含碰撞层）。

    每行 = 该状态下采 60 动作聚合成 next(posture,foot) 的概率。
    """
    states = [(p, f) for p in POSTURES for f in (-1, +1)]
    idx = {s: i for i, s in enumerate(states)}
    M = np.zeros((len(states), len(states)))
    for (p, f) in states:
        row = idx[(p, f)]
        st = SchedState(p, f, 0)          # 矩阵按 t_prev=0 分支物化
        for o, pr in sch.order_probs(st).items():
            M[row, idx[(o.target, o.foot)]] += pr
    return states, M


def stationary(M: np.ndarray, iters: int = 400) -> np.ndarray:
    """稳态分布（幂迭代）。行随机矩阵 → pi @ M → pi。"""
    pi = np.full(M.shape[0], 1.0 / M.shape[0])
    for _ in range(iters):
        pi = pi @ M
    return pi / pi.sum()


# ---------------------------------------------------------------------------
# 自测：造一段序列看边际统计是否符合旋钮
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import collections

    sch = StepScheduler(seed=0)
    seq = sch.simulate(SchedState.initial(FLAT), n=20000)
    print("== 边际统计（n=20000）==")
    for k, v in StepScheduler.stats(seq).items():
        print(f"  {k:>12s}: {v:.4f}" if isinstance(v, float)
              else f"  {k:>12s}: {v}")

    print("\n== 前 30 条步令 ==")
    cnt = collections.Counter(order_name(o) for _, o in seq)
    for st, o in seq[:30]:
        print(f"  {st.posture.name:>4s} → {order_name(o)}")

    print("\n== 姿态频率 top10 ==")
    pcnt = collections.Counter(st.posture.name for st, _ in seq)
    for name, c in pcnt.most_common(10):
        print(f"  {name:>4s}: {c/len(seq):.3f}")

    states, M = transition_matrix(sch)
    print(f"\n== 20×20 矩阵 == 行和 err={abs(M.sum(1)-1).max():.2e}")
    pi = stationary(M)
    for s, p in sorted(zip(states, pi), key=lambda x: -x[1])[:6]:
        print(f"  {s[0].name:>4s}/f={s[1]:+d}: {p:.4f}")
