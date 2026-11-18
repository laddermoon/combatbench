"""步态姿态集与转移矩阵（S4 步令调度基础）。

`STEP_GAIT.md` 的代码化：10 姿态 = 5 交错档 × 2 宽距；
单步转移 = 恰好一脚移动（矢状 ±1/±2 档，可选横向 ±1 换宽距）；
同一转移的等价步令集 = 双脚对称指令（如平步→左领 =
{左前小, 右后小}）。转向为修饰符，不在表内。

⚠ 本表为**几何可达性枚举**，Markov 权重/裁剪规则未定稿。

用法：
    TRANSITIONS[FLAT][L_LEAD]                 # → 等价步令 tuple
    transitions_from(FLAT)                    # → {目标姿态: 步令集}
    order_name(order)                         # → "左前小" 读法名
    order.to_vec()                            # → 6 维 D7 编码
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 姿态集
# ---------------------------------------------------------------------------

# 交错档：正号 = 左脚在前；单位 = 小步长档
_L2, _L1, _E, _R1, _R2 = 2, 1, 0, -1, -2
_N, _W = 0, 1


@dataclass(frozen=True)
class GaitPosture:
    """步态姿态：交错档 × 宽距。"""

    stagger: int          # ∈ {-2..+2}，正 = 左脚在前
    width: int            # 0 = 常距 N，1 = 宽距 W

    @property
    def code(self) -> str:
        s_name = {_L2: "L2", _L1: "L1", _E: "E", _R1: "R1", _R2: "R2"}
        return f"{s_name[self.stagger]}{'w' if self.width else 'n'}"

    @property
    def name(self) -> str:
        s_name = {_L2: "左弓", _L1: "左领", _E: "平步", _R1: "右领", _R2: "右弓"}
        if (self.stagger, self.width) == (_E, _W):
            return "马步"
        return s_name[self.stagger] + ("宽" if self.width else "")


# 10 姿态常量（交错升序命名）
L2n = GaitPosture(_L2, _N)
L1n = GaitPosture(_L1, _N)
FLAT = GaitPosture(_E, _N)     # 平步——枢纽/基准
R1n = GaitPosture(_R1, _N)
R2n = GaitPosture(_R2, _N)
L2w = GaitPosture(_L2, _W)
L1w = GaitPosture(_L1, _W)
HORSE = GaitPosture(_E, _W)    # 马步
R1w = GaitPosture(_R1, _W)
R2w = GaitPosture(_R2, _W)

#: 全部姿态（文档矩阵顺序：常距系 | 宽距系）
POSTURES: Tuple[GaitPosture, ...] = (
    L2n, L1n, FLAT, R1n, R2n,
    L2w, L1w, HORSE, R1w, R2w,
)

# ---------------------------------------------------------------------------
# 步令（6 维 D7 编码）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepOrder:
    """一条步令 = 到目标姿态的单步转移指令。

    语义按 D7：脚(−1左/0/+1右) 矢状(−1后/+1前) 横向(−1内/+1外,
    anatomical) 步幅(−1小/+1大) 转向(−1左/+1右) 预留(0)。
    转向为修饰符——不改变目标姿态，叠加在同一转移上。
    """

    foot: int             # −1 左 / 0 无 / +1 右
    sag: int              # −1 后 / 0 / +1 前
    lat: int              # −1 内 / 0 / +1 外
    stride: int           # −1 小 / +1 大
    turn: int = 0         # −1 左转 / 0 / +1 右转

    def to_vec(self) -> np.ndarray:
        return np.array(
            [self.foot, self.sag, self.lat, self.stride, self.turn, 0],
            dtype=np.float32,
        )


#: 脚=0 自环步令（原地碎步保平衡）
IDLE = StepOrder(0, 0, 0, -1, 0)


def order_name(o: StepOrder) -> str:
    """步令读法：`脚+矢状+横向+幅·转`；纯横向（矢状=0）幅省略。"""
    if o.foot == 0:
        base = "原宽" if o.lat > 0 else "原"
        if o.turn:
            base += "·左" if o.turn < 0 else "·右"
        return base
    foot = "左" if o.foot < 0 else "右"
    sag = {-1: "后", 0: "", 1: "前"}[o.sag]
    lat = {-1: "内", 0: "", 1: "外"}[o.lat]
    mag = "" if o.sag == 0 else ("大" if o.stride > 0 else "小")
    turn = "" if o.turn == 0 else ("·左" if o.turn < 0 else "·右")
    return f"{foot}{sag}{lat}{mag}{turn}"


# ---------------------------------------------------------------------------
# 转移矩阵（程序化枚举：单步 = 一脚矢状±1/±2 可选横向±1）
# ---------------------------------------------------------------------------


def _build_transitions() -> Dict[GaitPosture, Dict[GaitPosture, Tuple[StepOrder, ...]]]:
    """枚举所有 (姿态 → 姿态) 的一步可达转移及其等价步令集。

    左脚前移 a 档 → 交错档 +a；右脚前移 a 档 → 交错档 −a
    （stagger 正号 = 左前）。横向 anatomical 外→宽+1 / 内→宽−1。
    """
    table: Dict[GaitPosture, Dict[GaitPosture, Tuple[StepOrder, ...]]] = {
        p: {} for p in POSTURES
    }
    for src in POSTURES:
        for foot, sign in ((-1, +1), (+1, -1)):          # 左脚 +1, 右脚 −1
            for a in (-2, -1, 0, 1, 2):                  # 矢状位移（档），+前 −后
                for b in (-1, 0, 1):                     # 横向：+1 外 / −1 内
                    if a == 0 and b == 0:
                        continue
                    ns = src.stagger + sign * a
                    nw = src.width + b
                    if not (-2 <= ns <= 2) or nw not in (0, 1):
                        continue
                    order = StepOrder(
                        foot=foot,
                        sag=(a > 0) - (a < 0),
                        lat=b,
                        stride=+1 if abs(a) == 2 else -1,
                    )
                    dst = GaitPosture(ns, nw)
                    table[src][dst] = table[src].get(dst, ()) + (order,)
    return table


#: TRANSITIONS[src][dst] = 等价步令 tuple；dst 缺席 = 一步不可达
TRANSITIONS = _build_transitions()


def transitions_from(src: GaitPosture) -> Dict[GaitPosture, Tuple[StepOrder, ...]]:
    """src 姿态一步可达的全部目标及其等价步令集。"""
    return TRANSITIONS[src]


__all__ = [
    "GaitPosture",
    "StepOrder",
    "IDLE",
    "POSTURES",
    "TRANSITIONS",
    "transitions_from",
    "order_name",
    "L2n", "L1n", "FLAT", "R1n", "R2n",
    "L2w", "L1w", "HORSE", "R1w", "R2w",
]
