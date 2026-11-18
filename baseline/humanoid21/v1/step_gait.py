"""步态姿态集与步令编码（S4 步令调度基础，`STEP_GAIT.md` 代码化）。

**步令 = 声明式终态指令**（6 维）：脚动(±1) 转向(±1/0)
终态(前脚±1/0 × 纵档±1/0 × 横宽±1)。描述完成后的姿态与朝向，
不描述动作轨迹。脚动维 0 不合法——每条步令必有一脚迈动；
全零向量是"无指令"哨兵（IDLE_VEC），不是步令。

用法：
    POSTURE_ORDERS[L1n]              # → (左→左领, 右→左领)
    orders_to(HORSE, turn=+1)        # → 目标马步带右转的全部步令
    o = StepOrder(-1, 0, +1, +1, -1) # 左脚动→右弓常距
    o.target                          # → GaitPosture 右弓
    o.to_vec()                        # → (6,) D7 编码填 step_cur/next
    step_cost(FLAT, L2n)              # → 动脚位移代价(档)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 姿态集
# ---------------------------------------------------------------------------

# 交错档符号 = 维2 编码：负 = 左脚在前，正 = 右脚在前，0 = 齐平
_N, _W = 0, 1


@dataclass(frozen=True)
class GaitPosture:
    """步态姿态：交错档(负=左前) × 宽距(0=N 1=W)。"""

    stagger: int          # ∈ {-2..+2}，负=左前/正=右前/0=齐平
    width: int            # 0 = 常距 N，1 = 宽距 W

    @property
    def lead(self) -> int:
        """维2 编码：−1 左前 / 0 齐 / +1 右前。"""
        return (self.stagger > 0) - (self.stagger < 0)

    @property
    def sag(self) -> int:
        """维3 编码：−1 小 / +1 大 / 0 不迈(lead=0 时必0)。"""
        a = abs(self.stagger)
        return 0 if a == 0 else (-1 if a == 1 else +1)

    @property
    def code(self) -> str:
        s_name = {-2: "L2", -1: "L1", 0: "E", 1: "R1", 2: "R2"}
        return f"{s_name[self.stagger]}{'w' if self.width else 'n'}"

    @property
    def name(self) -> str:
        s_name = {-2: "左弓", -1: "左领", 0: "平步", 1: "右领", 2: "右弓"}
        if (self.stagger, self.width) == (0, _W):
            return "马步"
        return s_name[self.stagger] + ("宽" if self.width else "")


# 10 姿态常量
L2n = GaitPosture(-2, _N)
L1n = GaitPosture(-1, _N)
FLAT = GaitPosture(0, _N)      # 平步——基准/枢纽
R1n = GaitPosture(+1, _N)
R2n = GaitPosture(+2, _N)
L2w = GaitPosture(-2, _W)
L1w = GaitPosture(-1, _W)
HORSE = GaitPosture(0, _W)     # 马步
R1w = GaitPosture(+1, _W)
R2w = GaitPosture(+2, _W)

#: 全部姿态（文档顺序：常距系 | 宽距系）
POSTURES: Tuple[GaitPosture, ...] = (
    L2n, L1n, FLAT, R1n, R2n,
    L2w, L1w, HORSE, R1w, R2w,
)

# ---------------------------------------------------------------------------
# 步令（6 维终态编码）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepOrder:
    """一条步令 = 脚动 + 转向 + 终态姿态。

    foot: −1 左脚动 / +1 右脚动（0 不合法）
    turn: −1 左转 / 0 不转 / +1 右转
    lead: −1 终态左脚前 / 0 齐平 / +1 右脚前
    sag:  −1 纵小 / 0 不迈 / +1 纵大（lead=0 时必 0）
    width: −1 常距 / +1 宽距
    """

    foot: int
    turn: int
    lead: int
    sag: int
    width: int

    def __post_init__(self) -> None:
        assert self.foot in (-1, +1), f"脚动维非法: {self.foot}"
        assert self.turn in (-1, 0, +1)
        assert self.lead in (-1, 0, +1)
        assert self.sag in (-1, 0, +1)
        assert self.width in (-1, +1)
        assert not (self.lead == 0 and self.sag != 0), "齐平时纵档必为0"

    @property
    def target(self) -> GaitPosture:
        """终态姿态。"""
        a = 0 if self.lead == 0 else (1 if self.sag < 0 else 2)
        return GaitPosture(self.lead * a, 0 if self.width < 0 else 1)

    def to_vec(self) -> np.ndarray:
        return np.array(
            [self.foot, self.turn, self.lead, self.sag, self.width, 0],
            dtype=np.float32,
        )

    def with_turn(self, turn: int) -> "StepOrder":
        return StepOrder(self.foot, turn, self.lead, self.sag, self.width)


#: 无指令哨兵（全零，非合法步令——碎步保平衡默认）
IDLE_VEC = np.zeros(6, dtype=np.float32)


def order_name(o: StepOrder) -> str:
    """步令读法：`脚→终态姿态[·转]`。"""
    foot = "左" if o.foot < 0 else "右"
    turn = "" if o.turn == 0 else ("·左" if o.turn < 0 else "·右")
    return f"{foot}→{o.target.name}{turn}"


# ---------------------------------------------------------------------------
# 姿态→步令生成表（列均匀：终态只由指令决定，可达性不交表管）
# ---------------------------------------------------------------------------


def orders_to(dst: GaitPosture, turn: int = 0) -> Tuple[StepOrder, StepOrder]:
    """达成目标姿态的全部步令：左/右脚各一（转向可叠加）。"""
    return (
        StepOrder(-1, turn, dst.lead, dst.sag, -1 if dst.width == 0 else +1),
        StepOrder(+1, turn, dst.lead, dst.sag, -1 if dst.width == 0 else +1),
    )


#: POSTURE_ORDERS[p] = 达成 p 的无转向等价步令对
POSTURE_ORDERS: Dict[GaitPosture, Tuple[StepOrder, StepOrder]] = {
    p: orders_to(p) for p in POSTURES
}


def step_cost(src: GaitPosture, dst: GaitPosture) -> int:
    """动脚位移代价（采样权重输入）：矢状 |Δ交错档| + 宽距切换 1。"""
    return abs(src.stagger - dst.stagger) + (src.width != dst.width)


# ---------------------------------------------------------------------------
# 几何参数（拍定 v1 — s42i 实测依据，render_gait_poses.py 同值）
# ---------------------------------------------------------------------------

WIDTH_N = 0.76      # 常距横向脚距 m：s42i 双支撑中位 0.764
WIDTH_W = 1.00      # 宽距横向脚距 m：~1.3×，s42i 无样本外推待校
STAGGER_UNIT = 0.22 # 一档前后错位 m：双支撑 |stagger| 中位~0.25
ANKLE_Z = 0.107     # 踝锚（ankle_y xanchor）站姿态高度 m
TURN_RAD = 0.52     # 每步转向角 rad（±30°，暂定）


__all__ = [
    "GaitPosture",
    "StepOrder",
    "IDLE_VEC",
    "POSTURES",
    "POSTURE_ORDERS",
    "orders_to",
    "step_cost",
    "order_name",
    "WIDTH_N", "WIDTH_W", "STAGGER_UNIT", "ANKLE_Z", "TURN_RAD",
    "L2n", "L1n", "FLAT", "R1n", "R2n",
    "L2w", "L1w", "HORSE", "R1w", "R2w",
]
