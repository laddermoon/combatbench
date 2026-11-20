"""V1 步令实验指令生成器：连续目标采样（D20 残差编码的指令源）。

每条令 = (动脚, 终态槽位偏移, Δyaw)——动脚落点以 stance 脚锚定，
发令时刻钉死世界系目标；obs 侧残差由 sim 每帧按真值重算，
本模块只负责"目标怎么来"。

采样规则（V1 定稿，对应 STEP_CMD.md 无相位伺服语义）：
    ① 动脚   P(交替)=0.8 / P(同脚)=0.1 / P(保持)=0.1
             保持 = 全零指令（"原地不动"合法指令）
    ② 方向   P(前)=0.8 / P(后)=0.2 —— 终态交错偏移符号
    ③ 纵向   前 U(0.15,0.53) / 后 U(0.10,0.40) m（相对 stance 脚）
    ④ 横向   脚距 U(0.35,0.55) m（常距 0.475 附近）
    ⑤ 转向   P=0.25，Δyaw ~ ±U(5°,30°)
    ⑥ 撞墙   collide_fn(cmd) 拒绝重采 ≤8 次，兜底=保持

与旧 StepScheduler 的区别：指令空间连续化——不再从 10 姿态
词表选终态，直接采连续槽位；姿态集仅作几何参照（SAG/LAT 范围
即旧档位的连续化）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 采样参数（V1 定稿值）
# ---------------------------------------------------------------------------

P_SAME = 0.1            # 同脚再动（调整步）
P_ALT = 0.8             # 交替脚
P_HOLD = 0.1            # 不动 = 保持指令（全零）
P_FWD = 0.8             # 向前迈步
SAG_FWD = (0.15, 0.53)  # 前迈终态交错偏移范围 m（≈旧 0.5–2 档）
SAG_BWD = (0.10, 0.40)  # 后撤步幅范围 m
LAT_RANGE = (0.35, 0.55)  # 终态横向脚距 m（常距 0.475 附近）
FOOT_DIST_MAX = 0.62    # 终态双脚欧氏距离上限 m（腿长 ~0.66，
                        # 超出即深弓步，拒绝重采）
P_TURN = 0.25           # 该令附带转向的概率
DYAW_RANGE = (math.radians(5), math.radians(30))
MAX_RESAMPLE = 8


@dataclass(frozen=True)
class StepCmd:
    """一条步令的内部表示（不是 obs 编码——obs 写残差）。

    foot: −1 左 / +1 右 / 0 保持
    sag : 动脚相对 stance 脚的终态纵向偏移（+ 前）m
    lat : 终态横向脚距 m（恒正；左动脚落 stance 左侧）
    dyaw: 目标 yaw 增量 rad（+ 左转）
    """

    foot: int = 0
    sag: float = 0.0
    lat: float = 0.0
    dyaw: float = 0.0


HOLD_CMD = StepCmd()


def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def sample_cmd(
    rng: np.random.Generator,
    last_foot: int,
    collide_fn: Optional[Callable[[StepCmd], bool]] = None,
) -> StepCmd:
    """采一条令；collide_fn(cmd) 返回 True 表示落点出界，重采后兜底保持。"""
    for _ in range(MAX_RESAMPLE):
        cmd = _draw(rng, last_foot)
        if cmd.foot == 0 or collide_fn is None or not collide_fn(cmd):
            return cmd
    return HOLD_CMD


def _draw(rng: np.random.Generator, last_foot: int) -> StepCmd:
    u = rng.uniform()
    if u < P_HOLD:
        return HOLD_CMD
    base = last_foot if last_foot != 0 else int(rng.choice([-1, +1]))
    foot = base if u < P_HOLD + P_SAME else -base

    fwd = rng.uniform() < P_FWD
    lat = sag = 0.0
    for _ in range(MAX_RESAMPLE):
        mag = rng.uniform(*(SAG_FWD if fwd else SAG_BWD))
        s = mag if fwd else -mag
        l = float(rng.uniform(*LAT_RANGE))
        if math.hypot(s, l) <= FOOT_DIST_MAX:
            sag, lat = s, l
            break
    else:                                # 兜底：留 lat、纵向收到上限
        lat = l
        sag = math.copysign(
            math.sqrt(max(FOOT_DIST_MAX ** 2 - lat ** 2, 0.0)),
            1.0 if fwd else -1.0)
    dyaw = 0.0
    if rng.uniform() < P_TURN:
        dyaw = float(rng.uniform(*DYAW_RANGE) * rng.choice([-1, +1]))
    return StepCmd(foot=foot, sag=sag, lat=lat, dyaw=dyaw)


def cmd_target(
    foot_l: np.ndarray,
    foot_r: np.ndarray,
    yaw: float,
    cmd: StepCmd,
) -> Tuple[Optional[np.ndarray], float]:
    """发令时刻的世界系锚定目标：(动脚落点 xy | None, 目标 yaw)。

    stance 脚原地锚定；动脚落点 = stance + R(yaw+dyaw)·(sag, ±lat)。
    保持令返回 (None, yaw)——无动脚目标，残差恒 0。
    """
    if cmd.foot == 0:
        return None, float(yaw)
    yaw_t = yaw + cmd.dyaw
    stance = np.asarray(foot_r if cmd.foot < 0 else foot_l, dtype=float)
    lat_s = +cmd.lat if cmd.foot < 0 else -cmd.lat
    tgt = stance + _rot(yaw_t) @ np.array([cmd.sag, lat_s])
    return tgt, float(yaw_t)


def residual(
    ankle_mv: np.ndarray,
    yaw: float,
    tgt: np.ndarray,
    yaw_t: float,
) -> Tuple[float, float, float]:
    """obs 每帧写入的残差（本体系，前向=躯干 yaw 投影）：(Δx, Δy, Δψ)。"""
    d = _rot(yaw).T @ (np.asarray(tgt) - np.asarray(ankle_mv))
    dyaw = (yaw_t - yaw + math.pi) % (2 * math.pi) - math.pi
    return float(d[0]), float(d[1]), float(dyaw)
