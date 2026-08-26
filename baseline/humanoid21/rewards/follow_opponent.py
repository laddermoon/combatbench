'''
一个Reward 基于以下想法：
如何让机器人跟踪对手到一个范围内
要做合理的归因，如果只取两个机器人之间的距离做为奖励信号，这时候对手的移动是一个干扰因素。

所以在这个实现中要做精细化的归因。
1. 当与目标机器人的距离大于阈值时， 此时我们希望机器人朝着对手移动（给奖励）， 这种移动不能是相对的，而应该是主动的。
最应该奖励的是机器人本身朝着对手的方面移动。
所以计算每一时刻相对于对手的移动相量



另一个思路，与摔倒的逻辑类似：
与对手在一个范围内给奖励 1 ，否则按距离给惩罚 ， 或者不给惩罚
'''
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext

# Debug visualization imports (lazy-loaded when debug=True)
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# 距离带半径（米）：root 的 2D（地面投影 xy）距离 <= 此值视为“在范围内”。
# 全模块统一使用水平距离，忽略高度差。
FOLLOW_DIST_MAX = 0.9

# 接近奖励系数（在 trainer 侧后处理 compute_approach_rewards 中使用）。
# 径向（朝对手）平滑位移奖励系数。
APPROACH_RADIAL_COEF = 1.0
# 切向（绕圈）惩罚系数。默认 0.0 = 关闭（门控开关）。开启后也仅在“原地打转”
# （|径向位移|<progress_eps）时才施加，背向对手由径向负分处理、不叠罚。
APPROACH_TANGENTIAL_COEF = 0.0
# 单步（平滑后）位移模长上限（米）。正常走路每个 action step 位移约 0.1m；
# 摔倒/数值异常可能产生瞬时大跳，clip 掉以避免奖励尖峰污染训练。
APPROACH_DISP_CLIP = 0.5
# 轨迹平滑窗宽（action-step 数）。dt=1/CONTROL_FREQUENCY=0.05s 时，
# N=17 ≈ 0.85s ≈ 一个步态周期，足以抵消一次完整的左右摆动。取奇数最佳。
APPROACH_SMOOTH_WINDOW = 17
# “原地打转”判定阈值（米）：每步平滑径向位移绝对值小于此值视为径向无进展，
# 仅此时才考虑施加切向惩罚。
APPROACH_RADIAL_PROGRESS_EPS = 0.01


class InZoneHoldRewarder(BaseObserverPlugin):
    """范围内保持奖励（稀疏指示，用于“留在对手身边”）。

    与摔倒惩罚同构的稀疏方案::

        r_t = 1.0   if  dist_2d(self, opp) <= dist_max
              0.0   otherwise

    其中 ``dist_2d`` 为 root 位置在地面投影（xy）上的水平距离。

    设计意图：势差/接近奖励负责“领进去”，本奖励负责“留下来”——它**不**
    telescoping，在区内提供**持续**正反馈，区外恒为 0（不提供方向梯度，
    避免和走路步态的逐步距离压力冲突）。

    暴露 ``.in_zone`` 布尔属性供课程门控读取；``get_output`` 同时输出
    ``reward`` 与 ``in_zone``（与 ``OpponentRelationRewarder`` 一致的 dict 形式）。
    """

    def __init__(
        self,
        agent_id: str,
        dist_max: float = FOLLOW_DIST_MAX,
    ) -> None:
        self.agent_id = str(agent_id)
        self.opponent_id = "robot_b" if self.agent_id == "robot_a" else "robot_a"
        self.dist_max = float(dist_max)
        self.in_zone: bool = False
        self._output: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self.in_zone = False
        self._output = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()
        # 统一使用 2D（地面投影）距离：只取 xy 分量，忽略高度差。这样
        # “在范围内”的判定与 ApproachVelocityRewarder 的水平距离一致，
        # 且不受蹲伏/起跳等竖直姿态变化的干扰。
        self_xy = np.asarray(
            core_state[self.agent_id]["root_pos"], dtype=np.float64
        )[:2]
        opp_xy = np.asarray(
            core_state[self.opponent_id]["root_pos"], dtype=np.float64
        )[:2]
        distance = float(np.linalg.norm(opp_xy - self_xy))
        self.in_zone = distance <= self.dist_max
        self._output = 1.0 if self.in_zone else 0.0

    def get_output(self) -> Dict[str, float]:
        return {
            "reward": float(self._output),
            "in_zone": float(self.in_zone),
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "dist_max": self.dist_max,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "InZoneHoldRewarder":
        return cls(**config)


class ApproachVelocityRewarder(BaseObserverPlugin):
    """轨迹记录器（接近奖励改由 trainer 侧后处理计算）。

    设计变更（为什么从“逐步算 reward”改成“只记录位置”）
    -------------------------------------------------------
    步态的左右摆动会污染逐步径向/切向投影——尤其切向取**模长**（恒 >=0）
    不会在求和中抵消，会把正常摆腿误判成“绕圈”而持续扣分，导致机器人
    不敢摆动、学不会正常走路。正确做法是先对整条轨迹做**非因果平滑**去掉
    摆动，再求位移/速度并投影。

    但观测器的 ``get_output`` 是被 recorder **逐步快照**进 Episode 的
    （``on_post_episode`` 不回填已写入的逐步值），无法在观测器内做非因果
    平滑。因此本观测器退化为**纯位置记录器**：每步只输出自身与对手的 2D
    位置；真正的接近奖励交给 :func:`compute_approach_rewards` 在 PPO 构建
    buffer 时对整条 ``(T, 2)`` 序列计算（那里能拿到全序列、可居中平滑）。

    每步输出（世界系 xy，单位米）::

        self_x, self_y   本体 root 水平位置
        opp_x,  opp_y    对手 root 水平位置
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = str(agent_id)
        self.opponent_id = "robot_b" if self.agent_id == "robot_a" else "robot_a"
        self._self_xy: np.ndarray = np.zeros(2, dtype=np.float64)
        self._opp_xy: np.ndarray = np.zeros(2, dtype=np.float64)

    def _read_xy(self, ctx: ReadOnlySimContext, agent: str) -> np.ndarray:
        core_state = ctx.accessor.get_core_state()
        return np.asarray(core_state[agent]["root_pos"], dtype=np.float64)[:2].copy()

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        try:
            self._self_xy = self._read_xy(ctx, self.agent_id)
            self._opp_xy = self._read_xy(ctx, self.opponent_id)
        except Exception:
            self._self_xy = np.zeros(2, dtype=np.float64)
            self._opp_xy = np.zeros(2, dtype=np.float64)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        self._self_xy = self._read_xy(ctx, self.agent_id)
        self._opp_xy = self._read_xy(ctx, self.opponent_id)

    def get_output(self) -> Dict[str, float]:
        return {
            "self_x": float(self._self_xy[0]),
            "self_y": float(self._self_xy[1]),
            "opp_x": float(self._opp_xy[0]),
            "opp_y": float(self._opp_xy[1]),
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ApproachVelocityRewarder":
        return cls(**config)


# ---------------------------------------------------------------------------
# Trainer-side post-processing (acausal smoothing → velocity → projection)
# ---------------------------------------------------------------------------
def _centered_moving_average(xy: np.ndarray, window: int) -> np.ndarray:
    """对 ``(T, 2)`` 轨迹做居中移动平均；边界窗口自动收缩（零相位）。"""
    arr = np.asarray(xy, dtype=np.float64)
    T = arr.shape[0]
    if window <= 1 or T == 0:
        return arr.copy()
    half = int(window) // 2
    csum = np.cumsum(arr, axis=0)
    out = np.empty_like(arr)
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T - 1, t + half)
        s = csum[hi] - (csum[lo - 1] if lo > 0 else 0.0)
        out[t] = s / (hi - lo + 1)
    return out


def compute_approach_rewards(
    self_xy: np.ndarray,
    opp_xy: np.ndarray,
    *,
    dist_max: float = FOLLOW_DIST_MAX,
    smooth_window: int = APPROACH_SMOOTH_WINDOW,
    radial_coef: float = APPROACH_RADIAL_COEF,
    tangential_coef: float = APPROACH_TANGENTIAL_COEF,
    disp_clip: float = APPROACH_DISP_CLIP,
    progress_eps: float = APPROACH_RADIAL_PROGRESS_EPS,
    debug: bool = False,
    debug_title: Optional[str] = None,
    debug_save_path: Optional[str] = None,
):
    """两阶段计算逐步接近奖励：先平滑去摆动，再求位移投影。

    阶段 1（去摆动）：对自身 ``(T, 2)`` 轨迹做居中移动平均得平滑轨迹 ``p̃``。
    阶段 2（求位移/速度）：居中差分 ``disp_t = (p̃_{t+1} - p̃_{t-1}) / 2`` 作为
      该步的净位移向量（与速度仅差常数 ``dt``，``dt`` 折进 ``radial_coef``）。

    方向系数（几何构造，统一表达"有效接近度"）：
      设平滑位置 ``P=sm``、对手 ``O``、距离 ``d=|O-P|``、速度单位向量 ``v̂``。
      沿速度方向走距离 ``d`` 得端点 ``E = P + d·v̂``，记 ``c = |O - E|``::

          coef = (d - c) / d        # 等价于 1 - 2·sin(θ/2)，θ=∠(v̂, O-P)
          radial = radial_coef * |disp| * coef

      其中 ``θ`` 为速度方向与"指向对手"方向的夹角::

          θ = 0°   → coef = +1   正对冲刺（最强接近）
          θ < 60°  → coef > 0    有效接近
          θ = 60°  → coef = 0    临界
          θ > 60°  → coef < 0    无效（净远离）
          θ = 180° → coef = -1   背向逃离（最强惩罚）

    门控：
      * 仅在**区外**（``distance > dist_max``）给信号；区内交给
        :class:`InZoneHoldRewarder` 的保持奖励。
      * 静止（``|disp|≈0``）或贴脸（``d≈0``）时 coef=0，奖励为 0。

    返回 ``(radial, tangential)``，均为 ``(T,)`` float32。
    **tangential 恒为 0**（仅为兼容旧二元接口而保留；该分量已废弃，
    其 stage 权重应一并置 0）。``tangential_coef``/``progress_eps`` 参数保留
    但不再生效。

    Args:
        debug: 若为 True，则绘制原始/平滑轨迹、对手轨迹及奖励曲线（默认 False）。
        debug_title: 调试图的标题前缀（可选）。
        debug_save_path: 图片保存路径。若为目录则自动生成带时间戳的文件名；
                         若为完整路径（含扩展名如 .png/.jpg）则直接使用。默认 None（不保存）。
    """
    if debug and not _HAS_MPL:
        raise RuntimeError("matplotlib is required for debug visualization")

    self_xy = np.asarray(self_xy, dtype=np.float64)
    opp_xy = np.asarray(opp_xy, dtype=np.float64)
    T = self_xy.shape[0]
    radial = np.zeros(T, dtype=np.float32)
    tangential = np.zeros(T, dtype=np.float32)
    if T < 2:
        return radial, tangential

    # 阶段 1：平滑自身轨迹。
    sm = _centered_moving_average(self_xy, smooth_window)
    # 阶段 2：居中差分求每步净位移。
    disp = np.empty_like(sm)
    disp[1:-1] = (sm[2:] - sm[:-2]) * 0.5
    disp[0] = sm[1] - sm[0]
    disp[-1] = sm[-1] - sm[-2]
    # clip 位移模长，防数值异常的瞬时大跳。
    mag = np.linalg.norm(disp, axis=1)
    big = mag > disp_clip
    if np.any(big):
        disp[big] *= (disp_clip / np.maximum(mag[big], 1e-9))[:, None]

    # 平滑后净位移的模长（速度大小，dt 折进 radial_coef）与单位方向。
    speed = np.linalg.norm(disp, axis=1)
    moving = speed > 1e-9
    v_hat = np.zeros_like(disp)
    v_hat[moving] = disp[moving] / speed[moving, None]

    # 指向对手的方向：用**平滑后**位置 sm 计算（与速度同源，几何一致）。
    to_opp = opp_xy - sm
    dist = np.linalg.norm(to_opp, axis=1)

    # ---- 几何方向系数构造 ----
    # 从 sm 沿速度方向走距离 d=dist 得端点 E = sm + d * v_hat；
    # c = |O - E| 为该端点到对手的距离。
    #   coef = (d - c) / d            （等价于 1 - 2*sin(θ/2)）
    #   θ < 60° → coef > 0（有效接近）；θ = 60° → 0；θ > 60° → 负（越走越远）。
    valid = (dist > 1e-6) & moving
    E = sm + dist[:, None] * v_hat
    c = np.linalg.norm(opp_xy - E, axis=1)
    coef = np.zeros(T, dtype=np.float64)
    coef[valid] = (dist[valid] - c[valid]) / dist[valid]

    # 最终奖励 = 速度模长 * 方向系数（朝对手为正，背离为负）。
    out_zone = dist > max(float(dist_max), 1e-6)
    mask = out_zone & valid
    radial[mask] = (radial_coef * speed[mask] * coef[mask]).astype(np.float32)

    # tangential 始终为 0：保留二元返回仅为兼容旧接口；该项已废弃，
    # 其 stage 权重应一并置 0（见 common_v2.STAGE_WEIGHTS）。

    # ---------------------------------------------------------
    # Debug visualization (raw/smooth trajectories + rewards)
    # ---------------------------------------------------------
    if debug:
        title_prefix = debug_title if debug_title else "ApproachReward"
        t = np.arange(T)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{title_prefix} | T={T}, win={smooth_window}", fontsize=12)

        # Axis 0: Raw vs Smoothed Self Trajectory (2D XY)
        ax = axes[0, 0]
        ax.plot(self_xy[:, 0], self_xy[:, 1], "b.-", alpha=0.4, label="raw self", markersize=3)
        ax.plot(sm[:, 0], sm[:, 1], "b-", linewidth=2, label="smoothed self")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Self Trajectory (World XY)")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)

        # Axis 1: Opponent Trajectory (2D XY)
        ax = axes[0, 1]
        ax.plot(opp_xy[:, 0], opp_xy[:, 1], "r.-", alpha=0.5, label="opponent", markersize=3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Opponent Trajectory (World XY)")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)

        # Axis 2: Trajectory overlay (both self and opp)
        ax = axes[1, 0]
        ax.plot(self_xy[:, 0], self_xy[:, 1], "b.-", alpha=0.3, label="self raw", markersize=2)
        ax.plot(sm[:, 0], sm[:, 1], "b-", linewidth=2, label="self smooth")
        ax.plot(opp_xy[:, 0], opp_xy[:, 1], "r.-", alpha=0.5, label="opponent", markersize=2)
        # Mark start/end
        ax.scatter([self_xy[0, 0]], [self_xy[0, 1]], c="blue", marker="o", s=50, zorder=5)
        ax.scatter([self_xy[-1, 0]], [self_xy[-1, 1]], c="blue", marker="x", s=50, zorder=5)
        ax.scatter([opp_xy[0, 0]], [opp_xy[0, 1]], c="red", marker="o", s=50, zorder=5)
        ax.scatter([opp_xy[-1, 0]], [opp_xy[-1, 1]], c="red", marker="x", s=50, zorder=5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Overlay (o=start, x=end)")
        ax.legend(loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)

        # Axis 3: Rewards over time
        ax = axes[1, 1]
        ax.step(t, radial, where="mid", label="radial", alpha=0.8)
        ax.step(t, tangential, where="mid", label="tangential", alpha=0.6, linestyle="--")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("time step")
        ax.set_ylabel("reward")
        ax.set_title(f"Rewards (radial_coef={radial_coef}, tangential_coef={tangential_coef})")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save to file if path provided, otherwise show interactively
        if debug_save_path:
            from pathlib import Path
            import time

            save_path = Path(debug_save_path)
            # If it's a directory, auto-generate filename with timestamp
            if save_path.is_dir() or debug_save_path.endswith(('/', '\\')):
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = f"{title_prefix.replace(' ', '_')}_{ts}.png"
                save_path = save_path / fname
            else:
                # Ensure .png extension if none provided
                if not save_path.suffix:
                    save_path = save_path.with_suffix('.png')

            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"[DEBUG] Saved approach reward plot to: {save_path}")
            plt.close(fig)
        else:
            plt.show()

    return radial, tangential


# ---------------------------------------------------------------------------
# Velocity-decomposition approach (replaces compute_approach_rewards in new code)
# ---------------------------------------------------------------------------
def compute_radial_tangential_rewards(
    self_xy: np.ndarray,
    opp_xy: np.ndarray,
    *,
    dist_max: float = FOLLOW_DIST_MAX,
    smooth_window: int = APPROACH_SMOOTH_WINDOW,
    radial_coef: float = 1.0,
    tangential_coef: float = 1.0,
    disp_clip: float = APPROACH_DISP_CLIP,
    gate: bool = True,
):
    """平滑轨迹 → 速度 → 分解为径向（朝对手）与切向（横向）两个独立奖励。

    流程::

        1. 对自身轨迹做居中移动平均（去步态摆动）→ sm
        2. 居中差分得每步净位移向量 disp（速度代理）
        3. 计算"指向对手"单位向量 d_hat = (opp - sm) / |opp - sm|
        4. 分解位移:
             径向标量  v_rad = disp · d_hat        （正值=接近，负值=远离）
             切向量    v_tan = disp - v_rad * d_hat
             切向模长  |v_tan|                       （恒 ≥ 0）

    奖励::

        radial     =  radial_coef * v_rad            （接近时为正奖励，远离时为负惩罚）
        tangential = -tangential_coef * |v_tan|      （恒为惩罚，横向移动越快惩罚越大）

    门控: 当 ``gate=True``（默认，旧行为）时，仅在区外（distance > dist_max）
    时给信号。当 ``gate=False`` 时，返回无门控的物理 reward，由调用方在
    actor_weight 中自行加距离门控。

    返回 ``(radial, tangential)``，均为 ``(T,)`` float32。
    """
    """平滑轨迹 → 速度 → 分解为径向（朝对手）与切向（横向）两个独立奖励。

    流程::

        1. 对自身轨迹做居中移动平均（去步态摆动）→ sm
        2. 居中差分得每步净位移向量 disp（速度代理）
        3. 计算"指向对手"单位向量 d_hat = (opp - sm) / |opp - sm|
        4. 分解位移:
             径向标量  v_rad = disp · d_hat        （正值=接近，负值=远离）
             切向量    v_tan = disp - v_rad * d_hat
             切向模长  |v_tan|                       （恒 ≥ 0）

    奖励::
        radial     =  radial_coef * v_rad            （接近时为正奖励，远离时为负惩罚）
        tangential = -tangential_coef * |v_tan|      （恒为惩罚，横向移动越快惩罚越大）

    门控: 仅在区外（distance > dist_max）时给信号。

    返回 ``(radial, tangential)``，均为 ``(T,)`` float32。
    """
    self_xy = np.asarray(self_xy, dtype=np.float64)
    opp_xy = np.asarray(opp_xy, dtype=np.float64)
    T = self_xy.shape[0]
    radial = np.zeros(T, dtype=np.float32)
    tangential = np.zeros(T, dtype=np.float32)
    if T < 2:
        return radial, tangential

    # 阶段 1：平滑自身轨迹（去步态摆动）。
    sm = _centered_moving_average(self_xy, smooth_window)

    # 阶段 2：居中差分求每步净位移（速度代理）。
    disp = np.empty_like(sm)
    disp[1:-1] = (sm[2:] - sm[:-2]) * 0.5
    disp[0] = sm[1] - sm[0]
    disp[-1] = sm[-1] - sm[-2]

    # clip 位移模长，防数值异常的瞬时大跳。
    mag = np.linalg.norm(disp, axis=1)
    big = mag > disp_clip
    if np.any(big):
        disp[big] *= (disp_clip / np.maximum(mag[big], 1e-9))[:, None]

    speed = np.linalg.norm(disp, axis=1)
    moving = speed > 1e-9

    # 指向对手的方向（用平滑后位置计算，与速度同源）。
    to_opp = opp_xy - sm
    dist = np.linalg.norm(to_opp, axis=1)

    valid = (dist > 1e-6) & moving

    # 分解位移为径向 + 切向。
    to_opp_hat = np.zeros((T, 2), dtype=np.float64)
    to_opp_hat[valid] = to_opp[valid] / dist[valid, None]

    v_radial_scalar = np.zeros(T, dtype=np.float64)
    v_radial_scalar[valid] = np.sum(disp[valid] * to_opp_hat[valid], axis=1)

    v_tangential_vec = disp - v_radial_scalar[:, None] * to_opp_hat
    v_tangential_mag = np.linalg.norm(v_tangential_vec, axis=1)

    # 门控：仅在区外给信号（旧行为）。gate=False 时返回无门控 reward。
    out_zone = dist > max(float(dist_max), 1e-6)
    if gate:
        mask = out_zone & valid
        radial[mask] = (radial_coef * v_radial_scalar[mask]).astype(np.float32)
        tangential[mask] = (-tangential_coef * v_tangential_mag[mask]).astype(np.float32)
    else:
        radial[valid] = (radial_coef * v_radial_scalar[valid]).astype(np.float32)
        tangential[valid] = (-tangential_coef * v_tangential_mag[valid]).astype(np.float32)

    return radial, tangential
