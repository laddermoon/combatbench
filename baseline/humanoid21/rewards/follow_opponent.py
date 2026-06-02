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

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# 距离带半径（米）：root-to-root 距离 <= 此值视为“在范围内”。与
# opponent_relation 的 dist_max 保持一致语义。
FOLLOW_DIST_MAX = 0.9

# ApproachVelocityRewarder 系数。
# 径向（朝对手）位移奖励系数；切向（绕圈）位移惩罚系数。
APPROACH_RADIAL_COEF = 1.0
APPROACH_TANGENTIAL_COEF = 1.0
# 单步位移模长上限（米）。正常走路每个 action step 位移约 0.1m；摔倒/数值
# 异常可能产生瞬时大跳，clip 掉以避免奖励尖峰污染训练（防 NaN/爆梯度）。
APPROACH_DISP_CLIP = 0.5


class InZoneHoldRewarder(BaseObserverPlugin):
    """范围内保持奖励（稀疏指示，用于“留在对手身边”）。

    与摔倒惩罚同构的稀疏方案::

        r_t = 1.0   if  distance(self, opp) <= dist_max
              0.0   otherwise

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
        self_pos = np.asarray(core_state[self.agent_id]["root_pos"], dtype=np.float64)
        opp_pos = np.asarray(core_state[self.opponent_id]["root_pos"], dtype=np.float64)
        distance = float(np.linalg.norm(opp_pos - self_pos))
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
    """主动接近奖励（自身运动归因 + 径向/切向分解）。

    每步用**自身 root 的水平位移**作为“主动移动向量”（世界系，归因到自身，
    不含对手运动这一干扰项）::

        disp_t = self_xy(t) - self_xy(t-1)
        u_t    = (opp_xy - self_xy) / ||·||      # 指向对手的水平单位向量

    将位移分解为径向（朝对手）+ 切向（绕圈）两个正交分量，输出两个值：

      * ``radial``            = radial_coef * (disp · u)
            径向分量，**带符号**：朝对手为正、远离为负。这是“主动接近”的奖励。
      * ``tangential_penalty`` = -tangential_coef * ||disp - (disp·u) u||
            切向分量模长，**恒 <= 0**：直接给“绕圈/横移”记负分，掐死转圈现象。

    仅在 ``distance > dist_max``（区外）时给信号；区内置零（交给
    :class:`InZoneHoldRewarder` 的保持奖励）。聚合交给 PPO 的 γ/GAE——单步
    投影标量求和 ≈ 自身朝对手的净位移，摆腿相的瞬时反向在求和中自然抵消。

    设计依据（为什么用位移而非 ``root_vel_local``）：``root_vel_local`` 是
    **局部坐标系**速度，需用 root_rot 旋转到世界系才能与 ``u`` 比较；用
    root_pos 的逐步位移天然是世界系，直接可投影，且就是“移动向量”本身。
    """

    def __init__(
        self,
        agent_id: str,
        dist_max: float = FOLLOW_DIST_MAX,
        radial_coef: float = APPROACH_RADIAL_COEF,
        tangential_coef: float = APPROACH_TANGENTIAL_COEF,
        disp_clip: float = APPROACH_DISP_CLIP,
    ) -> None:
        self.agent_id = str(agent_id)
        self.opponent_id = "robot_b" if self.agent_id == "robot_a" else "robot_a"
        self.dist_max = float(dist_max)
        self.radial_coef = float(radial_coef)
        self.tangential_coef = float(tangential_coef)
        self.disp_clip = float(disp_clip)
        self._radial: float = 0.0
        self._tangential_penalty: float = 0.0
        self._prev_self_xy: np.ndarray = np.zeros(2, dtype=np.float64)

    def _self_xy(self, ctx: ReadOnlySimContext) -> np.ndarray:
        core_state = ctx.accessor.get_core_state()
        return np.asarray(
            core_state[self.agent_id]["root_pos"], dtype=np.float64
        )[:2].copy()

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._radial = 0.0
        self._tangential_penalty = 0.0
        # 用 reset 状态初始化 prev，使第一步就有真实位移。
        try:
            self._prev_self_xy = self._self_xy(ctx)
        except Exception:
            self._prev_self_xy = np.zeros(2, dtype=np.float64)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()
        self_xy = np.asarray(
            core_state[self.agent_id]["root_pos"], dtype=np.float64
        )[:2]
        opp_xy = np.asarray(
            core_state[self.opponent_id]["root_pos"], dtype=np.float64
        )[:2]

        # 主动移动向量（自身水平位移），并刷新 prev。
        disp = self_xy - self._prev_self_xy
        self._prev_self_xy = self_xy.copy()

        to_opp = opp_xy - self_xy
        distance = float(np.linalg.norm(to_opp))

        # 区内或方向退化 -> 不给接近信号。
        if distance <= self.dist_max or distance < 1e-6:
            self._radial = 0.0
            self._tangential_penalty = 0.0
            return

        # clip 位移模长，防摔倒/数值异常的瞬时大跳产生奖励尖峰。
        disp_mag = float(np.linalg.norm(disp))
        if disp_mag > self.disp_clip and disp_mag > 1e-9:
            disp = disp * (self.disp_clip / disp_mag)

        u = to_opp / distance
        v_radial = float(np.dot(disp, u))             # 带符号：朝对手为正
        tangential_vec = disp - v_radial * u
        tangential_mag = float(np.linalg.norm(tangential_vec))

        self._radial = self.radial_coef * v_radial
        self._tangential_penalty = -self.tangential_coef * tangential_mag

    def get_output(self) -> Dict[str, float]:
        # 两个值：径向接近奖励（带符号）+ 切向绕圈惩罚（<=0）。
        return {
            "radial": float(self._radial),
            "tangential_penalty": float(self._tangential_penalty),
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "dist_max": self.dist_max,
            "radial_coef": self.radial_coef,
            "tangential_coef": self.tangential_coef,
            "disp_clip": self.disp_clip,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "ApproachVelocityRewarder":
        return cls(**config)