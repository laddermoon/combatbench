"""Opponent relation reward plugin for humanoid21.

Provides:
  * :class:`OpponentRelationRewarder` — Penalty-only reward based on proximity
    and heading toward opponent.

Hook conventions
----------------
Observers use the framework's canonical dispatch hooks:
``on_pre_episode`` / ``on_post_action_step``.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Opponent-relation reward (课程二：接近并朝向对手)
# 无损失时输出0， 有损失时输出小于零的损失值
# 1) 距离损失：距离在 max_dist 内为 0，超出则线性惩罚, 有一个惩罚系数参数
# 2) 朝向损失：朝向误差角 <= max_angle（角度计） 时为 0，超出则线性惩罚，有一个惩罚系数参数
OPP_REL_DIST_MAX = 0.9
OPP_REL_HEADING_MAX_ANGLE_DEG = 45.0
OPP_REL_DIST_PENALTY_COEF = 0.33 # 距离3.7米时为1
OPP_REL_HEADING_PENALTY_COEF = 0.02  # deg * coef: max ~155deg * 0.02 ≈ 3, same scale as dist penalty
# 势差 shaping 的折扣因子。1.0 = 纯几何势差 Φ(s_t)-Φ(s_{t-1})（最直观的“进步量”）。
# 若要严格符合 PBRS 定理，应设为与该奖励 critic 的 γ 一致。
OPP_REL_SHAPING_GAMMA = 0.98
# 持续水平项系数 α：在势差之上叠加 α·Φ(s_t)。势差会 telescoping（总回报只看
# 首尾，不奖励“停在好位置”），加一个小的水平项给“在区内/靠近”提供**持续**反馈，
# 让策略有理由进区并停留。0.0 = 退回纯势差。
OPP_REL_LEVEL_COEF = 0.05

class OpponentRelationRewarder(BaseObserverPlugin):
    """对手关系奖励（纯势差 / potential-based shaping）。

    定义势函数（势能越高越理想，<=0）::

        Φ(s) = -max(0, distance - dist_max) * dist_penalty_coef

    即：距离在 ``dist_max`` 内 Φ=0（理想区，无惩罚），超出则线性变负。

    每步输出的奖励是**势差** + 一个小的**持续水平项**::

        r_t = shaping_gamma * Φ(s_t) - Φ(s_{t-1})  +  level_coef * Φ(s_t)
               └────────── 势差：进步量 ──────────┘   └─ 水平：持续激励 ─┘

    势差项含义：这一步**靠近**对手（Φ 上升）→ 正；**远离**（Φ 下降）→ 负。
    它编码“进步方向”，信用分配符号天然正确，但会 telescoping（折扣总回报
    只取决于首尾势能 ``γ^T Φ(s_T) - Φ(s_0)``），因此**单用势差不奖励“停在好
    位置”**，只奖励净位移。

    水平项 ``level_coef * Φ(s_t)`` 不 telescope：在区外它是持续的小负惩罚
    （越远越疼），在区内 Φ=0 → 不罚。这给策略**持续的**进区/停留激励，弥补
    纯势差“站着不动净回报为 0”的盲区。``level_coef=0`` 退回纯势差。

    暴露 ``.in_non_penalty_zone`` 布尔属性（当前是否在理想区内），供课程门控读取。
    """

    def __init__(
        self,
        agent_id: str,
        dist_max: float = OPP_REL_DIST_MAX,
        dist_penalty_coef: float = OPP_REL_DIST_PENALTY_COEF,
        shaping_gamma: float = OPP_REL_SHAPING_GAMMA,
        level_coef: float = OPP_REL_LEVEL_COEF,
    ) -> None:
        self.agent_id = str(agent_id)
        self.opponent_id = "robot_b" if self.agent_id == "robot_a" else "robot_a"
        self.dist_max = float(dist_max)
        self.dist_penalty_coef = float(dist_penalty_coef)
        self.shaping_gamma = float(shaping_gamma)
        self.level_coef = float(level_coef)
        self.in_non_penalty_zone: bool = False
        self._output: float = 0.0
        self._prev_phi: float = 0.0

    def _compute_phi(self, ctx: ReadOnlySimContext) -> float:
        """当前状态的势能 Φ(s) = -max(0, distance - dist_max) * coef，并刷新 in_zone。"""
        core_state = ctx.accessor.get_core_state()
        self_state = core_state[self.agent_id]
        opp_state = core_state[self.opponent_id]

        self_pos = np.asarray(self_state["root_pos"], dtype=np.float64)
        opp_pos = np.asarray(opp_state["root_pos"], dtype=np.float64)
        distance = float(np.linalg.norm(opp_pos - self_pos))

        dist_excess = max(0.0, distance - self.dist_max)
        self.in_non_penalty_zone = dist_excess == 0.0
        return float(-(dist_excess * self.dist_penalty_coef))

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self.in_non_penalty_zone = False
        self._output = 0.0
        # 用 reset 状态初始化 Φ_prev，使第一步就有真实势差。
        try:
            self._prev_phi = self._compute_phi(ctx)
        except Exception:
            self._prev_phi = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        phi = self._compute_phi(ctx)
        # r_t = γ·Φ(s_t) - Φ(s_{t-1})  +  level_coef·Φ(s_t)
        self._output = float(
            self.shaping_gamma * phi - self._prev_phi + self.level_coef * phi
        )
        self._prev_phi = phi

    def get_output(self) -> Dict[str, float]:
        # 同时输出势差奖励和**绝对**区内标志。后者无法从势差反推
        # （势差只携带增量，缺 Φ_0），所以必须显式携带，供课程门控的
        # final_in_zone 统计使用。
        return {
            "reward": float(self._output),
            "in_zone": float(self.in_non_penalty_zone),
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "dist_max": self.dist_max,
            "dist_penalty_coef": self.dist_penalty_coef,
            "shaping_gamma": self.shaping_gamma,
            "level_coef": self.level_coef,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "OpponentRelationRewarder":
        return cls(**config)
'''

class OpponentRelationRewarder(BaseObserverPlugin):
    """对手关系惩罚奖励。

    两个惩罚项，无违规时均输出 0，违规时输出负值：

    1. **距离惩罚**：距离（root 到 root 的三维距离）> ``dist_max`` 时线性惩罚，
       超出量 * ``dist_penalty_coef``，无上限。
    2. **朝向惩罚**：torso 局部 x 轴在世界坐标系的三维方向，与自身 root 指向对手 root
       的三维向量之间的夹角（0-180°）> ``heading_max_angle_deg`` 时线性惩罚，
       超出量 * ``heading_penalty_coef``，无上限。

    每步输出 <= 0，无上限。

    暴露 ``.in_non_penalty_zone`` 布尔属性（距离与朝向均无惩罚），
    供课程门控读取。
    """

    def __init__(
        self,
        agent_id: str,
        dist_max: float = OPP_REL_DIST_MAX,
        heading_max_angle_deg: float = OPP_REL_HEADING_MAX_ANGLE_DEG,
        dist_penalty_coef: float = OPP_REL_DIST_PENALTY_COEF,
        heading_penalty_coef: float = OPP_REL_HEADING_PENALTY_COEF,
    ) -> None:
        self.agent_id = str(agent_id)
        self.opponent_id = "robot_b" if self.agent_id == "robot_a" else "robot_a"
        self.dist_max = float(dist_max)
        self.heading_max_angle_deg = float(heading_max_angle_deg)
        self.dist_penalty_coef = float(dist_penalty_coef)
        self.heading_penalty_coef = float(heading_penalty_coef)
        self.in_non_penalty_zone: bool = False
        self._output: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self.in_non_penalty_zone = False
        self._output = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()
        self_state = core_state[self.agent_id]
        opp_state = core_state[self.opponent_id]

        self_pos = np.asarray(self_state["root_pos"], dtype=np.float64)
        opp_pos = np.asarray(opp_state["root_pos"], dtype=np.float64)
        delta = opp_pos - self_pos
        distance = float(np.linalg.norm(delta))

        # ---- 1) Distance penalty (3-D root-to-root distance) --------
        dist_excess = max(0.0, distance - self.dist_max)
        dist_penalty = dist_excess * self.dist_penalty_coef
        dist_in_zone = dist_excess == 0.0

        # ---- 2) Heading penalty -------------------------------------
        # Angle between the torso's 3-D forward axis (world frame) and the
        # 3-D vector from self root_pos to opponent root_pos. Range: 0-180°.
        if distance < 1e-6:
            heading_penalty = 0.0
            heading_in_zone = True
        else:
            to_opp_unit = delta / distance

            # Torso 3-D forward: rotate local x-axis [1,0,0] by torso world quat.
            derived_state = ctx.accessor.get_derived_state()
            self_derived = derived_state.get(self.agent_id, {})
            torso_quat_raw = None
            body_xquat = self_derived.get("body_xquat")
            if body_xquat:
                static_data = ctx.accessor.get_static_data()
                torso_name = static_data.get(self.agent_id, {}).get(
                    "keypoint_body_names", {}
                ).get("torso")
                if torso_name and torso_name in body_xquat:
                    torso_quat_raw = body_xquat[torso_name]

            if torso_quat_raw is not None:
                # MuJoCo quat: [w, x, y, z] -> scipy: [x, y, z, w]
                q = np.asarray(torso_quat_raw, dtype=np.float64)
                torso_rot = R.from_quat([q[1], q[2], q[3], q[0]])
                forward_3d = torso_rot.apply([1.0, 0.0, 0.0])
            else:
                # Fallback: use root quaternion, project to ground plane
                q = np.asarray(self_state["root_rot"], dtype=np.float64)
                norm = float(np.linalg.norm(q))
                if norm > 1e-8:
                    q = q / norm
                    torso_rot = R.from_quat([q[1], q[2], q[3], q[0]])
                    forward_3d = torso_rot.apply([1.0, 0.0, 0.0])
                else:
                    forward_3d = np.array([1.0, 0.0, 0.0])

            f_norm = float(np.linalg.norm(forward_3d))
            forward_unit = forward_3d / f_norm if f_norm > 1e-8 else np.array([1.0, 0.0, 0.0])

            cosang = float(np.clip(np.dot(forward_unit, to_opp_unit), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cosang)))
            angle_excess = max(0.0, angle_deg - self.heading_max_angle_deg)
            heading_penalty = angle_excess * self.heading_penalty_coef
            heading_in_zone = angle_excess == 0.0

        self.in_non_penalty_zone = bool(dist_in_zone and heading_in_zone)
        self._output = float(-(dist_penalty + heading_penalty))

    def get_output(self) -> float:
        return float(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "OpponentRelationRewarder":
        return cls(**config)
'''