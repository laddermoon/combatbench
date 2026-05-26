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
OPP_REL_DIST_MAX = 2.2
OPP_REL_HEADING_MAX_ANGLE_DEG = 25.0
OPP_REL_DIST_PENALTY_COEF = 1.0
OPP_REL_HEADING_PENALTY_COEF = 0.02  # deg * coef: max ~155deg * 0.02 ≈ 3, same scale as dist penalty


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

