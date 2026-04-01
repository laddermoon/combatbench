"""
Rewarder 模块
基于新框架的 ReadOnlySimContext 计算奖励
"""
from typing import Any, Dict, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from combatbench.envs.framework.runtime_plugin import BaseObserverPlugin
from combatbench.envs.framework.context import ReadOnlySimContext, TerminationReason

from .reward_config import (
    AttackerRewardConfig,
    DistanceStageRewardConfig,
    compute_attacker_reward,
    compute_distance_stage_reward,
    zero_reward_terms,
)


class Humanoid21Rewarder(BaseObserverPlugin):
    """
    21自由度人形机器人奖励计算器

    从 ReadOnlySimContext 提取指标并计算奖励，支持两种课程模式：
    - "attack": 攻击模式，鼓励造成伤害
    - "distance_stage1": 距离阶段模式，先学会接近对手
    """

    ACTION_DIM = 21

    def __init__(
        self,
        agent_id: str,
        reward_config: Optional[AttackerRewardConfig] = None,
        distance_stage_config: Optional[DistanceStageRewardConfig] = None,
        curriculum_stage: str = "attack",
    ):
        if agent_id not in {"robot_a", "robot_b"}:
            raise ValueError(f"Unsupported agent_id: {agent_id}")
        self.agent_id = agent_id
        self.opponent_id = "robot_b" if agent_id == "robot_a" else "robot_a"
        self.reward_config = reward_config or AttackerRewardConfig()
        self.distance_stage_config = distance_stage_config or DistanceStageRewardConfig()
        self.curriculum_stage = curriculum_stage

        # 内部状态
        self._output = 0.0
        self._prev_metrics: Optional[Dict[str, float]] = None
        self._episode_damage_dealt = 0.0
        self._episode_damage_received = 0.0
        self._episode_hits_dealt = 0
        self._episode_hits_received = 0
        self._episode_clamp_count = 0
        self._episode_min_horizontal_distance = float("inf")

    def on_reset(self, ctx: ReadOnlySimContext) -> None:
        self._output = 0.0
        self._prev_metrics = self._extract_metrics(ctx)
        self._episode_damage_dealt = 0.0
        self._episode_damage_received = 0.0
        self._episode_hits_dealt = 0
        self._episode_hits_received = 0
        self._episode_clamp_count = 0
        self._episode_min_horizontal_distance = float("inf")

    def on_post_step(self, ctx: ReadOnlySimContext) -> None:
        curr_metrics = self._extract_metrics(ctx)

        # 更新累计统计
        self._episode_damage_dealt += curr_metrics["damage_dealt"]
        self._episode_damage_received += curr_metrics["damage_received"]
        self._episode_hits_dealt += int(curr_metrics["hits_dealt"])
        self._episode_hits_received += int(curr_metrics["hits_received"])
        self._episode_clamp_count += int(curr_metrics["clamp_count"])
        self._episode_min_horizontal_distance = min(
            self._episode_min_horizontal_distance,
            curr_metrics["horizontal_distance"],
        )

        # 添加累计统计到指标
        curr_metrics["episode_damage_dealt"] = self._episode_damage_dealt
        curr_metrics["episode_damage_received"] = self._episode_damage_received
        curr_metrics["episode_hits_dealt"] = float(self._episode_hits_dealt)
        curr_metrics["episode_hits_received"] = float(self._episode_hits_received)
        curr_metrics["episode_clamp_count"] = float(self._episode_clamp_count)
        curr_metrics["episode_min_horizontal_distance"] = self._episode_min_horizontal_distance

        # 计算奖励
        if self.curriculum_stage == "distance_stage1":
            reward, _ = compute_distance_stage_reward(curr_metrics, self.distance_stage_config)
        else:
            reward, _ = compute_attacker_reward(curr_metrics, self.reward_config)

        self._output = reward
        self._prev_metrics = curr_metrics

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = 0.0

    def get_output(self) -> Any:
        return self._output

    def _extract_metrics(self, ctx: ReadOnlySimContext) -> Dict[str, float]:
        """从 ReadOnlySimContext 提取奖励计算所需的指标"""
        accessor = ctx.accessor
        static_data = accessor.get_static_data()
        core_state = accessor.get_core_state()
        derived_state = accessor.get_derived_state()
        action_data = accessor.get_action()

        robot_info = static_data.get("robot_info", {})
        if self.agent_id not in robot_info or self.opponent_id not in robot_info:
            return self._zero_metrics()

        # 获取 qpos/qvel 数组
        qpos_array = core_state.get("qpos", np.array([]))
        qvel_array = core_state.get("qvel", np.array([]))
        data = accessor.data
        model = accessor.model

        # 提取机器人状态
        agent_state = core_state.get(self.agent_id, {})
        opponent_state = core_state.get(self.opponent_id, {})

        # 根部位置和姿态
        agent_root_pos = agent_state.get("root_position", np.zeros(3))
        agent_root_quat_wxyz = agent_state.get("root_orientation", np.array([1.0, 0.0, 0.0, 0.0]))
        opponent_root_pos = opponent_state.get("root_position", np.zeros(3))

        # 计算距离指标
        horizontal_distance = np.linalg.norm(agent_root_pos[:2] - opponent_root_pos[:2])
        distance = np.linalg.norm(agent_root_pos - opponent_root_pos)

        # 计算朝向指标
        facing_opponent = self._compute_facing_ratio(agent_root_quat_wxyz, agent_root_pos, opponent_root_pos)

        # 计算 uprightness
        uprightness = self._compute_uprightness(agent_root_quat_wxyz)

        # 从 metrics 获取伤害和 clamp 信息
        metrics = ctx.metrics
        damage_dealt = 0.0
        damage_received = 0.0

        # 从 events 获取命中信息
        hits_dealt = 0
        hits_received = 0
        for event in ctx.events:
            if event.get("type") == "hit":
                attacker = event.get("attacker")
                defender = event.get("defender")
                dmg = event.get("damage", 0.0)
                if attacker == self.agent_id:
                    damage_dealt += dmg
                    hits_dealt += 1
                if defender == self.agent_id:
                    damage_received += dmg
                    hits_received += 1

        # 获取当前步的 clamp count
        clamp_key = f"{self.agent_id}_clamp_count"
        current_step_clamp = 1.0 if metrics.get(clamp_key, 0) > self._episode_clamp_count else 0.0

        # 获取动作
        action = action_data.get(self.agent_id, np.zeros(self.ACTION_DIM))
        action_magnitude = float(np.mean(np.abs(action)))

        # 获取上一个动作用于计算 delta
        prev_action = np.zeros(self.ACTION_DIM)
        if self._prev_metrics is not None:
            # 这里简化处理，实际应该从 ctx.accessor 获取上一步动作
            # 暂时使用当前动作代替
            pass
        action_delta = float(np.mean(np.abs(action - prev_action)))

        # 计算距离增量
        prev_horizontal_distance = self._prev_metrics.get("horizontal_distance", horizontal_distance) if self._prev_metrics else horizontal_distance
        horizontal_distance_delta = prev_horizontal_distance - horizontal_distance

        # 计算 facing delta
        prev_facing = self._prev_metrics.get("facing_opponent", facing_opponent) if self._prev_metrics else facing_opponent
        facing_delta = facing_opponent - prev_facing

        # 计算 uprightness delta
        prev_uprightness = self._prev_metrics.get("uprightness", uprightness) if self._prev_metrics else uprightness
        uprightness_delta = uprightness - prev_uprightness

        # 距离误差（用于 distance_stage1）
        distance_error = abs(horizontal_distance - self.distance_stage_config.target_distance)
        if self._prev_metrics is not None:
            prev_distance_error = self._prev_metrics.get("distance_error", distance_error)
            distance_error_delta = prev_distance_error - distance_error
        else:
            distance_error_delta = 0.0

        # 判断胜负
        win = 0.0
        loss = 0.0
        if ctx.is_terminated:
            health_a = metrics.get("health_a", 100.0)
            health_b = metrics.get("health_b", 100.0)
            if TerminationReason.KO in ctx.termination_proposals:
                if health_a <= 0 and health_b > 0:
                    if self.agent_id == "robot_b":
                        win = 1.0
                    else:
                        loss = 1.0
                elif health_b <= 0 and health_a > 0:
                    if self.agent_id == "robot_a":
                        win = 1.0
                    else:
                        loss = 1.0
            elif TerminationReason.TIMEOUT in ctx.termination_proposals:
                if health_a > health_b:
                    if self.agent_id == "robot_a":
                        win = 1.0
                    else:
                        loss = 1.0
                elif health_b > health_a:
                    if self.agent_id == "robot_b":
                        win = 1.0
                    else:
                        loss = 1.0

        return {
            "damage_dealt": damage_dealt,
            "damage_received": damage_received,
            "distance": distance,
            "horizontal_distance": horizontal_distance,
            "horizontal_distance_delta": horizontal_distance_delta,
            "distance_error": distance_error,
            "distance_error_delta": distance_error_delta,
            "facing_opponent": facing_opponent,
            "facing_delta": facing_delta,
            "uprightness": uprightness,
            "uprightness_delta": uprightness_delta,
            "hits_dealt": float(hits_dealt),
            "hits_received": float(hits_received),
            "action_magnitude": action_magnitude,
            "action_delta": action_delta,
            "clamp_count": current_step_clamp,
            "win": win,
            "loss": loss,
        }

    def _zero_metrics(self) -> Dict[str, float]:
        """返回零指标（当数据不可用时）"""
        return {
            "damage_dealt": 0.0,
            "damage_received": 0.0,
            "distance": 0.0,
            "horizontal_distance": 0.0,
            "horizontal_distance_delta": 0.0,
            "distance_error": 0.0,
            "distance_error_delta": 0.0,
            "facing_opponent": 0.0,
            "facing_delta": 0.0,
            "uprightness": 1.0,
            "uprightness_delta": 0.0,
            "hits_dealt": 0.0,
            "hits_received": 0.0,
            "action_magnitude": 0.0,
            "action_delta": 0.0,
            "clamp_count": 0.0,
            "win": 0.0,
            "loss": 0.0,
        }

    def _compute_facing_ratio(
        self,
        root_quat_wxyz: np.ndarray,
        root_pos: np.ndarray,
        opponent_pos: np.ndarray,
    ) -> float:
        """计算朝向对手的程度 [0, 1]"""
        try:
            # 转换四元数格式 wxyz -> xyzw
            root_quat_xyzw = np.array([
                root_quat_wxyz[1], root_quat_wxyz[2],
                root_quat_wxyz[3], root_quat_wxyz[0]
            ])
            rot = R.from_quat(root_quat_xyzw)
            forward_dir = rot.as_matrix()[:, 0]  # x轴为前方

            to_opponent = opponent_pos - root_pos
            to_opponent[2] = 0  # 只看水平方向
            to_opponent_norm = np.linalg.norm(to_opponent)

            if to_opponent_norm < 1e-6:
                return 0.0

            to_opponent_unit = to_opponent / to_opponent_norm
            forward_dir[2] = 0  # 只看水平方向
            forward_dir = forward_dir / (np.linalg.norm(forward_dir) + 1e-8)

            return float(max(0.0, np.dot(forward_dir, to_opponent_unit)))
        except Exception:
            return 0.0

    def _compute_uprightness(self, root_quat_wxyz: np.ndarray) -> float:
        """计算直立程度 [0, 1]，1表示完全直立"""
        try:
            root_quat_xyzw = np.array([
                root_quat_wxyz[1], root_quat_wxyz[2],
                root_quat_wxyz[3], root_quat_wxyz[0]
            ])
            rot = R.from_quat(root_quat_xyzw)
            up_dir = rot.as_matrix()[:, 2]  # z轴为上方
            return float(max(0.0, up_dir[2]))
        except Exception:
            return 1.0
