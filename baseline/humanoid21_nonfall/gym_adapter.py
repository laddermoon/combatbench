"""
Gym 适配器模块
将 EnvRuntime 包装为标准的 Gymnasium 环境，用于 RL 训练
"""
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from combatbench.envs.humanoid21 import MujocoCombatSimulator, Humanoid21Observer
from combatbench.envs.framework import EnvRuntime, TerminationReason
from combatbench.envs.humanoid21.plugins import CombatScoringPlugin, NonFallConstraintPlugin

from .opponents import BaseCombatPolicy, make_opponent_policy
from .rewarder import Humanoid21Rewarder
from .reward_config import AttackerRewardConfig, DistanceStageRewardConfig


class SingleAgentAttackerEnv(gym.Env):
    """
    单智能体攻击者环境

    将 EnvRuntime 包装为单智能体 Gym 环境，robot_a 是训练的策略，
    robot_b 是对手策略。
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        arena_xml: str,
        *,
        # 环境参数
        dt: float = 0.002,
        control_frequency: int = 20,
        match_duration: float = 30.0,
        initial_distance: float = 2.0,
        # 对手参数
        opponent: Any = "standing",
        opponent_seed: Optional[int] = None,
        opponent_random_scale: float = 0.1,
        # 奖励参数
        curriculum_stage: str = "attack",
        reward_config: Optional[AttackerRewardConfig] = None,
        distance_stage_config: Optional[DistanceStageRewardConfig] = None,
        # 约束参数
        non_fall_mode: bool = True,
        non_fall_pitch_limit_deg: float = 5.0,
        non_fall_roll_limit_deg: float = 5.0,
        # 战斗参数
        damage_scale: float = 100.0,
        initial_health: float = 100.0,
        initial_health_a: Optional[float] = None,
        initial_health_b: Optional[float] = None,
        # 渲染参数
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # 计算物理步数
        sim_frequency = 1.0 / dt
        self.phy_steps_per_action = max(1, int(round(sim_frequency / control_frequency)))
        self.max_steps = int(match_duration * control_frequency)

        # 创建底层模拟器
        self.simulator = MujocoCombatSimulator(
            arena_xml=arena_xml,
            dt=dt,
            initial_distance=initial_distance,
        )

        # 创建对手策略
        self.opponent_policy = make_opponent_policy(
            opponent,
            seed=opponent_seed,
            random_scale=opponent_random_scale,
        )

        # 奖励配置
        self.reward_config = reward_config or AttackerRewardConfig()
        self.distance_stage_config = distance_stage_config or DistanceStageRewardConfig()
        self.curriculum_stage = curriculum_stage

        # 创建插件列表
        plugins = [
            CombatScoringPlugin(
                initial_health=initial_health,
                initial_health_a=initial_health_a,
                initial_health_b=initial_health_b,
                damage_scale=damage_scale,
            ),
        ]
        if non_fall_mode:
            plugins.append(
                NonFallConstraintPlugin(
                    pitch_limit_deg=non_fall_pitch_limit_deg,
                    roll_limit_deg=non_fall_roll_limit_deg,
                )
            )

        # 创建观察器插件
        observer_plugins = {
            "robot_a_obs": Humanoid21Observer("robot_a"),
            "robot_b_obs": Humanoid21Observer("robot_b"),
            "robot_a_reward": Humanoid21Rewarder(
                "robot_a",
                reward_config=self.reward_config,
                distance_stage_config=self.distance_stage_config,
                curriculum_stage=self.curriculum_stage,
            ),
        }

        # 创建运行时
        self.runtime = EnvRuntime(
            simulator=self.simulator,
            observer_plugins=observer_plugins,
            plugins=plugins,
            phy_steps_per_action=self.phy_steps_per_action,
            max_steps=self.max_steps,
        )

        # 设置 Gym 空间
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(Humanoid21Observer.ACTION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(Humanoid21Observer.OBS_DIM,),
            dtype=np.float32,
        )

        # 渲染模式
        self.render_mode = render_mode

        # 内部状态
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)

        # 重置运行时
        self.runtime.reset(seed=seed, options=options)

        # 重置对手策略
        if hasattr(self.opponent_policy, "reset"):
            self.opponent_policy.reset()

        # 重置内部状态
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # 获取初始观测
        obs = self.runtime.get_observer_output("robot_a_obs")

        # 构建初始 info
        info = self._build_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步"""
        # 裁剪动作
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )

        # 获取对手动作
        opponent_obs = self.runtime.get_observer_output("robot_b_obs")
        opponent_action = self.opponent_policy.act(opponent_obs)
        opponent_action = np.clip(
            np.asarray(opponent_action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )

        # 执行一步
        self.runtime.step(action, opponent_action)

        # 获取观测和奖励
        obs = self.runtime.get_observer_output("robot_a_obs")
        reward = self.runtime.get_observer_output("robot_a_reward")

        # 获取终止标志
        terminated, truncated = self.runtime.get_termination_flags()

        # 构建信息
        info = self._build_info()

        # 保存动作
        self._last_action = action.copy()

        return obs, float(reward), terminated, truncated, info

    def _build_info(self) -> Dict[str, Any]:
        """构建 info 字典"""
        shared_info = self.runtime.get_shared_info()
        metrics = shared_info.get("metrics", {})

        # 从共享信息中提取统计数据
        info = {
            "scores": {
                "robot_a": metrics.get("health_a", 100.0),
                "robot_b": metrics.get("health_b", 100.0),
            },
            "relative_metrics": {
                "robot_a": {
                    "horizontal_distance": self._get_horizontal_distance(),
                    "facing_opponent": self._get_facing_opponent(),
                },
                "robot_b": {
                    "horizontal_distance": self._get_horizontal_distance(),
                    "facing_opponent": self._get_facing_opponent(),
                },
            },
            "robot_states": {
                "robot_a": {
                    "uprightness": self._get_uprightness("robot_a"),
                },
                "robot_b": {
                    "uprightness": self._get_uprightness("robot_b"),
                },
            },
            "health": {
                "robot_a": metrics.get("health_a", 100.0),
                "robot_b": metrics.get("health_b", 100.0),
            },
            "damage_taken": {
                "robot_a": metrics.get("damage_taken_a", 0.0),
                "robot_b": metrics.get("damage_taken_b", 0.0),
            },
            "non_fall_mode": {
                "clamp_counts": {
                    "current_step": {
                        "robot_a": metrics.get("robot_a_clamp_count", 0),
                        "robot_b": metrics.get("robot_b_clamp_count", 0),
                    },
                    "episode": {
                        "robot_a": metrics.get("robot_a_clamp_count", 0),
                        "robot_b": metrics.get("robot_b_clamp_count", 0),
                    },
                },
            },
            "episode_stats": {
                "clamp_count": metrics.get("robot_a_clamp_count", 0),
                "damage_dealt": metrics.get("damage_taken_b", 0.0),
                "damage_received": metrics.get("damage_taken_a", 0.0),
                "min_horizontal_distance": self._get_horizontal_distance(),
            },
            "attacker_metrics": {
                "horizontal_distance": self._get_horizontal_distance(),
                "uprightness": self._get_uprightness("robot_a"),
            },
            "reward_terms": {},
            "winner": shared_info.get("winner"),
        }

        return info

    def _get_horizontal_distance(self) -> float:
        """获取水平距离"""
        try:
            core_state = self.runtime.simulator.get_core_state()
            pos_a = core_state["robot_a"]["root_position"]
            pos_b = core_state["robot_b"]["root_position"]
            return float(np.linalg.norm(pos_a[:2] - pos_b[:2]))
        except Exception:
            return 0.0

    def _get_facing_opponent(self) -> float:
        """获取朝向对手的程度"""
        try:
            core_state = self.runtime.simulator.get_core_state()
            pos_a = core_state["robot_a"]["root_position"]
            pos_b = core_state["robot_b"]["root_position"]
            quat_a = core_state["robot_a"]["root_orientation"]

            from scipy.spatial.transform import Rotation as R
            quat_xyzw = np.array([quat_a[1], quat_a[2], quat_a[3], quat_a[0]])
            rot = R.from_quat(quat_xyzw)
            forward_dir = rot.as_matrix()[:, 0]

            to_opponent = pos_b - pos_a
            to_opponent[2] = 0
            to_opponent_norm = np.linalg.norm(to_opponent)
            if to_opponent_norm < 1e-6:
                return 0.0

            forward_dir[2] = 0
            forward_dir = forward_dir / (np.linalg.norm(forward_dir) + 1e-8)
            return float(max(0.0, np.dot(forward_dir, to_opponent / to_opponent_norm)))
        except Exception:
            return 0.0

    def _get_uprightness(self, robot_id: str) -> float:
        """获取直立程度"""
        try:
            core_state = self.runtime.simulator.get_core_state()
            quat = core_state[robot_id]["root_orientation"]

            from scipy.spatial.transform import Rotation as R
            quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
            rot = R.from_quat(quat_xyzw)
            up_dir = rot.as_matrix()[:, 2]
            return float(max(0.0, up_dir[2]))
        except Exception:
            return 1.0

    def render(self) -> Optional[np.ndarray]:
        """渲染环境"""
        if self.render_mode == "rgb_array":
            return self.runtime.render()
        return None

    def close(self) -> None:
        """关闭环境"""
        self.runtime.close()
