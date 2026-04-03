"""
Humanoid21 观测插件

严格按照 DATASPEC.md 和 OBSERVATION_zh.md 实现 96 维观测空间：
- 模块一：本体感知 (42维) - joint_pos_norm, joint_vel_norm
- 模块二：全局状态 (13维) - height, local_orientation, linear_vel, angular_vel
- 模块三：触觉力反馈 (2维) - feet_forces
- 模块四：对手观测 (39维) - basic_pose, keypoint_pos, keypoint_vel
"""

from typing import Any, Dict

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import BaseObserverPlugin, ReadOnlySimContext, TerminationReason


class Humanoid21Observer(BaseObserverPlugin):
    """Humanoid21 96维观测空间"""

    ACTION_DIM = 21
    OBS_DIM = 96  # 更新为 96 维

    def __init__(self, agent_id: str):
        if agent_id not in {"robot_a", "robot_b"}:
            raise ValueError(f"Unsupported agent_id: {agent_id}")
        self.agent_id = agent_id
        self._output: Any = None

    def on_reset(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_observation(ctx, self.agent_id)

    def on_post_step(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_observation(ctx, self.agent_id)

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_observation(ctx, self.agent_id)

    def get_output(self) -> Any:
        return self._output

    @classmethod
    def get_observation_space(cls) -> spaces.Dict:
        return spaces.Dict({
            "robot_a": spaces.Box(low=-np.inf, high=np.inf, shape=(cls.OBS_DIM,), dtype=np.float32),
            "robot_b": spaces.Box(low=-np.inf, high=np.inf, shape=(cls.OBS_DIM,), dtype=np.float32),
        })

    @classmethod
    def get_action_space(cls) -> spaces.Dict:
        return spaces.Dict({
            "robot_a": spaces.Box(low=-1.0, high=1.0, shape=(cls.ACTION_DIM,), dtype=np.float32),
            "robot_b": spaces.Box(low=-1.0, high=1.0, shape=(cls.ACTION_DIM,), dtype=np.float32),
        })

    @classmethod
    def _build_observation(cls, ctx: ReadOnlySimContext, agent_id: str) -> np.ndarray:
        """
        构建 96 维观测空间

        按照 DATASPEC.md 规范：
        - 模块一：本体感知 (42维) - 索引 [0:42]
        - 模块二：全局状态 (13维) - 索引 [42:55]
        - 模块三：触觉力反馈 (2维) - 索引 [55:57]
        - 模块四：对手观测 (39维) - 索引 [57:96]
        """
        accessor = ctx.accessor
        derived_state = accessor.get_derived_state()

        # 直接从 derived_state 获取完整观测
        observation = derived_state[agent_id]['observation']

        return observation.astype(np.float32)


class Humanoid21Rewarder(BaseObserverPlugin):
    """Humanoid21 奖励计算插件"""

    def __init__(self, agent_id: str):
        if agent_id not in {"robot_a", "robot_b"}:
            raise ValueError(f"Unsupported agent_id: {agent_id}")
        self.agent_id = agent_id
        self._output = 0.0

    def on_reset(self, ctx: ReadOnlySimContext) -> None:
        self._output = 0.0

    def on_post_step(self, ctx: ReadOnlySimContext) -> None:
        self._output = 0.0

    def on_post_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = 0.0

    def get_output(self) -> Any:
        return self._output


def build_shared_runtime_info(ctx: ReadOnlySimContext) -> Dict[str, Any]:
    """构建共享运行时信息"""
    info: Dict[str, Any] = {
        "health": {
            "robot_a": float(ctx.metrics.get("health_a", 100.0)),
            "robot_b": float(ctx.metrics.get("health_b", 100.0)),
        },
        "damage_taken": {
            "robot_a": float(ctx.metrics.get("damage_taken_a", 0.0)),
            "robot_b": float(ctx.metrics.get("damage_taken_b", 0.0)),
        },
        "winner": None,
    }
    if ctx.is_terminated:
        proposals = ctx.termination_proposals
        health_a = info["health"]["robot_a"]
        health_b = info["health"]["robot_b"]
        if TerminationReason.KO in proposals:
            if health_a <= 0 and health_b > 0:
                info["winner"] = "robot_b"
            elif health_b <= 0 and health_a > 0:
                info["winner"] = "robot_a"
            else:
                info["winner"] = "draw"
        elif TerminationReason.TIMEOUT in proposals:
            if health_a > health_b:
                info["winner"] = "robot_a"
            elif health_b > health_a:
                info["winner"] = "robot_b"
            else:
                info["winner"] = "draw"
    return info
