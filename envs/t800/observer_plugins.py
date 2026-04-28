"""
T800 观测插件

基础版 Observer：直接从 derived_state 读取完整 observation 向量。
"""

from typing import Any

import numpy as np
from gymnasium import spaces

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework import BaseObserverPlugin, ReadOnlySimContext


class T800Observer(BaseObserverPlugin):
    """T800 104维观测空间"""

    ACTION_DIM = 25
    OBS_DIM = 104

    def __init__(self, agent_id: str):
        if agent_id not in {"robot_a", "robot_b"}:
            raise ValueError(f"Unsupported agent_id: {agent_id}")
        self.agent_id = agent_id
        self._output: Any = None

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._build_observation(ctx, self.agent_id)

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
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
        accessor = ctx.accessor
        derived_state = accessor.get_derived_state()
        observation = derived_state[agent_id]["observation"]
        return np.asarray(observation, dtype=np.float32)
