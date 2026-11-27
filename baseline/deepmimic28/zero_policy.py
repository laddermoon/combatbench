"""零动作陪衬策略：robot_b 输出归一化 0（= action bounds center 保持姿态）。"""
from typing import Any, Optional

import numpy as np

from envs.framework.policy import Policy


class ZeroPolicy(Policy):
    def __init__(self, action_dim: int = 28):
        self._action_dim = action_dim

    @property
    def name(self) -> str:
        return "zero"

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, observation: Any, *, want_extra: bool = False):
        return np.zeros(self._action_dim, dtype=np.float32), None

    def sample(self, observation: Any, *, ctx=None, want_extra: bool = False):
        return np.zeros(self._action_dim, dtype=np.float32), None
