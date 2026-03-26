from typing import Any, Callable, Dict, Optional

import numpy as np


class BaseControlMode:
    name = "base"

    def reset(self, env: Any, robot_id: str) -> None:
        pass

    def resolve_action(self, env: Any, robot_id: str, action: Optional[np.ndarray]) -> np.ndarray:
        if action is None:
            return np.zeros(env.action_space[robot_id].shape, dtype=np.float32)
        return np.asarray(action, dtype=np.float32).reshape(env.action_space[robot_id].shape)


class PolicyControlMode(BaseControlMode):
    name = "policy"


class ZeroActionControlMode(BaseControlMode):
    name = "zero_action"

    def resolve_action(self, env: Any, robot_id: str, action: Optional[np.ndarray]) -> np.ndarray:
        return np.zeros(env.action_space[robot_id].shape, dtype=np.float32)


class FixedActionControlMode(BaseControlMode):
    name = "fixed_action"

    def __init__(self, action: np.ndarray):
        self.action = np.asarray(action, dtype=np.float32)

    def resolve_action(self, env: Any, robot_id: str, action: Optional[np.ndarray]) -> np.ndarray:
        return np.asarray(self.action, dtype=np.float32).reshape(env.action_space[robot_id].shape)


class CallbackControlMode(BaseControlMode):
    name = "callback"

    def __init__(self, callback: Callable[[Any, str, Optional[np.ndarray]], np.ndarray]):
        self.callback = callback

    def resolve_action(self, env: Any, robot_id: str, action: Optional[np.ndarray]) -> np.ndarray:
        result = self.callback(env, robot_id, action)
        return np.asarray(result, dtype=np.float32).reshape(env.action_space[robot_id].shape)


def build_default_control_modes() -> Dict[str, BaseControlMode]:
    return {
        "robot_a": PolicyControlMode(),
        "robot_b": PolicyControlMode(),
    }
