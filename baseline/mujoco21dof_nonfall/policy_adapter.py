from typing import Any, Optional

import numpy as np
from stable_baselines3 import PPO

from combatbench.policy import BaseCombatPolicy


class SB3PPOCombatPolicy(BaseCombatPolicy):
    def __init__(
        self,
        model_path: str,
        observation_space: Optional[Any] = None,
        action_space: Optional[Any] = None,
        device: str = "auto",
        deterministic: bool = True,
        **kwargs,
    ):
        super().__init__(observation_space=observation_space, action_space=action_space, **kwargs)
        self.model_path = str(model_path)
        self.device = device
        self.deterministic = bool(deterministic)
        self.model = PPO.load(self.model_path, device=self.device)

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32)
        action, _ = self.model.predict(obs_array, deterministic=self.deterministic)
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0).reshape(self.ACTION_DIM)

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model_path={self.model_path!r}, "
            f"device={self.device!r}, deterministic={self.deterministic!r})"
        )
