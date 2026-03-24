from typing import Any, Optional

import numpy as np
import torch

from combatbench.baseline.mujoco21dof_nonfall.grpo import load_grpo_checkpoint, resolve_device
from combatbench.policy import BaseCombatPolicy


class GRPOCombatPolicy(BaseCombatPolicy):
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
        self.device = resolve_device(device)
        self.deterministic = bool(deterministic)
        self.actor, self.checkpoint = load_grpo_checkpoint(self.model_path, device=self.device)

    def act(self, obs: np.ndarray, info: Optional[dict] = None) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_tensor = self.actor.act(obs_tensor, deterministic=self.deterministic)
        action = action_tensor.detach().cpu().numpy()[0].astype(np.float32)
        return np.clip(action, -1.0, 1.0).reshape(self.ACTION_DIM)

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model_path={self.model_path!r}, "
            f"device={str(self.device)!r}, deterministic={self.deterministic!r})"
        )
