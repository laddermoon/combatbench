import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch import nn

# Ensure the combatbench repo root (containing ``envs/`` and ``policy/``)
# is importable, regardless of whether this module is loaded as a package
# member or via load_policy's file-based import.
for parent in Path(__file__).resolve().parents:
    if (parent / "envs" / "framework" / "policy.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break
    if (parent / "combatbench" / "envs" / "framework" / "policy.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

try:
    from envs.framework.policy import Policy
except ImportError:
    from combatbench.envs.framework.policy import Policy  # type: ignore[no-redef]


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class StandingCombatPolicy(Policy):
    def __init__(self, model_path: Optional[str] = None, **_ignored: Any):
        # Silently drop unknown kwargs (e.g. observation_space / action_space
        # if a caller passes them) so load_policy's query-string plumbing is
        # forgiving.
        payload_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parent / "model.pt"
        payload = torch.load(payload_path, map_location="cpu")
        hidden_dim = int(payload.get("hidden_dim", payload.get("actor_hidden_dim", 256)))
        self.actor = Actor(payload["obs_dim"], payload["action_dim"], hidden_dim)
        model_state_dict = self.actor.state_dict()
        filtered_state_dict = {
            key: value
            for key, value in payload["state_dict"].items()
            if key in model_state_dict
        }
        incompatible = self.actor.load_state_dict(filtered_state_dict, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(f"Missing keys in exported standing policy: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected keys in exported standing policy: {incompatible.unexpected_keys}")
        self.actor.eval()

    def act(
        self,
        observation: Any,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, None]:
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor)
        return action.squeeze(0).cpu().numpy().astype(np.float32), None

    def reset(self, seed: Optional[int] = None) -> None:
        # Stateless deterministic policy; ignore the seed.
        return None
