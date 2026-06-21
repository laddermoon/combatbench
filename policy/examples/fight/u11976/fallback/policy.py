"""Policy module - imports from repo to reuse TanhGaussianMLPPolicy."""
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch

# Import from repo - requires baseline/ to be on sys.path
from baseline.common.policies.tanh_gaussian_mlp import TanhGaussianMLPPolicy
from envs.framework.policy import Policy


class ExportedMLPPolicy(Policy):
    """Runtime-loadable policy backed by a ``model.pt`` checkpoint.

    Uses :class:`TanhGaussianMLPPolicy` from the training repo for
    consistent architecture and behavior.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        stochastic: bool = False,
        **_ignored: Any,
    ):
        payload_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parent / "model.pt"
        payload = torch.load(payload_path, map_location="cpu")

        hidden_dim = int(payload.get("hidden_dim", payload.get("actor_hidden_dim", 256)))

        # Reuse training-time policy class (no code duplication).
        self._policy = TanhGaussianMLPPolicy(
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            hidden_dim=hidden_dim,
            log_std_min=float(payload.get("log_std_min", -4.0)),
            log_std_max=float(payload.get("log_std_max", 0.0)),
        )
        self._policy.load_state_dict(payload["state_dict"], strict=False)
        self._policy.eval()
        self.stochastic = bool(stochastic)

    def act(
        self,
        observation: Any,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, None]:
        """Return action for given observation."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if self.stochastic:
                action, _ = self._policy.sample_action(obs_tensor)
            else:
                action = self._policy.deterministic_action(obs_tensor)
        return action.squeeze(0).cpu().numpy().astype(np.float32), None

    def reset(self, seed: Optional[int] = None) -> None:
        """Optional: reseed RNG for reproducible rollouts."""
        if seed is not None:
            torch.manual_seed(seed)
        return None


# Backward compatibility alias
Policy = ExportedMLPPolicy
