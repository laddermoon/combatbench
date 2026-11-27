"""ExportedGaussianMLPPolicy — self-contained deployment artifact.

No imports from baseline.* / envs.* — only torch/numpy/math/stdlib.
Rebuilds the MLP mean network from model.pt and exposes act()/sample().
"""
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_LOG_STD_MIN, _LOG_STD_MAX = -5.0, 2.0


class ExportedGaussianMLPPolicy(nn.Module):
    """Deployment-side mirror of GaussianMLPPolicy (inference only)."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        super().__init__()
        if model_path is None:
            model_path = str(Path(__file__).resolve().parent / "model.pt")
        payload = torch.load(model_path, map_location="cpu", weights_only=False)

        arch = payload.get("arch", {})
        self.obs_dim = int(arch.get("obs_dim", payload["obs_dim"]))
        self.action_dim = int(arch.get("action_dim", payload["action_dim"]))
        self.hidden_dim = int(arch.get("hidden_dim", payload["hidden_dim"]))

        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(self.action_dim), requires_grad=False)
        self.load_state_dict(payload["state_dict"])
        self.to(torch.device(device))
        self.eval()

        self._gen = torch.Generator(device=torch.device(device))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._gen = torch.Generator(device=self.device)
            self._gen.manual_seed(int(seed))

    def _sigma(self) -> torch.Tensor:
        return torch.clamp(self.log_std, _LOG_STD_MIN, _LOG_STD_MAX).exp()

    def act(self, observation: Any, *, want_extra: bool = False):
        obs_t = torch.as_tensor(np.asarray(observation, dtype=np.float32),
                                dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.net(obs_t)
        return action.squeeze(0).cpu().numpy().astype(np.float32), None

    def sample(self, observation: Any, *, ctx=None, want_extra: bool = False):
        obs_t = torch.as_tensor(np.asarray(observation, dtype=np.float32),
                                dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean = self.net(obs_t)
            sigma = self._sigma().expand_as(mean)
            eps = torch.randn(mean.shape, generator=self._gen,
                              device=mean.device, dtype=mean.dtype)
            action = mean + sigma * eps
        return action.squeeze(0).cpu().numpy().astype(np.float32), None


# PolicyBlueprint duck-type alias
GaussianMLPPolicy = ExportedGaussianMLPPolicy
