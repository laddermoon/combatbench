"""Runtime-loadable teacher policy wrapping an exported MLP model.pt."""
from pathlib import Path
from typing import Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from envs.framework.policy import Policy


class TeacherMLPPolicy(Policy):
    def __init__(self, model_path: Optional[str] = None,
                 stochastic: bool = False, **_ignored: Any):
        payload_path = (Path(model_path) if model_path is not None
                        else Path(__file__).resolve().parent / "model.pt")
        p = torch.load(payload_path, map_location="cpu")
        h = int(p.get("hidden_dim", 256))
        self.net = nn.Sequential(
            nn.Linear(int(p["obs_dim"]), h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, int(p["action_dim"])),
        )
        self.net.load_state_dict(
            {k[4:]: v for k, v in p["state_dict"].items() if k.startswith("net.")})
        self.net.eval()
        self._det = not stochastic
        self.log_std = p["state_dict"].get("log_std")

    def act(self, observation: Any, explore_factor: float = 0.5,
            want_extra: bool = False) -> Tuple[np.ndarray, Optional[dict]]:
        with torch.no_grad():
            a = torch.tanh(self.net(torch.as_tensor(
                observation, dtype=torch.float32))).numpy()
        return a, None

    def set_deterministic(self, deterministic: bool) -> None:
        self._det = bool(deterministic)
