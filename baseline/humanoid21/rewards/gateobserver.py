"""Observer plugin to run the trained Gating MLP model.

Predicts the safety state (p_safe) at every step and runs the exact same
state machine logic as MixedPolicy to output which model is currently active
(gating_mode: 1.0 = primary/chaser, 0.0 = fallback/recovery).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from envs.framework import BaseObserverPlugin, ReadOnlySimContext
from baseline.humanoid21.curriculum.train_gating_network import GatingMLP


class GateObserver(BaseObserverPlugin):
    """Observer plugin to monitor gating predictions and determine active mode."""

    def __init__(
        self,
        agent_id: str = "robot_a",
        model_dir: str = "/data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_model_plus",
        threshold: float = 0.65,
        release_threshold: float = 0.90,
    ) -> None:
        self.agent_id = str(agent_id)
        self.threshold = float(threshold)
        self.release_threshold = float(release_threshold)
        self.model_dir = Path(model_dir)

        # Load Gating MLP model structure & weights
        config_path = self.model_dir / "gating_config.json"
        model_path = self.model_dir / "gating_model.pt"

        with open(config_path, "r") as f:
            config = json.load(f)

        self.model = GatingMLP(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"]
        )
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.active_mode = "primary"  # "primary" or "fallback"
        self._output = {"gating_mode": 1.0, "p_safe": 1.0}

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        """Reset state machine on episode start."""
        self.active_mode = "primary"
        self._output = {"gating_mode": 1.0, "p_safe": 1.0}

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        """Run Gating MLP forward and update active mode state machine."""
        obs_dict = ctx.accessor.get_observation()
        if self.agent_id not in obs_dict:
            return

        # Get observation vector for agent
        obs = np.asarray(obs_dict[self.agent_id], dtype=np.float32)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            p_safe = self.model.predict_probability(obs_tensor).item()

        # State machine transition logic (hysteresis - MUST match MixedPolicy exactly)
        if self.active_mode == "primary" and p_safe < self.threshold:
            self.active_mode = "fallback"
        elif self.active_mode == "fallback" and p_safe > self.release_threshold:
            self.active_mode = "primary"

        gating_mode = 1.0 if self.active_mode == "primary" else 0.0
        self._output = {
            "gating_mode": gating_mode,
            "p_safe": p_safe,
        }

    def get_output(self) -> Dict[str, Any]:
        """Expose current step outputs to extract_rewards."""
        return self._output
