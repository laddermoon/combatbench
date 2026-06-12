"""Policy that combines a primary learning policy with a frozen fallback recovery policy.

Uses the trained Gating MLP model to monitor stability and switch control
to the recovery policy when a fall is predicted, and switch back once balance is recovered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from envs.framework.policy import Policy, PolicyBlueprint
from baseline.humanoid21.curriculum.train_gating_network import GatingMLP


class MixedPolicy(Policy):
    """Dynamic switching composite policy utilizing a Gating MLP shield."""

    def __init__(
        self,
        primary_policy_bp: Dict[str, Any] | PolicyBlueprint,
        fallback_policy_bp: Dict[str, Any] | PolicyBlueprint,
        gating_model_dir: str = "/data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_model_plus",
        threshold: float = 0.65,
        release_threshold: float = 0.90,
        **kwargs: Any,
    ) -> None:
        # 1. Rebuild primary policy blueprint and live instance
        if isinstance(primary_policy_bp, dict):
            self.primary_policy_bp = PolicyBlueprint.from_dict(primary_policy_bp)
        else:
            self.primary_policy_bp = primary_policy_bp
        self.primary_policy = self.primary_policy_bp.build()

        # 2. Rebuild fallback/recovery policy blueprint and live instance
        if isinstance(fallback_policy_bp, dict):
            self.fallback_policy_bp = PolicyBlueprint.from_dict(fallback_policy_bp)
        else:
            self.fallback_policy_bp = fallback_policy_bp
        self.fallback_policy = self.fallback_policy_bp.build()

        self.threshold = float(threshold)
        self.release_threshold = float(release_threshold)
        self.gating_model_dir = Path(gating_model_dir)

        # 3. Load Gating MLP model structure & weights
        config_path = self.gating_model_dir / "gating_config.json"
        model_path = self.gating_model_dir / "gating_model.pt"

        with open(config_path, "r") as f:
            config = json.load(f)

        self.gating_network = GatingMLP(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"]
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.gating_network.load_state_dict(checkpoint["state_dict"])
        self.gating_network.eval()

        self.active_mode = "primary"  # "primary" or "fallback"

    def act(self, observation: Any, want_extra: bool = False) -> Tuple[Any, Any | None]:
        """Runs the gating network to route observation to the appropriate active policy."""
        # 1. Predict safety probability from observation
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p_safe = self.gating_network.predict_probability(obs_tensor).item()

        # 2. State machine transition logic (hysteresis - MUST match GateObserver exactly)
        if self.active_mode == "primary" and p_safe < self.threshold:
            self.active_mode = "fallback"
        elif self.active_mode == "fallback" and p_safe > self.release_threshold:
            self.active_mode = "primary"

        # 3. Dispatch action & extras
        if self.active_mode == "primary":
            action, extra = self.primary_policy.act(observation, want_extra)
            if want_extra and isinstance(extra, dict):
                extra["gating_mode"] = 1.0
                extra["p_safe"] = p_safe
        else:
            action, extra = self.fallback_policy.act(observation, want_extra)
            if want_extra and isinstance(extra, dict):
                extra["gating_mode"] = 0.0
                extra["p_safe"] = p_safe

        return action, extra

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal state machine and child policies."""
        if hasattr(self.primary_policy, "reset"):
            self.primary_policy.reset(seed)
        if hasattr(self.fallback_policy, "reset"):
            self.fallback_policy.reset(seed)
        self.active_mode = "primary"

    def close(self) -> None:
        """Release any resources held by child policies."""
        if hasattr(self.primary_policy, "close"):
            self.primary_policy.close()
        if hasattr(self.fallback_policy, "close"):
            self.fallback_policy.close()
