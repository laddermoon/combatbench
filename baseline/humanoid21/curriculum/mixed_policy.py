"""Policy that combines a primary learning policy with a frozen fallback recovery policy.

Uses the trained Gating MLP model to monitor stability and switch control
to the recovery policy when a fall is predicted, and switch back once balance is recovered.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from envs.framework.policy import Policy, PolicyBlueprint
from baseline.humanoid21.curriculum.train_gating_network import GatingMLP

_DEBUG = os.environ.get("MIXED_POLICY_DEBUG", "0") == "1"


class MixedPolicy(Policy):
    """Dynamic switching composite policy utilizing a Gating MLP shield."""

    def __init__(
        self,
        primary_policy_bp: str | Dict[str, Any] | PolicyBlueprint,
        fallback_policy_bp: str | Dict[str, Any] | PolicyBlueprint,
        gating_model_dir: str = "/data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_model_plus",
        threshold: float = 0.65,
        release_threshold: float = 0.90,
        release_patience: int = 10,
        **kwargs: Any,
    ) -> None:
        # 1. Rebuild primary policy blueprint and live instance
        self.primary_policy_bp = self._resolve_bp(primary_policy_bp)
        print(f"[MixedPolicy] primary_policy_bp: cls={self.primary_policy_bp.cls}", flush=True)
        self.primary_policy = self.primary_policy_bp.build()

        # 2. Rebuild fallback/recovery policy blueprint and live instance
        self.fallback_policy_bp = self._resolve_bp(fallback_policy_bp)
        print(f"[MixedPolicy] fallback_policy_bp: cls={self.fallback_policy_bp.cls}", flush=True)
        self.fallback_policy = self.fallback_policy_bp.build()

        self.threshold = float(threshold)
        self.release_threshold = float(release_threshold)
        self.release_patience = int(release_patience)
        self.gating_model_dir = Path(gating_model_dir)

        # 3. Load Gating MLP model structure & weights
        config_path = self.gating_model_dir / "gating_config.json"
        model_path = self.gating_model_dir / "gating_model.pt"

        print(f"[MixedPolicy] loading gating model from {self.gating_model_dir}", flush=True)
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"[MixedPolicy] gating config: input_dim={config['input_dim']}, hidden_dims={config['hidden_dims']}", flush=True)

        self.gating_network = GatingMLP(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"]
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.gating_network.load_state_dict(checkpoint["state_dict"])
        self.gating_network.eval()
        print(f"[MixedPolicy] gating model loaded, threshold={self.threshold}, release_threshold={self.release_threshold}", flush=True)

        self._step_count = 0
        self._recovery_count = 0
        self.active_mode = "primary"  # "primary" or "fallback"

    @staticmethod
    def _resolve_bp(bp: str | Dict[str, Any] | PolicyBlueprint) -> PolicyBlueprint:
        """Normalise a blueprint source to a :class:`PolicyBlueprint`.

        Accepts:
        * ``str`` — path to a YAML/JSON blueprint file.
        * ``dict`` — raw blueprint document (``PolicyBlueprint.from_dict``).
        * ``PolicyBlueprint`` — already resolved.
        """
        if isinstance(bp, PolicyBlueprint):
            return bp
        if isinstance(bp, str):
            return PolicyBlueprint.load(bp)
        return PolicyBlueprint.from_dict(bp)

    def act(self, observation: Any, want_extra: bool = False) -> Tuple[Any, Any | None]:
        """Runs the gating network to route observation to the appropriate active policy."""
        # 1. Predict safety probability from observation
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p_safe = self.gating_network.predict_probability(obs_tensor).item()

        # 2. State machine transition logic (hysteresis with patience)
        prev_mode = self.active_mode
        if self.active_mode == "primary" and p_safe < self.threshold:
            self.active_mode = "fallback"
            self._recovery_count = 0
        elif self.active_mode == "fallback":
            if p_safe > self.release_threshold:
                self._recovery_count += 1
                if self._recovery_count >= self.release_patience:
                    self.active_mode = "primary"
                    self._recovery_count = 0
                    if hasattr(self.primary_policy, "reset"):
                        self.primary_policy.reset()
            else:
                self._recovery_count = 0

        if _DEBUG:
            self._step_count += 1
            if self.active_mode != prev_mode or self._step_count % 1 == 0:
                print(
                    f"[MixedPolicy] step={self._step_count} p_safe={p_safe:.4f} "
                    f"mode={prev_mode}->{self.active_mode} "
                    f"recovery={self._recovery_count}/{self.release_patience} "
                    f"threshold={self.threshold} release={self.release_threshold}",
                    flush=True,
                )

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
        if _DEBUG:
            print(f"[MixedPolicy] reset(seed={seed}), mode={self.active_mode}->primary", flush=True)
        if hasattr(self.primary_policy, "reset"):
            self.primary_policy.reset(seed)
        if hasattr(self.fallback_policy, "reset"):
            self.fallback_policy.reset(seed)
        self.active_mode = "primary"
        self._step_count = 0
        self._recovery_count = 0

    def close(self) -> None:
        """Release any resources held by child policies."""
        if hasattr(self.primary_policy, "close"):
            self.primary_policy.close()
        if hasattr(self.fallback_policy, "close"):
            self.fallback_policy.close()
