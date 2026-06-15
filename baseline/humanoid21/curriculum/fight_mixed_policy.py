"""Policy that combines a primary learning Fight policy, a frozen Follow policy, and a frozen Recover policy.

Uses the trained Gating MLP model to monitor stability and switch control to the recovery policy
when a fall is predicted, and a distance-based hysteresis mechanism to switch to Follow when out of range.
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

_DEBUG = os.environ.get("FIGHT_MIXED_POLICY_DEBUG", "0") == "1"


class FightMixedPolicy(Policy):
    """Dynamic three-way switching composite policy utilizing a Gating MLP safety shield and distance-based Follow fallback."""

    def __init__(
        self,
        primary_policy_bp: str | Dict[str, Any] | PolicyBlueprint,
        follow_policy_bp: str | Dict[str, Any] | PolicyBlueprint,
        fallback_policy_bp: str | Dict[str, Any] | PolicyBlueprint,
        gating_model_dir: str = "/data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_model_plus_mix_level",
        threshold: float = 0.65,
        release_threshold: float = 0.90,
        release_patience: int = 10,
        distance_fallback_threshold: float = 1.3,
        distance_recover_threshold: float = 1.0,
        **kwargs: Any,
    ) -> None:
        # 1. Rebuild primary (Fight) policy blueprint and live instance
        self.primary_policy_bp = self._resolve_bp(primary_policy_bp)
        self.primary_policy = self.primary_policy_bp.build()

        # 2. Rebuild Follow policy blueprint and live instance
        self.follow_policy_bp = self._resolve_bp(follow_policy_bp)
        self.follow_policy = self.follow_policy_bp.build()

        # 3. Rebuild fallback/recovery policy blueprint and live instance
        self.fallback_policy_bp = self._resolve_bp(fallback_policy_bp)
        self.fallback_policy = self.fallback_policy_bp.build()

        self.threshold = float(threshold)
        self.release_threshold = float(release_threshold)
        self.release_patience = int(release_patience)
        self.distance_fallback_threshold = float(distance_fallback_threshold)
        self.distance_recover_threshold = float(distance_recover_threshold)
        self.gating_model_dir = Path(gating_model_dir)

        # 4. Load Gating MLP model structure & weights
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

        self._step_count = 0
        self._recovery_count = 0
        self.active_mode = "fight"  # "fight", "follow", or "recover"

    @staticmethod
    def _resolve_bp(bp: str | Dict[str, Any] | PolicyBlueprint) -> PolicyBlueprint:
        """Normalise a blueprint source to a :class:`PolicyBlueprint`."""
        if isinstance(bp, PolicyBlueprint):
            return bp
        if isinstance(bp, str):
            return PolicyBlueprint.load(bp)
        return PolicyBlueprint.from_dict(bp)

    def act(self, observation: Any, want_extra: bool = False) -> Tuple[Any, Any | None]:
        """Runs the safety shield and distance metrics to route observation to the appropriate policy."""
        # 1. Predict safety probability from observation
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p_safe = self.gating_network.predict_probability(obs_tensor).item()

        # 2. Extract relative opponent 2D distance
        # Opponent section starts at index 57 in the 96-dim observation.
        # Opponent relative root position is indices [57, 58, 59] (Ego's frame).
        opp_rel_x = float(observation[57])
        opp_rel_y = float(observation[58])
        dist = float(np.sqrt(opp_rel_x**2 + opp_rel_y**2))

        # 3. State machine transition logic
        prev_mode = self.active_mode

        # Trigger recovery immediately if balance is lost (highest priority)
        if p_safe < self.threshold:
            if self.active_mode != "recover":
                self.active_mode = "recover"
                self._recovery_count = 0
                if hasattr(self.fallback_policy, "reset"):
                    self.fallback_policy.reset()

        # If in recover mode: stay until stable, then choose fight or follow based on distance
        elif self.active_mode == "recover":
            if p_safe > self.release_threshold:
                self._recovery_count += 1
                if self._recovery_count >= self.release_patience:
                    if dist > self.distance_fallback_threshold:
                        self.active_mode = "follow"
                        if hasattr(self.follow_policy, "reset"):
                            self.follow_policy.reset()
                    else:
                        self.active_mode = "fight"
                        if hasattr(self.primary_policy, "reset"):
                            self.primary_policy.reset()
                    self._recovery_count = 0
            else:
                self._recovery_count = 0

        # If in fight mode: switch to follow if opponent evades too far
        elif self.active_mode == "fight":
            if dist > self.distance_fallback_threshold:
                self.active_mode = "follow"
                if hasattr(self.follow_policy, "reset"):
                    self.follow_policy.reset()

        # If in follow mode: switch back to fight once opponent is within reach
        elif self.active_mode == "follow":
            if dist <= self.distance_recover_threshold:
                self.active_mode = "fight"
                if hasattr(self.primary_policy, "reset"):
                    self.primary_policy.reset()

        if _DEBUG:
            self._step_count += 1
            if self.active_mode != prev_mode or self._step_count % 10 == 0:
                print(
                    f"[FightMixedPolicy] step={self._step_count} p_safe={p_safe:.4f} dist={dist:.2f}m "
                    f"mode={prev_mode}->{self.active_mode} recovery={self._recovery_count}/{self.release_patience}",
                    flush=True,
                )

        # 4. Dispatch action & extra info
        # gating_mode definitions:
        #   1.0  -> fight (primary, only this is trained on)
        #   0.0  -> recover (fallback shield, excluded from training)
        #   -1.0 -> follow (fallback locomotion, excluded from training)
        if self.active_mode == "fight":
            action, extra = self.primary_policy.act(observation, want_extra)
            if want_extra:
                if not isinstance(extra, dict):
                    extra = {}
                extra.setdefault("log_prob", 0.0)
                extra["gating_mode"] = 1.0
                extra["p_safe"] = p_safe
        elif self.active_mode == "follow":
            action, extra = self.follow_policy.act(observation, want_extra)
            if want_extra:
                if not isinstance(extra, dict):
                    extra = {}
                extra.setdefault("log_prob", 0.0)
                extra["gating_mode"] = -1.0
                extra["p_safe"] = p_safe
        else:  # recover
            action, extra = self.fallback_policy.act(observation, want_extra)
            if want_extra:
                if not isinstance(extra, dict):
                    extra = {}
                extra.setdefault("log_prob", 0.0)
                extra["gating_mode"] = 0.0
                extra["p_safe"] = p_safe

        return action, extra

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal state machine and child policies."""
        if _DEBUG:
            print(f"[FightMixedPolicy] reset(seed={seed}), mode={self.active_mode}->fight", flush=True)
        if hasattr(self.primary_policy, "reset"):
            self.primary_policy.reset(seed)
        if hasattr(self.follow_policy, "reset"):
            self.follow_policy.reset(seed)
        if hasattr(self.fallback_policy, "reset"):
            self.fallback_policy.reset(seed)
        self.active_mode = "fight"
        self._step_count = 0
        self._recovery_count = 0

    def close(self) -> None:
        """Release any resources held by child policies."""
        if hasattr(self.primary_policy, "close"):
            self.primary_policy.close()
        if hasattr(self.follow_policy, "close"):
            self.follow_policy.close()
        if hasattr(self.fallback_policy, "close"):
            self.fallback_policy.close()
