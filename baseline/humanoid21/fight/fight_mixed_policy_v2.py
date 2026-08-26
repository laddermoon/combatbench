"""Policy that combines a primary learning Fight policy, a frozen Recover policy,
and a frozen Standup policy.

Uses the trained Gating MLP model to monitor stability and switch control to the
recovery policy when a fall is predicted.  When the robot is on the ground
(height below ``fall_height``), control switches to the Standup policy to get
the robot back on its feet.  A recover-timeout mechanism prevents the robot
from getting stuck in a half-recovered state.

State machine (gating_mode values in action_extras):
    1.0  -> fight   (primary, only this is trained on)
    0.0  -> recover (frozen fallback, excluded from training)
   -2.0  -> standup (frozen fallback, excluded from training)

Transitions:
    any      + height < fall_height            -> standup
    standup  + height > stand_height + p_safe  -> fight (stood up and stable)
    standup  + height > stand_height + !p_safe -> recover (stood up but wobbly)
    recover  + p_safe > release_threshold      -> fight (balance restored)
    recover  + recover_step_count >= timeout   -> standup (stuck, force standup)
    fight    + p_safe < threshold              -> recover (fall predicted)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from envs.framework.policy import Policy, PolicyBlueprint
from baseline.humanoid21.curriculum.train_gating_network import GatingMLP

_DEBUG = os.environ.get("FIGHT_MIXED_POLICY_V2_DEBUG", "0") == "1"

# Observation indices (humanoid21 96-dim observation)
_OBS_HEIGHT = 42  # root Z height


class FightMixedPolicyV2(Policy):
    """Three-way switching composite policy: fight / recover / standup.

    The primary fight policy is the only one that gets trained.  The recover
    and standup policies are frozen fallbacks that take over when the robot
    loses balance or falls to the ground.
    """

    def __init__(
        self,
        primary_policy_bp: str | Dict[str, Any] | PolicyBlueprint,
        fallback_policy_bp: str | Dict[str, Any] | PolicyBlueprint,
        standup_policy_bp: str | Dict[str, Any] | PolicyBlueprint,
        gating_model_dir: str = "/data1/mono/things/combatbench/baseline/humanoid21/curriculum/gating_model_plus_mix_level",
        threshold: float = 0.65,
        release_threshold: float = 0.90,
        fall_height: float = 0.5,
        stand_height: float = 1.27,
        recover_timeout: int = 50,
        **kwargs: Any,
    ) -> None:
        # 1. Rebuild primary (Fight) policy blueprint and live instance
        self.primary_policy_bp = self._resolve_bp(primary_policy_bp)
        self.primary_policy = self.primary_policy_bp.build()

        # 2. Rebuild fallback/recovery policy blueprint and live instance
        self.fallback_policy_bp = self._resolve_bp(fallback_policy_bp)
        self.fallback_policy = self.fallback_policy_bp.build()

        # 3. Rebuild standup policy blueprint and live instance
        self.standup_policy_bp = self._resolve_bp(standup_policy_bp)
        self.standup_policy = self.standup_policy_bp.build()

        self.threshold = float(threshold)
        self.release_threshold = float(release_threshold)
        self.fall_height = float(fall_height)
        self.stand_height = float(stand_height)
        self.recover_timeout = int(recover_timeout)
        self.gating_model_dir = Path(gating_model_dir)

        # 4. Load Gating MLP model structure & weights
        config_path = self.gating_model_dir / "gating_config.json"
        model_path = self.gating_model_dir / "gating_model.pt"

        import json
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
        self._recover_step_count = 0
        self.active_mode = "fight"  # "fight", "recover", or "standup"

    @staticmethod
    def _resolve_bp(bp: str | Dict[str, Any] | PolicyBlueprint) -> PolicyBlueprint:
        """Normalise a blueprint source to a :class:`PolicyBlueprint`."""
        if isinstance(bp, PolicyBlueprint):
            return bp
        if isinstance(bp, str):
            return PolicyBlueprint.load(bp)
        return PolicyBlueprint.from_dict(bp)

    def act(self, observation: Any, want_extra: bool = False) -> Tuple[Any, Any | None]:
        """Routes observation to the appropriate policy based on safety and height."""
        # 1. Predict safety probability from observation
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p_safe = self.gating_network.predict_probability(obs_tensor).item()

        # 2. Extract root height
        height = float(observation[_OBS_HEIGHT])

        # 3. State machine transition logic
        prev_mode = self.active_mode

        # ── Priority 1: fallen (highest priority, any mode) ──
        if height < self.fall_height:
            new_mode = "standup"

        # ── Priority 2: in standup — wait until standing height reached ──
        elif self.active_mode == "standup":
            if height > self.stand_height:
                # Stood up — check balance
                if p_safe > self.release_threshold:
                    new_mode = "fight"       # stable and standing → fight
                else:
                    new_mode = "recover"     # standing but wobbly → recover
            else:
                new_mode = "standup"         # still getting up

        # ── Priority 3: in recover — check release or timeout ──
        elif self.active_mode == "recover":
            if p_safe > self.release_threshold:
                new_mode = "fight"           # balance restored
            elif self._recover_step_count >= self.recover_timeout:
                new_mode = "standup"         # stuck too long → force standup
            else:
                new_mode = "recover"         # keep recovering

        # ── Priority 4: in fight — gating predicts imbalance ──
        elif self.active_mode == "fight":
            if p_safe < self.threshold:
                new_mode = "recover"
            else:
                new_mode = "fight"

        else:
            new_mode = "fight"  # safety fallback

        # 4. Apply mode transition
        if new_mode != self.active_mode:
            if new_mode == "recover":
                self._recover_step_count = 0
                if hasattr(self.fallback_policy, "reset"):
                    self.fallback_policy.reset()
            elif new_mode == "standup":
                self._recover_step_count = 0
                if hasattr(self.standup_policy, "reset"):
                    self.standup_policy.reset()
            elif new_mode == "fight":
                self._recover_step_count = 0
                if hasattr(self.primary_policy, "reset"):
                    self.primary_policy.reset()
            self.active_mode = new_mode

        # Increment recover counter when staying in recover
        if self.active_mode == "recover":
            self._recover_step_count += 1

        if _DEBUG:
            self._step_count += 1
            if self.active_mode != prev_mode or self._step_count % 10 == 0:
                print(
                    f"[FightMixedPolicyV2] step={self._step_count} p_safe={p_safe:.4f} "
                    f"h={height:.3f}m mode={prev_mode}->{self.active_mode} "
                    f"recover_steps={self._recover_step_count}/{self.recover_timeout}",
                    flush=True,
                )

        # 5. Dispatch action & extra info
        # gating_mode definitions:
        #   1.0  -> fight (primary, only this is trained on)
        #   0.0  -> recover (frozen fallback, excluded from training)
        #   -2.0 -> standup (frozen fallback, excluded from training)
        if self.active_mode == "fight":
            action, extra = self.primary_policy.act(observation, want_extra)
            if want_extra:
                if not isinstance(extra, dict):
                    extra = {}
                extra.setdefault("log_prob", 0.0)
                extra["gating_mode"] = 1.0
                extra["p_safe"] = p_safe
        elif self.active_mode == "recover":
            action, extra = self.fallback_policy.act(observation, want_extra)
            if want_extra:
                if not isinstance(extra, dict):
                    extra = {}
                extra.setdefault("log_prob", 0.0)
                extra["gating_mode"] = 0.0
                extra["p_safe"] = p_safe
        else:  # standup
            action, extra = self.standup_policy.act(observation, want_extra)
            if want_extra:
                if not isinstance(extra, dict):
                    extra = {}
                extra.setdefault("log_prob", 0.0)
                extra["gating_mode"] = -2.0
                extra["p_safe"] = p_safe

        return action, extra

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal state machine and child policies."""
        if _DEBUG:
            print(f"[FightMixedPolicyV2] reset(seed={seed}), mode={self.active_mode}->fight", flush=True)
        if hasattr(self.primary_policy, "reset"):
            self.primary_policy.reset(seed)
        if hasattr(self.fallback_policy, "reset"):
            self.fallback_policy.reset(seed)
        if hasattr(self.standup_policy, "reset"):
            self.standup_policy.reset(seed)
        self.active_mode = "fight"
        self._step_count = 0
        self._recover_step_count = 0

    def close(self) -> None:
        """Release any resources held by child policies."""
        if hasattr(self.primary_policy, "close"):
            self.primary_policy.close()
        if hasattr(self.fallback_policy, "close"):
            self.fallback_policy.close()
        if hasattr(self.standup_policy, "close"):
            self.standup_policy.close()
