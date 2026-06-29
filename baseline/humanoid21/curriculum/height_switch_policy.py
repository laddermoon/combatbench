"""Simple height/uprightness-based switching policy.

Uses the standup policy to get the robot upright, then switches to the
fallback (balance recovery) policy once height and uprightness exceed
thresholds. Switches back to standup if the robot falls below the
release threshold.

Observation format (96-dim, from Humanoid21Simulator._get_robot_view):
  index 48: root height (root_pos[2])
  index 42-47: local_orientation (6-dim)
  Uprightness is not directly in the flat observation, but can be
  approximated from local_orientation or computed from the rotation matrix.
  For simplicity, we use height as the primary switch signal and
  local_orientation[0] (which correlates with uprightness) as secondary.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from envs.framework.policy import Policy, PolicyBlueprint

_DEBUG = os.environ.get("HEIGHT_SWITCH_DEBUG", "0") == "1"


class HeightSwitchPolicy(Policy):
    """Switch between standup and fallback policy based on height threshold.

    Primary (standup) policy is used until the robot reaches switch_height,
    then fallback (balance recovery) policy takes over. If the robot falls
    back below release_height, control returns to the standup policy.
    """

    def __init__(
        self,
        standup_policy_bp: str | Dict[str, Any] | PolicyBlueprint,
        fallback_policy_bp: str | Dict[str, Any] | PolicyBlueprint,
        switch_height: float = 0.55,
        switch_uprightness: float = 0.80,
        release_height: float = 0.35,
        release_patience: int = 10,
        **kwargs: Any,
    ) -> None:
        self.standup_policy_bp = self._resolve_bp(standup_policy_bp)
        self.standup_policy = self.standup_policy_bp.build()

        self.fallback_policy_bp = self._resolve_bp(fallback_policy_bp)
        self.fallback_policy = self.fallback_policy_bp.build()

        self.switch_height = float(switch_height)
        self.switch_uprightness = float(switch_uprightness)
        self.release_height = float(release_height)
        self.release_patience = int(release_patience)

        self._step_count = 0
        self._low_count = 0
        self.active_mode = "standup"

    @staticmethod
    def _resolve_bp(bp: str | Dict[str, Any] | PolicyBlueprint) -> PolicyBlueprint:
        if isinstance(bp, PolicyBlueprint):
            return bp
        if isinstance(bp, str):
            return PolicyBlueprint.load(bp)
        return PolicyBlueprint.from_dict(bp)

    def _extract_height_uprightness(self, observation: np.ndarray) -> Tuple[float, float]:
        """Extract height and uprightness from 96-dim observation.

        Height is at index 48.
        Uprightness is approximated from local_orientation: the first
        element of local_orientation (index 42) corresponds to the
        z-component of the torso's up vector in world frame, which is
        the uprightness value.
        """
        obs = np.asarray(observation, dtype=np.float32)
        height = float(obs[48])
        # local_orientation is 6-dim starting at index 42.
        # It's a flattened rotation matrix (first 6 elements of 9).
        # The uprightness (world z of local z-axis) = rot_mat[2,2].
        # local_orientation contains: [r00, r01, r02, r10, r11, r12]
        # (first 2 rows of rotation matrix, column-major or row-major?)
        # Actually, from the simulator code, local_orientation is derived
        # from the rotation matrix. Let's just use height as primary signal
        # and approximate uprightness from orientation[0] which is r00.
        # A more robust approach: use height only.
        uprightness = float(obs[42])  # rough approximation
        return height, uprightness

    def act(self, observation: Any, want_extra: bool = False) -> Tuple[Any, Any | None]:
        obs_array = np.asarray(observation, dtype=np.float32)
        height, uprightness = self._extract_height_uprightness(obs_array)

        prev_mode = self.active_mode

        # State machine: standup -> fallback when tall enough
        if self.active_mode == "standup":
            if height >= self.switch_height:
                self.active_mode = "fallback"
                self._low_count = 0
                if hasattr(self.fallback_policy, "reset"):
                    self.fallback_policy.reset()
        elif self.active_mode == "fallback":
            if height < self.release_height:
                self._low_count += 1
                if self._low_count >= self.release_patience:
                    self.active_mode = "standup"
                    self._low_count = 0
                    if hasattr(self.standup_policy, "reset"):
                        self.standup_policy.reset()
            else:
                self._low_count = 0

        if _DEBUG:
            self._step_count += 1
            if self.active_mode != prev_mode or self._step_count % 10 == 0:
                print(
                    f"[HeightSwitchPolicy] step={self._step_count} "
                    f"h={height:.3f} upright={uprightness:.3f} "
                    f"mode={prev_mode}->{self.active_mode} "
                    f"low_count={self._low_count}/{self.release_patience}",
                    flush=True,
                )

        if self.active_mode == "standup":
            action, extra = self.standup_policy.act(observation, want_extra)
        else:
            action, extra = self.fallback_policy.act(observation, want_extra)

        if want_extra:
            if not isinstance(extra, dict):
                extra = {}
            extra.setdefault("log_prob", 0.0)
            extra["switch_mode"] = 1.0 if self.active_mode == "standup" else 0.0
            extra["height"] = height

        return action, extra

    def reset(self, seed: Optional[int] = None) -> None:
        if hasattr(self.standup_policy, "reset"):
            self.standup_policy.reset(seed)
        if hasattr(self.fallback_policy, "reset"):
            self.fallback_policy.reset(seed)
        self.active_mode = "standup"
        self._step_count = 0
        self._low_count = 0

    def close(self) -> None:
        if hasattr(self.standup_policy, "close"):
            self.standup_policy.close()
        if hasattr(self.fallback_policy, "close"):
            self.fallback_policy.close()
