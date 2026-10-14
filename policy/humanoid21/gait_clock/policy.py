"""Gait-clock adapter policy — deploys a 99-dim step-lineage policy on
the canonical 96-dim platform environment.

Problem: every experiment in the step chain (step / balance / follow /
attack_prep / strike / fight_step) trains against
``GaitClockSimulator``, which appends 3 dims to the simulator's 96-dim
observation (``cmd_L``, ``cmd_R``, window progress — a deterministic
function of the action-step counter).  The canonical platform
environment (``envs/humanoid21/blueprint.yaml``, ``fight_env.yaml`` …)
uses bare ``Humanoid21Simulator`` and produces only 96 dims — feeding
that straight into an exported 99-dim policy crashes on the input
matrix shape.

This adapter folds the augmentation INTO the policy artifact: it
accepts the standard 96-dim observation, keeps its own action-step
counter (identical convention to ``GaitClockSimulator`` — the clock for
frame t is computed before the action for frame t is applied), appends
the 3 clock dims, and delegates to the inner 99-dim policy.  Opponent
policies and the platform see a normal 96-dim policy; the clock becomes
an implementation detail of ours.

Timing note: ``GaitClockSimulator._action_step`` counts ``set_action``
*calls*, not env frames — and the runtime calls it once per agent, so in
a 2-agent env the counter advances by 2 per frame (verified empirically:
env obs clock pos == 2*frame % 40).  ``counter_step`` replicates that;
keep it at 2 for the standard 2-robot combat envs.

Blueprint usage::

    version: 1
    cls: "policy.humanoid21.gait_clock.policy:GaitClockAdapterPolicy"
    config:
      gait_period: 40           # must match the training env
      counter_step: 2           # set_action calls per frame (1/agent)
      inner:
        version: 1
        cls: "file:/path/to/export/policy.py:ExportedPolicyClass"
        config: {}
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np

# Repo-root bootstrap — same convention as policy/humanoid21/standing.
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
    from envs.framework.policy import Policy, PolicyBlueprint
except ImportError:
    from combatbench.envs.framework.policy import (  # type: ignore[no-redef]
        Policy, PolicyBlueprint)


class GaitClockAdapterPolicy(Policy):
    """96-dim in → gait-clock append → inner 99-dim policy."""

    BASE_DIM = 96
    CLOCK_DIM = 3

    def __init__(
        self,
        inner: Mapping[str, Any],
        gait_period: int = 40,
        counter_step: int = 2,
    ) -> None:
        self.gait_period = int(gait_period)
        self.counter_step = int(counter_step)
        self._inner = PolicyBlueprint.from_dict(dict(inner)).build()
        self._step = 0

    def _clock(self) -> np.ndarray:
        """Mirror GaitClockSimulator.gait_clock() exactly."""
        half = self.gait_period // 2
        pos = self._step % self.gait_period
        if pos < half:
            return np.array([1.0, 0.0, pos / half], dtype=np.float32)
        return np.array(
            [0.0, 1.0, (pos - half) / half], dtype=np.float32)

    def _augment(self, observation: Any) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        dim = obs.shape[-1]
        if dim == self.BASE_DIM:
            return np.concatenate([obs, self._clock()])
        # Already augmented (e.g. training env with GaitClockSimulator)
        # — pass through untouched; counter still advances.
        if dim == self.BASE_DIM + self.CLOCK_DIM:
            return obs
        raise ValueError(
            f"GaitClockAdapterPolicy expects a {self.BASE_DIM}- or "
            f"{self.BASE_DIM + self.CLOCK_DIM}-dim observation, got {dim}")

    def act(
        self, observation: Any, *, want_extra: bool = False,
    ) -> Tuple[Any, Any]:
        obs99 = self._augment(observation)
        self._step += self.counter_step
        return self._inner.act(obs99, want_extra=want_extra)

    def sample(self, observation: Any, *, ctx=None, want_extra=False):
        obs99 = self._augment(observation)
        self._step += self.counter_step
        if hasattr(self._inner, "sample"):
            return self._inner.sample(obs99, ctx=ctx, want_extra=want_extra)
        return self._inner.act(obs99, want_extra=want_extra)

    def reset(self, seed: Optional[int] = None) -> None:
        self._step = 0
        if hasattr(self._inner, "reset"):
            self._inner.reset(seed)
