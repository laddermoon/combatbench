"""Policy-driven initial-state generator (RSI-style, online).

Modelled on ``RandomFallenStatePlugin``: keeps a lazy internal
``Humanoid21Simulator``, rolls an exported policy inside it for a
random number of action steps at episode start, and injects the
resulting core state into the real environment.  Produces fresh
guard-stance start poses straight from the S2 policy's own state
distribution — no offline state set needed.

Blueprint params::

    - cls: "baseline.humanoid21.plugins.policy_init_state:PolicyInitStatePlugin"
      config:
        target_robots: ["robot_a", "robot_b"]   # or one robot
        policy_dir: "baseline/runs/hier_s2_guard_96d_s1/policy"
        min_steps: 30
        max_steps: 120
        disturb: false        # v1: no disturbance (placeholder)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from envs.framework import BasePlugin, SimContext
from envs.framework.observer_plugin import OBSERVER_DISPATCHER_PRIORITY
from envs.framework.policy import PolicyBlueprint


class PolicyInitStatePlugin(BasePlugin):
    """Inject states produced by rolling a policy in an internal sim."""

    def __init__(self, policy_dir: str,
                 target_robots: Sequence[str] = ("robot_a", "robot_b"),
                 min_steps: int = 30, max_steps: int = 120,
                 disturb: bool = False,
                 random_seed: Optional[int] = None) -> None:
        self.policy_dir = str(policy_dir)
        self._targets = tuple(target_robots)
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps)
        self.disturb = bool(disturb)
        self._rng = np.random.RandomState(random_seed)
        self._internal_sim: Optional[Any] = None
        self._policies: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "policy_init_state"

    @property
    def require_mutator(self) -> bool:
        return True

    @property
    def priority(self) -> int:
        return OBSERVER_DISPATCHER_PRIORITY + 2

    def set_episode_seed(self, seed: int) -> None:
        self._rng = np.random.RandomState(int(seed))

    # ------------------------------------------------------------------

    def _ensure_internal(self):
        if self._internal_sim is None:
            from envs.humanoid21.simulator import Humanoid21Simulator
            self._internal_sim = Humanoid21Simulator()
            spec = self.policy_dir
            if Path(spec).is_dir():
                spec = f"file:{Path(spec).resolve() / 'policy.py'}:" \
                       "ExportedStateMixtureBoundedStdTruncNormPolicy"
            bp = PolicyBlueprint.from_dict({"cls": spec, "config": {}})
            for rid in self._targets:
                self._policies[rid] = bp.build()
        return self._internal_sim

    def on_detach(self) -> None:
        if self._internal_sim is not None:
            self._internal_sim.close()
            self._internal_sim = None

    def on_pre_episode(self, ctx: SimContext) -> None:
        sim = self._ensure_internal()
        n_steps = int(self._rng.randint(self.min_steps,
                                        self.max_steps + 1))
        sim.reset(seed=int(self._rng.randint(1 << 30)))
        for rid, pol in self._policies.items():
            pol.reset(seed=int(self._rng.randint(1 << 30)))
        phy_per_action = int(getattr(sim, "get_physical_frequency",
                                     lambda: 500)() * 0.05) or 25
        for _ in range(n_steps):
            obs = sim.get_observation()
            acts = {}
            for rid in ("robot_a", "robot_b"):
                if rid in self._targets:
                    out = self._policies[rid].act(obs[rid])
                    acts[rid] = np.asarray(
                        out[0] if isinstance(out, tuple) else out,
                        dtype=np.float32)
                else:
                    acts[rid] = np.zeros(21, dtype=np.float32)
            sim.set_action(acts)
            for _ in range(phy_per_action):
                sim.physical_step()
        cs = sim.get_core_state()
        ctx.mutator.set_core_state(
            {rid: cs[rid] for rid in self._targets})

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "policy_dir": self.policy_dir,
            "target_robots": list(self._targets),
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "disturb": self.disturb,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "PolicyInitStatePlugin":
        return cls(**config)
