"""Mixed-level perturbation curriculum (refine) — prevents catastrophic forgetting.

Subclasses :class:`BalanceRecoverConfig`. During training rollouts each episode
samples a perturbation level **uniformly** from {0, 1, ..., current_level}, so
every unlocked difficulty tier gets ~1/(N+1) of the episodes.  This keeps the
policy's recovery behaviour anchored to low-perturbation regimes even as it
learns harder ones — avoiding the "can handle big kicks, forgets small ones"
failure mode.

Evaluation still uses only the current (hardest unlocked) level so that
promotion decisions are based on mastering the frontier difficulty.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.exp_balance_recover_plus import BalanceRecoverConfig
from envs.framework.blueprint import EnvBlueprint
from envs.framework.policy import PolicyBlueprint


class BalanceRecoverConfigRefine(BalanceRecoverConfig):
    """Progressive perturbation curriculum with **uniform mixed-level rollouts**.

    Everything is inherited from :class:`BalanceRecoverConfig` (PPO knobs,
    reward extraction, level promotion logic, eval job construction, …).
    Only :meth:`build_rollout_jobs` is overridden so that training data
    always spans all unlocked levels.
    """

    name = "balance_recover_plus_refine"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _perturb_params_for_level(self, level: int) -> Dict[str, float]:
        """Full-strength perturbation magnitudes scaled for curriculum *level*."""
        idx = max(0, min(int(level), len(self.LEVEL_SCALES) - 1))
        scale = float(self.LEVEL_SCALES[idx])
        return {k: float(v) * scale for k, v in self.PERTURB_FULL.items()}

    # ------------------------------------------------------------------
    # Rollout construction — uniform mix of levels 0..current
    # ------------------------------------------------------------------
    def build_rollout_jobs(
        self, policy_bp: PolicyBlueprint, base_seed: int
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        """Build training jobs with per-episode random level sampling.

        At level *N* each episode independently draws a level from
        ``{0, ..., N}`` (uniform), giving equal expected proportion to every
        tier.  Materialized env blueprints are cached per ``(level, agent_id)``
        so at most ``(N+1) × 2`` blueprints exist simultaneously.
        """
        max_steps = self.custom_config["max_steps"]
        env_pb = self._env_pb()
        n_episodes = self.episodes_per_update
        rng = np.random.default_rng(base_seed)

        bp_cache: Dict[Tuple[int, str], EnvBlueprint] = {}

        def _get_env_bp(level: int, agent_id: str) -> EnvBlueprint:
            key = (level, agent_id)
            if key not in bp_cache:
                perturb = self._perturb_params_for_level(level)
                bp_cache[key] = env_pb.materialize(
                    max_steps=max_steps, agent_id=agent_id, **perturb
                )
            return bp_cache[key]

        jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            initial_distance = float(
                rng.uniform(
                    self.custom_config["rollout_distance_min"],
                    self.custom_config["rollout_distance_max"],
                )
            )
            # Uniformly pick a difficulty from every unlocked level.
            sampled_level = int(rng.integers(0, self._level + 1))
            env_bp_i = _get_env_bp(sampled_level, agent_id)
            jobs.append((
                policy_bp, policy_bp,
                env_bp_i, seed,
                {
                    "agent_id": agent_id,
                    "initial_distance": initial_distance,
                    "perturb_level": float(sampled_level),
                },
            ))
        return jobs


# Singleton instance for the registry
EXPERIMENT = BalanceRecoverConfigRefine()
