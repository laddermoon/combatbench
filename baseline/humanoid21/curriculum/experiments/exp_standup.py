"""Curriculum training experiment for humanoid21 Standup task.

This experiment uses a composite policy (StandupMixedPolicy) to first run a
random policy to make the robot fall down. Once the pelvis height drops below
a threshold (default 0.35m), it switches to the primary stand-up policy to train.
The random policy steps are excluded from training via prepare_training_segments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.framework.config import ExperimentConfig
from baseline.humanoid21.curriculum.framework.ppo_trainer import (
    _extract_per_step_field,
    _extract_per_step_scalar,
)
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import Policy, PolicyBlueprint


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_RANDOM_POLICY_BP = PolicyBlueprint.load(
    _PROJECT_ROOT / "policy" / "blueprints" / "random.yaml"
)


class StandupMixedPolicy(Policy):
    """Dynamic switching policy for Stand-up training.

    Starts with a fallback (random) policy to fall down, and permanently switches
    to the learning primary policy once torso height is below threshold.
    """

    def __init__(
        self,
        primary_policy_bp: PolicyBlueprint | Dict[str, Any],
        fallback_policy_bp: PolicyBlueprint | Dict[str, Any],
        fall_height_threshold: float = 0.35,
        **kwargs: Any,
    ) -> None:
        self.primary_policy_bp = self._resolve_bp(primary_policy_bp)
        self.primary_policy = self.primary_policy_bp.build()

        self.fallback_policy_bp = self._resolve_bp(fallback_policy_bp)
        self.fallback_policy = self.fallback_policy_bp.build()

        self.fall_height_threshold = float(fall_height_threshold)
        self.active_mode = "random"

    @staticmethod
    def _resolve_bp(bp: str | Dict[str, Any] | PolicyBlueprint) -> PolicyBlueprint:
        if isinstance(bp, PolicyBlueprint):
            return bp
        if isinstance(bp, str):
            return PolicyBlueprint.load(bp)
        return PolicyBlueprint.from_dict(bp)

    def act(self, observation: Any, want_extra: bool = False) -> Tuple[Any, Any | None]:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        # Parse height. Proprioception (42-dim) + Local Orientation (6-dim) -> Height at index 48.
        height = float(obs[48]) if len(obs) > 48 else 0.9

        # State transition: Random (to fall down) -> Primary (learning standup)
        if self.active_mode == "random" and height < self.fall_height_threshold:
            self.active_mode = "primary"

        if self.active_mode == "random":
            action, extra = self.fallback_policy.act(observation, want_extra=want_extra)
        else:
            action, extra = self.primary_policy.act(observation, want_extra=want_extra)

        if extra is None:
            extra = {}
        # gating_mode: 1.0 represents the trainable primary standup phase, 0.0 is the random fall phase
        extra["gating_mode"] = 1.0 if self.active_mode == "primary" else 0.0

        return action, extra


class StandupConfig(ExperimentConfig):
    """Standup curriculum experiment with potential-difference rewards."""

    name = "standup"
    weight_target_total: float = 200.0
    weight_cap: float = 10.0
    reward_keys = ("r_potential", "r_cross")
    gammas = {
        "r_potential": 0.99,
        "r_cross": 0.99,
    }

    BLUEPRINT = "standup_env.yaml"

    max_updates: int = 15000

    # --- PPO tuning ---
    log_std_min: float = -1.8
    learning_rate: float = 3e-5
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096 * 4
    entropy_coef: float = 1.5e-3

    # --- Rollout schedule ---
    episodes_per_update: int = 512
    eval_episodes: int = 64
    eval_interval: int = 2

    # --- Video recording ---
    video_eval_interval: int = 2

    # --- Stateful metrics ---
    _success_rate: float = 0.0

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def _materialize_env(self, agent_id: str) -> EnvBlueprint:
        return self._env_pb().materialize(
            agent_id=agent_id,
        )

    def video_env_blueprint(self):
        return self._materialize_env("robot_a")

    # ---- Policy blueprint helpers -----------------------------------------

    def _make_mixed_bp(self, primary_bp: PolicyBlueprint) -> PolicyBlueprint:
        """Wrap *primary_bp* in StandupMixedPolicy with random fall fallback."""
        return PolicyBlueprint(
            cls="baseline.humanoid21.curriculum.experiments.exp_standup:StandupMixedPolicy",
            config={
                "primary_policy_bp": primary_bp.to_dict(),
                "fallback_policy_bp": _RANDOM_POLICY_BP.to_dict(),
                "fall_height_threshold": 0.35,
            },
        )

    # ---- Job construction -------------------------------------------------

    def _build_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        mixed_bp = self._make_mixed_bp(policy_bp)
        env_bps: Dict[str, EnvBlueprint] = {
            aid: self._materialize_env(aid)
            for aid in ("robot_a", "robot_b")
        }

        jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            # Stand-up only trains robot_a
            agent_id = "robot_a"
            
            jobs.append((
                mixed_bp, _RANDOM_POLICY_BP,
                env_bps[agent_id], seed,
                {"agent_id": agent_id, "initial_distance": 2.0},
            ))
        return jobs

    def build_rollout_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.eval_episodes)

    # ---- Eval comparison --------------------------------------------------

    def compare_eval(self, esum, best_esum):
        if not best_esum:
            return True
        return esum.get("success", 0.0) > best_esum.get("success", 0.0)

    # ---- Scheduler --------------------------------------------------------

    def initial_weights(self) -> Tuple[float, ...]:
        return (1.0, 0.1)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        self._success_rate = float(eval_metrics.get("success", 0.0))
        return (1.0, 0.1)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        """Extract potential-difference reward and cross-support balance reward."""
        T = episode.num_frames
        oo = episode.observer_outputs

        # Extract potential values from the StandupPotentialRewarder observer plugin
        potentials = _extract_per_step_field(oo, "standup", "potential", T)
        r_potential = np.zeros(T, dtype=np.float32)
        if potentials is not None:
            # r_potential[t] = potentials[t] - potentials[t-1] (Potential Difference)
            r_potential[1:] = potentials[1:] - potentials[:-1]
            r_potential[0] = potentials[0] - 0.0
            
            # Scale potential difference so the total possible reward sum is 10.0
            scale = float(self.custom_config.get("potential_reward_scale", 10.0))
            r_potential *= scale

        # Extract cross support balance reward
        r_cross = _extract_per_step_scalar(oo, "cross_support", T)
        if r_cross is None:
            r_cross = np.zeros(T, dtype=np.float32)

        return {
            "r_potential": r_potential,
            "r_cross": r_cross,
        }

    # ---- Episode metrics --------------------------------------------------

    def prepare_training_segments(
        self, episode,
    ) -> List[Tuple[int, int, float]]:
        """Split episode at fall boundary, keeping only primary (stand-up) steps.

        This ensures the robot is never trained on actions produced by the
        random policy during the fall phase.
        """
        T = episode.num_frames
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)
        if extras is None or "gating_mode" not in extras:
            w = min(self.weight_target_total / T, self.weight_cap)
            return [(0, T, w)]

        gating_mode = np.asarray(extras["gating_mode"], dtype=np.float32).reshape(-1)
        L = min(T, len(gating_mode))
        is_primary = gating_mode[:L] >= 0.5

        segments: List[Tuple[int, int]] = []
        start = None
        for t in range(L):
            if is_primary[t]:
                if start is None:
                    start = t
            else:
                if start is not None:
                    segments.append((start, t))
                    start = None
        if start is not None:
            segments.append((start, L))

        return [
            (s, e, min(self.weight_target_total / (e - s), self.weight_cap))
            for s, e in segments
        ]

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        """Compute metrics for success monitoring and curriculum progression."""
        T = episode.num_frames
        oo = episode.observer_outputs

        # Filter out initial standing/falling phases to prevent metric pollution!
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)
        
        is_primary = None
        if extras is not None and "gating_mode" in extras:
            gating_mode = np.asarray(extras["gating_mode"], dtype=np.float32).reshape(-1)
            L = min(T, len(gating_mode))
            is_primary = gating_mode[:L] >= 0.5
            
        stages = _extract_per_step_field(oo, "standup", "stage", T)
        potentials = _extract_per_step_field(oo, "standup", "potential", T)

        if is_primary is not None and np.any(is_primary):
            valid_stages = stages[:len(is_primary)][is_primary] if stages is not None else None
            valid_pots = potentials[:len(is_primary)][is_primary] if potentials is not None else None
        else:
            valid_stages = stages
            valid_pots = potentials

        if valid_stages is not None and len(valid_stages) > 0:
            max_stage = float(np.max(valid_stages))
            # Reaching Stage 5 (Perfect Standing) represents full success!
            success = 1.0 if max_stage >= 5.0 else 0.0
            avg_stage = float(np.mean(valid_stages))
        else:
            max_stage = 0.0
            success = 0.0
            avg_stage = 0.0

        max_potential = float(np.max(valid_pots)) if valid_pots is not None and len(valid_pots) > 0 else 0.0

        return {
            "success": success,
            "max_stage": max_stage,
            "avg_stage": avg_stage,
            "max_potential": max_potential,
        }

    # ---- Scheduler state --------------------------------------------------

    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "success_rate": round(self._success_rate, 3),
        }

    def scheduler_state(self) -> dict:
        return {
            "success_rate": self._success_rate,
        }

    def load_scheduler_state(self, state: dict) -> None:
        self._success_rate = float(state.get("success_rate", 0.0))


# Register singleton config for the registry
EXPERIMENT = StandupConfig()