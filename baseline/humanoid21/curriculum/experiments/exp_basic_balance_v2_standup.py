"""Basic balance v2 with standup fallback — no early termination.

Variant of basic_balance_v2 that wraps the trainable actor in
StandupFallbackPolicy.  When the robot falls, the frozen standup
policy takes over; once standing again, control returns to the
primary (trainable) policy.  The episode always runs to timeout.

Training data is split into sub-episodes at gating_mode boundaries:
only primary (gating_mode == 1.0) segments are kept; standup
(gating_mode == -1.0) segments are discarded.

Purpose:
  1. Validate sub-episode splitting with prepare_training_segments.
  2. Test whether fixed-length episodes + standup fallback improves
     balance training efficiency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.exp_basic_balance_v2 import BasicBalanceV2Config
from baseline.framework.ppo_trainer import _extract_per_step_field, _extract_per_step_scalar
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


# Frozen standup checkpoint used as fallback.
_STANDUP_POLICY_BP = PolicyBlueprint.load(
    "/data1/mono/things/combatbench/baseline/runs/"
    "train_standing_balance_4stage_dense_ppo_resume5k_20260730_211100/"
    "policy_exports/u04935/policy_blueprint.yaml"
)


class BasicBalanceV2StandupConfig(BasicBalanceV2Config):
    """Balance training with StandupFallbackPolicy and sub-episode splitting."""

    name = "basic_balance_v2_standup"
    reward_keys = ("r_fall", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot")
    gammas = {
        "r_fall": 0.99,
        "r_cross": 0.99,
        "r_joint": 0.99,
        "r_vel": 0.99,
        "r_tilt": 0.99,
        "r_foot": 0.99,
    }

    BLUEPRINT = "basic_balance_v2_standup_env.yaml"

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def _materialize_env(self, agent_id: str) -> EnvBlueprint:
        return self._env_pb().materialize(
            max_steps=self.custom_config["max_steps"],
            agent_id=agent_id,
        )

    def video_env_blueprint(self):
        return self._materialize_env("robot_a")

    # ---- Policy blueprint helpers -----------------------------------------

    @staticmethod
    def _make_standup_bp(primary_bp: PolicyBlueprint) -> PolicyBlueprint:
        """Wrap *primary_bp* in StandupFallbackPolicy with frozen standup."""
        return PolicyBlueprint(
            cls="baseline.humanoid21.curriculum.standup_fallback_policy:StandupFallbackPolicy",
            config={
                "primary_policy_bp": primary_bp.to_dict(),
                "standup_policy_bp": _STANDUP_POLICY_BP.to_dict(),
                "fall_height": 0.5,
                "stand_height": 1.25,
            },
        )

    # ---- Job construction -------------------------------------------------

    def _build_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        standup_bp = self._make_standup_bp(policy_bp)

        env_bps: Dict[str, EnvBlueprint] = {
            aid: self._materialize_env(aid)
            for aid in ("robot_a", "robot_b")
        }

        jobs: List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            initial_distance = float(
                np.random.default_rng(seed).uniform(
                    self.custom_config["rollout_distance_min"],
                    self.custom_config["rollout_distance_max"],
                )
            )
            jobs.append((
                standup_bp, standup_bp,
                env_bps[agent_id], seed,
                {"agent_id": agent_id, "initial_distance": initial_distance},
            ))
        return jobs

    def build_rollout_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.eval_episodes)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        """Same reward structure as basic_balance_v2, but:
        - Fall is detected via gating_mode transition (1.0 -> -1.0), not env termination.
        - Standup segments (gating_mode < 0) have all rewards zeroed.
        """
        T = episode.num_frames
        oo = episode.observer_outputs

        # Default per-step survival bonus
        r_fall = np.full(T, self.per_step_survival_reward, dtype=np.float32)
        penalty = float(self.custom_config["terminal_fall_penalty"])

        # Detect fall via gating_mode: transition from primary to standup
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)
        gating_mode = None
        if extras is not None and "gating_mode" in extras:
            gating_mode = np.asarray(extras["gating_mode"], dtype=np.float32).reshape(-1)

        if gating_mode is not None and len(gating_mode) > 0:
            L = min(T, len(gating_mode))
            for t in range(L - 1):
                # Fall: primary -> standup
                if gating_mode[t] >= 0.5 and gating_mode[t + 1] < 0.5:
                    r_fall[t] = -penalty
            # If episode ends in standup, penalize last primary step
            if gating_mode[L - 1] < 0.5:
                # Find last primary step
                for t in range(L - 1, -1, -1):
                    if gating_mode[t] >= 0.5:
                        r_fall[t] = -penalty
                        break
        else:
            # No gating_mode info — fallback to old behavior
            fell = all(r.startswith("imbalance") for r in episode.agent_termination_reason.values())
            if fell:
                r_fall[-1] = -penalty
            else:
                r_fall[-1] = penalty

        # Other rewards (same as parent)
        r_cross = _extract_per_step_scalar(oo, "cross_support", T)

        joint_dev_arr = _extract_per_step_field(oo, "posture", "joint_deviation", T)
        joint_vel_arr = _extract_per_step_field(oo, "posture", "joint_vel", T)
        torso_tilt_arr = _extract_per_step_field(oo, "posture", "torso_tilt", T)
        foot_height_arr = _extract_per_step_field(oo, "posture", "foot_height", T)

        if joint_dev_arr is None:
            joint_dev_arr = np.zeros(T, dtype=np.float32)
        if joint_vel_arr is None:
            joint_vel_arr = np.zeros(T, dtype=np.float32)
        if torso_tilt_arr is None:
            torso_tilt_arr = np.zeros(T, dtype=np.float32)
        if foot_height_arr is None:
            foot_height_arr = np.zeros(T, dtype=np.float32)

        excess_joint = np.maximum(0.0, joint_dev_arr - 0.1)
        r_joint = np.where(excess_joint == 0.0, 0.01, 0.01 - 5.0 * excess_joint)

        excess_vel = np.maximum(0.0, joint_vel_arr - 0.1)
        r_vel = np.where(excess_vel == 0.0, 0.01, 0.01 - 1.0 * excess_vel)

        excess_tilt = np.maximum(0.0, torso_tilt_arr - 0.26)
        r_tilt = np.where(excess_tilt == 0.0, 0.01, 0.01 - 3.0 * excess_tilt)

        excess_foot = np.maximum(0.0, foot_height_arr - 0.10)
        r_foot = np.where(excess_foot == 0.0, 0.01, 0.01 - 5.0 * excess_foot)

        # Zero out standup segments for ALL rewards
        if gating_mode is not None and len(gating_mode) > 0:
            L = min(T, len(gating_mode))
            standup_mask = gating_mode[:L] < 0.5
            r_fall[:L][standup_mask] = 0.0
            r_cross[:L][standup_mask] = 0.0
            r_joint[:L][standup_mask] = 0.0
            r_vel[:L][standup_mask] = 0.0
            r_tilt[:L][standup_mask] = 0.0
            r_foot[:L][standup_mask] = 0.0

        return {
            "r_fall": r_fall,
            "r_cross": r_cross,
            "r_joint": r_joint,
            "r_vel": r_vel,
            "r_tilt": r_tilt,
            "r_foot": r_foot,
        }

    # ---- Sub-episode splitting --------------------------------------------

    def prepare_training_segments(
        self, episode,
    ) -> List[Tuple[int, int, float]]:
        """Split episode at standup boundaries, keeping only primary steps."""
        T = episode.num_frames
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)
        if extras is None or "gating_mode" not in extras:
            w = min(self.weight_target_total / T, self.weight_cap) if hasattr(self, "weight_target_total") else 1.0
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

        weight_total = getattr(self, "weight_target_total", 200.0)
        weight_cap = getattr(self, "weight_cap", 10.0)
        return [
            (s, e, min(weight_total / (e - s), weight_cap))
            for s, e in segments
            if e - s > 1
        ]

    # ---- Episode metrics --------------------------------------------------

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        """survived = 1.0 if no standup was triggered during the episode."""
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        extras = episode.action_extras.get(ep_target)

        fell = False
        primary_ratio = 1.0
        standup_count = 0.0

        if extras is not None and "gating_mode" in extras:
            gating_mode = np.asarray(extras["gating_mode"], dtype=np.float32).reshape(-1)
            if len(gating_mode) > 0:
                primary_ratio = float(np.mean(gating_mode >= 0.5))
                # Count transitions into standup
                for t in range(len(gating_mode) - 1):
                    if gating_mode[t] >= 0.5 and gating_mode[t + 1] < 0.5:
                        standup_count += 1.0
                fell = standup_count > 0

        return {
            "survived": 0.0 if fell else 1.0,
            "primary_ratio": primary_ratio,
            "standup_count": standup_count,
        }

    # ---- Eval comparison --------------------------------------------------

    def compare_eval(self, esum, best_esum):
        if not best_esum:
            return True
        return esum.get("survived", 0.0) > best_esum.get("survived", 0.0)

    # ---- Scheduler --------------------------------------------------------

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        self._survival_rate = float(eval_metrics.get("survived", 0.0))
        return (3.0, 1.0, 0.2, 0.2, 0.2, 0.2)


# Singleton instance for the registry
EXPERIMENT = BasicBalanceV2StandupConfig()
