"""Hybrid standup + balance experiment.

Trains a HybridActor with two sub-networks:
  - standup_net: initialized from standup_v2_r14 checkpoint, learns to stand up fast
  - balance_net: initialized from follow_v2 fallback checkpoint, learns stepping balance

Reward structure (7 reward keys, each only active during its phase):
  - r_standup: PBRS potential + per-step survival (standup phase)
  - r_success: success bonus at standup→balance transition (standup phase)
  - r_fall: per-step survival + fall/survival terminal (balance phase)
  - r_cross: cross-support balance reward (balance phase)
  - r_joint: joint deviation penalty (balance phase)
  - r_vel: joint velocity penalty (balance phase)
  - r_tilt: torso tilt penalty (balance phase)
  - r_foot: foot height penalty (balance phase)

Balance reward keys match balance_recover_v2 exactly; fall signal comes from
mode transition (balance→standup) instead of ImbalanceTermination.

Routing is based on uprightness (cos of torso tilt), computed from obs:
  uprightness = obs[42]*obs[46] - obs[43]*obs[45]
  standup -> balance when uprightness >= 0.97 (~15°)
  balance -> standup when uprightness < 0.30 (~72°, fallen)

Episode never terminates early — robot can fall and stand up repeatedly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.humanoid21.curriculum.framework.ppo_trainer import (
    _extract_per_step_scalar,
    _extract_per_step_field,
)
from baseline.humanoid21.curriculum.hybrid_actor import HybridActor, compute_uprightness
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint

# Observation indices for uprightness
_OBS_R00 = 42
_OBS_R10 = 43
_OBS_R01 = 45
_OBS_R11 = 46

# Default checkpoint paths
_STANDUP_CKPT = (
    "/data1/mono/things/combatbench/baseline/humanoid21/runs/"
    "standup_v2_r14/checkpoints/checkpoint_u04615.pt"
)
_BALANCE_MODEL = (
    "/data1/mono/things/combatbench/policy/baseline/follow_v2/u09168/fallback/model.pt"
)


class HybridStandupBalanceConfig(CombatExperimentBase):
    """Hybrid standup + balance experiment with dual-network training."""

    name = "hybrid_standup_balance"
    reward_keys = ("r_standup", "r_success", "r_fall", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot")
    gammas = {
        "r_standup": 0.99,
        "r_success": 0.99,
        "r_fall": 0.99,
        "r_cross": 0.99,
        "r_joint": 0.99,
        "r_vel": 0.99,
        "r_tilt": 0.99,
        "r_foot": 0.99,
    }

    BLUEPRINT = "hybrid_env.yaml"

    # Switch thresholds (uprightness = cos of torso tilt from vertical)
    switch_uprightness: float = 0.97   # ~15°: standup -> balance
    fall_uprightness: float = 0.30     # ~72°: balance -> standup (fallen)

    # PPO knobs — very conservative for fine-tuning pretrained models
    learning_rate: float = 3e-6
    critic_learning_rate: float = 1e-5
    log_std_min: float = -1.8
    update_epochs: int = 4
    minibatch_size: int = 4096 * 4
    entropy_coef: float = 2e-3
    grad_clip_norm: float = 0.5

    # Rollout schedule
    episodes_per_update: int = 512
    max_updates: int = 5000
    eval_interval: int = 5
    eval_episodes: int = 32

    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "rollout_distance_min": 1.5,
        "rollout_distance_max": 3.5,
        "max_steps": 200,
        "terminal_fall_penalty": 1.0,
        "per_step_survival_reward": 0.01,
        "potential_reward_scale": 5.0,
        "standup_ckpt": _STANDUP_CKPT,
        "balance_model": _BALANCE_MODEL,
    }
    custom_config: Dict[str, Any] = DEFAULT_CUSTOM_CONFIG


    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def video_env_blueprint(self) -> EnvBlueprint:
        return self._env_pb().materialize(
            max_steps=self.custom_config["max_steps"],
            agent_id="robot_a",
        )

    # --- Model construction ---

    def build_actor(self, device: torch.device) -> HybridActor:
        """Build HybridActor with pretrained sub-networks."""
        actor = HybridActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.actor_hidden_dim,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            switch_uprightness=self.switch_uprightness,
            standup_model_path=self.custom_config["standup_ckpt"],
            balance_model_path=self.custom_config["balance_model"],
            device=device,
        )
        return actor.to(device)

    def build_critic(self, reward_key: str, device: torch.device) -> nn.Module:
        from baseline.common.policies import CriticMLP
        return CriticMLP(
            obs_dim=self.obs_dim, hidden_dim=self.critic_hidden_dim,
        ).to(device)

    # --- Rollout jobs ---

    def _build_jobs(
        self,
        policy_bp: PolicyBlueprint,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[PolicyBlueprint, PolicyBlueprint, EnvBlueprint, int, Dict[str, Any]]]:
        max_steps = self.custom_config["max_steps"]
        env_pb = self._env_pb()
        rng = np.random.default_rng(base_seed)

        env_bps: Dict[str, EnvBlueprint] = {
            aid: env_pb.materialize(max_steps=max_steps, agent_id=aid)
            for aid in ("robot_a", "robot_b")
        }

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
            jobs.append((
                policy_bp, policy_bp,
                env_bps[agent_id], seed,
                {"agent_id": agent_id, "initial_distance": initial_distance},
            ))
        return jobs

    def build_rollout_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp: PolicyBlueprint, base_seed: int):
        return self._build_jobs(policy_bp, base_seed, self.eval_episodes)

    # --- Weights ---

    def initial_weights(self) -> Tuple[float, ...]:
        return (1.0, 6.0, 6.0, 1.0, 0.2, 0.2, 0.2, 0.2)

    def next_weights(
        self,
        eval_metrics: Dict[str, float],
        current_weights: Tuple[float, ...],
    ) -> Tuple[float, ...]:
        return current_weights

    # --- Sub-episode segmentation ---

    def prepare_training_segments(self, episode):
        """Split episode at mode transitions recorded in action_extras.

        Returns 4-tuples ``(start, end, weight, mode)`` where mode is a
        float from the rollout policy's extra (1.0=standup, 0.0=balance).
        Falls back to a single 3-tuple if mode info is unavailable.
        """
        T = episode.num_frames
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        mode_arr = episode.action_extras.get(ep_target, {}).get("mode")
        if mode_arr is None:
            return [(0, T, 1.0)]

        mode_arr = np.asarray(mode_arr, dtype=np.float32).ravel()
        segments = []
        seg_start = 0
        for t in range(1, T):
            if mode_arr[t] != mode_arr[seg_start]:
                segments.append((seg_start, t, 1.0, float(mode_arr[seg_start])))
                seg_start = t
        segments.append((seg_start, T, 1.0, float(mode_arr[seg_start])))
        return segments

    # --- Reward extraction ---

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        """Extract per-step rewards for both phases.

        Standup phase (r_standup):
          - PBRS potential difference
          - Per-step survival bonus
          - Success bonus (+penalty) at standup→balance transition
          - Terminal fall penalty if episode ends fallen

        Balance phase (r_fall, r_cross, r_joint, r_vel, r_tilt, r_foot):
          Matches balance_recover_v2 exactly.  Fall signal comes from
          balance→standup mode transition instead of ImbalanceTermination.
        """
        T = episode.num_frames
        oo = episode.observer_outputs

        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))

        # Mode mask from action_extras (ground truth from rollout policy)
        mode_arr = episode.action_extras.get(ep_target, {}).get("mode")
        if mode_arr is not None:
            mode_arr = np.asarray(mode_arr, dtype=np.float32).ravel()
            balance_mask = mode_arr < 0.5  # 0.0=balance, 1.0=standup
        else:
            # Fallback: compute from uprightness with hysteresis
            obs = episode.observations.get(ep_target)
            if obs is None:
                obs = np.zeros((T, self.obs_dim), dtype=np.float32)
            obs_arr = np.asarray(obs, dtype=np.float32)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr.reshape(1, -1)
            upright = obs_arr[:, _OBS_R00] * obs_arr[:, _OBS_R11] - obs_arr[:, _OBS_R10] * obs_arr[:, _OBS_R01]
            mode = "standup"
            balance_mask = np.zeros(T, dtype=bool)
            for t in range(T):
                if mode == "standup" and upright[t] >= self.switch_uprightness:
                    mode = "balance"
                elif mode == "balance" and upright[t] < self.fall_uprightness:
                    mode = "standup"
                balance_mask[t] = (mode == "balance")

        standup_mask = ~balance_mask
        penalty = float(self.custom_config["terminal_fall_penalty"])
        survival = float(self.custom_config.get("per_step_survival_reward", 0.01))

        # --- r_standup: PBRS + survival + success bonus (standup phase only) ---
        potentials = _extract_per_step_field(oo, "standup", "potential", T)
        pot_scale = float(self.custom_config.get("potential_reward_scale", 5.0))

        r_standup = np.zeros(T, dtype=np.float32)
        if potentials is not None:
            r_standup[1:] += pot_scale * (potentials[1:] - potentials[:-1])
            r_standup[0] += pot_scale * (potentials[0] - 0.0)
        r_standup += survival  # per-step survival for all steps
        # Zero out during balance phase
        r_standup[balance_mask] = 0.0

        # --- r_success: success bonus (standup phase only) ---
        r_success = np.zeros(T, dtype=np.float32)
        # Success bonus: at the LAST standup step before standup→balance transition
        if T > 1:
            success_transitions = np.where(standup_mask[:-1] & balance_mask[1:])[0]
            r_success[success_transitions] += penalty
        r_success[balance_mask] = 0.0

        # --- r_fall: per-step survival + fall/survival terminal (balance phase) ---
        r_fall = np.full(T, survival, dtype=np.float32)
        # Fall: at the LAST balance step before balance→standup transition
        if T > 1:
            fall_transitions = np.where(balance_mask[:-1] & ~balance_mask[1:])[0]
            r_fall[fall_transitions] = -penalty
        # Survival bonus: still in balance at episode end
        if T > 0 and balance_mask[-1]:
            r_fall[-1] = penalty
        # Zero out during standup phase
        r_fall[standup_mask] = 0.0

        # --- r_cross: cross-support balance reward (balance phase) ---
        r_cross = _extract_per_step_scalar(oo, "cross_support", T)
        if r_cross is None:
            r_cross = np.zeros(T, dtype=np.float32)
        r_cross = r_cross.copy()
        r_cross[standup_mask] = 0.0

        # --- r_joint, r_vel, r_tilt, r_foot: from posture observer (balance phase) ---
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

        # Same formulas as balance_recover_v2
        excess_joint = np.maximum(0.0, joint_dev_arr - 0.1)
        r_joint = np.where(excess_joint == 0.0, 0.01, 0.01 - 5.0 * excess_joint)

        excess_vel = np.maximum(0.0, joint_vel_arr - 0.1)
        r_vel = np.where(excess_vel == 0.0, 0.01, 0.01 - 1.0 * excess_vel)

        excess_tilt = np.maximum(0.0, torso_tilt_arr - 0.26)
        r_tilt = np.where(excess_tilt == 0.0, 0.01, 0.01 - 3.0 * excess_tilt)

        excess_foot = np.maximum(0.0, foot_height_arr - 0.10)
        r_foot = np.where(excess_foot == 0.0, 0.01, 0.01 - 5.0 * excess_foot)

        # Zero out during standup phase
        r_joint[standup_mask] = 0.0
        r_vel[standup_mask] = 0.0
        r_tilt[standup_mask] = 0.0
        r_foot[standup_mask] = 0.0

        return {
            "r_standup": r_standup,
            "r_success": r_success,
            "r_fall": r_fall,
            "r_cross": r_cross,
            "r_joint": r_joint,
            "r_vel": r_vel,
            "r_tilt": r_tilt,
            "r_foot": r_foot,
        }

    # --- Advantage combination ---

    def combine_advantages(
        self,
        advs: Dict[str, np.ndarray],
        stage_weights: Tuple[float, ...],
    ) -> Optional[np.ndarray]:
        """Mode-aware advantage combination.

        Standup steps get ONLY r_standup advantage (normalized over standup steps).
        Balance steps get weighted sum of 6 balance advantages (normalized over
        balance steps), matching balance_recover_v2's weight structure.
        """
        adv_standup = advs["r_standup"]
        adv_success = advs["r_success"]
        balance_keys = ("r_fall", "r_cross", "r_joint", "r_vel", "r_tilt", "r_foot")

        # Infer mode mask: balance steps have non-zero r_fall advantage
        balance_mask = np.abs(advs["r_fall"]) > 1e-10
        standup_mask = ~balance_mask

        def _norm_masked(a, mask):
            active = a[mask]
            if len(active) == 0 or float(active.std()) < 1e-8:
                return np.zeros_like(a, dtype=np.float32)
            mean = float(active.mean())
            std = float(active.std())
            result = np.zeros_like(a, dtype=np.float32)
            result[mask] = ((a[mask] - mean) / std).astype(np.float32)
            return result

        combined = np.zeros_like(adv_standup, dtype=np.float32)

        # Standup steps: r_standup + r_success advantage
        w_s = stage_weights[0]
        w_succ = stage_weights[1]
        normed_s = _norm_masked(adv_standup, standup_mask)
        normed_succ = _norm_masked(adv_success, standup_mask)
        combined[standup_mask] = w_s * normed_s[standup_mask] + w_succ * normed_succ[standup_mask]

        # Balance steps: weighted sum of 6 balance advantages
        for i, key in enumerate(balance_keys):
            w_b = stage_weights[2 + i]
            normed_b = _norm_masked(advs[key], balance_mask)
            combined[balance_mask] += w_b * normed_b[balance_mask]

        return combined

    # --- Episode metrics ---

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        """Compute metrics for eval and logging.

        Returns per-episode metrics that the framework aggregates as batch
        means in ``bsum``.  The expanded set tracks both sub-networks
        independently so degradation can be localized early.
        """
        T = episode.num_frames
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        obs = episode.observations.get(ep_target)
        if obs is None:
            return {
                "survived": 0.0, "balance_ratio": 0.0, "standup_count": 0.0,
                "final_uprightness": 0.0, "max_uprightness": 0.0,
                "time_to_first_balance": float(T),
                "mean_balance_duration": 0.0, "mean_standup_duration": float(T),
                "no_balance": 1.0,
                "uprightness_at_switch": 0.0, "uprightness_at_fall": 0.0,
            }

        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        upright = obs_arr[:, _OBS_R00] * obs_arr[:, _OBS_R11] - obs_arr[:, _OBS_R10] * obs_arr[:, _OBS_R01]

        # Walk through mode transitions with hysteresis
        mode = "standup"
        balance_steps = 0
        standup_transitions = 0
        time_to_first_balance = float(T)  # default: never reached
        balance_durations: List[float] = []
        standup_durations: List[float] = []
        seg_start = 0
        switch_uprightnesss: List[float] = []
        fall_uprightnesss: List[float] = []

        for t in range(T):
            if mode == "standup" and upright[t] >= self.switch_uprightness:
                mode = "balance"
                standup_durations.append(float(t - seg_start))
                seg_start = t
                if time_to_first_balance == float(T):
                    time_to_first_balance = float(t)
                switch_uprightnesss.append(float(upright[t]))
            elif mode == "balance" and upright[t] < self.fall_uprightness:
                mode = "standup"
                standup_transitions += 1
                balance_durations.append(float(t - seg_start))
                seg_start = t
                fall_uprightnesss.append(float(upright[t]))
            if mode == "balance":
                balance_steps += 1

        # Close last segment
        if mode == "balance":
            balance_durations.append(float(T - seg_start))
        else:
            standup_durations.append(float(T - seg_start))

        final_upright = float(upright[-1]) if T > 0 else 0.0
        no_balance = 1.0 if balance_steps == 0 else 0.0

        return {
            # Existing
            "survived": 1.0 if final_upright >= self.fall_uprightness else 0.0,
            "balance_ratio": float(balance_steps / max(1, T)),
            "standup_count": float(standup_transitions),
            "final_uprightness": final_upright,
            "max_uprightness": float(np.max(upright)) if T > 0 else 0.0,
            # New: standup_net diagnostics
            "time_to_first_balance": time_to_first_balance,
            "mean_standup_duration": float(np.mean(standup_durations)) if standup_durations else float(T),
            "no_balance": no_balance,
            "uprightness_at_switch": float(np.mean(switch_uprightnesss)) if switch_uprightnesss else 0.0,
            # New: balance_net diagnostics
            "mean_balance_duration": float(np.mean(balance_durations)) if balance_durations else 0.0,
            "uprightness_at_fall": float(np.mean(fall_uprightnesss)) if fall_uprightnesss else 0.0,
        }

    def compare_eval(self, esum: Dict[str, float], best_esum: Dict[str, float]) -> bool:
        if not best_esum:
            return True
        # Prioritize balance_ratio (more time in balance = better)
        return esum.get("balance_ratio", 0.0) > best_esum.get("balance_ratio", 0.0)

    def scheduler_info(self) -> Dict[str, Any]:
        return {
            "switch_uprightness": self.switch_uprightness,
            "fall_uprightness": self.fall_uprightness,
        }

    def scheduler_state(self) -> dict:
        return {}

    def load_scheduler_state(self, state: dict) -> None:
        pass


# Singleton instance for the registry
EXPERIMENT = HybridStandupBalanceConfig()
