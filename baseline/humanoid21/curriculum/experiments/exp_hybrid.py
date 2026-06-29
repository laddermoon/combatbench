"""Hybrid standup + balance experiment.

Trains a HybridActor with two sub-networks:
  - standup_net: initialized from standup_v2_r14 checkpoint, learns to stand up fast
  - balance_net: initialized from follow_v2 fallback checkpoint, learns stepping balance

Reward structure (two reward keys, each only active during its phase):
  - r_standup: PBRS potential + per-step survival (standup phase only)
  - r_balance: cross-support + tilt + height (balance phase only)

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
    reward_keys = ("r_standup", "r_balance")
    gammas = {
        "r_standup": 0.99,
        "r_balance": 0.99,
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
        "max_steps": 600,
        "terminal_fall_penalty": 1.0,
        "per_step_survival_reward": 0.01,
        "potential_reward_scale": 5.0,
        "balance_reward_scale": 0.1,
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
        return (1.0, 1.0)

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

        r_standup: active during standup mode
          - PBRS potential difference
          - Per-step survival bonus

        r_balance: active during balance mode
          - Per-step survival bonus (maintain standing)
          - Fall penalty at balance→standup transition
          - Survival bonus if still in balance at episode end
          - Cross-support balance reward (scaled)
          - Torso tilt penalty
          - Height reward
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

        # --- r_standup: PBRS + survival (standup phase only) ---
        potentials = _extract_per_step_field(oo, "standup", "potential", T)
        pot_scale = float(self.custom_config.get("potential_reward_scale", 5.0))
        survival = float(self.custom_config.get("per_step_survival_reward", 0.01))

        r_standup = np.zeros(T, dtype=np.float32)
        if potentials is not None:
            r_standup[1:] += pot_scale * (potentials[1:] - potentials[:-1])
            r_standup[0] += pot_scale * (potentials[0] - 0.0)
        r_standup += survival  # per-step survival for all steps
        # Zero out during balance phase
        r_standup[balance_mask] = 0.0

        # Terminal penalty: if episode ends with robot fallen
        obs = episode.observations.get(ep_target)
        if obs is not None and T > 0:
            obs_arr = np.asarray(obs, dtype=np.float32)
            if obs_arr.ndim == 2:
                final_upright = obs_arr[-1, _OBS_R00] * obs_arr[-1, _OBS_R11] - obs_arr[-1, _OBS_R10] * obs_arr[-1, _OBS_R01]
                if final_upright < self.fall_uprightness:
                    r_standup[-1] -= float(self.custom_config["terminal_fall_penalty"])

        # --- r_balance: survival + fall penalty + cross-support + tilt + height ---
        bal_scale = float(self.custom_config.get("balance_reward_scale", 0.1))
        r_cross = _extract_per_step_scalar(oo, "cross_support", T)

        # Tilt from posture observer
        torso_tilt_arr = _extract_per_step_field(oo, "posture", "torso_tilt", T)
        if torso_tilt_arr is None:
            torso_tilt_arr = np.zeros(T, dtype=np.float32)

        # Height from height observer
        heights = _extract_per_step_field(oo, "height", "height", T)
        if heights is None:
            heights = np.zeros(T, dtype=np.float32)

        # Tilt penalty: penalize when tilt > 0.26 rad (~15°)
        excess_tilt = np.maximum(0.0, torso_tilt_arr - 0.26)
        r_tilt = np.where(excess_tilt == 0.0, 0.01, 0.01 - 3.0 * excess_tilt)

        # Height reward: linear penalty below 0.55m, bonus above
        r_height = (heights - 0.55) * 0.1

        # Per-step survival bonus during balance (maintain standing)
        r_balance_survival = np.zeros(T, dtype=np.float32)
        r_balance_survival[balance_mask] = survival

        # Fall penalty: balance→standup transition points
        r_balance_fall = np.zeros(T, dtype=np.float32)
        if T > 1:
            fall_transitions = np.where(balance_mask[:-1] & ~balance_mask[1:])[0] + 1
            r_balance_fall[fall_transitions] = -float(self.custom_config["terminal_fall_penalty"])

        # Survival bonus: still in balance at episode end
        if T > 0 and balance_mask[-1]:
            r_balance_fall[-1] = float(self.custom_config["terminal_fall_penalty"])

        r_balance = (
            bal_scale * (r_cross + r_tilt + r_height)
            + r_balance_survival
            + r_balance_fall
        )
        # Zero out during standup phase
        r_balance[standup_mask] = 0.0

        return {
            "r_standup": r_standup,
            "r_balance": r_balance,
        }

    # --- Advantage combination ---

    def combine_advantages(
        self,
        advs: Dict[str, np.ndarray],
        stage_weights: Tuple[float, ...],
    ) -> Optional[np.ndarray]:
        """Mode-aware advantage combination.

        Standup steps get ONLY r_standup advantage (zeroed for balance steps).
        Balance steps get ONLY r_balance advantage (zeroed for standup steps).
        Mode is inferred from which advantage is non-zero (rewards are
        zeroed in the inactive phase during extract_rewards).
        """
        adv_standup = advs["r_standup"]
        adv_balance = advs["r_balance"]

        # Infer mode mask from advantage magnitudes (rewards are zeroed
        # in inactive phases, so advantages should be ~0 there)
        balance_mask = np.abs(adv_balance) > 1e-10
        standup_mask = ~balance_mask

        # Normalize each advantage ONLY over its active steps
        def _norm_masked(a, mask):
            active = a[mask]
            if len(active) == 0 or float(active.std()) < 1e-8:
                return np.zeros_like(a, dtype=np.float32)
            mean = float(active.mean())
            std = float(active.std())
            result = np.zeros_like(a, dtype=np.float32)
            result[mask] = ((a[mask] - mean) / std).astype(np.float32)
            return result

        normed_s = _norm_masked(adv_standup, standup_mask)
        normed_b = _norm_masked(adv_balance, balance_mask)

        # Combine: standup steps get standup advantage, balance steps get balance advantage
        w_s, w_b = stage_weights
        combined = np.zeros_like(adv_standup, dtype=np.float32)
        combined[standup_mask] = w_s * normed_s[standup_mask]
        combined[balance_mask] = w_b * normed_b[balance_mask]
        return combined

    # --- Episode metrics ---

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        """Compute metrics for eval and logging."""
        T = episode.num_frames
        ep_target = str(episode.episode_options.get("agent_id", "robot_a"))
        obs = episode.observations.get(ep_target)
        if obs is None:
            return {"survived": 0.0, "balance_ratio": 0.0, "standup_count": 0.0}

        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        upright = obs_arr[:, _OBS_R00] * obs_arr[:, _OBS_R11] - obs_arr[:, _OBS_R10] * obs_arr[:, _OBS_R01]

        # Count balance vs standup steps
        mode = "standup"
        balance_steps = 0
        standup_transitions = 0
        for t in range(T):
            if mode == "standup" and upright[t] >= self.switch_uprightness:
                mode = "balance"
            elif mode == "balance" and upright[t] < self.fall_uprightness:
                mode = "standup"
                standup_transitions += 1
            if mode == "balance":
                balance_steps += 1

        # Final uprightness
        final_upright = float(upright[-1]) if T > 0 else 0.0

        return {
            "survived": 1.0 if final_upright >= self.fall_uprightness else 0.0,
            "balance_ratio": float(balance_steps / max(1, T)),
            "standup_count": float(standup_transitions),
            "final_uprightness": final_upright,
            "max_uprightness": float(np.max(upright)) if T > 0 else 0.0,
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
