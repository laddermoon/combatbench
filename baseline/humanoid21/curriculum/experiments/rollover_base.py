"""Base class for rollover ablation: Delta vs PBRS.

Pure orientation signal (rollover_score = clip((f_down+1)/2, 0, 1)).
No base reward, no terminal bonus, no termination plugin.

Two reward modes:
  - "delta":  r_t = φ(t) - φ(t-1)
  - "pbrs":   r_t = γ·φ(t) - φ(t-1)

All other settings identical to standup_4stage_a.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.humanoid21.curriculum.experiments.base import CombatExperimentBase
from baseline.framework.rollout import extract_per_step_field
from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint


class RolloverBase(CombatExperimentBase):
    """Base class for rollover Delta-vs-PBRS ablation."""

    reward_keys = ("r_standup",)
    gammas = {"r_standup": 0.99}

    BLUEPRINT = "rollover_env.yaml"
    reward_mode: str = "delta"  # overridden by subclasses

    max_updates: int = 300
    episodes_per_update: int = 512
    eval_episodes: int = 64
    eval_interval: int = 5
    video_eval_interval: int = 5

    # PPO params — identical to standup_4stage_a
    log_std_min: float = -2.5
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    grad_clip_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 4096
    entropy_coef: float = 1e-3

    DEFAULT_CUSTOM_CONFIG: Dict[str, Any] = {
        "max_steps": 200,
        "potential_reward_scale": 1.0,
        "success_threshold": 0.97,
        "success_bonus": 1.0,
        "maintain_reward": 0.01,
    }
    custom_config: Dict[str, Any] = DEFAULT_CUSTOM_CONFIG

    _success_rate: float = 0.0

    # ---- Blueprint helpers ------------------------------------------------

    def _env_pb(self) -> ParameterizedEnvBlueprint:
        return ParameterizedEnvBlueprint.load(
            Path(__file__).resolve().parent.parent.parent / "blueprints" / self.BLUEPRINT
        )

    def _materialize_env(self, agent_id: str) -> EnvBlueprint:
        return self._env_pb().materialize(
            agent_id=agent_id,
            max_steps=self.custom_config["max_steps"],
        )

    def video_env_blueprint(self):
        return self._materialize_env("robot_a")

    # ---- Job construction -------------------------------------------------

    def _build_jobs(self, policy_bp, base_seed, n_episodes):
        env_bp = self._materialize_env("robot_a")
        jobs = []
        for i in range(n_episodes):
            jobs.append((
                policy_bp, policy_bp, env_bp, int(base_seed + i),
                {"agent_id": "robot_a", "initial_distance": 2.0},
            ))
        return jobs

    def build_rollout_jobs(self, policy_bp, base_seed):
        return self._build_jobs(policy_bp, base_seed, self.episodes_per_update)

    def build_eval_jobs(self, policy_bp, base_seed):
        return self._build_jobs(policy_bp, base_seed, self.eval_episodes)

    # ---- Eval comparison --------------------------------------------------

    def compare_eval(self, esum, best_esum):
        if not best_esum:
            return True
        return esum.get("max_potential", 0.0) > best_esum.get("max_potential", 0.0)

    # ---- Scheduler --------------------------------------------------------

    def initial_weights(self) -> Tuple[float, ...]:
        return (1.0,)

    def next_weights(self, eval_metrics, current_weights):
        self._success_rate = float(eval_metrics.get("success", 0.0))
        return (1.0,)

    # ---- Reward extraction ------------------------------------------------

    def extract_rewards(self, episode) -> Dict[str, np.ndarray]:
        T = episode.num_frames
        oo = episode.observer_outputs

        potentials = extract_per_step_field(oo, "standup", "potential", T)
        pot_scale = float(self.custom_config.get("potential_reward_scale", 1.0))
        gamma = self.gammas["r_standup"]

        r = np.zeros(T, dtype=np.float32)

        if potentials is not None:
            if self.reward_mode == "delta":
                # r_t = φ(t) - φ(t-1)
                r[1:] = pot_scale * (potentials[1:] - potentials[:-1])
                r[0] = pot_scale * (potentials[0] - 0.0)
            elif self.reward_mode == "pbrs":
                # r_t = γ·φ(t) - φ(t-1)
                r[1:] = pot_scale * (gamma * potentials[1:] - potentials[:-1])
                r[0] = pot_scale * (gamma * potentials[0] - 0.0)
            elif self.reward_mode == "dense_potential":
                # r_t = (1-γ)·φ(t)  — direct dense potential reward.
                # Discounted return = (1-γ)·Σγ^t·φ(t), identical (up to the
                # constant factor already applied here) to Delta's Abel-summed
                # return.  Used to test whether Delta is merely dense potential
                # reward in disguise (§7.1 of the ablation report).
                r[:] = pot_scale * (1.0 - gamma) * potentials[:]
            elif self.reward_mode == "delta_plus_dense":
                # r_t = [φ(t) - φ(t-1)] + c·φ(t)
                # Delta (high-SNR progress signal) + continuous dense base
                # proportional to φ.  The continuous term provides a constant
                # marginal pull toward higher φ even at the top (unlike
                # threshold-gated base reward), targeting Delta's weakness of
                # "satisficing at φ≈0.96".  Coefficient c is configurable via
                # custom_config["dense_base_coef"] (default 0.01).
                c = float(self.custom_config.get("dense_base_coef", 0.01))
                r[1:] = pot_scale * (potentials[1:] - potentials[:-1])
                r[0] = pot_scale * (potentials[0] - 0.0)
                r[:] += c * pot_scale * potentials[:]
            elif self.reward_mode == "generalized_shaping":
                # r_t = γ_s·φ(t) - φ(t-1)
                # Unified single-parameter shaping: γ_s=1 → Delta,
                # γ_s=γ → PBRS, γ_s>1 → Delta+Dense, γ_s<γ → anti-learning.
                # Coefficient γ_s via custom_config["shaping_gamma"] (default 1.0).
                gs = float(self.custom_config.get("shaping_gamma", 1.0))
                r[1:] = pot_scale * (gs * potentials[1:] - potentials[:-1])
                r[0] = pot_scale * (gs * potentials[0] - 0.0)
            elif self.reward_mode == "pbrs_base_flag":
                # PBRS + base reward (方案A: one-time success bonus via flag)
                # PBRS shaping
                r[1:] = pot_scale * (gamma * potentials[1:] - potentials[:-1])
                r[0] = pot_scale * (gamma * potentials[0] - 0.0)
                # Base reward: +1.0 on first crossing threshold (once per episode),
                #              +maintain_reward per step while above threshold
                thr = float(self.custom_config.get("success_threshold", 0.97))
                bonus = float(self.custom_config.get("success_bonus", 1.0))
                maint = float(self.custom_config.get("maintain_reward", 0.01))
                awarded = False
                for t in range(T):
                    if potentials[t] >= thr:
                        if not awarded and (t == 0 or potentials[t - 1] < thr):
                            r[t] += bonus
                            awarded = True
                        r[t] += maint
            elif self.reward_mode == "pbrs_base_symmetric":
                # PBRS + base reward (方案B: symmetric ±1.0 on cross/leave)
                # PBRS shaping
                r[1:] = pot_scale * (gamma * potentials[1:] - potentials[:-1])
                r[0] = pot_scale * (gamma * potentials[0] - 0.0)
                # Base reward: +1.0 each time crossing up, -1.0 each time crossing down,
                #              +maintain_reward per step while above threshold
                thr = float(self.custom_config.get("success_threshold", 0.97))
                bonus = float(self.custom_config.get("success_bonus", 1.0))
                maint = float(self.custom_config.get("maintain_reward", 0.01))
                for t in range(T):
                    prev_below = (t == 0) or (potentials[t - 1] < thr)
                    curr_above = potentials[t] >= thr
                    if curr_above and prev_below:
                        r[t] += bonus
                    elif not curr_above and not prev_below:
                        r[t] -= bonus
                    if curr_above:
                        r[t] += maint
            else:
                raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        return {"r_standup": r}

    # ---- Episode metrics --------------------------------------------------

    def compute_episode_metrics(self, episode) -> Dict[str, float]:
        T = episode.num_frames
        oo = episode.observer_outputs
        potentials = extract_per_step_field(oo, "standup", "potential", T)

        max_pot = float(np.max(potentials)) if potentials is not None and len(potentials) > 0 else 0.0
        final_pot = float(potentials[-1]) if potentials is not None and len(potentials) > 0 else 0.0
        avg_pot = float(np.mean(potentials)) if potentials is not None and len(potentials) > 0 else 0.0

        success = 1.0 if max_pot >= 0.9 else 0.0

        return {
            "success": success,
            "max_potential": max_pot,
            "final_potential": final_pot,
            "avg_potential": avg_pot,
        }

    # ---- Scheduler state --------------------------------------------------

    def scheduler_info(self) -> Dict[str, Any]:
        return {"success_rate": round(self._success_rate, 3)}

    def scheduler_state(self) -> dict:
        return {"success_rate": self._success_rate}

    def load_scheduler_state(self, state: dict) -> None:
        self._success_rate = float(state.get("success_rate", 0.0))
