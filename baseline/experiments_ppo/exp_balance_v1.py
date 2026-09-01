"""V2 end-to-end balance reinforcement: step rewards + perturbation.

Combines ``exp_step.py`` (per-foot stepping rewards) with ``exp_standup_balance.py``
(StandingTriggeredForcePlugin + 12-level force/duration curriculum).

From random fallen state → stand up → step alternately → withstand external
force perturbations.  Built on top of the step policy (warm-start via
--resume-from).

Reward channels (same as exp_step):
  r_potential  = 0.01 × φ(t),              γ=0.99, aw = 3.0 (fixed)
  r_left_foot  = clip(h_left, -0.05, 0.05), γ=0.90, aw = state machine × φ²
  r_right_foot = clip(h_right, -0.05, 0.05), γ=0.90, aw = state machine × φ²

Perturbation curriculum (same as exp_standup_balance):
  12 levels across 3 force tiers (40N / 100N / 200N).
  Level 0-3:  40N  (dur 1-10, 11-20, 21-30, 31-40)
  Level 4-7:  100N (dur 1-10, 11-20, 21-30, 31-40)
  Level 8-11: 200N (dur 1-10, 11-20, 21-30, 31-40)
  Promotion when recovery_rate (1 - fall_count/push_count) >= 0.8.

Blueprint: baseline/humanoid21/end2end/balance_v1_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.common.rollout import extract_per_step_field

from baseline.humanoid21.end2end.stepping_state_machine import (
    compute_foot_weights,
)

from .base import CombatExperimentPPOBase


class BalanceV1(CombatExperimentPPOBase):
    """End-to-end balance reinforcement: step rewards + perturbation.

    Dual-agent: both robots get RandomFallenStatePlugin and
    StandingTriggeredForcePlugin, train simultaneously.  No early
    termination — robot can fall and recover under perturbation.
    """

    name = "balance_v1"

    # --- Network ---
    obs_dim: int = 96
    action_dim: int = 21

    # --- Reward channels ---
    _channel_names = ("r_potential", "r_left_foot", "r_right_foot")
    _channel_gammas = {
        "r_potential": 0.99,
        "r_left_foot": 0.90,
        "r_right_foot": 0.90,
    }
    _gae_lambda: float = 0.95

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01

    # --- Foot height reward saturation ---
    foot_height_clip: float = 0.05

    # --- r_potential actor weight (fixed) ---
    r_potential_actor_weight: float = 3.0

    # --- Env ---
    env_blueprint = ""  # overridden via _env_pb()
    agent_used = "both"
    max_steps: int = 400

    _AGENT_OBS = (
        ("robot_a", "foot_state_a", "standing_balance_a"),
        ("robot_b", "foot_state_b", "standing_balance_b"),
    )
    _AGENT_IDS = ("robot_a", "robot_b")

    # --- PPO tuning (conservative for warm-start) ---
    learning_rate: float = 3e-5
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.03
    update_epochs: int = 4
    minibatch_size: int = 4096
    entropy_coef: float = 1e-3

    # --- Rollout schedule ---
    episodes_per_update: int = 512
    max_updates: int = 5000
    eval_interval: int = 3
    eval_episodes: int = 64

    # --- Video recording ---
    video_eval_interval: int = 2

    # --- Early stop ---
    _no_improvement_limit: int = 200
    _min_updates: int = 600

    # --- Curriculum: 12 levels, 3 force tiers × 4 duration tiers ---
    LEVEL_FORCES: Tuple[float, ...] = (
        40.0, 40.0, 40.0, 40.0,       # level 0-3
        100.0, 100.0, 100.0, 100.0,   # level 4-7
        200.0, 200.0, 200.0, 200.0,   # level 8-11
    )
    LEVEL_DURATION_RANGES: Tuple[Tuple[int, int], ...] = (
        (1, 10), (11, 20), (21, 30), (31, 40),   # 40N
        (1, 10), (11, 20), (21, 30), (31, 40),   # 100N
        (1, 10), (11, 20), (21, 30), (31, 40),   # 200N
    )
    PROMOTE_RECOVERY_RATE: float = 0.8
    """晋级阈值：扰动后不摔倒的比例（recovery_rate = 1 - fall/push）。"""
    PROMOTE_PATIENCE: int = 2

    # --- Stateful metrics ---
    _best_potential: float = -1.0
    _best_survived: float = -1.0
    _success_rate: float = 0.0
    _last_best_update: int = 0

    # --- Curriculum state ---
    _level: int = 0
    _consecutive_pass: int = 0
    _balance_ratio: float = 0.0
    _recovery_rate: float = 0.0
    _best_level: int = -1
    _best_recovery_rate: float = -1.0

    # ------------------------------------------------------------------
    # Blueprint loading — from end2end/ directory
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "end2end" / "balance_v1_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    @property
    def current_force(self) -> float:
        idx = max(0, min(self._level, len(self.LEVEL_FORCES) - 1))
        return float(self.LEVEL_FORCES[idx])

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return tuple(
            RewardChannel(
                name=k,
                gamma=self._channel_gammas[k],
                gae_lambda=self._gae_lambda,
            )
            for k in self._channel_names
        )

    # ------------------------------------------------------------------
    # Job construction — inject impulse_params per episode
    # ------------------------------------------------------------------

    def build_jobs(
        self,
        policy_bp,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[Any, Any, Any, int, Dict[str, Any]]]:
        env_pb = self._env_pb()
        env_bp = env_pb.materialize(max_steps=self.max_steps)

        force = self.current_force
        idx = max(0, min(self._level, len(self.LEVEL_DURATION_RANGES) - 1))
        dur_min, dur_max = self.LEVEL_DURATION_RANGES[idx]

        jobs: List[Tuple[Any, Any, Any, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            impulse_params = {}
            for rid in self._AGENT_IDS:
                impulse_params[rid] = {
                    "force": force,
                    "duration_min": dur_min,
                    "duration_max": dur_max,
                    "body": "torso",
                    "seed": seed,
                }
            jobs.append((
                policy_bp, policy_bp,
                env_bp, seed,
                {"impulse_params": impulse_params},
            ))
        return jobs

    # ------------------------------------------------------------------
    # Trajectory building — same as exp_step
    # ------------------------------------------------------------------

    def _build_agent_trajectory(
        self,
        episode,
        agent_id: str,
        foot_key: str,
        phi_key: str,
    ) -> List[Trajectory]:
        T_full = episode.num_frames
        if T_full == 0:
            return []

        obs_all = episode.observations.get(agent_id)
        acts_all = episode.actions.get(agent_id)
        fin_obs = episode.final_observation.get(agent_id)

        if obs_all is None or acts_all is None or fin_obs is None:
            return []

        obs_all = np.asarray(obs_all, dtype=np.float32)
        acts_all = np.asarray(acts_all, dtype=np.float32)

        # --- Extract φ (4-stage standing potential) ---
        phi_arr = extract_per_step_field(
            episode.observer_outputs, phi_key, "potential", T_full,
        )
        if phi_arr is not None:
            phi_arr = phi_arr[:T_full]
        else:
            phi_arr = np.zeros(T_full, dtype=np.float32)
        phi_arr = np.clip(phi_arr, 0.0, 1.0).astype(np.float32)

        # --- r_potential: 0.01 × φ(t) per step ---
        r_potential = (self.per_step_phi_coef * phi_arr).astype(np.float32)

        # --- Foot heights (saturated) ---
        h_left = extract_per_step_field(
            episode.observer_outputs, foot_key, "h_left_foot", T_full,
        )
        h_right = extract_per_step_field(
            episode.observer_outputs, foot_key, "h_right_foot", T_full,
        )
        if h_left is not None:
            r_left_foot = np.clip(
                np.asarray(h_left[:T_full], dtype=np.float32),
                -self.foot_height_clip, self.foot_height_clip,
            )
        else:
            r_left_foot = np.zeros(T_full, dtype=np.float32)
        if h_right is not None:
            r_right_foot = np.clip(
                np.asarray(h_right[:T_full], dtype=np.float32),
                -self.foot_height_clip, self.foot_height_clip,
            )
        else:
            r_right_foot = np.zeros(T_full, dtype=np.float32)

        # --- Contacts → stepping state machine → foot actor weights ---
        contact_l = extract_per_step_field(
            episode.observer_outputs, foot_key, "left_foot_contact", T_full,
        )
        contact_r = extract_per_step_field(
            episode.observer_outputs, foot_key, "right_foot_contact", T_full,
        )
        if contact_l is not None and contact_r is not None:
            w_left, w_right = compute_foot_weights(
                np.asarray(contact_l[:T_full], dtype=bool),
                np.asarray(contact_r[:T_full], dtype=bool),
                T_full,
                h_left=np.asarray(h_left[:T_full], dtype=np.float32) if h_left is not None else None,
                h_right=np.asarray(h_right[:T_full], dtype=np.float32) if h_right is not None else None,
            )
        else:
            w_left = np.zeros(T_full, dtype=np.float32)
            w_right = np.zeros(T_full, dtype=np.float32)

        # --- No early termination: robot can fall and get back up ---
        is_terminated = False

        # --- Actor weights: r_potential fixed, foot channels gated by φ² ---
        phi_sq = (phi_arr ** 2).astype(np.float32)
        actor_weights = {
            "r_potential": np.full(T_full, self.r_potential_actor_weight, dtype=np.float32),
            "r_left_foot": (w_left * phi_sq),
            "r_right_foot": (w_right * phi_sq),
        }

        all_rewards = {
            "r_potential": r_potential,
            "r_left_foot": r_left_foot,
            "r_right_foot": r_right_foot,
        }

        channels: Dict[str, ChannelData] = {}
        for key in self._channel_names:
            channels[key] = ChannelData(
                reward=all_rewards[key].astype(np.float32),
                is_terminated=is_terminated,
                actor_weight=actor_weights[key],
            )

        return [Trajectory(
            obs=obs_all,
            actions=acts_all,
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            mode=None,
        )]

    def build_trajectories(self, episodes) -> List[Trajectory]:
        all_trajs: List[Trajectory] = []
        for episode in episodes:
            for agent_id, foot_key, phi_key in self._AGENT_OBS:
                trajs = self._build_agent_trajectory(
                    episode, agent_id, foot_key, phi_key,
                )
                all_trajs.extend(trajs)
        return all_trajs

    # ------------------------------------------------------------------
    # Eval — balance_ratio + curriculum promotion
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        max_pots = []
        final_pots = []
        success_count = 0
        n_agents = 0
        balance_ratios: List[float] = []
        total_push = 0
        total_fall = 0

        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue

            # --- 从 episode_metrics 提取 push/fall count ---
            em = dict(ep.episode_metrics) if hasattr(ep, "episode_metrics") else {}
            for rid in self._AGENT_IDS:
                total_push += int(em.get(f"{rid}_push_count", 0))
                total_fall += int(em.get(f"{rid}_fall_count", 0))

            for agent_id, _, phi_key in self._AGENT_OBS:
                n_agents += 1
                phi = extract_per_step_field(
                    ep.observer_outputs, phi_key, "potential", T,
                )
                if phi is not None and len(phi) > 0:
                    mx = float(np.max(phi))
                    fn = float(phi[-1])
                else:
                    mx = 0.0
                    fn = 0.0
                max_pots.append(mx)
                final_pots.append(fn)
                if mx >= 0.9:
                    success_count += 1

                # balance_ratio: fraction of steps with h_torso > 1.0
                h_torso = extract_per_step_field(
                    ep.observer_outputs, phi_key, "h_torso", T,
                )
                if h_torso is not None and len(h_torso) > 0:
                    balance_ratios.append(float(np.mean(np.asarray(h_torso) > 1.0)))

        n = max(len(max_pots), 1)
        mean_max_pot = sum(max_pots) / n if max_pots else 0.0
        mean_final_pot = sum(final_pots) / n if final_pots else 0.0
        success_rate = success_count / n
        mean_balance_ratio = float(np.mean(balance_ratios)) if balance_ratios else 0.0

        # --- recovery_rate: 扰动后不摔倒的比例 ---
        recovery_rate = float(1.0 - total_fall / max(total_push, 1)) if total_push > 0 else 1.0

        self._success_rate = success_rate
        self._balance_ratio = mean_balance_ratio
        self._recovery_rate = recovery_rate

        # --- Curriculum promotion: 基于 recovery_rate ---
        prev_level = self._level
        if self._level < len(self.LEVEL_FORCES) - 1:
            if recovery_rate >= self.PROMOTE_RECOVERY_RATE:
                self._consecutive_pass += 1
                if self._consecutive_pass >= self.PROMOTE_PATIENCE:
                    self._level += 1
                    self._consecutive_pass = 0
            else:
                self._consecutive_pass = 0

        # 晋级时重置 best 指标，让每个 level 独立计算 best
        if self._level > prev_level:
            self._best_recovery_rate = -1.0
            self._best_potential = -1.0

        # --- Best-of-run: level > recovery_rate > max_pot ---
        current_level = self._level
        is_new_best = (
            current_level > self._best_level
            or (current_level == self._best_level
                and recovery_rate > self._best_recovery_rate)
            or (current_level == self._best_level
                and recovery_rate == self._best_recovery_rate
                and mean_max_pot > self._best_potential)
        )
        if is_new_best:
            self._best_level = current_level
            self._best_recovery_rate = recovery_rate
            self._best_potential = mean_max_pot
            self._last_best_update = update

        # --- No early stop: train until user stops or max updates ---
        stop_training = False

        return {
            "is_new_best": is_new_best,
            "stop_training": stop_training,
            "info": {
                "max_pot": round(mean_max_pot, 3),
                "final_pot": round(mean_final_pot, 3),
                "success": round(success_rate, 3),
                "balance_ratio": round(mean_balance_ratio, 3),
                "recovery_rate": round(recovery_rate, 3),
                "push_count": total_push,
                "fall_count": total_fall,
                "level": float(self._level),
                "force": round(self.current_force, 1),
            },
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "best_potential": self._best_potential,
            "success_rate": self._success_rate,
            "last_best_update": self._last_best_update,
            "level": self._level,
            "consecutive_pass": self._consecutive_pass,
            "balance_ratio": self._balance_ratio,
            "recovery_rate": self._recovery_rate,
            "best_level": self._best_level,
            "best_recovery_rate": self._best_recovery_rate,
        }

    def load_state(self, state: dict) -> None:
        self._best_potential = float(state.get("best_potential", -1.0))
        self._success_rate = float(state.get("success_rate", 0.0))
        self._last_best_update = int(state.get("last_best_update", 0))
        self._level = int(state.get("level", 0))
        self._consecutive_pass = int(state.get("consecutive_pass", 0))
        self._balance_ratio = float(state.get("balance_ratio", 0.0))
        self._recovery_rate = float(state.get("recovery_rate", 0.0))
        self._best_level = int(state.get("best_level", -1))
        self._best_recovery_rate = float(state.get("best_recovery_rate", -1.0))


EXPERIMENT_CLASS = BalanceV1
