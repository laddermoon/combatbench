"""Follow experiment: chase a scripted moving target while maintaining balance.

Single-agent curriculum experiment.  The opponent is controlled by
``RandomMovePlugin`` (teleport-based, never falls).  The learning agent
must approach and stay near the opponent without losing balance.

No MixedPolicy / fallback — the learning policy faces the environment
directly and must learn balance on its own.

Reward channels (4):
  - r_fall:       0.01 × φ(t) per step (dense survival, no terminal penalty)
                  actor_weight = 3.0 (fixed)
  - r_cross:      CrossSupportBalanceRewarder output
                  actor_weight = 1.0 × φ²  (balance matters less when about to fall)
  - r_radial:     radial approach velocity (toward opponent, active outside 0.9m)
                  actor_weight = 3.0 × φ²
  - r_tangential: tangential movement penalty (active outside 0.9m)
                  actor_weight = 1.0 × φ²

φ = uprightness × (height / standing_height)  from HeightPhiObserver.

Curriculum: opponent movement speed increases through 8 levels
(0.0 → 0.7 m/s).  Promotion when hold_ratio ≥ 0.5 for 1 consecutive eval.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.common.rollout import (
    extract_per_step_field,
    extract_per_step_scalar,
)
from baseline.humanoid21.rewards.follow_opponent import (
    compute_radial_tangential_rewards,
)

from .base import CombatExperimentPPOBase


class Follow(CombatExperimentPPOBase):

    name = "follow"

    # --- Reward channels ---
    _channel_names = ("r_fall", "r_cross", "r_radial", "r_tangential")
    _gamma = 0.99
    _gae_lambda = 0.95

    # --- Env / rollout config ---
    agent_used = "random"
    max_steps = 600
    INITIAL_DISTANCE: float = 2.0

    episodes_per_update: int = 1024
    eval_episodes: int = 128
    eval_interval: int = 2
    video_eval_interval: int = 2
    max_updates: int = 20000

    # --- PPO tuning (match V1 follow_v2) ---
    log_std_min: float = -1.8
    learning_rate: float = 3e-5
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096 * 4
    entropy_coef: float = 1.5e-3

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01

    # --- Base actor weights (r_fall fixed, others gated by φ²) ---
    _base_actor_weights: Tuple[float, ...] = (3.0, 1.0, 3.0, 1.0)

    # --- Curriculum: opponent movement speed per level (m/s) ---
    LEVEL_SPEEDS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5)
    PROMOTE_HOLD_RATIO: float = 0.5
    PROMOTE_PATIENCE: int = 1

    # --- Early stop ---
    _no_improvement_limit: int = 200
    _min_updates: int = 600

    # --- Stateful scheduler ---
    _level: int = 0
    _consecutive_pass: int = 0
    _hold_ratio: float = 0.0
    _survival_rate: float = 0.0
    _best_survived: float = -1.0
    _best_level: int = -1
    _best_hold_ratio: float = -1.0
    _last_best_update: int = 0

    _AGENT_IDS = ("robot_a", "robot_b")

    # ------------------------------------------------------------------
    # Env blueprint
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "follow" / "follow_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    @property
    def current_speed(self) -> float:
        idx = max(0, min(self._level, len(self.LEVEL_SPEEDS) - 1))
        return float(self.LEVEL_SPEEDS[idx])

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return tuple(
            RewardChannel(name=k, gamma=self._gamma, gae_lambda=self._gae_lambda)
            for k in self._channel_names
        )

    # ------------------------------------------------------------------
    # Job construction — per-episode env materialization with speed
    # ------------------------------------------------------------------

    def build_jobs(
        self,
        policy_bp,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[Any, Any, Any, int, Dict[str, Any]]]:
        env_pb = self._env_pb()
        speed = self.current_speed

        jobs: List[Tuple[Any, Any, Any, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            oppo_agent_id = "robot_b" if agent_id == "robot_a" else "robot_a"
            env_bp = env_pb.materialize(
                max_steps=self.max_steps,
                agent_id=agent_id,
                oppo_agent_id=oppo_agent_id,
                random_move_speed=speed,
            )
            jobs.append((
                policy_bp, policy_bp,
                env_bp, seed,
                {"agent_id": agent_id, "initial_distance": self.INITIAL_DISTANCE},
            ))
        return jobs

    # ------------------------------------------------------------------
    # Trajectory building
    # ------------------------------------------------------------------

    def _build_agent_trajectory(
        self,
        episode,
        agent_id: str,
    ) -> List[Trajectory]:
        T_full = episode.num_frames
        if T_full == 0:
            return []

        # --- Truncate at agent's termination step ---
        records = episode.agent_termination_proposal_records.get(agent_id, ())
        if records:
            first_reason, term_step = records[0]
            fell = first_reason.startswith("imbalance")
            T = term_step if fell else T_full
        else:
            fell = False
            T = T_full

        if T == 0:
            return []

        obs_all = episode.observations.get(agent_id)
        acts_all = episode.actions.get(agent_id)
        fin_obs = episode.final_observation.get(agent_id)

        if obs_all is None or acts_all is None or fin_obs is None:
            return []

        obs_all = np.asarray(obs_all, dtype=np.float32)
        acts_all = np.asarray(acts_all, dtype=np.float32)

        oo = episode.observer_outputs

        # --- Extract φ per step ---
        phi_arr = extract_per_step_field(oo, "height_phi", "phi", T_full)
        if phi_arr is not None:
            phi_arr = phi_arr[:T]
        else:
            phi_arr = np.ones(T, dtype=np.float32)
        phi_arr = np.clip(phi_arr, 0.0, 1.0).astype(np.float32)

        # --- r_fall: 0.01 × φ(t) per step (no terminal penalty) ---
        r_fall = (self.per_step_phi_coef * phi_arr).astype(np.float32)

        # --- r_cross ---
        r_cross = extract_per_step_scalar(oo, "cross_support", T_full)
        if r_cross is not None:
            r_cross = r_cross[:T]
        else:
            r_cross = np.zeros(T, dtype=np.float32)

        # --- r_radial / r_tangential: velocity decomposition ---
        self_x = extract_per_step_field(oo, "approach_velocity", "self_x", T_full)
        self_y = extract_per_step_field(oo, "approach_velocity", "self_y", T_full)
        opp_x = extract_per_step_field(oo, "approach_velocity", "opp_x", T_full)
        opp_y = extract_per_step_field(oo, "approach_velocity", "opp_y", T_full)

        if self_x is None or self_y is None or opp_x is None or opp_y is None:
            r_radial = np.zeros(T, dtype=np.float32)
            r_tangential = np.zeros(T, dtype=np.float32)
        else:
            self_xy = np.stack([self_x[:T], self_y[:T]], axis=1)
            opp_xy = np.stack([opp_x[:T], opp_y[:T]], axis=1)
            r_radial, r_tangential = compute_radial_tangential_rewards(
                self_xy, opp_xy,
            )

        # --- Actor weights: r_fall fixed, others gated by φ² ---
        phi_sq = (phi_arr ** 2).astype(np.float32)
        actor_weights = {
            "r_fall": np.full(T, self._base_actor_weights[0], dtype=np.float32),
            "r_cross": (self._base_actor_weights[1] * phi_sq),
            "r_radial": (self._base_actor_weights[2] * phi_sq),
            "r_tangential": (self._base_actor_weights[3] * phi_sq),
        }

        is_terminated = fell

        # --- Build channels ---
        all_rewards = {
            "r_fall": r_fall,
            "r_cross": r_cross.astype(np.float32),
            "r_radial": r_radial.astype(np.float32),
            "r_tangential": r_tangential.astype(np.float32),
        }

        channels: Dict[str, ChannelData] = {}
        for key in self._channel_names:
            channels[key] = ChannelData(
                reward=all_rewards[key].astype(np.float32),
                is_terminated=is_terminated,
                actor_weight=actor_weights[key],
            )

        return [Trajectory(
            obs=np.asarray(obs_all[:T], dtype=np.float32),
            actions=np.asarray(acts_all[:T], dtype=np.float32),
            last_obs=np.asarray(fin_obs, dtype=np.float32),
            channels=channels,
            importance=1.0,
            mode=None,
        )]

    def build_trajectories(self, episodes) -> List[Trajectory]:
        all_trajs: List[Trajectory] = []
        for episode in episodes:
            agent_id = str(episode.episode_options.get("agent_id", "robot_a"))
            agent_trajs = self._build_agent_trajectory(episode, agent_id)
            all_trajs.extend(agent_trajs)
        return all_trajs

    # ------------------------------------------------------------------
    # Eval — metrics, curriculum promotion, best-of-run, early stop
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        survived_count = 0
        total_agents = 0
        hold_ratios: List[float] = []

        for ep in episodes:
            agent_id = str(ep.episode_options.get("agent_id", "robot_a"))
            total_agents += 1

            # Survival check
            term_reason = ep.agent_termination_reason.get(agent_id, "")
            if not term_reason.startswith("imbalance"):
                survived_count += 1

            # hold_ratio: fraction of steps within 1.1m of opponent
            T = ep.num_frames
            oo = ep.observer_outputs
            self_x = extract_per_step_field(oo, "approach_velocity", "self_x", T)
            self_y = extract_per_step_field(oo, "approach_velocity", "self_y", T)
            opp_x = extract_per_step_field(oo, "approach_velocity", "opp_x", T)
            opp_y = extract_per_step_field(oo, "approach_velocity", "opp_y", T)

            if all(v is not None for v in (self_x, self_y, opp_x, opp_y)):
                raw_dist = np.sqrt(
                    (self_x - opp_x) ** 2 + (self_y - opp_y) ** 2
                )
                if len(raw_dist) > 0:
                    hold_ratios.append(float(np.mean(raw_dist <= 1.1)))

        survival_rate = float(survived_count / max(total_agents, 1))
        mean_hold_ratio = float(np.mean(hold_ratios)) if hold_ratios else 0.0

        self._survival_rate = survival_rate
        self._hold_ratio = mean_hold_ratio

        # --- Curriculum promotion ---
        if self._level < len(self.LEVEL_SPEEDS) - 1:
            if mean_hold_ratio >= self.PROMOTE_HOLD_RATIO:
                self._consecutive_pass += 1
                if self._consecutive_pass >= self.PROMOTE_PATIENCE:
                    self._level += 1
                    self._consecutive_pass = 0
            else:
                self._consecutive_pass = 0

        # --- Best-of-run: 3-level priority (survived > level > hold_ratio) ---
        survived_metric = float(survived_count)
        current_level = self._level
        is_new_best = (
            survived_metric > self._best_survived
            or (survived_metric == self._best_survived and current_level > self._best_level)
            or (survived_metric == self._best_survived and current_level == self._best_level
                and mean_hold_ratio > self._best_hold_ratio)
        )
        if is_new_best:
            self._best_survived = survived_metric
            self._best_level = current_level
            self._best_hold_ratio = mean_hold_ratio
            self._last_best_update = update

        # --- Early stop ---
        no_improvement = update - self._last_best_update
        stop_training = (
            no_improvement >= self._no_improvement_limit
            and update >= self._min_updates
        )

        return {
            "is_new_best": is_new_best,
            "stop_training": stop_training,
            "info": {
                "survived": survived_metric,
                "survival_rate": round(survival_rate, 3),
                "hold_ratio": round(mean_hold_ratio, 3),
                "level": float(self._level),
                "opp_speed": round(self.current_speed, 3),
            },
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "level": self._level,
            "consecutive_pass": self._consecutive_pass,
            "hold_ratio": self._hold_ratio,
            "survival_rate": self._survival_rate,
            "best_survived": self._best_survived,
            "best_level": self._best_level,
            "best_hold_ratio": self._best_hold_ratio,
            "last_best_update": self._last_best_update,
        }

    def load_state(self, state: dict) -> None:
        self._level = int(state.get("level", 0))
        self._consecutive_pass = int(state.get("consecutive_pass", 0))
        self._hold_ratio = float(state.get("hold_ratio", 0.0))
        self._survival_rate = float(state.get("survival_rate", 0.0))
        self._best_survived = float(state.get("best_survived", -1.0))
        self._best_level = int(state.get("best_level", -1))
        self._best_hold_ratio = float(state.get("best_hold_ratio", -1.0))
        self._last_best_update = int(state.get("last_best_update", 0))


EXPERIMENT_CLASS = Follow
