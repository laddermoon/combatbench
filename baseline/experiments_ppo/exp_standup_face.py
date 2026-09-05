"""V2 end-to-end step 4: standup + follow + face opponent.

On top of standup_follow, adds r_face: a conditional reward that
activates when within D_FACE of the opponent, rewarding the robot for
facing the opponent (torso forward axis pointing toward opponent).

Reward channels (5):
  - r_fall:       0.01 × φ(t) per step, actor_weight = 3.0 (fixed)
  - r_cross:      CrossSupportBalanceRewarder, actor_weight = 1.0 × φ²
  - r_radial:     radial approach velocity, actor_weight = 3.0 × φ²
  - r_tangential: tangential penalty, actor_weight = 1.0 × φ²
  - r_face:       facing_score × dist_gate, actor_weight = 1.0 × φ²

φ is the 4-stage standing potential. No imbalance termination.
Curriculum: 13 levels (0.0 → 1.5 m/s).

Blueprint: baseline/humanoid21/end2end/standup_face_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import (
    extract_per_step_field,
    extract_per_step_scalar,
)
from baseline.humanoid21.rewards.follow_opponent import (
    compute_radial_tangential_rewards,
)

from .base import CombatExperimentPPOBase
from baseline.framework.rollout.job import Job


# --- Face reward constants ---
D_FACE: float = 1.5     # m — face reward starts activating
D_STRIKE: float = 0.7   # m — face reward fully active


class StandupFace(CombatExperimentPPOBase):
    """End-to-end step 4: standup + follow + face opponent.

    Single-agent curriculum.  Opponent controlled by RandomMovePlugin.
    Learning agent starts from random fallen state, must stand up, follow,
    and face the opponent when close.
    """

    name = "standup_face"

    # --- Reward channels ---
    _channel_names = ("r_fall", "r_cross", "r_radial", "r_tangential", "r_face")
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

    # --- PPO tuning (match follow) ---
    log_std_min: float = -1.8
    learning_rate: float = 3e-5
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096 * 4
    entropy_coef: float = 1.5e-3

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01

    # --- Base actor weights (r_fall fixed, others gated by φ²) ---
    _base_actor_weights: Tuple[float, ...] = (3.0, 1.0, 3.0, 1.0, 1.0)

    # --- Curriculum ---
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
    _facing_ratio: float = 0.0
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
            / "humanoid21" / "end2end" / "standup_face_env.yaml"
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
    # Job construction
    # ------------------------------------------------------------------

    def build_jobs(
        self,
        policy_bp,
        base_seed: int,
        n_episodes: int,
    ) -> List[Job]:
        env_pb = self._env_pb()
        speed = self.current_speed

        jobs: List[Job] = []
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
            jobs.append(Job(
    policy_a_bp=policy_bp,
    policy_b_bp=policy_bp,
    env_bp=env_bp,
    seed=seed,
    episode_options={"agent_id": agent_id, "initial_distance": self.INITIAL_DISTANCE},
    explore_intensity_a=self.explore_intensity,
    explore_intensity_b=self.explore_intensity,
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

        obs_all = episode.observations.get(agent_id)
        acts_all = episode.actions.get(agent_id)
        fin_obs = episode.final_observation.get(agent_id)

        if obs_all is None or acts_all is None or fin_obs is None:
            return []

        obs_all = np.asarray(obs_all, dtype=np.float32)
        acts_all = np.asarray(acts_all, dtype=np.float32)

        oo = episode.observer_outputs

        # --- Extract φ (4-stage standing potential) ---
        phi_arr = extract_per_step_field(oo, "standing_balance", "potential", T_full)
        if phi_arr is not None:
            phi_arr = phi_arr[:T_full]
        else:
            phi_arr = np.zeros(T_full, dtype=np.float32)
        phi_arr = np.clip(phi_arr, 0.0, 1.0).astype(np.float32)

        # --- r_fall ---
        r_fall = (self.per_step_phi_coef * phi_arr).astype(np.float32)

        # --- r_cross ---
        r_cross = extract_per_step_scalar(oo, "cross_support", T_full)
        if r_cross is not None:
            r_cross = r_cross[:T_full]
        else:
            r_cross = np.zeros(T_full, dtype=np.float32)

        # --- r_radial / r_tangential ---
        self_x = extract_per_step_field(oo, "approach_velocity", "self_x", T_full)
        self_y = extract_per_step_field(oo, "approach_velocity", "self_y", T_full)
        opp_x = extract_per_step_field(oo, "approach_velocity", "opp_x", T_full)
        opp_y = extract_per_step_field(oo, "approach_velocity", "opp_y", T_full)

        if self_x is None or self_y is None or opp_x is None or opp_y is None:
            r_radial = np.zeros(T_full, dtype=np.float32)
            r_tangential = np.zeros(T_full, dtype=np.float32)
            self_xy = np.zeros((T_full, 2), dtype=np.float64)
            opp_xy = np.zeros((T_full, 2), dtype=np.float64)
        else:
            self_xy = np.stack([self_x[:T_full], self_y[:T_full]], axis=1)
            opp_xy = np.stack([opp_x[:T_full], opp_y[:T_full]], axis=1)
            r_radial, r_tangential = compute_radial_tangential_rewards(
                self_xy, opp_xy,
            )

        # --- r_face: facing_score × dist_gate ---
        fwd_x = extract_per_step_field(oo, "face_opponent", "forward_x", T_full)
        fwd_y = extract_per_step_field(oo, "face_opponent", "forward_y", T_full)

        r_face = np.zeros(T_full, dtype=np.float32)
        if fwd_x is not None and fwd_y is not None and self_x is not None:
            fwd_x = np.asarray(fwd_x[:T_full], dtype=np.float64)
            fwd_y = np.asarray(fwd_y[:T_full], dtype=np.float64)
            fwd = np.stack([fwd_x, fwd_y], axis=1)  # (T, 2)

            # 2D distance
            dist = np.linalg.norm(opp_xy[:T_full] - self_xy[:T_full], axis=1)

            # Direction to opponent
            to_opp = opp_xy[:T_full] - self_xy[:T_full]
            to_opp_norm = np.linalg.norm(to_opp, axis=1)
            valid = to_opp_norm > 1e-6
            to_opp_hat = np.zeros((T_full, 2), dtype=np.float64)
            to_opp_hat[valid] = to_opp[valid] / to_opp_norm[valid, None]

            # cos_angle: 1 = facing opponent, -1 = facing away
            cos_angle = np.sum(fwd * to_opp_hat, axis=1)
            facing_score = np.maximum(0.0, cos_angle)

            # Distance gate: 0 beyond D_FACE, 1 at D_STRIKE, linear between
            dist_gate = np.clip(
                (D_FACE - dist) / (D_FACE - D_STRIKE), 0.0, 1.0
            )

            r_face = (facing_score * dist_gate).astype(np.float32)

        # --- No early termination ---
        is_terminated = False

        # --- Actor weights: r_fall fixed, others gated by φ² ---
        phi_sq = (phi_arr ** 2).astype(np.float32)
        actor_weights = {
            "r_fall": np.full(T_full, self._base_actor_weights[0], dtype=np.float32),
            "r_cross": (self._base_actor_weights[1] * phi_sq),
            "r_radial": (self._base_actor_weights[2] * phi_sq),
            "r_tangential": (self._base_actor_weights[3] * phi_sq),
            "r_face": (self._base_actor_weights[4] * phi_sq),
        }

        all_rewards = {
            "r_fall": r_fall,
            "r_cross": r_cross.astype(np.float32),
            "r_radial": r_radial.astype(np.float32),
            "r_tangential": r_tangential.astype(np.float32),
            "r_face": r_face,
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
            explore_intensity=self.extract_explore_intensity(episode, agent_id, T_full),
        )]

    def build_trajectories(self, episodes) -> List[Trajectory]:
        all_trajs: List[Trajectory] = []
        for episode in episodes:
            agent_id = str(episode.episode_options.get("agent_id", "robot_a"))
            agent_trajs = self._build_agent_trajectory(episode, agent_id)
            all_trajs.extend(agent_trajs)
        return all_trajs

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        survived_count = 0
        total_agents = 0
        hold_ratios: List[float] = []
        facing_ratios: List[float] = []

        for ep in episodes:
            agent_id = str(ep.episode_options.get("agent_id", "robot_a"))
            total_agents += 1

            term_reason = ep.agent_termination_reason.get(agent_id, "")
            if not term_reason.startswith("imbalance"):
                survived_count += 1

            T = ep.num_frames
            oo = ep.observer_outputs
            self_x = extract_per_step_field(oo, "approach_velocity", "self_x", T)
            self_y = extract_per_step_field(oo, "approach_velocity", "self_y", T)
            opp_x = extract_per_step_field(oo, "approach_velocity", "opp_x", T)
            opp_y = extract_per_step_field(oo, "approach_velocity", "opp_y", T)
            fwd_x = extract_per_step_field(oo, "face_opponent", "forward_x", T)
            fwd_y = extract_per_step_field(oo, "face_opponent", "forward_y", T)

            if all(v is not None for v in (self_x, self_y, opp_x, opp_y)):
                raw_dist = np.sqrt(
                    (self_x - opp_x) ** 2 + (self_y - opp_y) ** 2
                )
                if len(raw_dist) > 0:
                    hold_ratios.append(float(np.mean(raw_dist <= 1.1)))

                    # facing_ratio: among in-range steps (< D_FACE),
                    # fraction where cos_angle > 0.5 (within ~60° of opponent)
                    if fwd_x is not None and fwd_y is not None:
                        in_range = raw_dist < D_FACE
                        if np.any(in_range):
                            sx = np.asarray(self_x, dtype=np.float64)
                            sy = np.asarray(self_y, dtype=np.float64)
                            ox = np.asarray(opp_x, dtype=np.float64)
                            oy = np.asarray(opp_y, dtype=np.float64)
                            fx = np.asarray(fwd_x, dtype=np.float64)
                            fy = np.asarray(fwd_y, dtype=np.float64)

                            to_opp = np.stack([ox - sx, oy - sy], axis=1)
                            to_opp_norm = np.linalg.norm(to_opp, axis=1)
                            valid = to_opp_norm > 1e-6
                            to_opp_hat = np.zeros_like(to_opp)
                            to_opp_hat[valid] = to_opp[valid] / to_opp_norm[valid, None]

                            fwd = np.stack([fx, fy], axis=1)
                            cos_angle = np.sum(fwd * to_opp_hat, axis=1)

                            in_range_valid = in_range & valid
                            if np.any(in_range_valid):
                                facing_ratios.append(
                                    float(np.mean(cos_angle[in_range_valid] > 0.5))
                                )

        survival_rate = float(survived_count / max(total_agents, 1))
        mean_hold_ratio = float(np.mean(hold_ratios)) if hold_ratios else 0.0
        mean_facing_ratio = float(np.mean(facing_ratios)) if facing_ratios else 0.0

        self._survival_rate = survival_rate
        self._hold_ratio = mean_hold_ratio
        self._facing_ratio = mean_facing_ratio

        # --- Curriculum promotion ---
        if self._level < len(self.LEVEL_SPEEDS) - 1:
            if mean_hold_ratio >= self.PROMOTE_HOLD_RATIO:
                self._consecutive_pass += 1
                if self._consecutive_pass >= self.PROMOTE_PATIENCE:
                    self._level += 1
                    self._consecutive_pass = 0
            else:
                self._consecutive_pass = 0

        # --- Best-of-run: 3-level priority ---
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
                "facing_ratio": round(mean_facing_ratio, 3),
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
            "facing_ratio": self._facing_ratio,
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
        self._facing_ratio = float(state.get("facing_ratio", 0.0))
        self._survival_rate = float(state.get("survival_rate", 0.0))
        self._best_survived = float(state.get("best_survived", -1.0))
        self._best_level = int(state.get("best_level", -1))
        self._best_hold_ratio = float(state.get("best_hold_ratio", -1.0))
        self._last_best_update = int(state.get("last_best_update", 0))


EXPERIMENT_CLASS = StandupFace
