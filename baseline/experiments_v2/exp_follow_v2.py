"""V2 end-to-end: phase-switched standup + balance + stepping + follow + face.

Combines ``exp_balance_v2.py`` (phase-switched standup/balance with per-foot
stepping) with ``exp_standup_face.py`` (follow + face opponent).

Single learning agent starts from random fallen state, stands up, maintains
balance with per-foot stepping, follows the scripted moving target, and
when within 1.5m must face the opponent.

Two reward phases with hard switch based on torso height:

  STANDUP phase (h_torso < plateau):
    r_potential = (1-γ) × φ_4stage = 0.01 × φ_4stage,  weight = 3.0

  BALANCE phase (h_torso >= plateau):
    r_fall       = 0.01 × φ_height,         weight = 3.0 (fixed)
    r_left_foot  = clip(h_left,  -0.05, 0.05), weight = stepping state machine
    r_right_foot = clip(h_right, -0.05, 0.05), weight = stepping state machine

  Follow/face channels (always present, gated by φ_height² × BALANCE mask):
    r_radial     = radial approach vel,    weight = 3.0 × φ_height² × BALANCE
    r_tangential = tangential penalty,     weight = 1.0 × φ_height² × BALANCE
    r_face       = facing_score,             weight = 1.0 × dist_gate × φ_height² × BALANCE

Seven reward channels (each with independent critic):
  r_potential  — reward always present, aw=3.0 in STANDUP, 0 in BALANCE
  r_fall       — reward always present, aw=3.0 in BALANCE, 0 in STANDUP
  r_left_foot  — reward always present, aw = state machine (BALANCE only)
  r_right_foot — reward always present, aw = state machine (BALANCE only)
  r_radial     — reward always present, aw = 3.0 × φ_height² × BALANCE
  r_tangential — reward always present, aw = 1.0 × φ_height² × BALANCE
  r_face       — reward always present, aw = 1.0 × dist_gate × φ_height² × BALANCE

Rewards are NOT masked — critics learn at all times.  Only actor_weight
controls when each channel influences the policy update.

Curriculum: 13 levels (0.0 → 1.5 m/s opponent speed).

Blueprint: baseline/humanoid21/end2end/follow_v2_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.framework.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.ppo_trainer import _extract_per_step_field

from baseline.humanoid21.rewards.follow_opponent import (
    compute_radial_tangential_rewards,
)

from .base import CombatExperimentV2Base
from baseline.humanoid21.end2end.stepping_state_machine import (
    compute_foot_weights,
    FOOT_WEIGHT,
    PHASE_A_STEPS,
    PHASE_B_END,
    DOUBLE_GRACE_STEPS,
)


# --- Phase thresholds ---
H_BALANCE_LOW_THRESHOLD: float = 1.0
"""h_torso must be above this for plateau detection (entire window)."""
H_BALANCE_TO_STANDUP: float = 0.70
"""h_torso below this → fall back to STANDUP phase."""
PLATEAU_WINDOW: int = 20
"""Sliding window size (action steps) for plateau detection."""
PLATEAU_SLOPE_EPS: float = 0.005
"""Max |slope| (m/step) for plateau detection."""

# --- Face reward constants ---
D_FACE: float = 1.5     # m — face reward starts activating
D_STRIKE: float = 0.7   # m — face reward fully active


class FollowV2(CombatExperimentV2Base):
    """End-to-end: phase-switched standup + balance + stepping + follow + face.

    Single-agent curriculum.  Opponent controlled by RandomMovePlugin.
    Learning agent starts from random fallen state, must stand up, maintain
    balance with stepping, follow, and face the opponent when close.
    """

    name = "follow_v2"

    # --- Network ---
    obs_dim: int = 96
    action_dim: int = 21

    # --- Reward channels ---
    _channel_names = (
        "r_potential", "r_fall",
        "r_left_foot", "r_right_foot",
        "r_radial", "r_tangential", "r_face",
    )
    _channel_gammas = {
        "r_potential": 0.99,
        "r_fall": 0.99,
        "r_left_foot": 0.9,
        "r_right_foot": 0.9,
        "r_radial": 0.99,
        "r_tangential": 0.99,
        "r_face": 0.99,
    }
    _gae_lambda = 0.95

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01

    # --- Foot height reward saturation ---
    foot_height_clip: float = 0.05

    # --- r_fall actor weight (fixed, balance phase) ---
    r_fall_actor_weight: float = 3.0

    # --- r_potential actor weight (fixed, standup phase) ---
    r_potential_actor_weight: float = 3.0

    # --- Follow/face base actor weights (gated by φ_height²) ---
    r_radial_actor_weight: float = 3.0
    r_tangential_actor_weight: float = 1.0
    r_face_actor_weight: float = 1.0

    # --- Env / rollout config ---
    env_blueprint = ""  # overridden via _env_pb()
    agent_used = "random"
    max_steps: int = 600
    INITIAL_DISTANCE: float = 2.0

    episodes_per_update: int = 1024
    eval_episodes: int = 128
    eval_interval: int = 2
    video_eval_interval: int = 2
    max_updates: int = 20000

    # --- PPO tuning (match standup_face) ---
    log_std_min: float = -1.8
    learning_rate: float = 3e-5
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096 * 4
    entropy_coef: float = 1.5e-3

    # --- Curriculum ---
    LEVEL_SPEEDS: Tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5,
    )
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
            / "humanoid21" / "end2end" / "follow_v2_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    @property
    def current_speed(self) -> float:
        idx = max(0, min(self._level, len(self.LEVEL_SPEEDS) - 1))
        return float(self.LEVEL_SPEEDS[idx])

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
    # Job construction
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
    # Phase determination
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_phase_mask(
        h_torso: np.ndarray, T: int,
    ) -> np.ndarray:
        """Compute per-step phase mask (post-hoc, on full episode).

        Returns boolean array of shape (T,):
          True  = BALANCE phase
          False = STANDUP phase
        """
        phase = np.zeros(T, dtype=bool)  # False = STANDUP

        balance_start = None
        W = PLATEAU_WINDOW
        for t in range(W, T + 1):
            window = h_torso[t - W:t]
            if np.all(window >= H_BALANCE_LOW_THRESHOLD):
                x = np.arange(W, dtype=np.float64)
                y = window.astype(np.float64)
                x_mean = x.mean()
                y_mean = y.mean()
                denom = np.sum((x - x_mean) ** 2)
                if denom > 0:
                    slope = np.sum((x - x_mean) * (y - y_mean)) / denom
                else:
                    slope = 0.0
                if abs(slope) < PLATEAU_SLOPE_EPS:
                    balance_start = t - W
                    break

        if balance_start is None:
            return phase

        in_balance = True
        for t in range(balance_start, T):
            if in_balance:
                if float(h_torso[t]) < H_BALANCE_TO_STANDUP:
                    in_balance = False
            phase[t] = in_balance

        return phase

    # ------------------------------------------------------------------
    # Stepping state machine (phase-gated wrapper)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_foot_weights_masked(
        contact_l: np.ndarray,
        contact_r: np.ndarray,
        balance_mask: np.ndarray,
        T: int,
        h_left: Optional[np.ndarray] = None,
        h_right: Optional[np.ndarray] = None,
        weight: float = FOOT_WEIGHT,
        phase_a_steps: int = PHASE_A_STEPS,
        phase_b_end: int = PHASE_B_END,
        double_grace_steps: int = DOUBLE_GRACE_STEPS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance-gated foot weights."""
        w_left = np.zeros(T, dtype=np.float32)
        w_right = np.zeros(T, dtype=np.float32)

        seg_start = 0
        for t in range(T + 1):
            in_seg = t < T and bool(balance_mask[t])
            seg_active = t > seg_start and (t == T or not in_seg)
            if seg_active:
                seg_len = t - seg_start
                cl = np.asarray(contact_l[seg_start:t], dtype=np.float32)
                cr = np.asarray(contact_r[seg_start:t], dtype=np.float32)
                hl = np.asarray(h_left[seg_start:t], dtype=np.float32) if h_left is not None else None
                hr = np.asarray(h_right[seg_start:t], dtype=np.float32) if h_right is not None else None
                wl, wr = compute_foot_weights(
                    cl, cr, seg_len,
                    h_left=hl, h_right=hr,
                    weight=weight,
                    phase_a_steps=phase_a_steps,
                    phase_b_end=phase_b_end,
                    double_grace_steps=double_grace_steps,
                )
                w_left[seg_start:t] = wl
                w_right[seg_start:t] = wr
            if t < T and not in_seg:
                seg_start = t + 1

        return w_left, w_right

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

        # --- Extract φ_4stage (StandingBalance4StageRewarder "potential") ---
        phi4_arr = _extract_per_step_field(oo, "standing_balance", "potential", T_full)
        if phi4_arr is not None:
            phi4_arr = phi4_arr[:T_full]
        else:
            phi4_arr = np.zeros(T_full, dtype=np.float32)
        phi4_arr = np.clip(phi4_arr, 0.0, 1.0).astype(np.float32)

        # --- Extract φ_height (HeightPhiObserver "phi") ---
        phi_h_arr = _extract_per_step_field(oo, "height_phi", "phi", T_full)
        if phi_h_arr is not None:
            phi_h_arr = phi_h_arr[:T_full]
        else:
            phi_h_arr = np.zeros(T_full, dtype=np.float32)
        phi_h_arr = np.clip(phi_h_arr, 0.0, 1.0).astype(np.float32)

        # --- Extract h_torso for phase determination ---
        h_torso = _extract_per_step_field(oo, "standing_balance", "h_torso", T_full)
        if h_torso is not None:
            h_torso = h_torso[:T_full]
        else:
            h_torso = np.zeros(T_full, dtype=np.float32)

        # --- Compute phase mask ---
        balance_mask = self._compute_phase_mask(h_torso, T_full)
        standup_mask = ~balance_mask

        # --- r_potential: dense reward, critic learns at all times ---
        r_potential = (self.per_step_phi_coef * phi4_arr).astype(np.float32)

        # --- r_fall: dense reward, critic learns at all times ---
        r_fall = (self.per_step_phi_coef * phi_h_arr).astype(np.float32)

        # --- Foot heights (saturated) ---
        h_left = self._extract_foot_field(oo, "foot_state", "h_left_foot", T_full)
        h_right = self._extract_foot_field(oo, "foot_state", "h_right_foot", T_full)
        r_left = np.clip(h_left, -self.foot_height_clip, self.foot_height_clip).astype(np.float32)
        r_right = np.clip(h_right, -self.foot_height_clip, self.foot_height_clip).astype(np.float32)

        # --- Contacts → stepping state machine → foot actor weights ---
        contact_l = self._extract_foot_field(oo, "foot_state", "left_foot_contact", T_full)
        contact_r = self._extract_foot_field(oo, "foot_state", "right_foot_contact", T_full)
        w_left, w_right = self._compute_foot_weights_masked(
            contact_l.astype(bool), contact_r.astype(bool), balance_mask, T_full,
            h_left=h_left, h_right=h_right,
        )

        # --- r_radial / r_tangential ---
        self_x = _extract_per_step_field(oo, "approach_velocity", "self_x", T_full)
        self_y = _extract_per_step_field(oo, "approach_velocity", "self_y", T_full)
        opp_x = _extract_per_step_field(oo, "approach_velocity", "opp_x", T_full)
        opp_y = _extract_per_step_field(oo, "approach_velocity", "opp_y", T_full)

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

        # --- r_face: facing_score (reward) + dist_gate (actor weight) ---
        fwd_x = _extract_per_step_field(oo, "face_opponent", "forward_x", T_full)
        fwd_y = _extract_per_step_field(oo, "face_opponent", "forward_y", T_full)

        r_face = np.zeros(T_full, dtype=np.float32)
        dist_gate = np.zeros(T_full, dtype=np.float32)
        if fwd_x is not None and fwd_y is not None and self_x is not None:
            fwd_x = np.asarray(fwd_x[:T_full], dtype=np.float64)
            fwd_y = np.asarray(fwd_y[:T_full], dtype=np.float64)
            fwd = np.stack([fwd_x, fwd_y], axis=1)

            dist = np.linalg.norm(opp_xy[:T_full] - self_xy[:T_full], axis=1)

            to_opp = opp_xy[:T_full] - self_xy[:T_full]
            to_opp_norm = np.linalg.norm(to_opp, axis=1)
            valid = to_opp_norm > 1e-6
            to_opp_hat = np.zeros((T_full, 2), dtype=np.float64)
            to_opp_hat[valid] = to_opp[valid] / to_opp_norm[valid, None]

            cos_angle = np.sum(fwd * to_opp_hat, axis=1)
            facing_score = np.maximum(0.0, cos_angle)

            # dist_gate goes into actor_weight, not reward
            dist_gate = np.clip(
                (D_FACE - dist) / (D_FACE - D_STRIKE), 0.0, 1.0
            ).astype(np.float32)

            r_face = facing_score.astype(np.float32)

        # --- No early termination ---
        is_terminated = False

        # --- Actor weights ---
        # r_potential: STANDUP phase only
        # All other channels: BALANCE phase only
        # Follow/face channels additionally gated by φ_height²
        phi_h_sq = (phi_h_arr ** 2).astype(np.float32)
        actor_weights = {
            "r_potential": (self.r_potential_actor_weight * standup_mask).astype(np.float32),
            "r_fall": (self.r_fall_actor_weight * balance_mask).astype(np.float32),
            "r_left_foot": w_left,
            "r_right_foot": w_right,
            "r_radial": (self.r_radial_actor_weight * phi_h_sq * balance_mask).astype(np.float32),
            "r_tangential": (self.r_tangential_actor_weight * phi_h_sq * balance_mask).astype(np.float32),
            "r_face": (self.r_face_actor_weight * dist_gate * phi_h_sq * balance_mask).astype(np.float32),
        }

        all_rewards = {
            "r_potential": r_potential,
            "r_fall": r_fall,
            "r_left_foot": r_left,
            "r_right_foot": r_right,
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
            mode=None,
            log_prob=None,
        )]

    @staticmethod
    def _extract_foot_field(
        oo, observer_key: str, field: str, T_full: int,
    ) -> np.ndarray:
        """Extract a FootStateObserver field, truncated to ``T_full``."""
        arr = _extract_per_step_field(oo, observer_key, field, T_full)
        if arr is None:
            raise KeyError(
                f"_extract_foot_field: observer '{observer_key}' field '{field}' "
                f"missing from observer_outputs "
                f"(available observers={list(oo.keys())})"
            )
        return arr[:T_full]

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
            self_x = _extract_per_step_field(oo, "approach_velocity", "self_x", T)
            self_y = _extract_per_step_field(oo, "approach_velocity", "self_y", T)
            opp_x = _extract_per_step_field(oo, "approach_velocity", "opp_x", T)
            opp_y = _extract_per_step_field(oo, "approach_velocity", "opp_y", T)
            fwd_x = _extract_per_step_field(oo, "face_opponent", "forward_x", T)
            fwd_y = _extract_per_step_field(oo, "face_opponent", "forward_y", T)

            if all(v is not None for v in (self_x, self_y, opp_x, opp_y)):
                raw_dist = np.sqrt(
                    (self_x - opp_x) ** 2 + (self_y - opp_y) ** 2
                )
                if len(raw_dist) > 0:
                    hold_ratios.append(float(np.mean(raw_dist <= 1.1)))

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


EXPERIMENT_CLASS = FollowV2
