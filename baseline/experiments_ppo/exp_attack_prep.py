"""Attack-prep: follow_v2 + arm punch shaping against a fixed-speed moving target.

Extends ``exp_follow_v2.py`` with four arm shaping channels driven by a
three-state punch machine.  Everything else — standup, balance, stepping,
follow, face — is identical to follow_v2 so the previously learned behavior
is preserved.

Differences from follow_v2:
  - Opponent speed is FIXED at 0.5 m/s (no curriculum / level promotion).
  - Adds r_left_elbow / r_right_elbow / r_left_hand_dist / r_right_hand_dist.
  - Blueprint adds ArmStateObserver.
  - Best-of-run is judged by survival first, then punch count.

Reward phases (unchanged from follow_v2):

  STANDUP phase (h_torso < plateau):
    r_potential = 0.01 × φ_4stage,  aw = 3.0

  BALANCE phase (h_torso >= plateau):
    r_fall       = 0.01 × φ_height,          aw = 3.0
    r_left_foot  = clip(h_left,  ±0.05),     aw = stepping state machine
    r_right_foot = clip(h_right, ±0.05),     aw = stepping state machine

  Follow/face channels (gated by φ_height² × BALANCE):
    r_radial     = radial approach vel,      aw = 3.0 × out_zone × φ_h² × BALANCE
    r_tangential = tangential penalty,       aw = 1.0 × out_zone × φ_h² × BALANCE
    r_face       = facing_score,             aw = 1.0 × dist_gate × φ_h² × BALANCE

  Arm punch channels (NEW, gated by φ_height²; the state machine itself only
  runs on valid segments where dist ≤ 1.1 m AND facing within ±60° AND BALANCE):
    r_left_elbow      = (1 - left_elbow_norm) / 2   ∈ [0, 1]
    r_right_elbow     = (1 - right_elbow_norm) / 2  ∈ [0, 1]
    r_left_hand_dist  = left hand → opp head (m)
    r_right_hand_dist = right hand → opp head (m)

  Elbow reward is mapped so 1 = punched out (伸直), 0 = chambered (收回):
    ATTACK  aw = +W → reward↑ → elbow extends
    FLEX    aw = -W → reward↓ → elbow retracts
  Hand distance uses the raw distance:
    ATTACK  aw = -W → hand approaches opp head
    FLEX    aw = +W → hand withdraws

Eleven reward channels, each with an independent critic.  Rewards are NOT
masked — critics learn at all times.  Only actor_weight controls when a
channel influences the policy update.

Blueprint: baseline/humanoid21/end2end/attack_prep_env.yaml
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_field

from baseline.humanoid21.rewards.follow_opponent import (
    compute_radial_tangential_rewards,
    FOLLOW_DIST_MAX,
)

from .base import CombatExperimentPPOBase
from baseline.framework.rollout.job import EiSpec, Job
from baseline.humanoid21.end2end.stepping_state_machine import (
    compute_foot_weights,
    FOOT_WEIGHT,
    PHASE_A_STEPS,
    PHASE_B_END,
    DOUBLE_GRACE_STEPS,
)
from baseline.humanoid21.end2end.arm_state_machine import (
    compute_arm_weights_with_stats,
    ELBOW_EXTEND_THRESHOLD,
    ELBOW_FLEX_THRESHOLD,
    ARM_WEIGHT,
)


# --- Phase thresholds (same as follow_v2) ---
H_BALANCE_LOW_THRESHOLD: float = 1.0
"""h_torso must be above this for plateau detection (entire window)."""
H_BALANCE_TO_STANDUP: float = 0.70
"""h_torso below this → fall back to STANDUP phase."""
PLATEAU_WINDOW: int = 20
"""Sliding window size (action steps) for plateau detection."""
PLATEAU_SLOPE_EPS: float = 0.005
"""Max |slope| (m/step) for plateau detection."""

# --- Face reward constants (same as follow_v2) ---
D_FACE: float = 1.5     # m — face reward starts activating
D_STRIKE: float = 0.7   # m — face reward fully active

# --- Arm gate: striking range + facing cone ---
D_ARM_GATE: float = 1.1
"""m — arm state machine runs only when opponent is within this distance."""
FACING_ANGLE_DEG: float = 60.0
"""deg — arm state machine runs only within this angle off the opponent."""
FACING_COS_THRESHOLD: float = math.cos(math.radians(FACING_ANGLE_DEG))  # ≈ 0.866


class AttackPrep(CombatExperimentPPOBase):
    """follow_v2 + arm punch shaping against a fixed-speed moving target.

    Single learning agent.  Opponent controlled by RandomMovePlugin at a
    fixed speed.  The agent starts from a random fallen state, stands up,
    maintains balance with stepping, follows and faces the target, and
    practices alternating punches once inside striking range.
    """

    name = "attack_prep"

    # --- Network ---
    obs_dim: int = 96
    action_dim: int = 21

    # --- Reward channels ---
    _channel_names = (
        "r_potential", "r_fall",
        "r_left_foot", "r_right_foot",
        "r_radial", "r_tangential", "r_face",
        # --- Arm punch shaping ---
        "r_left_elbow", "r_right_elbow",
        "r_left_hand_dist", "r_right_hand_dist",
    )
    _channel_gammas = {
        "r_potential": 0.99,
        "r_fall": 0.99,
        "r_left_foot": 0.9,
        "r_right_foot": 0.9,
        "r_radial": 0.99,
        "r_tangential": 0.99,
        "r_face": 0.99,
        "r_left_elbow": 0.9,
        "r_right_elbow": 0.9,
        "r_left_hand_dist": 0.9,
        "r_right_hand_dist": 0.9,
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

    # --- Arm shaping config ---
    arm_weight: float = ARM_WEIGHT
    extend_threshold: float = ELBOW_EXTEND_THRESHOLD
    flex_threshold: float = ELBOW_FLEX_THRESHOLD

    # --- Env / rollout config ---
    env_blueprint = ""  # overridden via _env_pb()
    agent_used = "random"
    max_steps: int = 600
    INITIAL_DISTANCE: float = 2.0
    OPPONENT_SPEED: float = 0.5
    """Fixed opponent speed (m/s) — no curriculum."""

    episodes_per_update: int = 1024
    eval_episodes: int = 128
    eval_interval: int = 2
    video_eval_interval: int = 2
    max_updates: int = 20000

    # --- PPO tuning (match follow_v2) ---
    log_std_min: float = -1.8
    learning_rate: float = 3e-5
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096 * 4
    entropy_coef: float = 1.5e-3

    # --- Early stop ---
    _no_improvement_limit: int = 300
    _min_updates: int = 600

    # --- Stateful (best-of-run: survival first, then punch count) ---
    _best_survived: float = -1.0
    _best_punches: float = -1.0
    _last_best_update: int = 0
    # Monitoring only
    _survival_rate: float = 0.0
    _hold_ratio: float = 0.0
    _facing_ratio: float = 0.0
    _punches_per_ep: float = 0.0
    _arm_active_ratio: float = 0.0

    _AGENT_IDS = ("robot_a", "robot_b")

    # ------------------------------------------------------------------
    # Constructor (receives --set params)
    # ------------------------------------------------------------------

    def __init__(
        self,
        arm_weight: Optional[float] = None,
        extend_threshold: Optional[float] = None,
        flex_threshold: Optional[float] = None,
        opponent_speed: Optional[float] = None,
    ):
        if arm_weight is not None:
            self.arm_weight = float(arm_weight)
        if extend_threshold is not None:
            self.extend_threshold = float(extend_threshold)
        if flex_threshold is not None:
            self.flex_threshold = float(flex_threshold)
        if opponent_speed is not None:
            self.OPPONENT_SPEED = float(opponent_speed)

    # ------------------------------------------------------------------
    # Env blueprint
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "end2end" / "attack_prep_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

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
        *,
        explore_intensity: EiSpec = 0.0,
    ) -> List[Job]:
        env_pb = self._env_pb()

        jobs: List[Job] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            oppo_agent_id = "robot_b" if agent_id == "robot_a" else "robot_a"
            env_bp = env_pb.materialize(
                max_steps=self.max_steps,
                agent_id=agent_id,
                oppo_agent_id=oppo_agent_id,
                random_move_speed=self.OPPONENT_SPEED,
            )
            jobs.append(Job(
    policy_a_bp=policy_bp,
    policy_b_bp=policy_bp,
    env_bp=env_bp,
    seed=seed,
    episode_options={"agent_id": agent_id, "initial_distance": self.INITIAL_DISTANCE},
    explore_intensity_a=explore_intensity,
    explore_intensity_b=explore_intensity,
))
        return jobs

    # ------------------------------------------------------------------
    # Phase determination (same as follow_v2)
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
    # Stepping state machine (phase-gated wrapper, same as follow_v2)
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
    # Geometry helpers (shared by trajectory building and eval)
    # ------------------------------------------------------------------

    @staticmethod
    def _self_opp_xy(oo, T: int):
        """Extract (self_xy, opp_xy) or (None, None) if unavailable."""
        self_x = extract_per_step_field(oo, "approach_velocity", "self_x", T)
        self_y = extract_per_step_field(oo, "approach_velocity", "self_y", T)
        opp_x = extract_per_step_field(oo, "approach_velocity", "opp_x", T)
        opp_y = extract_per_step_field(oo, "approach_velocity", "opp_y", T)
        if self_x is None or self_y is None or opp_x is None or opp_y is None:
            return None, None
        self_xy = np.stack([self_x[:T], self_y[:T]], axis=1).astype(np.float64)
        opp_xy = np.stack([opp_x[:T], opp_y[:T]], axis=1).astype(np.float64)
        return self_xy, opp_xy

    @staticmethod
    def _facing_cos(oo, self_xy: np.ndarray, opp_xy: np.ndarray, T: int) -> np.ndarray:
        """cos(angle) between the agent's forward axis and the direction to the opponent."""
        fwd_x = extract_per_step_field(oo, "face_opponent", "forward_x", T)
        fwd_y = extract_per_step_field(oo, "face_opponent", "forward_y", T)
        if fwd_x is None or fwd_y is None:
            return np.zeros(T, dtype=np.float64)

        fwd = np.stack([
            np.asarray(fwd_x[:T], dtype=np.float64),
            np.asarray(fwd_y[:T], dtype=np.float64),
        ], axis=1)

        to_opp = opp_xy[:T] - self_xy[:T]
        to_opp_norm = np.linalg.norm(to_opp, axis=1)
        valid = to_opp_norm > 1e-6
        to_opp_hat = np.zeros((T, 2), dtype=np.float64)
        to_opp_hat[valid] = to_opp[valid] / to_opp_norm[valid, None]
        return np.sum(fwd * to_opp_hat, axis=1)

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
        phi4_arr = extract_per_step_field(oo, "standing_balance", "potential", T_full)
        if phi4_arr is not None:
            phi4_arr = phi4_arr[:T_full]
        else:
            phi4_arr = np.zeros(T_full, dtype=np.float32)
        phi4_arr = np.clip(phi4_arr, 0.0, 1.0).astype(np.float32)

        # --- Extract φ_height (HeightPhiObserver "phi") ---
        phi_h_arr = extract_per_step_field(oo, "height_phi", "phi", T_full)
        if phi_h_arr is not None:
            phi_h_arr = phi_h_arr[:T_full]
        else:
            phi_h_arr = np.zeros(T_full, dtype=np.float32)
        phi_h_arr = np.clip(phi_h_arr, 0.0, 1.0).astype(np.float32)

        # --- Extract h_torso for phase determination ---
        h_torso = extract_per_step_field(oo, "standing_balance", "h_torso", T_full)
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
        h_left = self._extract_field(oo, "foot_state", "h_left_foot", T_full)
        h_right = self._extract_field(oo, "foot_state", "h_right_foot", T_full)
        r_left = np.clip(h_left, -self.foot_height_clip, self.foot_height_clip).astype(np.float32)
        r_right = np.clip(h_right, -self.foot_height_clip, self.foot_height_clip).astype(np.float32)

        # --- Contacts → stepping state machine → foot actor weights ---
        contact_l = self._extract_field(oo, "foot_state", "left_foot_contact", T_full)
        contact_r = self._extract_field(oo, "foot_state", "right_foot_contact", T_full)
        w_left, w_right = self._compute_foot_weights_masked(
            contact_l.astype(bool), contact_r.astype(bool), balance_mask, T_full,
            h_left=h_left, h_right=h_right,
        )

        # --- r_radial / r_tangential ---
        self_xy, opp_xy = self._self_opp_xy(oo, T_full)
        if self_xy is None:
            r_radial = np.zeros(T_full, dtype=np.float32)
            r_tangential = np.zeros(T_full, dtype=np.float32)
            self_xy = np.zeros((T_full, 2), dtype=np.float64)
            opp_xy = np.zeros((T_full, 2), dtype=np.float64)
            dist = np.full(T_full, 1e9, dtype=np.float64)
            cos_angle = np.zeros(T_full, dtype=np.float64)
            has_geom = False
        else:
            r_radial, r_tangential = compute_radial_tangential_rewards(
                self_xy, opp_xy, gate=False,
            )
            dist = np.linalg.norm(opp_xy[:T_full] - self_xy[:T_full], axis=1)
            cos_angle = self._facing_cos(oo, self_xy, opp_xy, T_full)
            has_geom = True

        # --- out_zone gate for r_radial / r_tangential actor weights ---
        # Active when distance > FOLLOW_DIST_MAX (0.9 m), i.e. outside striking range.
        out_zone = (dist > FOLLOW_DIST_MAX).astype(np.float32)

        # --- r_face: facing_score (reward) + dist_gate (actor weight) ---
        if has_geom:
            r_face = np.maximum(0.0, cos_angle).astype(np.float32)
            face_dist_gate = np.clip(
                (D_FACE - dist) / (D_FACE - D_STRIKE), 0.0, 1.0
            ).astype(np.float32)
        else:
            r_face = np.zeros(T_full, dtype=np.float32)
            face_dist_gate = np.zeros(T_full, dtype=np.float32)

        # --- Arm state: elbow angles + hand → opp head distances ---
        left_elbow = self._extract_field(oo, "arm_state", "left_elbow_norm", T_full)
        right_elbow = self._extract_field(oo, "arm_state", "right_elbow_norm", T_full)
        left_hand_opp = self._extract_field(oo, "arm_state", "left_hand_to_opp_head", T_full)
        right_hand_opp = self._extract_field(oo, "arm_state", "right_hand_to_opp_head", T_full)

        # --- Arm rewards ---
        # Elbow: map norm [-1,+1] → [0,1] where 1 = punched out (伸直), 0 = chambered (收回).
        r_left_elbow = ((1.0 - left_elbow) * 0.5).astype(np.float32)
        r_right_elbow = ((1.0 - right_elbow) * 0.5).astype(np.float32)
        # Hand distance: raw 3D distance to the opponent head (m).
        r_left_hand_dist = left_hand_opp.astype(np.float32)
        r_right_hand_dist = right_hand_opp.astype(np.float32)

        # --- Arm valid segments: striking range + facing cone + BALANCE ---
        arm_valid_mask = (
            (dist <= D_ARM_GATE)
            & (cos_angle >= FACING_COS_THRESHOLD)
            & balance_mask
        )

        # --- Arm state machine → base actor weights (±W inside segments, 0 outside) ---
        w_le, w_re, w_lhd, w_rhd, _arm_stats = compute_arm_weights_with_stats(
            left_elbow, right_elbow,
            left_hand_opp, right_hand_opp,
            valid_mask=arm_valid_mask,
            extend_threshold=self.extend_threshold,
            flex_threshold=self.flex_threshold,
            arm_weight=self.arm_weight,
        )

        # --- No early termination ---
        is_terminated = False

        # --- Actor weights ---
        # r_potential: STANDUP phase only
        # All other channels: BALANCE phase only
        # Follow/face channels additionally gated by φ_height²
        # Arm channels: BALANCE + range + facing already folded into the
        # state machine's valid mask; only φ_height² is applied here.
        phi_h_sq = (phi_h_arr ** 2).astype(np.float32)
        actor_weights = {
            "r_potential": (self.r_potential_actor_weight * standup_mask).astype(np.float32),
            "r_fall": (self.r_fall_actor_weight * balance_mask).astype(np.float32),
            "r_left_foot": w_left,
            "r_right_foot": w_right,
            "r_radial": (self.r_radial_actor_weight * out_zone * phi_h_sq * balance_mask).astype(np.float32),
            "r_tangential": (self.r_tangential_actor_weight * out_zone * phi_h_sq * balance_mask).astype(np.float32),
            "r_face": (self.r_face_actor_weight * face_dist_gate * phi_h_sq * balance_mask).astype(np.float32),
            "r_left_elbow": (w_le * phi_h_sq).astype(np.float32),
            "r_right_elbow": (w_re * phi_h_sq).astype(np.float32),
            "r_left_hand_dist": (w_lhd * phi_h_sq).astype(np.float32),
            "r_right_hand_dist": (w_rhd * phi_h_sq).astype(np.float32),
        }

        all_rewards = {
            "r_potential": r_potential,
            "r_fall": r_fall,
            "r_left_foot": r_left,
            "r_right_foot": r_right,
            "r_radial": r_radial.astype(np.float32),
            "r_tangential": r_tangential.astype(np.float32),
            "r_face": r_face,
            "r_left_elbow": r_left_elbow,
            "r_right_elbow": r_right_elbow,
            "r_left_hand_dist": r_left_hand_dist,
            "r_right_hand_dist": r_right_hand_dist,
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

    @staticmethod
    def _extract_field(
        oo, observer_key: str, field: str, T_full: int,
    ) -> np.ndarray:
        """Extract an observer field, truncated to ``T_full``."""
        arr = extract_per_step_field(oo, observer_key, field, T_full)
        if arr is None:
            raise KeyError(
                f"_extract_field: observer '{observer_key}' field '{field}' "
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
    # Eval — survival first, then punch count
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        survived_count = 0
        total_agents = 0
        hold_ratios: List[float] = []
        facing_ratios: List[float] = []
        punch_counts: List[float] = []
        arm_active_ratios: List[float] = []

        for ep in episodes:
            agent_id = str(ep.episode_options.get("agent_id", "robot_a"))
            total_agents += 1

            term_reason = ep.agent_termination_reason.get(agent_id, "")
            if not term_reason.startswith("imbalance"):
                survived_count += 1

            T = ep.num_frames
            if T == 0:
                continue
            oo = ep.observer_outputs

            self_xy, opp_xy = self._self_opp_xy(oo, T)
            if self_xy is None:
                continue

            dist = np.linalg.norm(opp_xy - self_xy, axis=1)
            hold_ratios.append(float(np.mean(dist <= 1.1)))

            cos_angle = self._facing_cos(oo, self_xy, opp_xy, T)
            in_range = dist < D_FACE
            if np.any(in_range):
                facing_ratios.append(float(np.mean(cos_angle[in_range] > 0.5)))

            # --- Punch count: run the same state machine used in training ---
            h_torso = extract_per_step_field(oo, "standing_balance", "h_torso", T)
            if h_torso is None:
                continue
            balance_mask = self._compute_phase_mask(h_torso[:T], T)

            left_elbow = extract_per_step_field(oo, "arm_state", "left_elbow_norm", T)
            right_elbow = extract_per_step_field(oo, "arm_state", "right_elbow_norm", T)
            left_hand_opp = extract_per_step_field(oo, "arm_state", "left_hand_to_opp_head", T)
            right_hand_opp = extract_per_step_field(oo, "arm_state", "right_hand_to_opp_head", T)
            if any(v is None for v in (left_elbow, right_elbow, left_hand_opp, right_hand_opp)):
                continue

            arm_valid_mask = (
                (dist <= D_ARM_GATE)
                & (cos_angle >= FACING_COS_THRESHOLD)
                & balance_mask
            )
            _, _, _, _, stats = compute_arm_weights_with_stats(
                left_elbow[:T], right_elbow[:T],
                left_hand_opp[:T], right_hand_opp[:T],
                valid_mask=arm_valid_mask,
                extend_threshold=self.extend_threshold,
                flex_threshold=self.flex_threshold,
                arm_weight=self.arm_weight,
            )
            punch_counts.append(float(stats["n_punches"]))
            arm_active_ratios.append(float(stats["n_valid_steps"]) / float(T))

        survival_rate = float(survived_count / max(total_agents, 1))
        mean_hold_ratio = float(np.mean(hold_ratios)) if hold_ratios else 0.0
        mean_facing_ratio = float(np.mean(facing_ratios)) if facing_ratios else 0.0
        mean_punches = float(np.mean(punch_counts)) if punch_counts else 0.0
        mean_arm_active = float(np.mean(arm_active_ratios)) if arm_active_ratios else 0.0

        self._survival_rate = survival_rate
        self._hold_ratio = mean_hold_ratio
        self._facing_ratio = mean_facing_ratio
        self._punches_per_ep = mean_punches
        self._arm_active_ratio = mean_arm_active

        # --- Best-of-run: survival first, then punch count ---
        survived_metric = float(survived_count)
        is_new_best = (
            survived_metric > self._best_survived
            or (survived_metric == self._best_survived
                and mean_punches > self._best_punches)
        )
        if is_new_best:
            self._best_survived = survived_metric
            self._best_punches = mean_punches
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
                "punches": round(mean_punches, 2),
                "arm_active": round(mean_arm_active, 3),
                "hold_ratio": round(mean_hold_ratio, 3),
                "facing_ratio": round(mean_facing_ratio, 3),
                "opp_speed": round(self.OPPONENT_SPEED, 3),
            },
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "best_survived": self._best_survived,
            "best_punches": self._best_punches,
            "last_best_update": self._last_best_update,
            "survival_rate": self._survival_rate,
            "hold_ratio": self._hold_ratio,
            "facing_ratio": self._facing_ratio,
            "punches_per_ep": self._punches_per_ep,
            "arm_active_ratio": self._arm_active_ratio,
        }

    def load_state(self, state: dict) -> None:
        self._best_survived = float(state.get("best_survived", -1.0))
        self._best_punches = float(state.get("best_punches", -1.0))
        self._last_best_update = int(state.get("last_best_update", 0))
        self._survival_rate = float(state.get("survival_rate", 0.0))
        self._hold_ratio = float(state.get("hold_ratio", 0.0))
        self._facing_ratio = float(state.get("facing_ratio", 0.0))
        self._punches_per_ep = float(state.get("punches_per_ep", 0.0))
        self._arm_active_ratio = float(state.get("arm_active_ratio", 0.0))


EXPERIMENT_CLASS = AttackPrep
