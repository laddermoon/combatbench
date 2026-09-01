"""V2 end-to-end step 5: standup + follow + face + fight (self-play).

On top of ``follow_v2``, adds two damage reward channels and replaces
the scripted moving target (RandomMovePlugin) with a real opponent policy
sampled from an opponent pool.

Phase-switched rewards (same as follow_v2 / balance_v2):

  STANDUP phase (h_torso < plateau):
    r_potential = 0.01 × φ_4stage,  weight = 3.0

  BALANCE phase (h_torso >= plateau):
    r_fall       = 0.01 × φ_height,          weight = 3.0
    r_left_foot  = clip(h_left, -0.05, 0.05),  weight = stepping state machine
    r_right_foot = clip(h_right, -0.05, 0.05), weight = stepping state machine
    r_radial     = radial approach vel,        weight = 3.0 × φ_height²
    r_tangential = tangential penalty,         weight = 1.0 × φ_height²
    r_face       = facing_score,             weight = 1.0 × dist_gate × φ_height²
    r_damage_dealt = damage dealt,             weight = dealt_weight × dist_gate
    r_damage_taken = damage taken,             weight = taken_weight × dist_gate

All channels except r_potential are only active in BALANCE phase.
Damage channels additionally gated by a hard distance gate (dist ≤ 0.9 m).

Opponent pool
-------------
A JSON config file (``--set pool_config=path/to/pool.json``) lists policy
blueprint paths and a decay coefficient::

    {
        "policies": [
            "/abs/path/to/policy_blueprint.yaml",
            "/abs/path/to/u00100/policy_blueprint.yaml"
        ],
        "decay": 0.95
    }

The first entry is the initial policy (e.g. from ``follow_v2``).
New entries are appended by the iterative training script after each
training round.  During rollout, each episode samples an opponent from
the pool with weight ``decay^age`` (age=0 for the newest, larger for
older), so recent policies are sampled more often.

Nine reward channels (each with independent critic):
  r_potential     — aw=3.0 in STANDUP, 0 in BALANCE
  r_fall          — aw=3.0 in BALANCE, 0 in STANDUP
  r_left_foot     — aw = state machine (BALANCE only)
  r_right_foot    — aw = state machine (BALANCE only)
  r_radial        — aw = 3.0 × φ_height² × BALANCE
  r_tangential    — aw = 1.0 × φ_height² × BALANCE
  r_face          — aw = 1.0 × dist_gate × φ_height² × BALANCE
  r_damage_dealt  — aw = dealt_weight × dist_gate × BALANCE
  r_damage_taken  — aw = taken_weight × dist_gate × BALANCE

Rewards are NOT masked — critics learn at all times.  Only actor_weight
controls when each channel influences the policy update.

Blueprint: baseline/humanoid21/end2end/standup_fight_env.yaml
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.common.rollout import extract_per_step_field

from baseline.humanoid21.rewards.follow_opponent import (
    compute_radial_tangential_rewards,
    FOLLOW_DIST_MAX,
)
from baseline.humanoid21.end2end.stepping_state_machine import (
    compute_foot_weights,
    FOOT_WEIGHT,
    PHASE_A_STEPS,
    PHASE_B_END,
    DOUBLE_GRACE_STEPS,
)

from .base import CombatExperimentPPOBase


# --- Phase thresholds (same as balance_v2 / follow_v2) ---
H_BALANCE_LOW_THRESHOLD: float = 1.0
H_BALANCE_TO_STANDUP: float = 0.70
PLATEAU_WINDOW: int = 20
PLATEAU_SLOPE_EPS: float = 0.005

# --- Face reward constants (same as standup_face / follow_v2) ---
D_FACE: float = 1.5     # m — face reward starts activating
D_STRIKE: float = 0.7   # m — face reward fully active

# --- Damage gate: hard switch, active within this distance ---
D_DAMAGE_GATE: float = 0.9   # m — damage aw fully on when dist <= this, off otherwise


class StandupFight(CombatExperimentPPOBase):
    """End-to-end standup + follow + face + fight with opponent pool.

    Single-agent curriculum with self-play.  The learning agent starts
    from random fallen state, stands up, approaches the opponent (a real
    policy from the pool), faces it, and fights.
    """

    name = "fight"

    # --- Network ---
    obs_dim: int = 96
    action_dim: int = 21

    # --- Reward channels ---
    _channel_names = (
        "r_potential", "r_fall",
        "r_left_foot", "r_right_foot",
        "r_radial", "r_tangential", "r_face",
        "r_damage_dealt", "r_damage_taken",
    )
    _channel_gammas = {
        "r_potential": 0.99,
        "r_fall": 0.99,
        "r_left_foot": 0.90,
        "r_right_foot": 0.90,
        "r_radial": 0.99,
        "r_tangential": 0.99,
        "r_face": 0.99,
        "r_damage_dealt": 0.90,
        "r_damage_taken": 0.99,
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

    # --- Follow/face base actor weights (gated by φ_height² × BALANCE) ---
    r_radial_actor_weight: float = 3.0
    r_tangential_actor_weight: float = 1.0
    r_face_actor_weight: float = 1.0

    # --- Damage actor weights (configurable via --set) ---
    damage_dealt_weight: float = 3.0
    damage_taken_weight: float = 1.0

    # --- Env / rollout config ---
    env_blueprint = ""  # overridden via _env_pb()
    agent_used = "random"
    max_steps = 600
    INITIAL_DISTANCE: float = 2.0

    episodes_per_update: int = 1024
    eval_episodes: int = 128
    eval_interval: int = 2
    video_eval_interval: int = 2
    max_updates: int = 20000

    # --- PPO tuning (match standup_face / follow_v2) ---
    log_std_min: float = -1.8
    learning_rate: float = 3e-5
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096 * 4
    entropy_coef: float = 1.5e-3

    # --- Eval / early stop ---
    eval_target: Optional[float] = None
    """Mean net damage threshold for early stop.  None = no target."""

    eval_patience: int = 3
    """Number of consecutive evals above target to trigger early stop."""

    # --- Early stop (no-improvement fallback) ---
    _no_improvement_limit: int = 300
    _min_updates: int = 600

    # --- Stateful ---
    _best_net_damage: float = -1e9
    _consecutive_pass: int = 0
    _last_best_update: int = 0

    _AGENT_IDS = ("robot_a", "robot_b")

    # ------------------------------------------------------------------
    # Constructor (receives --set params)
    # ------------------------------------------------------------------

    def __init__(
        self,
        pool_config: Optional[str] = None,
        damage_dealt_weight: Optional[float] = None,
        damage_taken_weight: Optional[float] = None,
        eval_target: Optional[float] = None,
        eval_patience: Optional[int] = None,
    ):
        if damage_dealt_weight is not None:
            self.damage_dealt_weight = float(damage_dealt_weight)
        if damage_taken_weight is not None:
            self.damage_taken_weight = float(damage_taken_weight)
        if eval_target is not None:
            self.eval_target = float(eval_target)
        if eval_patience is not None:
            self.eval_patience = int(eval_patience)

        # --- Load opponent pool ---
        self._pool: List[Any] = []  # PolicyBlueprint instances
        self._pool_decay: float = 0.95
        self._pool_config_path: Optional[str] = None

        if pool_config is not None:
            self._pool_config_path = pool_config
            self._load_pool(pool_config)

        if not self._pool:
            raise ValueError(
                "standup_fight requires pool_config. "
                "Example: --set pool_config=/path/to/pool.json"
            )

    def _load_pool(self, config_path: str) -> None:
        """Load opponent pool from JSON config file."""
        from envs.framework.policy import PolicyBlueprint

        with open(config_path) as f:
            cfg = json.load(f)

        self._pool_decay = float(cfg.get("decay", 0.95))
        policies = cfg.get("policies", [])
        for p in policies:
            bp = PolicyBlueprint.load(p)
            self._pool.append(bp)

    def _pool_weights(self) -> np.ndarray:
        """Compute sampling weights: decay^age, newest = age 0."""
        n = len(self._pool)
        if n == 0:
            raise RuntimeError("Opponent pool is empty")
        ages = np.arange(n - 1, -1, -1, dtype=np.float64)  # newest=0, oldest=n-1
        weights = self._pool_decay ** ages
        weights /= weights.sum()
        return weights

    # ------------------------------------------------------------------
    # Env blueprint
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "end2end" / "standup_fight_env.yaml"
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
    # Job construction — sample opponent from pool
    # ------------------------------------------------------------------

    def build_jobs(
        self,
        policy_bp,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[Any, Any, Any, int, Dict[str, Any]]]:
        env_pb = self._env_pb()
        rng = np.random.default_rng(base_seed)
        pool_w = self._pool_weights()
        opponent_indices = rng.choice(len(self._pool), size=n_episodes, p=pool_w)

        jobs: List[Tuple[Any, Any, Any, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            agent_id = self._agent_from_rollout_seed(seed)
            env_bp = env_pb.materialize(
                max_steps=self.max_steps,
                agent_id=agent_id,
            )

            oppo_bp = self._pool[int(opponent_indices[i])]

            # Assign policies: learning agent gets training policy,
            # opponent gets sampled pool policy.
            if agent_id == "robot_a":
                pa, pb = policy_bp, oppo_bp
            else:
                pa, pb = oppo_bp, policy_bp

            initial_distance = float(rng.uniform(
                self.init_distance_min, self.init_distance_max,
            ))
            jobs.append((
                pa, pb, env_bp, seed,
                {"agent_id": agent_id, "initial_distance": initial_distance},
            ))
        return jobs

    # ------------------------------------------------------------------
    # Phase determination (same as balance_v2 / follow_v2)
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
        phase = np.zeros(T, dtype=bool)

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
    # Stepping state machine (phase-gated wrapper, same as balance_v2)
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
        self_x = extract_per_step_field(oo, "approach_velocity", "self_x", T_full)
        self_y = extract_per_step_field(oo, "approach_velocity", "self_y", T_full)
        opp_x = extract_per_step_field(oo, "approach_velocity", "opp_x", T_full)
        opp_y = extract_per_step_field(oo, "approach_velocity", "opp_y", T_full)

        if self_x is None or self_y is None or opp_x is None or opp_y is None:
            r_radial = np.zeros(T_full, dtype=np.float32)
            r_tangential = np.zeros(T_full, dtype=np.float32)
            self_xy = np.zeros((T_full, 2), dtype=np.float64)
            opp_xy = np.zeros((T_full, 2), dtype=np.float64)
            dist = np.full(T_full, 1e9, dtype=np.float64)
        else:
            self_xy = np.stack([self_x[:T_full], self_y[:T_full]], axis=1)
            opp_xy = np.stack([opp_x[:T_full], opp_y[:T_full]], axis=1)
            r_radial, r_tangential = compute_radial_tangential_rewards(
                self_xy, opp_xy, gate=False,
            )
            dist = np.linalg.norm(opp_xy[:T_full] - self_xy[:T_full], axis=1)

        # --- out_zone gate for r_radial / r_tangential actor weights ---
        # Active when distance > FOLLOW_DIST_MAX (0.9m), i.e. outside striking range.
        out_zone = (dist > FOLLOW_DIST_MAX).astype(np.float32)

        # --- r_face: facing_score (reward) + dist_gate (actor weight) ---
        fwd_x = extract_per_step_field(oo, "face_opponent", "forward_x", T_full)
        fwd_y = extract_per_step_field(oo, "face_opponent", "forward_y", T_full)

        r_face = np.zeros(T_full, dtype=np.float32)
        face_dist_gate = np.zeros(T_full, dtype=np.float32)
        if fwd_x is not None and fwd_y is not None and self_x is not None:
            fwd_x = np.asarray(fwd_x[:T_full], dtype=np.float64)
            fwd_y = np.asarray(fwd_y[:T_full], dtype=np.float64)
            fwd = np.stack([fwd_x, fwd_y], axis=1)

            to_opp = opp_xy[:T_full] - self_xy[:T_full]
            to_opp_norm = np.linalg.norm(to_opp, axis=1)
            valid = to_opp_norm > 1e-6
            to_opp_hat = np.zeros((T_full, 2), dtype=np.float64)
            to_opp_hat[valid] = to_opp[valid] / to_opp_norm[valid, None]

            cos_angle = np.sum(fwd * to_opp_hat, axis=1)
            facing_score = np.maximum(0.0, cos_angle)

            # dist_gate goes into actor_weight, not reward
            face_dist_gate = np.clip(
                (D_FACE - dist) / (D_FACE - D_STRIKE), 0.0, 1.0
            ).astype(np.float32)

            r_face = facing_score.astype(np.float32)

        # --- r_damage_dealt / r_damage_taken ---
        dealt = extract_per_step_field(oo, "damage_breakdown", "dealt", T_full)
        taken = extract_per_step_field(oo, "damage_breakdown", "taken", T_full)
        if dealt is not None:
            r_dealt = np.asarray(dealt[:T_full], dtype=np.float32)
        else:
            r_dealt = np.zeros(T_full, dtype=np.float32)
        if taken is not None:
            r_taken = np.asarray(taken[:T_full], dtype=np.float32)
        else:
            r_taken = np.zeros(T_full, dtype=np.float32)

        # --- No early termination ---
        is_terminated = False

        # --- Actor weights ---
        # r_potential: STANDUP phase only
        # All other channels: BALANCE phase only
        # Follow/face channels additionally gated by φ_height²
        # Damage channels additionally gated by hard distance gate
        phi_h_sq = (phi_h_arr ** 2).astype(np.float32)
        damage_gate = (dist <= D_DAMAGE_GATE).astype(np.float32)
        actor_weights = {
            "r_potential": (self.r_potential_actor_weight * standup_mask).astype(np.float32),
            "r_fall": (self.r_fall_actor_weight * balance_mask).astype(np.float32),
            "r_left_foot": w_left,
            "r_right_foot": w_right,
            "r_radial": (self.r_radial_actor_weight * out_zone * phi_h_sq * balance_mask).astype(np.float32),
            "r_tangential": (self.r_tangential_actor_weight * out_zone * phi_h_sq * balance_mask).astype(np.float32),
            "r_face": (self.r_face_actor_weight * face_dist_gate * phi_h_sq * balance_mask).astype(np.float32),
            "r_damage_dealt": (self.damage_dealt_weight * damage_gate * balance_mask).astype(np.float32),
            "r_damage_taken": (self.damage_taken_weight * damage_gate * balance_mask).astype(np.float32),
        }

        all_rewards = {
            "r_potential": r_potential,
            "r_fall": r_fall,
            "r_left_foot": r_left,
            "r_right_foot": r_right,
            "r_radial": r_radial.astype(np.float32),
            "r_tangential": r_tangential.astype(np.float32),
            "r_face": r_face,
            "r_damage_dealt": r_dealt,
            "r_damage_taken": r_taken,
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

    @staticmethod
    def _extract_foot_field(
        oo, observer_key: str, field: str, T_full: int,
    ) -> np.ndarray:
        """Extract a FootStateObserver field, truncated to ``T_full``."""
        arr = extract_per_step_field(oo, observer_key, field, T_full)
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
    # Eval — mean net damage (dealt − taken)
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        net_damages: List[float] = []
        survived_count = 0
        total_agents = 0

        for ep in episodes:
            agent_id = str(ep.episode_options.get("agent_id", "robot_a"))
            total_agents += 1
            T = ep.num_frames
            if T == 0:
                continue

            oo = ep.observer_outputs

            # --- Survival ---
            term_reason = ep.agent_termination_reason.get(agent_id, "")
            if not term_reason.startswith("imbalance"):
                survived_count += 1

            # --- Net damage ---
            dealt = extract_per_step_field(oo, "damage_breakdown", "dealt", T)
            taken = extract_per_step_field(oo, "damage_breakdown", "taken", T)
            if dealt is not None and taken is not None:
                total_dealt = float(np.sum(dealt[:T]))
                total_taken = float(np.sum(taken[:T]))
                net_damages.append(total_dealt - total_taken)
            else:
                net_damages.append(0.0)

        survival_rate = float(survived_count / max(total_agents, 1))
        mean_net_damage = float(np.mean(net_damages)) if net_damages else 0.0

        # --- Best-of-run ---
        is_new_best = mean_net_damage > self._best_net_damage
        if is_new_best:
            self._best_net_damage = mean_net_damage
            self._last_best_update = update

        # --- Early stop: eval target met for eval_patience consecutive evals ---
        stop_training = False
        if self.eval_target is not None:
            if mean_net_damage >= self.eval_target:
                self._consecutive_pass += 1
                if self._consecutive_pass >= self.eval_patience:
                    stop_training = True
            else:
                self._consecutive_pass = 0

        # --- Early stop: no improvement fallback ---
        if not stop_training:
            no_improvement = update - self._last_best_update
            if (no_improvement >= self._no_improvement_limit
                    and update >= self._min_updates):
                stop_training = True

        return {
            "is_new_best": is_new_best,
            "stop_training": stop_training,
            "info": {
                "net_damage": round(mean_net_damage, 3),
                "best_net_damage": round(self._best_net_damage, 3),
                "survival_rate": round(survival_rate, 3),
                "consecutive_pass": self._consecutive_pass,
            },
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "best_net_damage": self._best_net_damage,
            "consecutive_pass": self._consecutive_pass,
            "last_best_update": self._last_best_update,
        }

    def load_state(self, state: dict) -> None:
        self._best_net_damage = float(state.get("best_net_damage", -1e9))
        self._consecutive_pass = int(state.get("consecutive_pass", 0))
        self._last_best_update = int(state.get("last_best_update", 0))


EXPERIMENT_CLASS = StandupFight
