"""V2 end-to-end step 5: standup + follow + face + fight (self-play).

On top of ``standup_face``, adds two damage reward channels and replaces
the scripted moving target (RandomMovePlugin) with a real opponent policy
sampled from an opponent pool.

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

The first entry is the initial policy (e.g. from ``standup_face``).
New entries are appended by the iterative training script after each
training round.  During rollout, each episode samples an opponent from
the pool with weight ``decay^age`` (age=0 for the newest, larger for
older), so recent policies are sampled more often.

Reward channels (8, each with independent critic):
  - r_fall:          0.01 × φ(t),                aw = 3.0 (fixed)
  - r_left_foot:     clip(h_left, -0.1, 0.1),     aw = state machine × φ²
  - r_right_foot:    clip(h_right, -0.1, 0.1),    aw = state machine × φ²
  - r_radial:        radial approach vel,         aw = 3.0 × φ²
  - r_tangential:    tangential penalty,          aw = 1.0 × φ²
  - r_face:          facing_score × dist_gate,    aw = 1.0 × φ²
  - r_damage_dealt:  damage dealt to opponent,    aw = dealt_weight × dist_gate
  - r_damage_taken:  damage taken from opponent,  aw = taken_weight × dist_gate

dist_gate is a hard switch: 1.0 when distance to opponent ≤ 0.9 m, 0
otherwise.  Damage channels only influence the policy when the robots
are close enough to actually hit each other.

The foot channels use the v2 stepping state machine
(``stepping_state_machine.compute_foot_weights``) with Phase A/B/C,
DOUBLE grace, and FLIGHT continuation.  See
``baseline/humanoid21/end2end/stepping_state_machine.py`` for details.

φ is the 4-stage standing potential.  Damage values come from
``DamageBreakdownRewarder`` which reads ``CombatScoringPlugin`` metrics.

Eval metric: mean net damage (dealt − taken) across eval episodes.
Early stop when ``eval_target`` is met for ``eval_patience`` consecutive
evals.

Blueprint: baseline/humanoid21/end2end/standup_fight_env.yaml
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.framework.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.ppo_trainer import (
    _extract_per_step_field,
)
from baseline.humanoid21.rewards.follow_opponent import (
    compute_radial_tangential_rewards,
)
from baseline.humanoid21.end2end.stepping_state_machine import (
    compute_foot_weights,
    FOOT_HEIGHT_CLIP,
)

from .base import CombatExperimentV2Base


# --- Face reward constants (same as standup_face) ---
D_FACE: float = 1.5     # m — face reward starts activating
D_STRIKE: float = 0.7   # m — face reward fully active

# --- Damage gate: hard switch, active within this distance ---
D_DAMAGE_GATE: float = 0.9   # m — damage aw fully on when dist <= this, off otherwise


class StandupFight(CombatExperimentV2Base):
    """End-to-end standup + follow + face + fight with opponent pool.

    Single-agent curriculum with self-play.  The learning agent starts
    from random fallen state, stands up, approaches the opponent (a real
    policy from the pool), faces it, and fights.
    """

    name = "standup_fight"

    # --- Reward channels ---
    _channel_names = (
        "r_fall", "r_left_foot", "r_right_foot",
        "r_radial", "r_tangential",
        "r_face", "r_damage_dealt", "r_damage_taken",
    )
    _channel_gammas = {
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

    # --- Env / rollout config ---
    agent_used = "random"
    max_steps = 600
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

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01

    # --- Base actor weights (r_fall fixed, others gated by φ²) ---
    # r_left_foot / r_right_foot use the state machine (per-frame), not a
    # fixed scalar, so they are not in this tuple.
    # Order: r_fall, r_radial, r_tangential, r_face
    _base_actor_weights: Tuple[float, ...] = (3.0, 3.0, 1.0, 1.0)

    # --- Damage actor weights (configurable via --set) ---
    damage_dealt_weight: float = 3.0
    damage_taken_weight: float = 1.0

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
        phi_arr = _extract_per_step_field(oo, "standing_balance", "potential", T_full)
        if phi_arr is not None:
            phi_arr = phi_arr[:T_full]
        else:
            phi_arr = np.zeros(T_full, dtype=np.float32)
        phi_arr = np.clip(phi_arr, 0.0, 1.0).astype(np.float32)

        # --- r_fall ---
        r_fall = (self.per_step_phi_coef * phi_arr).astype(np.float32)

        # --- r_left_foot / r_right_foot (foot heights + state machine) ---
        h_left = _extract_per_step_field(oo, "foot_state", "h_left_foot", T_full)
        h_right = _extract_per_step_field(oo, "foot_state", "h_right_foot", T_full)
        contact_l = _extract_per_step_field(oo, "foot_state", "left_foot_contact", T_full)
        contact_r = _extract_per_step_field(oo, "foot_state", "right_foot_contact", T_full)

        if h_left is not None:
            r_left_foot = np.clip(
                np.asarray(h_left[:T_full], dtype=np.float32),
                -FOOT_HEIGHT_CLIP, FOOT_HEIGHT_CLIP,
            )
        else:
            r_left_foot = np.zeros(T_full, dtype=np.float32)
        if h_right is not None:
            r_right_foot = np.clip(
                np.asarray(h_right[:T_full], dtype=np.float32),
                -FOOT_HEIGHT_CLIP, FOOT_HEIGHT_CLIP,
            )
        else:
            r_right_foot = np.zeros(T_full, dtype=np.float32)

        if contact_l is not None and contact_r is not None:
            w_left_raw, w_right_raw = compute_foot_weights(
                np.asarray(contact_l[:T_full], dtype=bool),
                np.asarray(contact_r[:T_full], dtype=bool),
                T_full,
                h_left=np.asarray(h_left[:T_full], dtype=np.float32) if h_left is not None else None,
                h_right=np.asarray(h_right[:T_full], dtype=np.float32) if h_right is not None else None,
            )
        else:
            w_left_raw = np.zeros(T_full, dtype=np.float32)
            w_right_raw = np.zeros(T_full, dtype=np.float32)

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
            dist = np.full(T_full, 1e9, dtype=np.float64)
        else:
            self_xy = np.stack([self_x[:T_full], self_y[:T_full]], axis=1)
            opp_xy = np.stack([opp_x[:T_full], opp_y[:T_full]], axis=1)
            r_radial, r_tangential = compute_radial_tangential_rewards(
                self_xy, opp_xy,
            )
            dist = np.linalg.norm(opp_xy[:T_full] - self_xy[:T_full], axis=1)

        # --- r_face: facing_score × dist_gate ---
        fwd_x = _extract_per_step_field(oo, "face_opponent", "forward_x", T_full)
        fwd_y = _extract_per_step_field(oo, "face_opponent", "forward_y", T_full)

        r_face = np.zeros(T_full, dtype=np.float32)
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
            dist_gate = np.clip(
                (D_FACE - dist) / (D_FACE - D_STRIKE), 0.0, 1.0
            )
            r_face = (facing_score * dist_gate).astype(np.float32)

        # --- r_damage_dealt / r_damage_taken ---
        dealt = _extract_per_step_field(oo, "damage_breakdown", "dealt", T_full)
        taken = _extract_per_step_field(oo, "damage_breakdown", "taken", T_full)
        if dealt is not None:
            r_dealt = np.asarray(dealt[:T_full], dtype=np.float32)
        else:
            r_dealt = np.zeros(T_full, dtype=np.float32)
        if taken is not None:
            r_taken = np.asarray(taken[:T_full], dtype=np.float32)
        else:
            r_taken = np.zeros(T_full, dtype=np.float32)

        # --- No early termination (same as standup_face) ---
        is_terminated = False

        # --- Actor weights ---
        # r_fall: fixed.  Foot/radial/tangential/face: gated by φ².
        # Damage channels: hard distance gate (active when dist <= D_DAMAGE_GATE).
        phi_sq = (phi_arr ** 2).astype(np.float32)
        damage_gate = (dist <= D_DAMAGE_GATE).astype(np.float32)
        actor_weights = {
            "r_fall": np.full(T_full, self._base_actor_weights[0], dtype=np.float32),
            "r_left_foot": (w_left_raw * phi_sq),
            "r_right_foot": (w_right_raw * phi_sq),
            "r_radial": (self._base_actor_weights[1] * phi_sq),
            "r_tangential": (self._base_actor_weights[2] * phi_sq),
            "r_face": (self._base_actor_weights[3] * phi_sq),
            "r_damage_dealt": (self.damage_dealt_weight * damage_gate),
            "r_damage_taken": (self.damage_taken_weight * damage_gate),
        }

        all_rewards = {
            "r_fall": r_fall,
            "r_left_foot": r_left_foot,
            "r_right_foot": r_right_foot,
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
            log_prob=None,
        )]

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
            dealt = _extract_per_step_field(oo, "damage_breakdown", "dealt", T)
            taken = _extract_per_step_field(oo, "damage_breakdown", "taken", T)
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
