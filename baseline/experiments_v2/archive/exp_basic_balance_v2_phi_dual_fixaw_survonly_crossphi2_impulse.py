"""V2 dual-agent experiment: crossphi2 + impulse perturbation, warm-started.

Same reward design as exp_basic_balance_v2_phi_dual_fixaw_survonly_crossphi2.py
but uses ImpulsePerturbationPlugin to generate physically realistic perturbed
initial states via internal sim + reference policy.

  - r_fall: 0.01 × φ(t) per step, no fall penalty, no timeout bonus
  - r_cross: alternating step reward/penalty, actor weight = 1.0 × φ²
  - r_fall actor weight: fixed 3.0
  - Impulse: force=[50, 150]N, duration=[2, 4] steps, random horizontal direction
  - Warm-start: loads actor weights from BASE_POLICY_PATH checkpoint

Environment variables:
  POLICY_BLUEPRINT_PATH - path to reference policy_blueprint.yaml for internal sim
  BASE_POLICY_PATH      - path to checkpoint .pt for warm-start (optional)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from baseline.framework.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.ppo_trainer import _extract_per_step_scalar, _extract_per_step_field

from .base import CombatExperimentV2Base


class BasicBalanceV2PhiDualFixAWSurvOnlyCrossPhi2Impulse(CombatExperimentV2Base):

    name = "v2_basic_balance_v2_phi_dual_fixaw_survonly_crossphi2_impulse"

    _channel_names = ("r_fall", "r_cross")
    _gamma = 0.99
    _gae_lambda = 0.95

    env_blueprint = "basic_balance_v2_phi_dual_impulse_env.yaml"
    agent_used = "both"

    episodes_per_update: int = 256 * 4

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01

    # --- Base actor weights (r_fall fixed, r_cross gated by φ²) ---
    _base_actor_weights: Tuple[float, ...] = (3.0, 1.0)

    _AGENT_IDS = ("robot_a", "robot_b")

    _survival_rate: float = 0.0
    _best_survived: float = -1.0

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
        bp_path = Path(__file__).resolve().parent.parent / "humanoid21" / "balance_recover" / "basic_balance_v2_phi_dual_impulse_env.yaml"
        return ParameterizedEnvBlueprint.load(bp_path)

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return tuple(
            RewardChannel(name=k, gamma=self._gamma, gae_lambda=self._gae_lambda)
            for k in self._channel_names
        )

    # ------------------------------------------------------------------
    # Impulse perturbation parameters
    # ------------------------------------------------------------------

    def _impulse_params(self) -> Dict[str, Any]:
        policy_bp_path = os.environ.get("POLICY_BLUEPRINT_PATH")
        if not policy_bp_path:
            raise ValueError(
                "crossphi2_impulse requires POLICY_BLUEPRINT_PATH environment variable. "
                "Example: POLICY_BLUEPRINT_PATH=baseline/runs/.../policy/policy_blueprint.yaml"
            )
        return {
            "policy_blueprint_path": str(Path(policy_bp_path).resolve()),
            "impulse_body": "torso",
            "force_magnitude": [50, 150],
            "duration_action_steps": [2, 4],
            "direction_mode": "random_horizontal",
            "fixed_direction": None,
        }

    # ------------------------------------------------------------------
    # Warm-start from base policy checkpoint
    # ------------------------------------------------------------------

    def build_actor(self, device: torch.device) -> Any:
        from baseline.common.policies.tanh_gaussian_mlp import TanhGaussianMLPPolicy

        base_path = os.environ.get("BASE_POLICY_PATH")
        if not base_path:
            return super().build_actor(device)

        ckpt_path = Path(base_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hidden_dim = int(payload.get("hidden_dim", payload.get("actor_hidden_dim", 256)))

        actor = TanhGaussianMLPPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            log_std_min=float(self.log_std_min),
            log_std_max=float(self.log_std_max),
            device=device,
        )
        actor.load_state_dict(payload["actor_state_dict"], strict=False)
        actor = actor.to(device)
        actor.log_std_min = float(self.log_std_min)
        return actor

    # ------------------------------------------------------------------
    # Job construction — inject impulse params into env materialize
    # ------------------------------------------------------------------

    def build_jobs(
        self,
        policy_bp,
        base_seed: int,
        n_episodes: int,
    ) -> List[Tuple[Any, Any, Any, int, Dict[str, Any]]]:
        env_pb = self._env_pb()
        impulse = self._impulse_params()
        rng = np.random.default_rng(base_seed)

        env_bp = env_pb.materialize(max_steps=self.max_steps, **impulse)

        jobs: List[Tuple[Any, Any, Any, int, Dict[str, Any]]] = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            initial_distance = float(
                rng.uniform(self.init_distance_min, self.init_distance_max)
            )
            jobs.append((
                policy_bp, policy_bp,
                env_bp, seed,
                {"initial_distance": initial_distance},
            ))
        return jobs

    # ------------------------------------------------------------------
    # Trajectory building — identical to crossphi2
    # ------------------------------------------------------------------

    def _build_agent_trajectory(
        self,
        episode,
        agent_id: str,
        cross_key: str,
        posture_key: str,
        phi_key: str,
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

        # --- Extract φ per step ---
        phi_arr = _extract_per_step_field(episode.observer_outputs, phi_key, "phi", T_full)
        if phi_arr is not None:
            phi_arr = phi_arr[:T]
        else:
            phi_arr = np.ones(T, dtype=np.float32)
        phi_arr = np.clip(phi_arr, 0.0, 1.0).astype(np.float32)

        # --- r_fall: 0.01 × φ(t) per step only — no terminal signal ---
        r_fall = (self.per_step_phi_coef * phi_arr).astype(np.float32)

        # --- r_cross ---
        r_cross = _extract_per_step_scalar(episode.observer_outputs, cross_key, T_full)
        if r_cross is not None:
            r_cross = r_cross[:T]
        else:
            r_cross = np.zeros(T, dtype=np.float32)

        # --- Actor weights: r_fall fixed, r_cross gated by φ² ---
        is_terminated = fell

        # --- Build channels ---
        all_rewards = {
            "r_fall": r_fall,
            "r_cross": r_cross.astype(np.float32),
        }

        actor_weights = {
            "r_fall": np.full(T, self._base_actor_weights[0], dtype=np.float32),
            "r_cross": (self._base_actor_weights[1] * phi_arr ** 2).astype(np.float32),
        }

        channels: Dict[str, ChannelData] = {}
        for idx, key in enumerate(self._channel_names):
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
            log_prob=None,
        )]

    def build_trajectories(self, episodes) -> List[Trajectory]:
        agent_specs = [
            ("robot_a", "cross_support_a", "posture_a", "height_phi_a"),
            ("robot_b", "cross_support_b", "posture_b", "height_phi_b"),
        ]

        all_trajs: List[Trajectory] = []
        for episode in episodes:
            for agent_id, cross_key, posture_key, phi_key in agent_specs:
                agent_trajs = self._build_agent_trajectory(
                    episode, agent_id, cross_key, posture_key, phi_key,
                )
                all_trajs.extend(agent_trajs)
        return all_trajs

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        survived_count = 0
        total_agents = 0
        for ep in episodes:
            for aid in self._AGENT_IDS:
                total_agents += 1
                term_reason = ep.agent_termination_reason.get(aid, "")
                if not term_reason.startswith("imbalance"):
                    survived_count += 1

        survival_rate = float(survived_count / max(total_agents, 1))
        self._survival_rate = survival_rate

        survived_metric = float(survived_count)
        is_new_best = survived_metric > self._best_survived
        if is_new_best:
            self._best_survived = survived_metric

        return {
            "is_new_best": is_new_best,
            "info": {
                "survived": survived_metric,
                "survival_rate": round(survival_rate, 3),
            },
        }

    def state(self) -> dict:
        return {
            "survival_rate": self._survival_rate,
            "best_survived": self._best_survived,
        }

    def load_state(self, state: dict) -> None:
        self._survival_rate = float(state.get("survival_rate", 0.0))
        self._best_survived = float(state.get("best_survived", -1.0))


EXPERIMENT_CLASS = BasicBalanceV2PhiDualFixAWSurvOnlyCrossPhi2Impulse
