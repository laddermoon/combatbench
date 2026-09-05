"""V2 end-to-end step 2: standup + cross-support stepping.

From random fallen state → stand up → step alternately.
Built on top of the standup policy (warm-start via --resume-from).

Reward channels:
  r_fall  = 0.01 × φ(t),           actor_weight = 3.0 (fixed)
  r_cross = cross-support signal,   actor_weight = 1.0 × φ²

φ is the 4-stage standing potential from StandingBalance4StageRewarder
(same as standup experiment). r_cross uses φ² gating so stepping is
only rewarded after the robot is standing.

No imbalance termination — the robot can fall and get back up.
Every step is trainable (like standup, not like basic_balance).

Blueprint: baseline/humanoid21/end2end/standup_step_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_scalar, extract_per_step_field

from .base import CombatExperimentPPOBase


class StandupStep(CombatExperimentPPOBase):
    """End-to-end step 2: standup + cross-support stepping.

    Dual-agent: both robots get RandomFallenStatePlugin and train
    simultaneously.  No early termination — robot can fall and recover.
    """

    name = "standup_step"

    # --- Network ---
    obs_dim: int = 96
    action_dim: int = 21

    # --- Reward channels ---
    _channel_names = ("r_fall", "r_cross")
    _gamma: float = 0.99
    _gae_lambda: float = 0.95

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01

    # --- Base actor weights (r_fall fixed, r_cross gated by φ²) ---
    _base_actor_weights: Tuple[float, ...] = (3.0, 1.0)

    # --- Env ---
    env_blueprint = ""  # overridden via _env_pb()
    agent_used = "both"
    max_steps: int = 400

    _AGENT_OBS = (
        ("robot_a", "cross_support_a", "standing_balance_a"),
        ("robot_b", "cross_support_b", "standing_balance_b"),
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
    eval_interval: int = 5
    eval_episodes: int = 64

    # --- Video recording ---
    video_eval_interval: int = 5

    # --- Early stop ---
    _no_improvement_limit: int = 200
    _min_updates: int = 600

    # --- Stateful metrics ---
    _best_potential: float = -1.0
    _best_survived: float = -1.0
    _success_rate: float = 0.0
    _last_best_update: int = 0

    # ------------------------------------------------------------------
    # Blueprint loading — from end2end/ directory, not blueprints/
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
        bp_path = Path(__file__).resolve().parent.parent / "humanoid21" / "end2end" / "standup_step_env.yaml"
        return ParameterizedEnvBlueprint.load(bp_path)

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return tuple(
            RewardChannel(name=k, gamma=self._gamma, gae_lambda=self._gae_lambda)
            for k in self._channel_names
        )

    # ------------------------------------------------------------------
    # Trajectory building
    # ------------------------------------------------------------------

    def _build_agent_trajectory(
        self,
        episode,
        agent_id: str,
        cross_key: str,
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

        # --- r_fall: 0.01 × φ(t) per step ---
        r_fall = (self.per_step_phi_coef * phi_arr).astype(np.float32)

        # --- r_cross: cross-support signal ---
        r_cross = extract_per_step_scalar(
            episode.observer_outputs, cross_key, T_full,
        )
        if r_cross is not None:
            r_cross = r_cross[:T_full]
        else:
            r_cross = np.zeros(T_full, dtype=np.float32)

        # --- No early termination: robot can fall and get back up ---
        is_terminated = False

        # --- Actor weights: r_fall fixed, r_cross gated by φ² ---
        actor_weights = {
            "r_fall": np.full(T_full, self._base_actor_weights[0], dtype=np.float32),
            "r_cross": (self._base_actor_weights[1] * phi_arr ** 2).astype(np.float32),
        }

        all_rewards = {
            "r_fall": r_fall,
            "r_cross": r_cross.astype(np.float32),
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
            explore_intensity=self.extract_explore_intensity(episode, agent_id, T_full),
        )]

    def build_trajectories(self, episodes) -> List[Trajectory]:
        all_trajs: List[Trajectory] = []
        for episode in episodes:
            for agent_id, cross_key, phi_key in self._AGENT_OBS:
                trajs = self._build_agent_trajectory(
                    episode, agent_id, cross_key, phi_key,
                )
                all_trajs.extend(trajs)
        return all_trajs

    # ------------------------------------------------------------------
    # Eval — track standing success + cross-support quality
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        max_pots = []
        final_pots = []
        success_count = 0
        n_agents = 0

        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue

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

        n = max(len(max_pots), 1)
        mean_max_pot = sum(max_pots) / n if max_pots else 0.0
        mean_final_pot = sum(final_pots) / n if final_pots else 0.0
        success_rate = success_count / n

        self._success_rate = success_rate

        # --- Best: primary = mean_max_pot (standing achievement) ---
        is_new_best = mean_max_pot > self._best_potential
        if is_new_best:
            self._best_potential = mean_max_pot
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
                "max_pot": round(mean_max_pot, 3),
                "final_pot": round(mean_final_pot, 3),
                "success": round(success_rate, 3),
            },
        }

    def state(self) -> dict:
        return {
            "best_potential": self._best_potential,
            "success_rate": self._success_rate,
            "last_best_update": self._last_best_update,
        }

    def load_state(self, state: dict) -> None:
        self._best_potential = float(state.get("best_potential", -1.0))
        self._success_rate = float(state.get("success_rate", 0.0))
        self._last_best_update = int(state.get("last_best_update", 0))


EXPERIMENT_CLASS = StandupStep
