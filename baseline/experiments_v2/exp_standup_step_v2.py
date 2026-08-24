"""V2 end-to-end: phase-switched standup + balance.

From random fallen state → stand up → maintain balance + cross-support step.

Two reward phases with hard switch based on torso height:

  STANDUP phase (h_torso < 1.20):
    r_potential = (1-γ) × φ_4stage = 0.01 × φ_4stage,  weight = 1.0
    (same as exp_standup.py — pure 4-stage standing potential)

  BALANCE phase (h_torso >= 1.20):
    r_fall  = 0.01 × φ_height,        weight = 3.0 (fixed)
    r_cross = cross-support signal,    weight = 1.0 × φ_height²
    (same as exp_basic_balance.py — survival + cross-support gating)

  Phase transitions (per agent, per step):
    STANDUP → BALANCE:  h_torso >= 1.20
    BALANCE → STANDUP:  h_torso < 0.70  (fallen)

Three reward channels (each with independent critic):
  r_potential — active only in STANDUP phase
  r_fall      — active only in BALANCE phase
  r_cross     — active only in BALANCE phase

φ_4stage comes from StandingBalance4StageRewarder ("potential" field).
φ_height comes from HeightPhiObserver ("phi" field).

No imbalance termination — robot can fall and get back up.
Every step is trainable.

Blueprint: baseline/humanoid21/end2end/standup_step_v2_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from baseline.framework.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.ppo_trainer import _extract_per_step_scalar, _extract_per_step_field

from .base import CombatExperimentV2Base


# --- Phase thresholds ---
H_STANDUP_TO_BALANCE: float = 1.20
"""h_torso above this → enter BALANCE phase."""
H_BALANCE_TO_STANDUP: float = 0.70
"""h_torso below this → fall back to STANDUP phase."""


class StandupStepV2(CombatExperimentV2Base):
    """End-to-end standup + balance with phase-switched reward.

    Dual-agent: both robots get RandomFallenStatePlugin and train
    simultaneously.  No early termination — robot can fall and recover.
    """

    name = "standup_step_v2"

    # --- Network ---
    obs_dim: int = 96
    action_dim: int = 21

    # --- Reward channels ---
    _channel_names = ("r_potential", "r_fall", "r_cross")
    _gamma: float = 0.99
    _gae_lambda: float = 0.95

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01

    # --- Base actor weights ---
    # r_potential: weight 1.0 (standup phase, same as exp_standup)
    # r_fall: weight 1.0 (balance phase)
    # r_cross: weight 0.33 × φ_height² (balance phase, gated)
    _base_actor_weights: Tuple[float, ...] = (1.0, 1.0, 0.33)

    # --- Env ---
    env_blueprint = ""  # overridden via _env_pb()
    agent_used = "both"
    max_steps: int = 200

    # Observer keys: (agent_id, cross_key, phi4stage_key, phi_height_key)
    _AGENT_OBS = (
        ("robot_a", "cross_support_a", "standing_balance_a", "height_phi_a"),
        ("robot_b", "cross_support_b", "standing_balance_b", "height_phi_b"),
    )
    _AGENT_IDS = ("robot_a", "robot_b")

    # --- PPO tuning (aligned with exp_standup) ---
    log_std_min: float = -2.5
    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096
    entropy_coef: float = 1e-3

    # --- Rollout schedule ---
    episodes_per_update: int = 512
    max_updates: int = 5000
    eval_interval: int = 5
    eval_episodes: int = 64

    # --- Video recording ---
    video_eval_interval: int = 2

    # --- Stateful metrics ---
    _best_potential: float = -1.0
    _success_rate: float = 0.0

    # ------------------------------------------------------------------
    # Blueprint loading
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
        bp_path = Path(__file__).resolve().parent.parent / "humanoid21" / "end2end" / "standup_step_v2_env.yaml"
        return ParameterizedEnvBlueprint.load(bp_path)

    def reward_channels(self) -> Tuple[RewardChannel, ...]:
        return tuple(
            RewardChannel(name=k, gamma=self._gamma, gae_lambda=self._gae_lambda)
            for k in self._channel_names
        )

    # ------------------------------------------------------------------
    # Phase determination
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_phase_mask(
        h_torso: np.ndarray, T: int,
    ) -> np.ndarray:
        """Compute per-step phase mask.

        Returns boolean array of shape (T,):
          True  = BALANCE phase
          False = STANDUP phase

        Phase transitions:
          STANDUP → BALANCE: h_torso >= H_STANDUP_TO_BALANCE
          BALANCE → STANDUP: h_torso < H_BALANCE_TO_STANDUP
        """
        phase = np.zeros(T, dtype=bool)  # False = STANDUP
        in_balance = False
        for t in range(T):
            h = float(h_torso[t])
            if in_balance:
                if h < H_BALANCE_TO_STANDUP:
                    in_balance = False
            else:
                if h >= H_STANDUP_TO_BALANCE:
                    in_balance = True
            phase[t] = in_balance
        return phase

    # ------------------------------------------------------------------
    # Trajectory building
    # ------------------------------------------------------------------

    def _build_agent_trajectory(
        self,
        episode,
        agent_id: str,
        cross_key: str,
        phi4stage_key: str,
        phi_height_key: str,
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

        # --- Extract φ_4stage (StandingBalance4StageRewarder "potential") ---
        phi4_arr = _extract_per_step_field(
            episode.observer_outputs, phi4stage_key, "potential", T_full,
        )
        if phi4_arr is not None:
            phi4_arr = phi4_arr[:T_full]
        else:
            phi4_arr = np.zeros(T_full, dtype=np.float32)
        phi4_arr = np.clip(phi4_arr, 0.0, 1.0).astype(np.float32)

        # --- Extract φ_height (HeightPhiObserver "phi") ---
        phi_h_arr = _extract_per_step_field(
            episode.observer_outputs, phi_height_key, "phi", T_full,
        )
        if phi_h_arr is not None:
            phi_h_arr = phi_h_arr[:T_full]
        else:
            phi_h_arr = np.zeros(T_full, dtype=np.float32)
        phi_h_arr = np.clip(phi_h_arr, 0.0, 1.0).astype(np.float32)

        # --- Extract h_torso for phase determination ---
        h_torso = _extract_per_step_field(
            episode.observer_outputs, phi4stage_key, "h_torso", T_full,
        )
        if h_torso is not None:
            h_torso = h_torso[:T_full]
        else:
            h_torso = np.zeros(T_full, dtype=np.float32)

        # --- Extract r_cross signal ---
        r_cross_raw = _extract_per_step_scalar(
            episode.observer_outputs, cross_key, T_full,
        )
        if r_cross_raw is not None:
            r_cross_raw = r_cross_raw[:T_full]
        else:
            r_cross_raw = np.zeros(T_full, dtype=np.float32)

        # --- Compute phase mask ---
        balance_mask = self._compute_phase_mask(h_torso, T_full)
        standup_mask = ~balance_mask

        # --- r_potential: active in STANDUP phase only ---
        # r_potential = (1-γ) × φ_4stage = 0.01 × φ_4stage
        r_potential = (self.per_step_phi_coef * phi4_arr * standup_mask).astype(np.float32)

        # --- r_fall: active in BALANCE phase only ---
        # r_fall = 0.01 × φ_height
        r_fall = (self.per_step_phi_coef * phi_h_arr * balance_mask).astype(np.float32)

        # --- r_cross: active in BALANCE phase only ---
        r_cross = (r_cross_raw.astype(np.float32) * balance_mask).astype(np.float32)

        # --- No early termination ---
        is_terminated = False

        # --- Actor weights ---
        # r_potential: 1.0 in STANDUP, 0 in BALANCE
        # r_fall: 3.0 in BALANCE, 0 in STANDUP
        # r_cross: φ_height² in BALANCE, 0 in STANDUP
        actor_weights = {
            "r_potential": (self._base_actor_weights[0] * standup_mask).astype(np.float32),
            "r_fall": (self._base_actor_weights[1] * balance_mask).astype(np.float32),
            "r_cross": (self._base_actor_weights[2] * phi_h_arr ** 2 * balance_mask).astype(np.float32),
        }

        all_rewards = {
            "r_potential": r_potential,
            "r_fall": r_fall,
            "r_cross": r_cross,
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
            for agent_id, cross_key, phi4stage_key, phi_height_key in self._AGENT_OBS:
                trajs = self._build_agent_trajectory(
                    episode, agent_id, cross_key, phi4stage_key, phi_height_key,
                )
                all_trajs.extend(trajs)
        return all_trajs

    # ------------------------------------------------------------------
    # Eval
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

            for agent_id, _, phi4stage_key, _ in self._AGENT_OBS:
                n_agents += 1
                phi = _extract_per_step_field(
                    ep.observer_outputs, phi4stage_key, "potential", T,
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

        is_new_best = mean_max_pot > self._best_potential
        if is_new_best:
            self._best_potential = mean_max_pot

        return {
            "is_new_best": is_new_best,
            "stop_training": False,
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
        }

    def load_state(self, state: dict) -> None:
        self._best_potential = float(state.get("best_potential", -1.0))
        self._success_rate = float(state.get("success_rate", 0.0))


EXPERIMENT_CLASS = StandupStepV2
