"""V2 end-to-end balance reinforcement: phase-switched rewards + perturbation.

Combines ``exp_standup_step_v3.py`` (phase-switched standup/balance rewards
with per-foot stepping) with ``exp_standup_balance.py`` (StandingTriggeredForcePlugin
+ 12-level force/duration curriculum: 40N (4 levels × 10-step) + 100N (8 levels × 5-step)).

From random fallen state → stand up → maintain balance + stepping → withstand
external force perturbations.  Built on top of the standup_step_v3 policy
(warm-start via --resume-from).

Two reward phases with hard switch based on torso height:

  STANDUP phase (h_torso < plateau):
    r_potential = (1-γ) × φ_4stage = 0.01 × φ_4stage,  weight = 3.0
    (same as exp_standup.py — pure 4-stage standing potential)

  BALANCE phase (h_torso >= plateau):
    r_fall       = 0.01 × φ_height,         weight = 3.0 (fixed)
    r_left_foot  = clip(h_left,  -0.1, 0.1), weight = stepping state machine
    r_right_foot = clip(h_right, -0.1, 0.1), weight = stepping state machine
    (same as exp_basic_balance_step.py — survival + per-foot stepping)

  Phase transitions (per agent, per step):
    STANDUP → BALANCE:  plateau detection on h_torso
    BALANCE → STANDUP:  h_torso < 0.70  (fallen)

Four reward channels (each with independent critic):
  r_potential — reward always present, aw=3.0 in STANDUP, 0 in BALANCE
  r_fall      — reward always present, aw=3.0 in BALANCE, 0 in STANDUP
  r_left_foot — reward always present, aw = state machine (BALANCE only)
  r_right_foot— reward always present, aw = state machine (BALANCE only)

Rewards are NOT masked — the critic can learn from the physical signal
at all times.  Only actor_weight controls when each channel influences
the policy update.

Perturbation curriculum:
  12 levels: 40N (4 levels × 10-step duration) + 100N (8 levels × 5-step duration).
  Level 0-3:  40N  (dur 1-10, 11-20, 21-30, 31-40)
  Level 4-7:  100N (dur 1-10, 11-20, 21-30, 31-40)
  Level 8-11: 200N (dur 1-10, 11-20, 21-30, 31-40)
  Promotion when recovery_rate (1 - fall_count/push_count) >= 0.8.

Blueprint: baseline/humanoid21/end2end/balance_v2_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_field

from .base import CombatExperimentPPOBase
from baseline.humanoid21.end2end.stepping_state_machine import (
    compute_foot_weights,
    FOOT_WEIGHT,
    PHASE_A_STEPS,
    PHASE_B_END,
    DOUBLE_GRACE_STEPS,
    STATE_DOUBLE,
    STATE_SUPPORT_L,
    STATE_SUPPORT_R,
    STATE_FLIGHT,
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


class BalanceV2(CombatExperimentPPOBase):
    """End-to-end balance reinforcement: phase-switched rewards + perturbation.

    Dual-agent: both robots get RandomFallenStatePlugin and
    StandingTriggeredForcePlugin, train simultaneously.  No early
    termination — robot can fall and recover under perturbation.
    """

    name = "balance_v2"

    # --- Network ---
    obs_dim: int = 96
    action_dim: int = 21

    # --- Reward channels ---
    _channel_names = ("r_potential", "r_fall", "r_left_foot", "r_right_foot")
    _channel_gammas = {
        "r_potential": 0.99,
        "r_fall": 0.99,
        "r_left_foot": 0.9,
        "r_right_foot": 0.9,
    }
    _gae_lambda = 0.95

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01

    # --- Foot height reward saturation ---
    foot_height_clip: float = 0.05

    # --- r_fall actor weight (fixed, same as exp_basic_balance_step) ---
    r_fall_actor_weight: float = 3.0

    # --- r_potential actor weight (fixed, standup phase) ---
    r_potential_actor_weight: float = 3.0

    # --- Env ---
    env_blueprint = ""  # overridden via _env_pb()
    agent_used = "both"
    max_steps: int = 400

    # Observer keys: (agent_id, foot_key, phi4stage_key, phi_height_key)
    _AGENT_OBS = (
        ("robot_a", "foot_state_a", "standing_balance_a", "height_phi_a"),
        ("robot_b", "foot_state_b", "standing_balance_b", "height_phi_b"),
    )
    _AGENT_IDS = ("robot_a", "robot_b")

    # --- PPO tuning (aligned with exp_standup) ---
    log_std_min: float = -2.5
    learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    target_kl: float = 0.05
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

    # --- Curriculum: 12 levels, 40N (4 levels × 10-step) + 100N (8 levels × 5-step) ---
    LEVEL_FORCES: Tuple[float, ...] = (
        40.0, 40.0, 40.0, 40.0,                               # level 0-3
        100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,  # level 4-11
    )
    LEVEL_DURATION_RANGES: Tuple[Tuple[int, int], ...] = (
        (1, 10), (11, 20), (21, 30), (31, 40),               # 40N
        (1, 5), (6, 10), (11, 15), (16, 20),                 # 100N
        (21, 25), (26, 30), (31, 35), (36, 40),              # 100N
    )
    PROMOTE_RECOVERY_RATE: float = 0.8
    """晋级阈值：扰动后不摔倒的比例（recovery_rate = 1 - fall/push）。"""
    PROMOTE_PATIENCE: int = 2

    # --- Stateful metrics ---
    _best_potential: float = -1.0
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
            / "humanoid21" / "end2end" / "balance_v2_env.yaml"
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
        *,
        explore_intensity: float = 0.0,
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
                {"impulse_params": impulse_params, "explore_intensity": explore_intensity},
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

        STANDUP → BALANCE: plateau detection on h_torso.
          A sliding window of PLATEAU_WINDOW steps is scanned. When the
          entire window is above H_BALANCE_LOW_THRESHOLD and the linear
          regression slope is below PLATEAU_SLOPE_EPS, the window start
          is marked as the BALANCE entry point.

        BALANCE → STANDUP: h_torso < H_BALANCE_TO_STANDUP (fallen).
        """
        phase = np.zeros(T, dtype=bool)  # False = STANDUP

        # --- Find plateau entry point ---
        balance_start = None
        W = PLATEAU_WINDOW
        for t in range(W, T + 1):
            window = h_torso[t - W:t]
            if np.all(window >= H_BALANCE_LOW_THRESHOLD):
                # Linear regression slope
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
                    balance_start = t - W  # BALANCE starts at window start
                    break

        if balance_start is None:
            return phase  # never reached plateau, all STANDUP

        # --- Fill phase: BALANCE from plateau start, fall back if h < 0.7 ---
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
        """Balance-gated foot weights.

        Delegates to ``stepping_state_machine.compute_foot_weights`` on each
        contiguous BALANCE segment.  Non-BALANCE (STANDUP) frames get zero
        weight and the state machine resets at each segment boundary.

        Returns ``(w_left, w_right)``, each shape ``(T,)`` float32.
        """
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
        foot_key: str,
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
        phi4_arr = extract_per_step_field(
            episode.observer_outputs, phi4stage_key, "potential", T_full,
        )
        if phi4_arr is not None:
            phi4_arr = phi4_arr[:T_full]
        else:
            phi4_arr = np.zeros(T_full, dtype=np.float32)
        phi4_arr = np.clip(phi4_arr, 0.0, 1.0).astype(np.float32)

        # --- Extract φ_height (HeightPhiObserver "phi") ---
        phi_h_arr = extract_per_step_field(
            episode.observer_outputs, phi_height_key, "phi", T_full,
        )
        if phi_h_arr is not None:
            phi_h_arr = phi_h_arr[:T_full]
        else:
            phi_h_arr = np.zeros(T_full, dtype=np.float32)
        phi_h_arr = np.clip(phi_h_arr, 0.0, 1.0).astype(np.float32)

        # --- Extract h_torso for phase determination ---
        h_torso = extract_per_step_field(
            episode.observer_outputs, phi4stage_key, "h_torso", T_full,
        )
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
        h_left = self._extract_foot_field(episode, foot_key, "h_left_foot", T_full)
        h_right = self._extract_foot_field(episode, foot_key, "h_right_foot", T_full)
        r_left = np.clip(h_left, -self.foot_height_clip, self.foot_height_clip).astype(np.float32)
        r_right = np.clip(h_right, -self.foot_height_clip, self.foot_height_clip).astype(np.float32)

        # --- Contacts → stepping state machine → foot actor weights ---
        contact_l = self._extract_foot_field(episode, foot_key, "left_foot_contact", T_full)
        contact_r = self._extract_foot_field(episode, foot_key, "right_foot_contact", T_full)
        w_left, w_right = self._compute_foot_weights_masked(
            contact_l.astype(bool), contact_r.astype(bool), balance_mask, T_full,
            h_left=h_left, h_right=h_right,
        )

        # --- No early termination ---
        is_terminated = False

        # --- Actor weights ---
        actor_weights = {
            "r_potential": (self.r_potential_actor_weight * standup_mask).astype(np.float32),
            "r_fall": (self.r_fall_actor_weight * balance_mask).astype(np.float32),
            "r_left_foot": w_left,
            "r_right_foot": w_right,
        }

        all_rewards = {
            "r_potential": r_potential,
            "r_fall": r_fall,
            "r_left_foot": r_left,
            "r_right_foot": r_right,
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
    def _extract_foot_field(
        episode, foot_key: str, field: str, T_full: int,
    ) -> np.ndarray:
        """Extract a FootStateObserver field, truncated to ``T_full``.

        Raises if the observer or field is missing — a silent zero fallback
        would make the stepping signal vanish without any error.
        """
        arr = extract_per_step_field(
            episode.observer_outputs, foot_key, field, T_full,
        )
        if arr is None:
            raise KeyError(
                f"_extract_foot_field: observer '{foot_key}' field '{field}' "
                f"missing from episode.observer_outputs "
                f"(available observers={list(episode.observer_outputs.keys())})"
            )
        return arr[:T_full]

    def build_trajectories(self, episodes) -> List[Trajectory]:
        all_trajs: List[Trajectory] = []
        for episode in episodes:
            for agent_id, foot_key, phi4stage_key, phi_height_key in self._AGENT_OBS:
                trajs = self._build_agent_trajectory(
                    episode, agent_id, foot_key, phi4stage_key, phi_height_key,
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

            for agent_id, _, phi4stage_key, _ in self._AGENT_OBS:
                n_agents += 1
                phi = extract_per_step_field(
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

                # balance_ratio: fraction of steps with h_torso > 1.0
                h_torso = extract_per_step_field(
                    ep.observer_outputs, phi4stage_key, "h_torso", T,
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


EXPERIMENT_CLASS = BalanceV2
