"""V2 end-to-end: phase-switched standup + balance with per-foot stepping.

From random fallen state → stand up → maintain balance + stepping.

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

φ_4stage comes from StandingBalance4StageRewarder ("potential" field).
φ_height comes from HeightPhiObserver ("phi" field).
Foot heights and contacts come from FootStateObserver.

Stepping state machine
----------------------
The same 3-phase gait scheduler as exp_basic_balance_step, but gated by
the BALANCE phase mask.  The state machine resets each time the robot
enters a new BALANCE segment (after falling and re-standing).  During
STANDUP frames the foot actor weights are zero and the state machine
internal state (last_swing, support_steps, prev_state) is reset.

See exp_basic_balance_step.py for the full state machine documentation.

No imbalance termination — robot can fall and get back up.
Every step is trainable.

Blueprint: baseline/humanoid21/end2end/standup_step_v3_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_field

from .base import CombatExperimentPPOBase
from baseline.framework.ppo.experiment import ExplorationSpec
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


# --- Phase-dependent explore_factor (per-frame callable) ---
# obs[45] = h_torso (root body z, meters).
# STANDUP (h < 1.1): ef = -0.631 → σ × 0.5 (more deterministic, focus on standing)
# BALANCE (h ≥ 1.1): ef = +0.631 → σ × 2.0 (more random, explore stepping)
# Must be top-level for multiprocessing picklability.
import math as _math
_EF_STANDUP = _math.log(0.5) / _math.log(3)    # ≈ -0.631
_EF_BALANCE = _math.log(2.0) / _math.log(3)   # ≈ +0.631
_H_PHASE_SWITCH = 1.1


def phase_explore_factor(obs, step):
    """Per-frame explore_factor based on torso height.

    Returns a negative ef (σ × 0.5) when the robot is low (STANDUP)
    and a positive ef (σ × 2.0) when upright (BALANCE).
    """
    h_torso = float(obs[45])
    if h_torso >= _H_PHASE_SWITCH:
        return _EF_BALANCE
    return _EF_STANDUP


class StandupStepV3(CombatExperimentPPOBase):
    """End-to-end standup + balance with phase-switched reward.

    Dual-agent: both robots get RandomFallenStatePlugin and train
    simultaneously.  No early termination — robot can fall and recover.
    """

    name = "standup_step_v3"

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
    r_fall_actor_weight: float = 1.0

    # --- r_potential actor weight (fixed, standup phase) ---
    r_potential_actor_weight: float = 1.0

    # --- Env ---
    env_blueprint = ""  # overridden via _env_pb()
    agent_used = "both"
    max_steps: int = 200

    # Observer keys: (agent_id, foot_key, phi4stage_key, phi_height_key)
    _AGENT_OBS = (
        ("robot_a", "foot_state_a", "standing_balance_a", "height_phi_a"),
        ("robot_b", "foot_state_b", "standing_balance_b", "height_phi_b"),
    )
    _AGENT_IDS = ("robot_a", "robot_b")

    # --- PPO tuning (aligned with exp_standup) ---
    learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096

    # --- Rollout schedule ---
    episodes_per_update: int = 512
    max_updates: int = 5000
    eval_interval: int = 5
    eval_episodes: int = 64

    # --- Video recording ---
    video_eval_interval: int = 2

    # --- Exploration: two-stage uncertainty floor ---
    # Phase 1: floor loss pulls uncertainty up so the policy can explore
    #          foot-lifting.  floor_weight (balance_mask) restricts the
    #          floor loss to BALANCE-phase frames only.
    # Phase 2: once the robot demonstrates stepping in ALL eval episodes
    #          for step_confirm_updates consecutive evals, floor loss is
    #          permanently disabled (one-way switch).  PPO then learns
    #          purely from advantage and σ can naturally tighten.
    #
    # Stepping success (per eval episode, per agent): after entering the
    # BALANCE phase, BOTH feet must have lifted above step_lift_threshold
    # at some point.  An update is "stepping-successful" if ALL eval
    # episodes (all agents) succeed.
    uncertainty_floor: float = 0.4
    uncertainty_coef: float = 5.0
    step_lift_threshold: float = 0.05
    step_confirm_updates: int = 3

    # --- Stateful metrics ---
    _best_potential: float = -1.0
    _success_rate: float = 0.0
    _uncertainty_history: list = None
    _floor_disabled: bool = False
    _floor_disabled_at: int = -1
    _step_success_streak: int = 0

    # ------------------------------------------------------------------
    # Blueprint loading
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
        bp_path = Path(__file__).resolve().parent.parent / "humanoid21" / "end2end" / "standup_step_v3_env.yaml"
        return ParameterizedEnvBlueprint.load(bp_path)

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, stochastic=True):
        """Override to use phase-dependent per-frame explore_factor.

        ``self.explore_factor`` is ignored — the callable
        ``phase_explore_factor`` decides ef per frame from obs[45]
        (h_torso): σ×0.5 in STANDUP, σ×2.0 in BALANCE.
        """
        from baseline.framework.rollout import Job
        env_bp = self._env_pb().materialize(max_steps=self.max_steps)
        rng = np.random.default_rng(base_seed)
        jobs = []
        for i in range(n_episodes):
            seed = int(base_seed + i)
            initial_distance = float(
                rng.uniform(self.init_distance_min, self.init_distance_max)
            )
            jobs.append(Job(
                policy_a_bp=policy_bp,
                policy_b_bp=policy_bp,
                env_bp=env_bp,
                seed=seed,
                episode_options={"initial_distance": initial_distance},
                explore_factor_a=phase_explore_factor,
                explore_factor_b=phase_explore_factor,
                stochastic=stochastic,
            ))
        return jobs

    def on_update(self, stats, update: int) -> None:
        """Track uncertainty history for diagnostics.

        Floor disable is now driven by stepping success in ``on_eval``,
        not by uncertainty level here.
        """
        if self._uncertainty_history is None:
            self._uncertainty_history = []
        u = float(stats.policy_stats.get("uncertainty", 0.0))
        self._uncertainty_history.append(u)

    def exploration(self, update: int) -> ExplorationSpec:
        """Two-stage exploration spec.

        Phase 1 (floor active): configured floor/coef.
        Phase 2 (floor disabled): zero floor/coef.
        """
        if self._floor_disabled:
            return ExplorationSpec(uncertainty_floor=0.0, uncertainty_coef=0.0)
        return ExplorationSpec(
            uncertainty_floor=self.uncertainty_floor,
            uncertainty_coef=self.uncertainty_coef,
        )

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
            explore_factor=self.extract_explore_factor(episode, agent_id, T_full),
            floor_weight=balance_mask.astype(np.float32),
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
        if not getattr(self, "_ef_verified", False):
            self._ef_verified = True
            self._verify_explore_factor_flow(all_trajs)
        return all_trajs

    def _verify_explore_factor_flow(self, trajs: List[Trajectory]) -> None:
        """One-time diagnostic: verify per-frame explore_factor data flow.

        Checks (first trajectory only):
        1. explore_factor array exists and has correct length.
        2. ef < 0 (σ×0.5) on frames where obs[45] < 1.1 (STANDUP).
        3. ef > 0 (σ×2.0) on frames where obs[45] >= 1.1 (BALANCE).
        4. Fraction of BALANCE frames is plausible for this experiment.
        5. Expected eff/std ratio matches the logged eff_std_mean/std_mean.
        """
        if not trajs:
            return
        t0 = trajs[0]
        ef = t0.explore_factor
        obs = np.asarray(t0.obs, dtype=np.float32)
        if ef is None:
            print("  [ef-verify] FAIL: Trajectory.explore_factor is None "
                  "— rollout did not record per-frame ef", flush=True)
            return
        ef = np.asarray(ef, dtype=np.float32)
        h_obs = obs[:, 45]
        n = len(ef)
        n_balance = int((h_obs >= _H_PHASE_SWITCH).sum())
        n_standup = n - n_balance
        # Consistency: ef sign must match phase from obs[45]
        ok_standup = bool(np.all(ef[h_obs < _H_PHASE_SWITCH] < 0)) if n_standup else True
        ok_balance = bool(np.all(ef[h_obs >= _H_PHASE_SWITCH] > 0)) if n_balance else True
        exp_ratio = (n_balance * 2.0 + n_standup * 0.5) / n
        print(
            f"  [ef-verify] T={n} ef_min={ef.min():.3f} ef_max={ef.max():.3f} | "
            f"standup(h<1.1)={n_standup} ef<0: {'OK' if ok_standup else 'FAIL'} | "
            f"balance(h>=1.1)={n_balance} ef>0: {'OK' if ok_balance else 'FAIL'} | "
            f"expected eff/std ratio={exp_ratio:.3f}",
            flush=True,
        )
        if not (ok_standup and ok_balance):
            print("  [ef-verify] FAIL: ef sign inconsistent with h_torso phase",
                  flush=True)

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        max_pots = []
        final_pots = []
        success_count = 0
        n_agents = 0
        step_success_count = 0  # agents that demonstrated stepping

        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue

            for agent_id, foot_key, phi4stage_key, _ in self._AGENT_OBS:
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

                # --- Stepping detection ---
                # After entering BALANCE phase, both feet must have
                # lifted above step_lift_threshold at some point.
                step_ok = self._check_stepping(ep, foot_key, phi4stage_key, T)
                if step_ok:
                    step_success_count += 1

        n = max(len(max_pots), 1)
        mean_max_pot = sum(max_pots) / n if max_pots else 0.0
        mean_final_pot = sum(final_pots) / n if final_pots else 0.0
        success_rate = success_count / n
        step_success_rate = step_success_count / n if n_agents else 0.0

        self._success_rate = success_rate

        # --- Floor disable: stepping-based ---
        # An update is "stepping-successful" if ALL eval episodes (all
        # agents) demonstrated stepping.  After step_confirm_updates
        # consecutive successes, permanently disable the floor loss.
        if not self._floor_disabled:
            if n_agents > 0 and step_success_count == n_agents:
                self._step_success_streak += 1
                if self._step_success_streak >= self.step_confirm_updates:
                    self._floor_disabled = True
                    self._floor_disabled_at = update
                    print(
                        f"  [floor] stepping success for "
                        f"{self.step_confirm_updates} consecutive evals; "
                        f"disabling floor loss at update {update}",
                        flush=True,
                    )
            else:
                self._step_success_streak = 0

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
                "step": round(step_success_rate, 3),
            },
        }

    @staticmethod
    def _check_stepping(
        episode, foot_key: str, phi4stage_key: str, T: int,
    ) -> bool:
        """Check if the agent demonstrated stepping in this episode.

        Stepping = after entering the BALANCE phase, BOTH feet must
        have lifted above ``step_lift_threshold`` at some point.

        Uses ``_compute_phase_mask`` on h_torso (from the 4-stage
        rewarder) to find BALANCE frames, then checks max foot heights
        within those frames.
        """
        # --- h_torso for phase determination ---
        h_torso = extract_per_step_field(
            episode.observer_outputs, phi4stage_key, "h_torso", T,
        )
        if h_torso is None:
            return False
        h_torso = np.asarray(h_torso[:T], dtype=np.float32)

        # --- Foot heights ---
        h_left = extract_per_step_field(
            episode.observer_outputs, foot_key, "h_left_foot", T,
        )
        h_right = extract_per_step_field(
            episode.observer_outputs, foot_key, "h_right_foot", T,
        )
        if h_left is None or h_right is None:
            return False
        h_left = np.asarray(h_left[:T], dtype=np.float32)
        h_right = np.asarray(h_right[:T], dtype=np.float32)

        # --- BALANCE phase mask ---
        balance_mask = StandupStepV3._compute_phase_mask(h_torso, T)
        if not balance_mask.any():
            return False  # never reached BALANCE

        # --- Both feet must have lifted above threshold ---
        max_h_left = float(h_left[balance_mask].max())
        max_h_right = float(h_right[balance_mask].max())
        return (
            max_h_left >= StandupStepV3.step_lift_threshold
            and max_h_right >= StandupStepV3.step_lift_threshold
        )

    def state(self) -> dict:
        return {
            "best_potential": self._best_potential,
            "success_rate": self._success_rate,
            "floor_disabled": self._floor_disabled,
            "floor_disabled_at": self._floor_disabled_at,
            "step_success_streak": self._step_success_streak,
        }

    def load_state(self, state: dict) -> None:
        self._best_potential = float(state.get("best_potential", -1.0))
        self._success_rate = float(state.get("success_rate", 0.0))
        self._floor_disabled = bool(state.get("floor_disabled", False))
        self._floor_disabled_at = int(state.get("floor_disabled_at", -1))
        self._step_success_streak = int(state.get("step_success_streak", 0))


EXPERIMENT_CLASS = StandupStepV3
