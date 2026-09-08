"""V3 end-to-end: standup (pretrained) + balance with per-foot stepping.

from random fallen state → stand up → maintain balance + stepping.

This experiment is designed to be **resumed from a pretrained standup
checkpoint** (``--resume-from <standup_ckpt> --reset-update``).  The
standup policy is already converged (std_mean≈0.15, uncertainty≈0.17),
so the key challenge is **re-injecting exploration** so the policy can
discover stepping behaviour without forgetting how to stand.

Two reward phases with hard switch based on torso height:

  STANDUP phase (h_torso < plateau):
    r_potential = (1-γ) × φ_4stage = 0.01 × φ_4stage,  weight = 1.0
    (same as exp_standup.py — pure 4-stage standing potential)

  BALANCE phase (h_torso >= plateau):
    r_fall       = 0.01 × φ_height,         weight = 1.0 (fixed)
    r_left_foot  = clip(h_left,  -0.3, 0.3), weight = stepping state machine (W=1.0)
    r_right_foot = clip(h_right, -0.3, 0.3), weight = stepping state machine (W=1.0)

  Phase transitions (per agent, per step):
    STANDUP → BALANCE:  plateau detection on h_torso
    BALANCE → STANDUP:  h_torso < 0.70  (fallen)

Four reward channels (each with independent critic):
  r_potential — reward always present, aw=1.0 in STANDUP, 0 in BALANCE
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

Exploration re-injection (two-stage floor scheduling)
----------------------------------------------------
The pretrained standup policy has low std (≈0.15) and uncertainty (≈0.17).
Without re-injection, the policy is too deterministic to discover stepping.
We use the framework's built-in exploration controls:

  explore_factor = 0.0  →  no rollout σ scaling
    Rollout uses the policy's own σ directly.  Exploration is driven
    entirely by the two-stage uncertainty floor below, which controls
    the policy's own σ through training-side loss.  This keeps
    rollout noise and policy σ coupled — when the floor is disabled
    in phase 2, σ can tighten without any residual rollout inflation.

  Two-stage uncertainty floor:
    Phase 1 (floor active):
      uncertainty_floor = 0.35, uncertainty_coef = 5.0
      The floor loss pulls the policy's own uncertainty up from the
      converged standup value (≈0.17) toward 0.35, opening up σ enough
      to explore foot-lifting actions.

    Phase 2 (floor disabled, one-way switch):
      Once uncertainty has stayed at/above the floor for
      `floor_confirm_updates` (default 5) consecutive updates, the
      floor loss is permanently disabled.  PPO then learns purely
      from the advantage signal and σ can naturally tighten as the
      policy becomes confident about stepping.  This prevents the
      floor from holding the policy in an over-random state that
      blocks fine action learning.

  learning_rate = 1e-4  (same as standup)
    PPO updates are moderated by the uncertainty floor — no need to
    slow down the LR separately.

No imbalance termination — robot can fall and get back up.
Every step is trainable.

Blueprint: baseline/humanoid21/end2end/standup_step_v3_env.yaml

Usage (resuming from pretrained standup checkpoint):

  PYTHONPATH=. python3 baseline/framework/train.py \\
    --experiment standup_step_v3 --algo ppo \\
    --resume-from baseline/runs/train_standup_ppo_<...>/checkpoints/checkpoint_u01200.pt \\
    --reset-update --background
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


class StandupStepV3(CombatExperimentPPOBase):
    """End-to-end standup + balance with phase-switched reward.

    Dual-agent: both robots get RandomFallenStatePlugin and train
    simultaneously.  No early termination — robot can fall and recover.

    Designed to resume from a pretrained standup checkpoint with
    exploration re-injection (see module docstring).
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

    # --- r_fall actor weight (balance phase) ---
    # Fixed weight — no curriculum.  The balance survival reward is
    # always active during BALANCE phase, coexisting with foot rewards.
    r_fall_actor_weight: float = 1.0

    # --- r_potential actor weight (standup phase) ---
    # 1.0: needed to preserve standup behaviour.  Setting it to 0.0
    # caused the policy to forget how to stand (final_pot dropped from
    # 0.999 to 0.618 in 60 updates).  The pretrained weights alone are
    # not enough — PPO updates drift the mean action without a reward
    # anchor.  1.0 is a moderate value that preserves standup while
    # letting the foot channels (weight 5.0) dominate during BALANCE.
    r_potential_actor_weight: float = 1.0

    # --- Foot reward scaling ---
    # 0.30: 6x larger than the original 0.05.  The foot height reward
    # needs to be strong enough that even small foot lifts produce a
    # clear advantage signal.  With clip=0.30, a 1cm lift gives
    # reward=0.01, and a 30cm lift gives reward=0.30.
    foot_height_clip: float = 0.30

    # --- Foot actor weight override ---
    # The stepping state machine uses FOOT_WEIGHT=1.0 by default.
    foot_weight_override: float = 1.0

    # --- Double grace override ---
    # The state machine default is 6 steps (0.3s @ 20Hz).  We reduce it
    # to 2 steps so the foot-lifting encouragement starts almost
    # immediately when the robot enters the BALANCE phase, giving the
    # policy more time per episode to discover stepping.
    double_grace_override: int = 2

    # --- Env ---
    env_blueprint = ""  # overridden via _env_pb()
    agent_used = "both"
    max_steps: int = 400  # standup ~100 + balance/stepping ~300

    # Observer keys: (agent_id, foot_key, phi4stage_key, phi_height_key)
    _AGENT_OBS = (
        ("robot_a", "foot_state_a", "standing_balance_a", "height_phi_a"),
        ("robot_b", "foot_state_b", "standing_balance_b", "height_phi_b"),
    )
    _AGENT_IDS = ("robot_a", "robot_b")

    # --- Exploration (re-injection for pretrained standup policy) ---
    # 0.0 → no rollout σ scaling.  Exploration is driven entirely by
    # the two-stage uncertainty floor (uncertainty_floor/coef), which
    # controls the policy's own σ through training-side loss.  This
    # keeps rollout noise and policy σ coupled: when the floor is
    # disabled in phase 2, σ can tighten without residual rollout
    # inflation.
    explore_factor: float = 0.0
    # Phase-1 floor: pulls uncertainty up from standup value (≈0.17)
    # toward 0.35 so the policy can explore foot-lifting actions.
    # Disabled permanently once uncertainty reaches `phase2_threshold`
    # for `floor_confirm_updates` consecutive updates (see on_update /
    # exploration overrides below).
    # 5.0 → with quadratic hinge, GradDiag showed coef=0.1 gave
    # floor/pol ratio=0.02x (still dominated by policy gradient).
    # dU/dlog_std is inherently small for TruncatedNormalPolicy, so
    # coef must be large to compensate.  At 5.0 the ratio should
    # reach ~1.0x, giving floor enough leverage to hold σ up.
    uncertainty_coef: float = 5.0

    # --- PPO tuning ---
    learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    target_kl: float = 0.05
    update_epochs: int = 4
    minibatch_size: int = 4096

    # --- Rollout schedule ---
    episodes_per_update: int = 512
    max_updates: int = 3000
    eval_interval: int = 5
    eval_episodes: int = 64

    # --- Video recording ---
    video_eval_interval: int = 5

    # --- Two-stage floor scheduling ---
    # Phase 1: floor loss active, pulls uncertainty up from standup value
    #          (≈0.17) toward `uncertainty_floor` (0.35).
    # Phase 2: once uncertainty has stayed at/above `phase2_threshold`
    #          for `floor_confirm_updates` consecutive updates, the floor
    #          loss is permanently disabled so PPO can focus on learning
    #          stepping action and naturally tighten σ.
    # One-way switch: once disabled, stays disabled (no re-arming).
    #
    # Why phase2_threshold < uncertainty_floor:
    #   The floor loss is a quadratic hinge `coef * relu(floor - U)^2`,
    #   whose gradient `2*coef*(floor - U)` vanishes as U approaches floor.
    #   Near the target the floor push becomes too weak to overcome PPO
    #   gradient, so U typically plateaus below floor.  Decoupling the
    #   trigger threshold from the floor lets phase 2 fire at a level
    #   the floor can actually reach.
    uncertainty_floor: float = 0.35
    phase2_threshold: float = 0.30
    floor_confirm_updates: int = 5

    # --- Stateful metrics ---
    _best_potential: float = -1.0
    _success_rate: float = 0.0
    _best_step_metric: float = -1.0
    _uncertainty_history: list = None  # list[float], recent uncertainty
    _floor_disabled: bool = False
    _floor_disabled_at: int = -1  # update index when floor was disabled

    # ------------------------------------------------------------------
    # Blueprint loading
    # ------------------------------------------------------------------

    def on_update(self, stats, update: int) -> None:
        """Track uncertainty for two-stage floor scheduling.

        Appends the policy's uncertainty to ``_uncertainty_history``.
        Once uncertainty has stayed at/above ``phase2_threshold`` for
        ``floor_confirm_updates`` consecutive updates, sets
        ``_floor_disabled=True`` (one-way switch).

        Note: ``phase2_threshold`` is intentionally below
        ``uncertainty_floor`` because the quadratic hinge
        ``coef * relu(floor - U)^2`` has vanishing gradient as U
        approaches floor, so U typically plateaus below floor.
        """
        if self._uncertainty_history is None:
            self._uncertainty_history = []
        u = float(stats.policy_stats.get("uncertainty", 0.0))
        self._uncertainty_history.append(u)
        if self._floor_disabled:
            return
        window = self._uncertainty_history[-self.floor_confirm_updates:]
        if (
            len(window) >= self.floor_confirm_updates
            and all(v >= self.phase2_threshold for v in window)
        ):
            self._floor_disabled = True
            self._floor_disabled_at = update
            print(
                f"  [floor] uncertainty reached phase2_threshold="
                f"{self.phase2_threshold} for {self.floor_confirm_updates} "
                f"consecutive updates (u={u:.4f}); disabling floor loss "
                f"at update {update}",
                flush=True,
            )

    def exploration(self, update: int) -> ExplorationSpec:
        """Two-stage exploration spec.

        Phase 1 (floor active): returns the configured floor/coef so the
        floor loss pulls uncertainty up.
        Phase 2 (floor disabled): returns zero floor/coef so PPO learns
        purely from advantage signal and σ can naturally tighten.
        """
        if self._floor_disabled:
            return ExplorationSpec(uncertainty_floor=0.0, uncertainty_coef=0.0)
        return ExplorationSpec(
            uncertainty_floor=self.uncertainty_floor,
            uncertainty_coef=self.uncertainty_coef,
        )

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
        bp_path = Path(__file__).resolve().parent.parent / "humanoid21" / "end2end" / "standup_step_v3_env.yaml"
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
        startup_bias: bool = True,
        startup_bias_steps: int = 40,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance-gated foot weights.

        Delegates to ``stepping_state_machine.compute_foot_weights`` on each
        contiguous BALANCE segment.  Non-BALANCE (STANDUP) frames get zero
        weight and the state machine resets at each segment boundary.

        **Startup bias**: When enabled, the first ``startup_bias_steps``
        frames of each BALANCE segment get an asymmetric weight pattern
        that alternates between encouraging left and right foot lifting
        every 5 steps.  This breaks the chicken-and-egg problem where
        the policy never enters a support phase because it never lifts
        a foot.  The alternating pattern gives the policy a clear,
        time-varying signal to try lifting one foot at a time.

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

                # --- Startup bias: alternating foot-lift encouragement ---
                # During the first N steps of each BALANCE segment, for
                # any frame where the robot is in double support (both
                # feet down), override with an alternating pattern that
                # strongly encourages one foot at a time.  This breaks
                # the chicken-and-egg problem where the policy never
                # enters a support phase because it never lifts a foot.
                if startup_bias and seg_len > 0:
                    n_bias = min(seg_len, startup_bias_steps)
                    both_down = cl[:n_bias].astype(bool) & cr[:n_bias].astype(bool)
                    for i in range(n_bias):
                        if both_down[i]:
                            # Alternate every 5 steps: lift left, then right
                            if (i // 5) % 2 == 0:
                                wl[i] = weight * 2.0  # strong lift left
                                wr[i] = -weight * 0.5  # slight press right
                            else:
                                wr[i] = weight * 2.0  # strong lift right
                                wl[i] = -weight * 0.5  # slight press left

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
            weight=self.foot_weight_override,
            double_grace_steps=self.double_grace_override,
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
    # Eval
    # ------------------------------------------------------------------

    @staticmethod
    def _count_steps(
        contact_l: np.ndarray, contact_r: np.ndarray, T: int,
    ) -> int:
        """Count the number of gait steps (support transitions) in a segment.

        A "step" is a transition from SUPPORT_L to SUPPORT_R or vice versa,
        passing through DOUBLE or FLIGHT.  We count the number of times
        the support foot changes.
        """
        if T == 0:
            return 0
        steps = 0
        prev_support = None  # 'L' or 'R'
        for t in range(T):
            cl = bool(contact_l[t])
            cr = bool(contact_r[t])
            if cl and not cr:
                cur = 'L'
            elif cr and not cl:
                cur = 'R'
            else:
                cur = None  # DOUBLE or FLIGHT

            if cur is not None and cur != prev_support:
                if prev_support is not None:
                    steps += 1
                prev_support = cur
        return steps

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        max_pots = []
        final_pots = []
        max_hs = []
        step_counts = []
        balance_fracs = []
        success_count = 0
        n_agents = 0

        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue

            for agent_id, foot_key, phi4stage_key, _ in self._AGENT_OBS:
                n_agents += 1

                # --- Standup metrics ---
                phi = extract_per_step_field(
                    ep.observer_outputs, phi4stage_key, "potential", T,
                )
                h_torso = extract_per_step_field(
                    ep.observer_outputs, phi4stage_key, "h_torso", T,
                )
                if phi is not None and len(phi) > 0:
                    mx = float(np.max(phi))
                    fn = float(phi[-1])
                else:
                    mx = 0.0
                    fn = 0.0
                max_pots.append(mx)
                final_pots.append(fn)

                if h_torso is not None and len(h_torso) > 0:
                    max_hs.append(float(np.max(h_torso)))
                else:
                    max_hs.append(0.0)

                if mx >= 0.9:
                    success_count += 1

                # --- Phase mask ---
                if h_torso is not None and len(h_torso) > 0:
                    h_arr = np.asarray(h_torso[:T], dtype=np.float64)
                    bmask = self._compute_phase_mask(h_arr, T)
                else:
                    bmask = np.zeros(T, dtype=bool)
                balance_fracs.append(float(bmask.sum()) / max(T, 1))

                # --- Stepping metrics (only in BALANCE phase) ---
                try:
                    contact_l = self._extract_foot_field(ep, foot_key, "left_foot_contact", T)
                    contact_r = self._extract_foot_field(ep, foot_key, "right_foot_contact", T)
                    # Count steps in BALANCE segments only
                    total_steps = 0
                    seg_start = 0
                    for t in range(T + 1):
                        in_seg = t < T and bool(bmask[t])
                        seg_active = t > seg_start and (t == T or not in_seg)
                        if seg_active:
                            seg_len = t - seg_start
                            cl = np.asarray(contact_l[seg_start:t], dtype=bool)
                            cr = np.asarray(contact_r[seg_start:t], dtype=bool)
                            total_steps += self._count_steps(cl, cr, seg_len)
                        if t < T and not in_seg:
                            seg_start = t + 1
                    step_counts.append(total_steps)
                except KeyError:
                    step_counts.append(0)

        n = max(len(max_pots), 1)
        mean_max_pot = sum(max_pots) / n if max_pots else 0.0
        mean_final_pot = sum(final_pots) / n if final_pots else 0.0
        mean_max_h = sum(max_hs) / n if max_hs else 0.0
        mean_steps = sum(step_counts) / n if step_counts else 0.0
        mean_balance_frac = sum(balance_fracs) / n if balance_fracs else 0.0
        success_rate = success_count / n

        self._success_rate = success_rate

        is_new_best = mean_max_pot > self._best_potential
        if is_new_best:
            self._best_potential = mean_max_pot

        # Track best step metric separately
        is_best_steps = mean_steps > self._best_step_metric
        if is_best_steps:
            self._best_step_metric = mean_steps

        return {
            "is_new_best": is_new_best,
            "stop_training": False,
            "info": {
                "max_pot": round(mean_max_pot, 3),
                "final_pot": round(mean_final_pot, 3),
                "max_h": round(mean_max_h, 3),
                "success": round(success_rate, 3),
                "steps": round(mean_steps, 1),
                "bal_frac": round(mean_balance_frac, 3),
            },
        }

    def state(self) -> dict:
        return {
            "best_potential": self._best_potential,
            "success_rate": self._success_rate,
            "best_step_metric": self._best_step_metric,
        }

    def load_state(self, state: dict) -> None:
        self._best_potential = float(state.get("best_potential", -1.0))
        self._success_rate = float(state.get("success_rate", 0.0))
        self._best_step_metric = float(state.get("best_step_metric", -1.0))


EXPERIMENT_CLASS = StandupStepV3
