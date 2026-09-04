"""V3 end-to-end: standup (pretrained) + balance with per-foot stepping.

from random fallen state → stand up → maintain balance + stepping.

This experiment is designed to be **resumed from a pretrained standup
checkpoint** (``--resume-from <standup_ckpt> --reset-update``).  The
standup policy is already converged (std_mean≈0.18, entropy≈-7), so
the key challenge is **re-injecting exploration** so the policy can
discover stepping behaviour without forgetting how to stand.

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

Exploration re-injection
------------------------
The pretrained standup policy has very low std (≈0.18) and negative
entropy (≈-7 nats).  Without re-injection, the policy is too deterministic
to discover stepping.  We use:

  explore_intensity = 0.75  →  σ × exp(0.5 × 2.0) = σ × 2.72
    This roughly triples the effective std during rollout, giving the
    policy enough noise to try lifting feet while still being grounded
    in the standup behaviour.

  entropy_floor = 0.35
    Prevents the policy from collapsing back to pure-standup during
    training.  The floor is set above the converged standup entropy
    (≈0.30) so the policy is pushed to maintain *more* entropy than
    pure standing requires.

  entropy_coef = 0.01
    Standard coefficient for the floor hinge loss.

  learning_rate = 5e-5  (half of standup's 1e-4)
    Slower updates to preserve the standup behaviour while learning
    the new stepping skill.

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

    # --- Foot height reward saturation ---
    foot_height_clip: float = 0.05

    # --- r_fall actor weight (balance phase) ---
    # This is the INITIAL value; the curriculum schedule in on_update()
    # ramps it up as the policy learns to step.  Starting at 0 removes
    # the conservative bias that prevents foot lifting.
    r_fall_actor_weight: float = 0.0

    # --- r_fall curriculum target ---
    # The final r_fall weight after the curriculum ramp completes.
    r_fall_target_weight: float = 3.0

    # --- r_fall curriculum ramp ---
    # r_fall weight ramps from 0 to r_fall_target_weight over this many
    # updates, starting after the stepping curriculum warmup.  This gives
    # the policy time to learn stepping without the conservative bias of
    # the balance reward, then gradually reintroduces stability.
    r_fall_ramp_updates: int = 800

    # --- Stepping curriculum warmup ---
    # For the first N updates, r_fall weight = 0 and foot weight is at
    # its full value.  This forces the policy to learn stepping first.
    # After N updates, r_fall begins ramping up.
    stepping_warmup_updates: int = 200

    # --- r_potential actor weight (standup phase) ---
    # Reduced from 3.0 to 0.5: the standup behaviour is already learned
    # (pretrained), so the standup reward only needs a weak signal to
    # prevent forgetting.  This lets the stepping channels dominate
    # the policy gradient during the BALANCE phase.
    r_potential_actor_weight: float = 0.5

    # --- Foot reward scaling ---
    # 0.30: 6x larger than the original 0.05.  The foot height reward
    # needs to be strong enough that even small foot lifts produce a
    # clear advantage signal.  With clip=0.30, a 1cm lift gives
    # reward=0.01, and a 30cm lift gives reward=0.30.
    foot_height_clip: float = 0.30

    # --- Foot actor weight override ---
    # The stepping state machine uses FOOT_WEIGHT=1.0 by default.  We
    # override it to 5.0 to make the stepping gradient dominant during
    # the BALANCE phase.
    foot_weight_override: float = 5.0

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
    # 0.85 → σ × exp(0.7 × 2.0) = σ × 2.01, very strong noise.
    # The policy needs enough randomness to accidentally lift a foot and
    # discover the foot reward.  σ×2.0 means the effective σ is about
    # 0.35 (vs native 0.17), which should produce occasional large
    # enough deviations in leg joints to lift a foot off the ground.
    explore_intensity: float = 0.85
    # 0.40 → prevents collapse back to pure-standup entropy (≈0.29).
    # Higher than before because the stronger exploration (0.85) means
    # the policy needs a higher floor to not collapse back when the
    # exploration noise is removed during training updates.
    entropy_floor: float = 0.40
    # 0.05 → moderate coefficient, enough to counteract PPO's natural
    # entropy reduction without dominating the gradient.
    entropy_coef: float = 0.05

    # --- Sigma bounds (match standup training) ---
    log_std_min: float = -2.5
    log_std_max: float = 0.0

    # --- PPO tuning ---
    # Lower LR to preserve standup behaviour while learning stepping.
    learning_rate: float = 5e-5
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

    # --- Stateful metrics ---
    _best_potential: float = -1.0
    _success_rate: float = 0.0
    _best_step_metric: float = -1.0

    # ------------------------------------------------------------------
    # Blueprint loading
    # ------------------------------------------------------------------

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
            mode=None,
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
    # Curriculum: r_fall weight schedule
    # ------------------------------------------------------------------

    def _current_r_fall_weight(self, update: int) -> float:
        """Compute the r_fall actor weight for the given update number.

        Phase 1 (update < stepping_warmup_updates): r_fall = 0.
            The policy learns to lift feet without any stability bias.
        Phase 2 (warmup ≤ update < warmup + ramp): linear ramp from 0
            to r_fall_target_weight.  Stability is gradually reintroduced.
        Phase 3 (update ≥ warmup + ramp): r_fall = r_fall_target_weight.
            Full stability constraint, stepping must coexist with balance.
        """
        if update < self.stepping_warmup_updates:
            return 0.0
        ramp_start = self.stepping_warmup_updates
        ramp_end = self.stepping_warmup_updates + self.r_fall_ramp_updates
        if update >= ramp_end:
            return self.r_fall_target_weight
        frac = (update - ramp_start) / self.r_fall_ramp_updates
        return self.r_fall_target_weight * frac

    def on_update(self, stats, update: int) -> None:
        """Curriculum hook: update r_fall_actor_weight based on update number."""
        new_weight = self._current_r_fall_weight(update)
        if abs(new_weight - self.r_fall_actor_weight) > 1e-6:
            print(
                f"[curriculum] update={update} r_fall_weight: "
                f"{self.r_fall_actor_weight:.3f} → {new_weight:.3f}",
                flush=True,
            )
            self.r_fall_actor_weight = new_weight

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
