"""V2 end-to-end stepping experiment: standup + per-foot stepping.

From random fallen state → stand up → step alternately.
Built on top of the standup policy (warm-start via --resume-from).

Based on ``exp_standup_step.py``, replacing the sparse ``r_cross`` channel
(CrossSupportBalanceRewarder) with two dense per-foot height channels
using the v2 stepping state machine::

    r_potential  = 0.01 × φ(t),             γ=0.99, aw = 3.0 × (1 - φ_trail²·ss)
    r_left_foot  = clip(sole_l,  0, 0.05),   γ=0.90, aw = clock weights × φ²
    r_right_foot = clip(sole_r, 0, 0.05),   γ=0.90, aw = clock weights × φ²

The reward carries only *physical fact* — sole clearance (min capsule-
endpoint z − radius), which stays ~0 while any sole edge is at ground
level.  Run 064853 showed midpoint ``h`` is gameable: rocking the foot
on its toe/side edge lifts the midpoint >5cm without ever leaving the
ground; sole clearance cannot be faked that way.  The *intent* (which
foot should rise / descend right now) is carried BOTH by ``actor_weight``
AND by the observation: ``GaitClockSimulator`` appends cmd_L / cmd_R /
window-progress dims so the command is a learnable state feature.  The
reward side replays the identical schedule via ``clock_foot_weights``
(the u01651 dump showed invisible +W intents convert at only ~0.5%).

φ² gating on the foot channels ensures stepping is only rewarded after
the robot is standing.  r_potential is always active (fixed aw = 3.0).

Exploration is phase-dependent via per-frame ``explore_factor``:
σ×0.5 while low (protect the warm-started standup skill), σ×2.0 while
standing (the u50 dump showed +W lift intents convert to real lifts only
~0.5% of the time — isotropic σ≈0.30 never produces the ~5 cm
coordinated hip+knee excursion needed to create positive advantage).

No imbalance termination — the robot can fall and get back up.
Every step is trainable (like standup, not like basic_balance).

Blueprint: baseline/humanoid21/end2end/step_env.yaml
"""
from __future__ import annotations

import math as _math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.rollout import extract_per_step_field

from baseline.humanoid21.end2end.stepping_state_machine import (
    CONTACT_HOLD_STEPS,
    GAIT_PERIOD,
    clock_foot_weights,
    count_falls,
    detect_step_cycles,
    single_support_mask,
)

from .base import CombatExperimentPPOBase


# --- Phase-dependent explore_factor (per-frame callable) ---
# obs[45] = h_torso (root body z, meters).  σ multiplier = 3^ef.
# low      (h < 1.1): ef = -0.631 → σ × 0.5 (protect standup skill)
# standing (h ≥ 1.1): ef = +0.631 → σ × 2.0 (explore stepping)
# Must be top-level for multiprocessing picklability.
_EF_LOW = _math.log(0.5) / _math.log(3)    # ≈ -0.631
_EF_STAND = _math.log(2.0) / _math.log(3)  # ≈ +0.631
_H_EF_SWITCH = 1.1


def _phase_explore_factor(obs, step):
    """Per-frame explore_factor: quiet while low, loud while standing."""
    return _EF_STAND if float(obs[45]) >= _H_EF_SWITCH else _EF_LOW


class Step(CombatExperimentPPOBase):
    """End-to-end stepping: standup + per-foot stepping state machine.

    Dual-agent: both robots get RandomFallenStatePlugin and train
    simultaneously.  No early termination — robot can fall and recover.
    """

    name = "step"

    # --- Network ---
    # 96 base dims + 3 gait-clock dims (cmd_L / cmd_R / window progress)
    # appended by GaitClockSimulator — the stepping command is an
    # observable state feature, not invisible advantage shaping.
    obs_dim: int = 99
    action_dim: int = 21

    # --- Reward channels ---
    _channel_names = ("r_potential", "r_left_foot", "r_right_foot",
                      "r_torso")
    _channel_gammas = {
        "r_potential": 0.99,
        "r_left_foot": 0.90,
        "r_right_foot": 0.90,
        "r_torso": 0.99,
    }
    _gae_lambda: float = 0.95

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01

    # --- Foot height reward saturation (overrides stepping_state_machine default) ---
    foot_height_clip: float = 0.05

    # --- Eval stepping detection ---
    # A swing attempt: one foot airborne (debounced contact) for
    # >= min_air_frames while the other foot stays down, starting on a
    # standing frame (φ >= step_phi_gate).  A valid cycle additionally
    # reaches step_lift_threshold and lands on a standing frame.
    # See stepping_state_machine.detect_step_cycles.
    step_lift_threshold: float = 0.05
    step_phi_gate: float = 0.9
    step_min_air_frames: int = 3

    # --- Step-cycle completion bonus ---
    # The u50 dump showed the payoff ratio between a real step and a
    # micro-hop is only ~1.07× (dense clip(h,0,0.05) shaping), while the
    # residual veto on >5 cm frames keeps the mean policy at hmax≈9 mm.
    # A per-cycle bonus (spread over the airborne window) raises that
    # ratio to ~50× and aligns the reward with the eval metric exactly:
    # the same detect_step_cycles definition decides both.
    step_cycle_bonus: float = 0.5

    # --- Lift stability gate ---
    # The clock commands a lift every window regardless of balance; the
    # u1801 dump showed ~1.16 real falls per 20 s trajectory, 66% of
    # them right after a genuine swing — the policy lifts on schedule
    # even when wobbly (liftoff φ p5 = 0.07).  Below lift_phi_gate the
    # commanded foot's +W flips to -W so wobbly lifts are punished, not
    # just unrewarded.  0.8 splits the bimodal swing-φ distribution
    # (healthy swing ≈0.9-0.97 vs collapse <0.5).
    lift_phi_gate: float = 0.8

    # --- Residual potential pressure during healthy swings ---
    # Full exemption (exempt_frac=1.0) left zero "stay balanced" policy
    # gradient on stable single-support frames — falls during swing then
    # only get punished *after* φ already collapsed.  0.75 keeps a mild
    # aw≈0.75 potential pressure on healthy swings; collapsing frames
    # (φ<gate) always get the full aw.
    swing_exempt_frac: float = 0.75

    # --- Torso sway penalty (r_torso) ---
    # User video review: genuine lifts but violent torso sway and still
    # frequent falls.  u2100 dump: roll/pitch |ω| median ≈2.0 rad/s even
    # while "standing still" — the warm-started policy wobbles as its
    # native style, and stepping amplifies it.  Penalize the squared
    # sqrt-compressed obs dims (v² ∝ |ω|) on stable frames only —
    # recovery flailing while fallen stays unpunished.
    # coef dose-response: 0.008 even with the floor still crushed
    # stepping (v5: step 0.73, solepk 0.044) — this policy's gait
    # fundamentally *needs* sway for CoM transfer, so only a gentle
    # pressure is tolerable.
    torso_sway_coef: float = 0.003
    torso_actor_weight: float = 1.0
    # Flat penalty (v4: sway² · coef) crushed stepping — solepk fell to
    # 0.038 and step to 0.64 because natural gait sway was punished too.
    # Only the violent tail is noise: penalize sway ABOVE a floor so
    # ordinary stepping sway is free (u2100 dump: standing p50=1.28,
    # p75=1.85 in v² units).
    torso_sway_floor: float = 1.5

    # --- r_potential actor weight ---
    # Fixed 3.0 in general, partially exempted on stable single-support
    # frames (see swing_exempt_frac) and fully restored the moment
    # φ drops below lift_phi_gate: the u50 dump showed the potential
    # channel vetoes every real lift (combined adv -0.18/-0.73 on h>3cm
    # frames — the transient φ dip is punished 30× harder than the foot
    # channel rewards the lift), while the u1801 dump showed a pure
    # trailing-max gate keeps the veto off through entire collapses.
    r_potential_actor_weight: float = 3.0

    # --- Env ---
    env_blueprint = ""  # overridden via _env_pb()
    agent_used = "both"
    max_steps: int = 400

    _AGENT_OBS = (
        ("robot_a", "foot_state_a", "standing_balance_a"),
        ("robot_b", "foot_state_b", "standing_balance_b"),
    )
    _AGENT_IDS = ("robot_a", "robot_b")

    # --- PPO tuning (conservative for warm-start) ---
    learning_rate: float = 3e-5
    critic_learning_rate: float = 3e-4
    target_kl: float = 0.03
    update_epochs: int = 4
    minibatch_size: int = 4096
    # --- Exploration (aligned with exp_standup_floor04) ---
    # floor=0.3/coef=1e-3 was inert: the ckpt's U≈0.46 sits above the
    # hinge and coef was 3 orders too weak to resist σ collapse.  The
    # step task needs sustained exploration pressure — floor=0.4 engages
    # almost immediately and coef=1.0 actually bites.
    uncertainty_floor: float = 0.4
    uncertainty_coef: float = 1.0

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
    _best_quality: float = -1.0
    _best_survived: float = -1.0
    _success_rate: float = 0.0
    _last_best_update: int = 0

    # ------------------------------------------------------------------
    # Blueprint loading
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
        bp_path = Path(__file__).resolve().parent.parent / "humanoid21" / "end2end" / "step_env.yaml"
        return ParameterizedEnvBlueprint.load(bp_path)

    def build_actor(self, device):
        """Build actor with obs_dim=99 (96 base + 3 gait-clock dims).

        Warm-start compatible: ``load_checkpoint`` zero-pads the first
        layer's input columns, so the extended inputs start inert and
        the restored policy is initially identical to the standup ckpt.
        """
        from envs.framework.policy import PolicyBlueprint
        blueprint_dir = Path(__file__).resolve().parent.parent / "humanoid21" / "blueprints"
        bp = PolicyBlueprint.load(blueprint_dir / self.actor_blueprint)
        return bp.build(obs_dim=self.obs_dim).to(device)

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
    # Trajectory building
    # ------------------------------------------------------------------

    def _build_agent_trajectory(
        self,
        episode,
        agent_id: str,
        foot_key: str,
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

        # --- r_potential: 0.01 × φ(t) per step ---
        r_potential = (self.per_step_phi_coef * phi_arr).astype(np.float32)

        # --- r_torso: -coef × (v49² + v50²) per step ---
        # obs[49:52] is body angular velocity after sign·sqrt(|ω|/2)
        # compression, so v² = |ω|/2 — the squared obs dims are already
        # proportional to sway magnitude (roll+pitch; yaw is allowed).
        if obs_all is not None and obs_all.shape[1] >= 51:
            sway = (
                obs_all[:T_full, 49].astype(np.float32) ** 2
                + obs_all[:T_full, 50].astype(np.float32) ** 2
            )
            r_torso = (
                -self.torso_sway_coef
                * np.maximum(sway - self.torso_sway_floor, 0.0)
            ).astype(np.float32)
        else:
            r_torso = np.zeros(T_full, dtype=np.float32)

        # --- Foot heights / sole clearance (saturated) ---
        # Reward value uses sole clearance, NOT midpoint h: midpoint
        # rises when the foot pivots on its toe/side edge, which let
        # run 064853 "step" by rocking without ever leaving the ground.
        h_left = extract_per_step_field(
            episode.observer_outputs, foot_key, "h_left_foot", T_full,
        )
        h_right = extract_per_step_field(
            episode.observer_outputs, foot_key, "h_right_foot", T_full,
        )
        sole_left = extract_per_step_field(
            episode.observer_outputs, foot_key, "sole_clear_left", T_full,
        )
        sole_right = extract_per_step_field(
            episode.observer_outputs, foot_key, "sole_clear_right", T_full,
        )
        if sole_left is not None:
            r_left_foot = np.clip(
                np.asarray(sole_left[:T_full], dtype=np.float32),
                0.0, self.foot_height_clip,
            )
        else:
            r_left_foot = np.zeros(T_full, dtype=np.float32)
        if sole_right is not None:
            r_right_foot = np.clip(
                np.asarray(sole_right[:T_full], dtype=np.float32),
                0.0, self.foot_height_clip,
            )
        else:
            r_right_foot = np.zeros(T_full, dtype=np.float32)

        # --- Contacts → stepping state machine → foot actor weights ---
        contact_l = extract_per_step_field(
            episode.observer_outputs, foot_key, "left_foot_contact", T_full,
        )
        contact_r = extract_per_step_field(
            episode.observer_outputs, foot_key, "right_foot_contact", T_full,
        )

        # --- Step-cycle completion bonus ---
        # Same detector as eval: a swing reaching step_lift_threshold
        # that lands on a standing frame earns step_cycle_bonus spread
        # over its airborne window.
        if (
            self.step_cycle_bonus > 0
            and contact_l is not None and contact_r is not None
            and h_left is not None and h_right is not None
            and sole_left is not None and sole_right is not None
        ):
            det = detect_step_cycles(
                np.asarray(contact_l[:T_full], dtype=bool),
                np.asarray(contact_r[:T_full], dtype=bool),
                np.asarray(h_left[:T_full], dtype=np.float32),
                np.asarray(h_right[:T_full], dtype=np.float32),
                standing=phi_arr >= self.step_phi_gate,
                sole_left=np.asarray(sole_left[:T_full], dtype=np.float32),
                sole_right=np.asarray(sole_right[:T_full], dtype=np.float32),
                min_air_steps=self.step_min_air_frames,
                h_thresh=self.step_lift_threshold,
            )
            for foot, t_off, t_land, _h_pk in det["cycles"]:
                per_frame = np.float32(
                    self.step_cycle_bonus / max(1, t_land - t_off))
                if foot == "left":
                    r_left_foot[t_off:t_land] += per_frame
                else:
                    r_right_foot[t_off:t_land] += per_frame

        # --- Clock-driven foot commands (observable via obs[96:99]) ---
        # The commanded foot is a deterministic function of the frame
        # index — the policy sees it in obs, so +W intents become a
        # learnable state→action mapping.  Replaces the contact-reactive
        # state machine, whose commands converted to real lifts only
        # ~0.5% of the time (u01651 dump).  stable gates both the lift
        # command (below) and the swing exemption (aw_potential below).
        stable = (phi_arr >= self.lift_phi_gate)
        w_left, w_right = clock_foot_weights(
            T_full,
            sole_left=np.asarray(sole_left[:T_full], dtype=np.float32) if sole_left is not None else None,
            sole_right=np.asarray(sole_right[:T_full], dtype=np.float32) if sole_right is not None else None,
            period=GAIT_PERIOD,
            stable=stable,
        )

        # --- No early termination: robot can fall and get back up ---
        is_terminated = False

        # --- Actor weights: r_potential swing-exempt, foot channels φ² ---
        phi_sq = (phi_arr ** 2).astype(np.float32)
        if contact_l is not None and contact_r is not None:
            ss_mask = single_support_mask(
                np.asarray(contact_l[:T_full], dtype=bool),
                np.asarray(contact_r[:T_full], dtype=bool),
            ).astype(np.float32)
            # Dilate ±CONTACT_HOLD_STEPS: the debounce lags real liftoff/
            # touchdown by up to `hold` frames, so the raw ss_mask misses
            # the swing's boundary frames — exactly where h>thresh frames
            # showed up as !ss and kept the full r_potential veto (u50
            # dump: 334/780 of >5cm frames got aw=3 → contrib −0.26).
            if ss_mask.any():
                k = CONTACT_HOLD_STEPS
                pad_ss = np.pad(
                    ss_mask, (k, k), mode="edge")
                ss_mask = np.lib.stride_tricks.sliding_window_view(
                    pad_ss, 2 * k + 1,
                ).max(axis=1).astype(np.float32)
        else:
            ss_mask = np.zeros(T_full, dtype=np.float32)
        # Swing exemption requires the frame to be CURRENTLY stable
        # (φ ≥ lift_phi_gate), and even then is only PARTIAL: the old
        # trailing-max φ kept exempting ~15 frames into a collapse —
        # u1801 showed aw_potential ≈0.1 during the 15 frames preceding
        # mid-swing falls (79% of falls happen foot-still-airborne), so
        # the potential channel was silenced exactly while the fall
        # developed AND gave no "stay balanced during swing" gradient
        # on healthy swings.  Now: stable swing → aw = w·(1-exempt_frac)
        # (residual stability pressure), unstable → full veto.
        aw_potential = (
            self.r_potential_actor_weight
            * (1.0 - self.swing_exempt_frac * stable * ss_mask)
        ).astype(np.float32)
        # +W is φ²-gated (only encourage lifts while standing), but -W
        # must NOT be attenuated: punishment matters most exactly when φ
        # is low (wobbly lift-off / airborne-while-falling), and φ²
        # would shrink it to ~4% there — the stability gate's -W flip
        # would be toothless.  s42_stable u2020-2065 confirmed: falls
        # stayed ~1.8 while the gate was nominally active.
        aw_left = np.where(w_left > 0, w_left * phi_sq, w_left)
        aw_right = np.where(w_right > 0, w_right * phi_sq, w_right)
        actor_weights = {
            "r_potential": aw_potential,
            "r_left_foot": aw_left.astype(np.float32),
            "r_right_foot": aw_right.astype(np.float32),
            # Sway penalty only counts while standing — recovery flail
            # after a fall is legitimate motion, not gait noise.
            "r_torso": (self.torso_actor_weight * stable).astype(np.float32),
        }

        all_rewards = {
            "r_potential": r_potential,
            "r_left_foot": r_left_foot,
            "r_right_foot": r_right_foot,
            "r_torso": r_torso,
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

    def build_trajectories(self, episodes) -> List[Trajectory]:
        all_trajs: List[Trajectory] = []
        for episode in episodes:
            for agent_id, foot_key, phi_key in self._AGENT_OBS:
                trajs = self._build_agent_trajectory(
                    episode, agent_id, foot_key, phi_key,
                )
                all_trajs.extend(trajs)
        if not getattr(self, "_ef_verified", False):
            self._ef_verified = True
            self._verify_explore_factor_flow(all_trajs)
        return all_trajs

    # ------------------------------------------------------------------
    # Rollout jobs — phase-dependent explore_factor
    # ------------------------------------------------------------------

    def build_jobs(self, policy_bp, base_seed, n_episodes, *, stochastic=True):
        """Per-frame explore_factor: σ×0.5 while low, σ×2.0 while standing.

        Overrides the scalar ``self.explore_factor`` — the callable on
        obs[45] (h_torso) concentrates exploration on standing frames,
        where stepping must be discovered, and keeps the standup phase
        deterministic so the warm-started skill is not perturbed.
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
                explore_factor_a=_phase_explore_factor,
                explore_factor_b=_phase_explore_factor,
                stochastic=stochastic,
            ))
        return jobs

    def _verify_explore_factor_flow(self, trajs: List[Trajectory]) -> None:
        """One-time diagnostic: verify per-frame explore_factor data flow.

        Checks (first trajectory only): ef array exists, ef < 0 where
        obs[45] < 1.1 (low), ef > 0 where obs[45] >= 1.1 (standing).
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
        n_stand = int((h_obs >= _H_EF_SWITCH).sum())
        n_low = n - n_stand
        ok_low = bool(np.all(ef[h_obs < _H_EF_SWITCH] < 0)) if n_low else True
        ok_stand = bool(np.all(ef[h_obs >= _H_EF_SWITCH] > 0)) if n_stand else True
        exp_ratio = (n_stand * 2.0 + n_low * 0.5) / n
        print(
            f"  [ef-verify] T={n} ef_min={ef.min():.3f} ef_max={ef.max():.3f} | "
            f"low(h<1.1)={n_low} ef<0: {'OK' if ok_low else 'FAIL'} | "
            f"stand(h>=1.1)={n_stand} ef>0: {'OK' if ok_stand else 'FAIL'} | "
            f"expected eff/std ratio={exp_ratio:.3f}",
            flush=True,
        )
        if not (ok_low and ok_stand):
            print("  [ef-verify] FAIL: ef sign inconsistent with h_torso phase",
                  flush=True)

    # ------------------------------------------------------------------
    # Eval — track standing success
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        max_pots = []
        final_pots = []
        success_count = 0
        step_success_count = 0  # agents that demonstrated stepping
        all_cycles = []         # valid step cycles per agent
        all_swings = []         # swing attempts per agent
        all_hmax = []           # mean swing peak height per agent
        all_solepk = []         # mean swing peak sole clearance per agent
        all_alt = []            # alternation ratio per agent (>=2 cycles)
        all_falls = []          # real falls per agent (φ<0.5 for >=10 f)
        all_sway = []           # mean roll/pitch |ω| on stable frames
        n_agents = 0

        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue

            for agent_id, foot_key, phi_key in self._AGENT_OBS:
                n_agents += 1
                phi = extract_per_step_field(
                    ep.observer_outputs, phi_key, "potential", T,
                )
                if phi is not None and len(phi) > 0:
                    mx = float(np.max(phi))
                    fn = float(phi[-1])
                    all_falls.append(count_falls(phi))
                    obs_arr = ep.observations.get(agent_id)
                    if obs_arr is not None and len(obs_arr) >= T:
                        v = np.asarray(obs_arr[:T, 49:51],
                                       dtype=np.float32)
                        # |ω_axis| = 2·v² (undo sign·sqrt(|ω|/2));
                        # report |ωx|+|ωy| — monotone sway magnitude.
                        sway_mag = 2.0 * (v[:, 0] ** 2 + v[:, 1] ** 2)
                        m = np.asarray(phi) >= 0.8
                        if m.any():
                            all_sway.append(float(sway_mag[m].mean()))
                else:
                    mx = 0.0
                    fn = 0.0
                max_pots.append(mx)
                final_pots.append(fn)
                if mx >= 0.9:
                    success_count += 1

                det = self._detect_stepping(ep, foot_key, phi_key, T)
                if det is not None:
                    if det["stepped"]:
                        step_success_count += 1
                    all_cycles.append(
                        det["n_cycles_left"] + det["n_cycles_right"]
                    )
                    all_swings.append(det["n_swings"])
                    all_hmax.append(det["h_swing_max"])
                    if det["sole_swing_max"] is not None:
                        all_solepk.append(det["sole_swing_max"])
                    if det["alt_ratio"] is not None:
                        all_alt.append(det["alt_ratio"])

        n = max(len(max_pots), 1)
        mean_max_pot = sum(max_pots) / n if max_pots else 0.0
        mean_final_pot = sum(final_pots) / n if final_pots else 0.0
        success_rate = success_count / n
        step_success_rate = step_success_count / n if n_agents else 0.0

        self._success_rate = success_rate

        # --- Best: standing AND stepping tracked separately.
        # mean_max_pot saturates at 1.000 once the warm-started standup
        # holds — tracking it alone stalls best-tracking and early-stop
        # counts "no improvement" while stepping is still climbing
        # (run 064853 stopped at u1705 with step=0.85 still rising).
        # step rate saturates too (1.0), so the tracked metric is a
        # quality score: stepping minus falls.  A resumed converged
        # checkpoint can then keep improving gait stability instead of
        # dying instantly on a saturated step counter (s42_stable would
        # have stopped at u2020, 5 updates after resume).
        falls_mean = (
            sum(all_falls) / max(len(all_falls), 1) if all_falls else 0.0
        )
        quality = step_success_rate - 0.1 * falls_mean
        improved_pot = mean_max_pot > self._best_potential
        if improved_pot:
            self._best_potential = mean_max_pot
        improved_step = (
            success_rate >= 0.9
            and quality > self._best_quality
        )
        if improved_step:
            self._best_quality = quality
        is_new_best = improved_pot or improved_step
        if is_new_best:
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
                "step": round(step_success_rate, 3),
                "cycles": round(sum(all_cycles) / max(len(all_cycles), 1), 2),
                "swings": round(sum(all_swings) / max(len(all_swings), 1), 2),
                "hmax": round(sum(all_hmax) / max(len(all_hmax), 1), 3),
                "solepk": round(sum(all_solepk) / max(len(all_solepk), 1), 3)
                        if all_solepk else None,
                "alt": round(sum(all_alt) / max(len(all_alt), 1), 3)
                      if all_alt else None,
                "falls": round(falls_mean, 2)
                        if all_falls else None,
                "sway": round(sum(all_sway) / max(len(all_sway), 1), 2)
                       if all_sway else None,
            },
        }

    @staticmethod
    def _detect_stepping(
        episode, foot_key: str, phi_key: str, T: int,
    ) -> Optional[dict]:
        """Run step-cycle detection for one agent in one episode.

        Returns the ``detect_step_cycles`` result dict, or ``None`` when
        any required observer field is missing.
        """
        phi = extract_per_step_field(
            episode.observer_outputs, phi_key, "potential", T,
        )
        h_left = extract_per_step_field(
            episode.observer_outputs, foot_key, "h_left_foot", T,
        )
        h_right = extract_per_step_field(
            episode.observer_outputs, foot_key, "h_right_foot", T,
        )
        c_left = extract_per_step_field(
            episode.observer_outputs, foot_key, "left_foot_contact", T,
        )
        c_right = extract_per_step_field(
            episode.observer_outputs, foot_key, "right_foot_contact", T,
        )
        sole_left = extract_per_step_field(
            episode.observer_outputs, foot_key, "sole_clear_left", T,
        )
        sole_right = extract_per_step_field(
            episode.observer_outputs, foot_key, "sole_clear_right", T,
        )
        if phi is None or h_left is None or h_right is None \
                or c_left is None or c_right is None \
                or sole_left is None or sole_right is None:
            return None

        standing = np.asarray(phi[:T], dtype=np.float32) >= Step.step_phi_gate
        return detect_step_cycles(
            np.asarray(c_left[:T], dtype=bool),
            np.asarray(c_right[:T], dtype=bool),
            np.asarray(h_left[:T], dtype=np.float32),
            np.asarray(h_right[:T], dtype=np.float32),
            standing,
            sole_left=np.asarray(sole_left[:T], dtype=np.float32),
            sole_right=np.asarray(sole_right[:T], dtype=np.float32),
            min_air_steps=Step.step_min_air_frames,
            h_thresh=Step.step_lift_threshold,
        )

    def state(self) -> dict:
        return {
            "best_potential": self._best_potential,
            "best_quality": self._best_quality,
            "success_rate": self._success_rate,
            "last_best_update": self._last_best_update,
        }

    def load_state(self, state: dict) -> None:
        self._best_potential = float(state.get("best_potential", -1.0))
        # "best_quality" is absent in pre-stability-gate checkpoints —
        # defaulting to -1 lets a resumed run re-anchor improvement
        # tracking at its first eval instead of inheriting a saturated
        # step counter that can never improve (instant early-stop).
        self._best_quality = float(state.get("best_quality", -1.0))
        self._success_rate = float(state.get("success_rate", 0.0))
        self._last_best_update = int(state.get("last_best_update", 0))


EXPERIMENT_CLASS = Step
