"""V2 dual-agent experiment: r_fall + per-foot height channels with a
direction-scheduling actor_weight state machine.

Based on ``exp_basic_balance.py``, replacing the single ``r_cross`` channel
(CrossSupportBalanceRewarder's sparse penalty scalar) with two dense,
physically-grounded channels — one per foot::

    r_fall       = 0.01 × φ_height          γ=0.99, aw = 3.0 (fixed)
    r_left_foot  = clip(h_left,  -0.1, 0.1) γ=0.90, aw = state machine
    r_right_foot = clip(h_right, -0.1, 0.1) γ=0.90, aw = state machine

Design rationale
----------------
The reward carries only *physical fact* (foot height); the *intent* (which
foot should rise / descend right now) is carried by ``actor_weight``.  This
matters because:

1. The stepping stage depends on history counters that are NOT in the 96-dim
   Markov observation.  Baking the stage into the reward would force the
   critic to estimate an unobservable quantity → low explained variance →
   the framework's confidence weighting (``conf = clip(EV,0,1)**0.5``) would
   automatically down-weight the channel.  Keeping the reward pure
   (``r = h_foot``) makes the critic's target a clean function of the
   observation.

2. ``actor_weight`` is the *only* per-frame gating channel in the framework
   (``key_seg_active`` is per-trajectory).  A zero reward does NOT disable a
   channel — GAE still produces a nonzero advantage from bootstrapping.
   Only ``aw = 0`` yields exactly zero contribution to ``combined_adv``.

3. The schedule is known exactly at trajectory-build time, so applying it as
   an actor_weight constant costs nothing, whereas hiding it in the reward
   would require re-learning it.

Stepping state machine (post-hoc scan over the contact sequence)
----------------------------------------------------------------
Physical state from ``(contact_L, contact_R)``::

    (T, T) → DOUBLE      both feet down
    (T, F) → SUPPORT_L   left supports,  right swings
    (F, T) → SUPPORT_R   right supports, left swings
    (F, F) → FLIGHT      neither foot down

Bookkeeping, per frame::

    if state is SUPPORT_*:
        current_swing = the airborne foot
        last_swing    = current_swing        # updated unconditionally
        support_steps = support_steps + 1 if same state as previous else 1
    else:
        current_swing = None
        support_steps = 0

Weights (W = 1.0)::

    initial DOUBLE (last_swing is None)  →  w_L = +W, w_R = +W
    FLIGHT                               →  w_L =  0, w_R =  0
    SUPPORT_*  steps 1..2   (Phase A)    →  w[swing]   =  0
                                            w[support] = -W
    SUPPORT_*  steps 3..10  (Phase B)    →  w_L =  0, w_R =  0
    SUPPORT_*  steps 11+    (Phase C)    →  w[swing]   = -W
                                            w[support] =  0
    DOUBLE transition (last_swing set)   →  w[prev_support] = +W
                                            w[prev_swing]   = -W

Phase semantics
---------------
Phase A (steps 1..2, ~0.1 s @ 20 Hz):
    Press the support foot down only (w[support] = -W).  The swing foot
    is left alone (w = 0) — its lift was already encouraged by the
    preceding DOUBLE transition.  Goal: weight transfer onto the new
    support foot.

Phase B (steps 3..10, ~0.4 s):
    Coast — no encouragement.  Let the swing foot travel through the air
    naturally; only r_fall keeps the robot alive.

Phase C (steps 11+, ~0.55 s+):
    Encourage the swing foot to lower (prepare for landing).  The support
    foot is left alone (w = 0): it should stay planted, not start lifting
    prematurely.

DOUBLE transition (SUPPORT_* → DOUBLE, last_swing is set):
    Encourage the *previous support* foot to lift (begin the next step)
    and the *previous swing* foot to keep lowering (finish landing
    cleanly).  This persists for the entire DOUBLE interval until the
    robot lifts a foot into the next SUPPORT_*.

Self-correction property
------------------------
``last_swing`` is updated unconditionally to whichever foot is actually
airborne.  In the DOUBLE transition the weights point at
``opposite(last_swing)`` (previous support → lift) and ``last_swing``
(previous swing → lower), so if the robot lifted the *wrong* foot on the
previous step it is pushed back toward the correct foot during DOUBLE.

Inside SUPPORT_* the weights follow the *actual* swing/support feet, not
an expected-swing target.  This means that if the robot commits to
lifting the same foot twice, Phase A will reinforce that choice (+W on
the swing foot).  This is by design: once a foot is committed, let the
gait cycle complete; correction happens at the next DOUBLE transition.
Occasionally repeating the same foot is acceptable, and forcing a
mid-stride correction would fight the physics.

Negative actor_weight relies on the ``!= 0.0`` skip predicate in
``ppo_trainer_v2.ppo_update_v2`` (a channel whose weights are all <= 0 was
previously dropped silently).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baseline.framework.trajectory import ChannelData, RewardChannel, Trajectory
from baseline.framework.ppo_trainer import _extract_per_step_field

from .base import CombatExperimentV2Base


# --- Physical states ---
STATE_DOUBLE = "double"
STATE_SUPPORT_L = "support_l"   # left supports, right swings
STATE_SUPPORT_R = "support_r"   # right supports, left swings
STATE_FLIGHT = "flight"

# --- Stepping state machine parameters ---
FOOT_WEIGHT: float = 1.0
"""Base actor_weight magnitude W for the two foot channels."""

PHASE_A_STEPS: int = 2
"""Phase A duration (steps 1..PHASE_A_STEPS): encourage swing foot to lift
and support foot to lower — fast lift + weight transfer."""

PHASE_B_END: int = 10
"""Phase B ends at this step (steps PHASE_A_STEPS+1 .. PHASE_B_END): coast,
no encouragement — let the swing foot travel naturally."""

FOOT_HEIGHT_CLIP: float = 0.1
"""Foot height reward saturation (m).  Lifting beyond this earns nothing
more, preventing a degenerate 'raise the knee as high as possible' policy."""


class BasicBalanceStep(CombatExperimentV2Base):

    name = "basic_balance_step"

    _channel_names = ("r_fall", "r_left_foot", "r_right_foot")

    # Per-channel discount: r_fall is long-horizon (survival), the foot
    # channels are local/reactive (did this action lift the foot now?).
    _channel_gammas = {
        "r_fall": 0.99,
        "r_left_foot": 0.9,
        "r_right_foot": 0.9,
    }
    _gae_lambda = 0.95

    env_blueprint = ""  # overridden via _env_pb()
    agent_used = "both"

    episodes_per_update: int = 256 * 4

    # --- Reward constants ---
    per_step_phi_coef: float = 0.01

    # --- r_fall actor weight (fixed, same as exp_basic_balance) ---
    r_fall_actor_weight: float = 3.0

    _AGENT_IDS = ("robot_a", "robot_b")

    _survival_rate: float = 0.0
    _best_survived: float = -1.0

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "end2end" / "basic_balance_step_env.yaml"
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
    # Stepping state machine
    # ------------------------------------------------------------------
    #
    # Single support is split into three sub-phases by support_steps:
    #   Phase A (steps 1..PHASE_A_STEPS):  w[swing]=0, w[support]=-W
    #       → press support foot down (weight transfer)
    #   Phase B (steps PHASE_A_STEPS+1..PHASE_B_END):  w=0
    #       → coast, let physics carry the swing
    #   Phase C (steps PHASE_B_END+1..):  w[swing]=-W, w[support]=0
    #       → lower swing for landing, support stays planted
    #
    # DOUBLE transition (prev SUPPORT_* → DOUBLE): w[prev_support]=+W,
    # w[prev_swing]=-W — start the next step, finish the current landing.
    #
    # See the module-level docstring for the full design rationale.

    @staticmethod
    def _compute_foot_weights(
        contact_l: np.ndarray,
        contact_r: np.ndarray,
        T: int,
        weight: float = FOOT_WEIGHT,
        phase_a_steps: int = PHASE_A_STEPS,
        phase_b_end: int = PHASE_B_END,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Post-hoc scan producing per-frame actor weights for both feet.

        Returns ``(w_left, w_right)``, each shape ``(T,)`` float32.
        See the module docstring for the full rule table.
        """
        w_left = np.zeros(T, dtype=np.float32)
        w_right = np.zeros(T, dtype=np.float32)

        last_swing: Optional[str] = None
        prev_state: Optional[str] = None
        support_steps: int = 0

        for t in range(T):
            cl = bool(contact_l[t])
            cr = bool(contact_r[t])

            if cl and cr:
                state = STATE_DOUBLE
            elif cl and not cr:
                state = STATE_SUPPORT_L
            elif cr and not cl:
                state = STATE_SUPPORT_R
            else:
                state = STATE_FLIGHT

            # --- Bookkeeping ---
            if state == STATE_SUPPORT_L:
                current_swing = "right"
            elif state == STATE_SUPPORT_R:
                current_swing = "left"
            else:
                current_swing = None

            if current_swing is not None:
                last_swing = current_swing
                support_steps = support_steps + 1 if state == prev_state else 1
            else:
                support_steps = 0

            # --- Weights ---
            if state == STATE_FLIGHT:
                # Neither foot down: don't inject a direction, let r_fall lead.
                pass
            elif current_swing is not None:
                # Single support — three sub-phases based on support_steps.
                swing_is_left = current_swing == "left"
                if support_steps <= phase_a_steps:
                    # Phase A: press the support foot down only.
                    # The swing foot is left alone (w=0) — lifting it is
                    # driven by the DOUBLE transition that just ended.
                    if swing_is_left:
                        w_right[t] = -weight     # support down
                    else:
                        w_left[t] = -weight
                elif support_steps <= phase_b_end:
                    # Phase B: coast — no encouragement.
                    pass
                else:
                    # Phase C: encourage swing foot down, leave support alone.
                    if swing_is_left:
                        w_left[t] = -weight     # swing down
                    else:
                        w_right[t] = -weight
            elif last_swing is None:
                # Initial double support, no step taken yet: lift either foot.
                w_left[t] = weight
                w_right[t] = weight
            else:
                # DOUBLE transition: encourage the previous support foot to
                # lift (start the next step) and the previous swing foot to
                # keep lowering (finish landing).  previous_swing == last_swing,
                # previous_support == opposite(last_swing).
                if last_swing == "left":
                    w_left[t] = -weight         # previous swing down
                    w_right[t] = weight         # previous support up
                else:
                    w_right[t] = -weight
                    w_left[t] = weight

            prev_state = state

        return w_left, w_right

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

        # --- r_fall: 0.01 × φ(t) per step ---
        phi_arr = _extract_per_step_field(
            episode.observer_outputs, phi_key, "phi", T_full,
        )
        if phi_arr is not None:
            phi_arr = phi_arr[:T]
        else:
            phi_arr = np.ones(T, dtype=np.float32)
        phi_arr = np.clip(phi_arr, 0.0, 1.0).astype(np.float32)
        r_fall = (self.per_step_phi_coef * phi_arr).astype(np.float32)

        # --- Foot heights (saturated) ---
        h_left = self._extract_foot_field(episode, foot_key, "h_left_foot", T_full, T)
        h_right = self._extract_foot_field(episode, foot_key, "h_right_foot", T_full, T)
        r_left = np.clip(h_left, -FOOT_HEIGHT_CLIP, FOOT_HEIGHT_CLIP).astype(np.float32)
        r_right = np.clip(h_right, -FOOT_HEIGHT_CLIP, FOOT_HEIGHT_CLIP).astype(np.float32)

        # --- Contacts → stepping state machine → foot actor weights ---
        contact_l = self._extract_foot_field(
            episode, foot_key, "left_foot_contact", T_full, T,
        )
        contact_r = self._extract_foot_field(
            episode, foot_key, "right_foot_contact", T_full, T,
        )
        w_left, w_right = self._compute_foot_weights(
            contact_l > 0.5, contact_r > 0.5, T,
        )

        is_terminated = fell

        all_rewards = {
            "r_fall": r_fall,
            "r_left_foot": r_left,
            "r_right_foot": r_right,
        }
        actor_weights = {
            "r_fall": np.full(T, self.r_fall_actor_weight, dtype=np.float32),
            "r_left_foot": w_left,
            "r_right_foot": w_right,
        }

        channels: Dict[str, ChannelData] = {}
        for key in self._channel_names:
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

    @staticmethod
    def _extract_foot_field(
        episode, foot_key: str, field: str, T_full: int, T: int,
    ) -> np.ndarray:
        """Extract a FootStateObserver field, truncated to ``T``.

        Raises if the observer or field is missing — a silent zero fallback
        would make the stepping signal vanish without any error.
        """
        arr = _extract_per_step_field(
            episode.observer_outputs, foot_key, field, T_full,
        )
        if arr is None:
            raise KeyError(
                f"_extract_foot_field: observer '{foot_key}' field '{field}' "
                f"missing from episode.observer_outputs "
                f"(available observers={list(episode.observer_outputs.keys())})"
            )
        return arr[:T]

    def build_trajectories(self, episodes) -> List[Trajectory]:
        agent_specs = [
            ("robot_a", "foot_state_a", "height_phi_a"),
            ("robot_b", "foot_state_b", "height_phi_b"),
        ]

        all_trajs: List[Trajectory] = []
        for episode in episodes:
            for agent_id, foot_key, phi_key in agent_specs:
                all_trajs.extend(
                    self._build_agent_trajectory(
                        episode, agent_id, foot_key, phi_key,
                    )
                )
        return all_trajs

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

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


EXPERIMENT_CLASS = BasicBalanceStep
