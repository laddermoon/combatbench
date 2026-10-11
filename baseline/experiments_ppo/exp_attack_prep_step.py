"""Attack-prep on the new stack: follow_step + arm punch shaping.

Warm-start from ``follow_step`` (s42b, speed curriculum 12/12 — the
learner can chase a 1.2-1.5 m/s target while staying upright under
pinned 40N pushes).  The opponent stays a scripted kinematic target
(``RandomMovePlugin``) at FIXED 0.5 m/s — slow enough that the learner
spends long stretches inside striking range; punching form is learned
against a predictable evader before real combat.

Four new reward channels driven by the post-hoc arm state machine
(``arm_state_machine.compute_arm_weights_with_stats``, adapted from the
archived ``exp_attack_prep`` recipe):

  r_left_elbow / r_right_elbow   — ((1 - elbow_norm)/2)² ∈ [0,1]
                                   1 = punched out, 0 = chambered
                                   (quadratic: half-jabs pay little —
                                   kills the half-extension plateau)
  r_left_hand_dist /             — hand→opp-head gap RELATIVE to the
      r_right_hand_dist            shoulder:  hand→head − shoulder→head.
                                   Body-lunging moves both terms equally
                                   (net zero) — only actual arm
                                   extension pays.  (v2 fix: the raw
                                   hand→head distance let the policy
                                   exploit body translation instead of
                                   punching.)

The state machine (reactive per-arm READY/RETRACT cycle — see
``arm_state_machine.py`` docstring for why strict turn-taking was
dropped) emits ±arm_weight actor weights; it only runs on valid
segments where the learner is standing (φ ≥ 0.5), within
D_ARM_GATE=1.1m of the target, and facing within ±60°.  Outside
segments aw = 0; inside, aw *= φ² so recovery still outranks
punching when half-fallen.  Anti-lock: an already-extended arm is in
RETRACT and must chamber before earning extension weights (no fake
instant punches).

Best-of-run / early stop use a punch composite —
    quality = punches_per_ep + 0.5 × arm_active_ratio
recorded only when success ≥ 0.9 (standup still solid).  Speed
curriculum is frozen at 0.5 m/s; push stays pinned at 40N level 3.

Blueprint: baseline/humanoid21/end2end/attack_prep_step_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.framework.rollout import extract_per_step_field
from baseline.humanoid21.end2end.arm_state_machine import (
    ARM_WEIGHT,
    ATTACK_TIMEOUT_STEPS,
    ELBOW_EXTEND_THRESHOLD,
    ELBOW_FLEX_THRESHOLD,
    compute_arm_weights_with_stats,
)

from .exp_follow_step import FollowStep


class AttackPrepStep(FollowStep):
    """follow_step + alternating-punch shaping vs a fixed-speed target."""

    name = "attack_prep_step"
    agent_used = "random"

    # Resumed from the follow run (~u7400+) — headroom for punch
    # shaping to converge; early stop bounds the tail.
    max_updates: int = 15000

    # --- Arm punch channels (γ=0.99, same horizon as r_fall) ---
    _channel_names = FollowStep._channel_names + (
        "r_left_elbow", "r_right_elbow",
        "r_left_hand_dist", "r_right_hand_dist",
    )
    _channel_gammas = {
        **FollowStep._channel_gammas,
        "r_left_elbow": 0.99,
        "r_right_elbow": 0.99,
        "r_left_hand_dist": 0.99,
        "r_right_hand_dist": 0.99,
    }

    # --- Arm state machine parameters (exp_attack_prep recipe) ---
    arm_weight: float = ARM_WEIGHT
    # -0.6 not -0.95: a strong unilateral habit from the exploit phase
    # left one arm stalling ~-0.65 (visibly extended) below -0.95, so
    # it could never complete and the turn never alternated.  The
    # threshold only gates the state transition — the continuous
    # (1-norm)/2 reward still pays for deeper extension afterward.
    extend_threshold: float = -0.6
    # +0.7 not +0.95: the natural chamber after a punch sits at elbow
    # ~+0.75..+0.9 — +0.95 demanded near-joint-limit flexion, so a
    # punched arm sat in RETRACT forever and could never re-enter READY
    # (observed: 89 consecutive RETRACT frames at elbow +0.64..0.9).
    # +0.7 = retract to a real guard position before the next punch.
    flex_threshold: float = 0.7
    attack_timeout_steps: int = ATTACK_TIMEOUT_STEPS
    D_ARM_GATE: float = 1.1          # m — machine runs inside this dist
    FACING_COS_THRESHOLD: float = 0.5  # cos(60°) facing cone
    STAND_PHI_MIN: float = 0.5       # φ ≥ this counts as standing

    # --- Fixed opponent speed: no curriculum for attack-prep ---
    OPPONENT_SPEED: float = 0.5      # m/s — slow evader, max range time
    FOLLOW_PROMOTE_PATIENCE: int = 10**9  # freeze _speed_level at 0

    _ARM_OBS = {
        "robot_a": "arm_state_a",
        "robot_b": "arm_state_b",
    }

    # --- Attack-prep tracking state ---
    _ap_best_quality: float = -1.0
    _ap_quality_ema: float | None = None
    _ap_last_best_update: int = -1
    _punches_per_ep: float = 0.0
    _arm_active_ratio: float = 0.0

    # ------------------------------------------------------------------
    # Blueprint — follow_step env + ArmStateObserver
    # ------------------------------------------------------------------

    def _env_pb(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint,
        )
        bp_path = (
            Path(__file__).resolve().parent.parent
            / "humanoid21" / "end2end" / "attack_prep_step_env.yaml"
        )
        return ParameterizedEnvBlueprint.load(bp_path)

    @property
    def current_speed(self) -> float:
        return self.OPPONENT_SPEED

    # ------------------------------------------------------------------
    # Trajectories — follow recipe + 4 arm channels, learner only
    # ------------------------------------------------------------------

    def _build_agent_trajectory(
        self, episode, agent_id: str, foot_key: str, phi_key: str,
    ) -> List[Trajectory]:
        trajs = super()._build_agent_trajectory(
            episode, agent_id, foot_key, phi_key,
        )
        if not trajs:
            return trajs

        T_full = episode.num_frames
        oo = episode.observer_outputs
        arm_key = self._ARM_OBS.get(agent_id)
        app_key = self._APPROACH_OBS.get(agent_id)
        face_key = self._FACE_OBS.get(agent_id)

        left_elbow = extract_per_step_field(
            oo, arm_key, "left_elbow_norm", T_full)
        right_elbow = extract_per_step_field(
            oo, arm_key, "right_elbow_norm", T_full)
        left_hand_opp = extract_per_step_field(
            oo, arm_key, "left_hand_to_opp_head", T_full)
        right_hand_opp = extract_per_step_field(
            oo, arm_key, "right_hand_to_opp_head", T_full)
        opp_head_lsh = extract_per_step_field(
            oo, arm_key, "opp_head_to_left_shoulder", T_full)
        opp_head_rsh = extract_per_step_field(
            oo, arm_key, "opp_head_to_right_shoulder", T_full)
        self_x = extract_per_step_field(oo, app_key, "self_x", T_full)
        self_y = extract_per_step_field(oo, app_key, "self_y", T_full)
        opp_x = extract_per_step_field(oo, app_key, "opp_x", T_full)
        opp_y = extract_per_step_field(oo, app_key, "opp_y", T_full)
        fwd_x = extract_per_step_field(oo, face_key, "forward_x", T_full)
        fwd_y = extract_per_step_field(oo, face_key, "forward_y", T_full)
        phi = extract_per_step_field(oo, phi_key, "potential", T_full)

        arm_missing = any(
            v is None
            for v in (
                left_elbow, right_elbow,
                left_hand_opp, right_hand_opp,
                opp_head_lsh, opp_head_rsh,
                self_x, opp_x, phi,
            )
        )
        if arm_missing:
            # Observers absent — keep channels zero so critics still
            # exist (channel set must match reward_channels()).
            for t in trajs:
                T = t.obs.shape[0]
                z = np.zeros(T, dtype=np.float32)
                ref = t.channels["r_fall"]
                for key in (
                    "r_left_elbow", "r_right_elbow",
                    "r_left_hand_dist", "r_right_hand_dist",
                ):
                    t.channels[key] = ChannelData(
                        reward=z.copy(),
                        is_terminated=ref.is_terminated,
                        actor_weight=z.copy(),
                    )
            return trajs

        # --- Geometry: dist + facing cos (same as parent's r_face) ---
        self_xy = np.stack([self_x, self_y], axis=1).astype(np.float64)
        opp_xy = np.stack([opp_x, opp_y], axis=1).astype(np.float64)
        fwd = (
            np.stack([fwd_x, fwd_y], axis=1).astype(np.float64)
            if fwd_x is not None and fwd_y is not None
            else np.tile([1.0, 0.0], (T_full, 1))
        )
        dist = np.linalg.norm(opp_xy - self_xy, axis=1)
        to_opp_hat = (opp_xy - self_xy) / np.maximum(dist, 1e-6)[:, None]
        cos_angle = np.sum(fwd * to_opp_hat, axis=1)

        phi_arr = np.clip(np.asarray(phi, dtype=np.float32), 0.0, 1.0)
        phi2 = (phi_arr ** 2).astype(np.float32)
        standing = phi_arr >= self.STAND_PHI_MIN

        # --- Valid segments: striking range + facing cone + standing ---
        arm_valid = (
            (dist <= self.D_ARM_GATE)
            & (cos_angle >= self.FACING_COS_THRESHOLD)
            & standing
        )
        w_le, w_re, w_lhd, w_rhd, _stats = compute_arm_weights_with_stats(
            np.asarray(left_elbow), np.asarray(right_elbow),
            np.asarray(left_hand_opp), np.asarray(right_hand_opp),
            valid_mask=arm_valid,
            extend_threshold=self.extend_threshold,
            flex_threshold=self.flex_threshold,
            arm_weight=self.arm_weight,
            attack_timeout_steps=self.attack_timeout_steps,
        )

        # --- Arm rewards ---
        # Elbow: norm [-1,+1] → [0,1]²; 1 = punched out, 0 = chambered.
        # Quadratic: the linear version paid 0.4 for a lazy half-jab
        # (elbow ~+0.2) — the policy settled into a half-extension
        # equilibrium and never reached the punch threshold.  Squaring
        # cuts half-jab payoff to 0.16 while keeping full-extension at
        # 1.0, sharpening the gradient where it matters.
        r_left_elbow = ((1.0 - np.asarray(left_elbow)) * 0.5) ** 2
        r_right_elbow = ((1.0 - np.asarray(right_elbow)) * 0.5) ** 2
        # Hand gap relative to the shoulder line:
        #   r = hand→opp_head − opp_head→shoulder
        # Minimizing it = extending the arm toward the head; body-lunging
        # shifts both terms equally so translation can't fake a punch.
        # ATTACK aw=-W → extend;  FLEX aw=+W → retract.
        r_left_hd = np.asarray(left_hand_opp) - np.asarray(opp_head_lsh)
        r_right_hd = np.asarray(right_hand_opp) - np.asarray(opp_head_rsh)

        arm_channels = {
            "r_left_elbow": (r_left_elbow, w_le),
            "r_right_elbow": (r_right_elbow, w_re),
            "r_left_hand_dist": (r_left_hd, w_lhd),
            "r_right_hand_dist": (r_right_hd, w_rhd),
        }
        for t in trajs:
            T = t.obs.shape[0]
            ref = t.channels["r_fall"]
            for key, (reward, w) in arm_channels.items():
                t.channels[key] = ChannelData(
                    reward=reward[:T].astype(np.float32),
                    is_terminated=ref.is_terminated,
                    actor_weight=(w[:T] * phi2[:T]).astype(np.float32),
                )
        return trajs

    # ------------------------------------------------------------------
    # Eval — parent's gait/recovery/hold/facing + punch metrics
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        result = super().on_eval(episodes, update)
        info = result["info"]

        punch_counts: List[float] = []
        arm_active_ratios: List[float] = []
        for ep in episodes:
            T = ep.num_frames
            if T == 0:
                continue
            agent_id = str(ep.episode_options.get("agent_id", "robot_a"))
            oo = ep.observer_outputs
            arm_key = self._ARM_OBS.get(agent_id)
            app_key = self._APPROACH_OBS.get(agent_id)
            face_key = self._FACE_OBS.get(agent_id)
            _, phi_key = self._AGENT_OBS_MAP[agent_id]

            left_elbow = extract_per_step_field(
                oo, arm_key, "left_elbow_norm", T)
            right_elbow = extract_per_step_field(
                oo, arm_key, "right_elbow_norm", T)
            left_hand_opp = extract_per_step_field(
                oo, arm_key, "left_hand_to_opp_head", T)
            right_hand_opp = extract_per_step_field(
                oo, arm_key, "right_hand_to_opp_head", T)
            self_x = extract_per_step_field(oo, app_key, "self_x", T)
            self_y = extract_per_step_field(oo, app_key, "self_y", T)
            opp_x = extract_per_step_field(oo, app_key, "opp_x", T)
            opp_y = extract_per_step_field(oo, app_key, "opp_y", T)
            fwd_x = extract_per_step_field(oo, face_key, "forward_x", T)
            fwd_y = extract_per_step_field(oo, face_key, "forward_y", T)
            phi = extract_per_step_field(oo, phi_key, "potential", T)
            if any(v is None for v in (
                left_elbow, right_elbow, left_hand_opp, right_hand_opp,
                self_x, opp_x, phi,
            )):
                continue

            self_xy = np.stack([self_x, self_y], axis=1).astype(np.float64)
            opp_xy = np.stack([opp_x, opp_y], axis=1).astype(np.float64)
            dist = np.linalg.norm(opp_xy - self_xy, axis=1)
            fwd = (
                np.stack([fwd_x, fwd_y], axis=1).astype(np.float64)
                if fwd_x is not None and fwd_y is not None
                else np.tile([1.0, 0.0], (T, 1))
            )
            to_opp_hat = (opp_xy - self_xy) / np.maximum(
                dist, 1e-6)[:, None]
            cos_angle = np.sum(fwd * to_opp_hat, axis=1)
            standing = np.asarray(phi, dtype=np.float32) >= self.STAND_PHI_MIN

            arm_valid = (
                (dist <= self.D_ARM_GATE)
                & (cos_angle >= self.FACING_COS_THRESHOLD)
                & standing
            )
            *_, stats = compute_arm_weights_with_stats(
                np.asarray(left_elbow), np.asarray(right_elbow),
                np.asarray(left_hand_opp), np.asarray(right_hand_opp),
                valid_mask=arm_valid,
                extend_threshold=self.extend_threshold,
                flex_threshold=self.flex_threshold,
                arm_weight=self.arm_weight,
                attack_timeout_steps=self.attack_timeout_steps,
            )
            punch_counts.append(float(stats["n_punches"]))
            arm_active_ratios.append(float(stats["n_valid_steps"]) / T)

        punches = float(np.mean(punch_counts)) if punch_counts else 0.0
        arm_active = (
            float(np.mean(arm_active_ratios)) if arm_active_ratios else 0.0
        )
        self._punches_per_ep = punches
        self._arm_active_ratio = arm_active

        # --- Best-of-run / early stop: punch composite ---
        # Parent's own tracking still runs (harmless; slevel frozen) but
        # the exported "best" policy is punch-driven here.
        quality = punches + 0.5 * arm_active
        improved = (
            info.get("success", 0.0) >= 0.9
            and quality > self._ap_best_quality
        )
        if improved:
            self._ap_best_quality = quality
        if self._ap_quality_ema is None:
            self._ap_quality_ema = quality
        else:
            self._ap_quality_ema = (
                0.85 * self._ap_quality_ema + 0.15 * quality
            )
        trend_improved = (
            info.get("success", 0.0) >= 0.9
            and self._ap_quality_ema > self._ap_best_quality
        )
        if trend_improved:
            self._ap_best_quality = self._ap_quality_ema
        if improved or trend_improved:
            self._ap_last_best_update = update
        if self._ap_last_best_update < 0:
            self._ap_last_best_update = update

        no_improvement = update - self._ap_last_best_update
        stop_training = (
            no_improvement >= self._no_improvement_limit
            and update >= self._min_updates
        )

        info.update({
            "punches": round(punches, 2),
            "arm_active": round(arm_active, 3),
        })
        return {
            "is_new_best": improved or trend_improved,
            "stop_training": stop_training,
            "info": info,
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            **super().state(),
            "ap_best_quality": self._ap_best_quality,
            "ap_last_best_update": self._ap_last_best_update,
            "punches_per_ep": self._punches_per_ep,
            "arm_active_ratio": self._arm_active_ratio,
        }

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._ap_best_quality = -1.0
        self._ap_quality_ema = None
        self._ap_last_best_update = -1
        self._punches_per_ep = 0.0
        self._arm_active_ratio = 0.0


EXPERIMENT_CLASS = AttackPrepStep
