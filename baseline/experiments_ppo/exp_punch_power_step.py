"""Punch-power stage: attack_prep recipe + per-arm punch-speed reward.

Stage intent (user design): after ``attack_prep_step`` achieved "can
throw punches", this stage trains punch POWER only — like shadow-
boxing drills.  Timing/accuracy are explicitly NOT the target; the
moving scripted target is kept (0.5 m/s) so direction stays honest,
but the new signal rewards raw fist speed toward the opponent's head.

Two new channels driven by the same reactive arm state machine:

  r_left_punch_speed / r_right_punch_speed
      reward = max(0, v_hand · dir(hand→opp_head)) — positive hand
               speed toward the opponent head (m/s, unbounded), paid
               only on frames where the elbow is ACTIVELY EXTENDING
               inside a READY run that ends in a completed punch.
               Positive-only: a wind-up dip or recoil inside the swing
               earns zero instead of a penalty — with the signed
               version, negative-speed extension frames suppressed the
               punching action itself (punches fell as W rose).
               Own velocity only; the opponent's motion is excluded.
      aw     = +SPEED_WEIGHT × φ² on those strike frames, else 0.

The world-frame hand velocity means the whole kinetic chain pays: arm
swing + body lunge.  The retained hand_dist channels (shoulder-
relative) still punish pure body-charging without arm extension.

best-of-run / early stop composite:
    quality = mean positive closing speed on valid frames
              + 0.1 × punches_per_ep     (success >= 0.9 gate)

Blueprint: baseline/humanoid21/end2end/attack_prep_step_env.yaml
(same as attack_prep — ArmStateObserver gained the closing-speed
fields; no env change needed).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.trajectory import ChannelData, Trajectory
from baseline.framework.rollout import extract_per_step_field

from .exp_attack_prep_step import AttackPrepStep


_READY = 0
_RETRACT = 1


def _completed_swing_mask(
    elbow: np.ndarray,
    valid_mask: np.ndarray,
    extend_threshold: float,
    flex_threshold: float,
) -> np.ndarray:
    """Frames belonging to READY runs that end in a completed punch.

    Mirrors ``arm_state_machine._scan`` per-arm semantics: the arm is
    READY while chambered, a punch completes when elbow <=
    extend_threshold (only counted on valid frames), and the arm
    re-arms when elbow >= flex_threshold.  A completed run is the
    contiguous span of READY-state frames ending at the crossing.
    Only valid frames inside completed runs return True — the reward
    is for swinging *through* a real punch, not for waving.
    """
    T = len(elbow)
    mask = np.zeros(T, dtype=bool)
    if T == 0:
        return mask
    state = _RETRACT if float(elbow[0]) <= extend_threshold else _READY
    run_start = 0 if state == _READY else -1
    prev_e = float(elbow[0])
    for t in range(T):
        e = float(elbow[t])
        if state == _READY and e <= extend_threshold:
            if valid_mask[t] and run_start >= 0:
                mask[run_start:t + 1] = True
            state = _RETRACT
            run_start = -1
        elif state == _RETRACT and e >= flex_threshold:
            state = _READY
            run_start = t
        prev_e = e
    # Keep only actively-extending frames: elbow_norm decreasing means
    # the arm is mid-strike — paying speed there rewards the strike's
    # power rather than slow drift displacement.
    extending = np.zeros(T, dtype=bool)
    de = np.diff(elbow.astype(np.float64))
    extending[1:] = de < 0.0
    return mask & extending


class PunchPowerStep(AttackPrepStep):
    """attack_prep_step + per-arm fist closing-speed channels."""

    name = "punch_power_step"

    # --- Channels: attack_prep's 12 + punch speed ×2 ---
    _channel_names = AttackPrepStep._channel_names + (
        "r_left_punch_speed", "r_right_punch_speed",
    )
    _channel_gammas = {
        **AttackPrepStep._channel_gammas,
        # Short horizon: punch speed is an instantaneous-action signal;
        # a short gamma keeps credit tight around the swing itself.
        "r_left_punch_speed": 0.90,
        "r_right_punch_speed": 0.90,
    }

    # Speed reward is raw m/s (~1-8).  Strike frames are sparse (~17%
    # of rollout) so the effective gradient share is small — at W=1.0
    # mean speed stayed flat at ~1.6 m/s.  W=3.0 matches ARM_WEIGHT so
    # punch power gets comparable pressure to punch form.
    PUNCH_SPEED_WEIGHT: float = 3.0

    # --- Tracking state ---
    _pp_best_quality: float = -1.0
    _pp_quality_ema: float | None = None
    _pp_last_best_update: int = -1
    _punch_speed_mean: float = 0.0

    # ------------------------------------------------------------------
    # Trajectories — attack_prep recipe + 2 speed channels
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

        l_cs = extract_per_step_field(
            oo, arm_key, "left_hand_speed_to_opp", T_full)
        r_cs = extract_per_step_field(
            oo, arm_key, "right_hand_speed_to_opp", T_full)
        left_elbow = extract_per_step_field(
            oo, arm_key, "left_elbow_norm", T_full)
        right_elbow = extract_per_step_field(
            oo, arm_key, "right_elbow_norm", T_full)
        left_hand_opp = extract_per_step_field(
            oo, arm_key, "left_hand_to_opp_head", T_full)
        right_hand_opp = extract_per_step_field(
            oo, arm_key, "right_hand_to_opp_head", T_full)
        self_x = extract_per_step_field(oo, app_key, "self_x", T_full)
        self_y = extract_per_step_field(oo, app_key, "self_y", T_full)
        opp_x = extract_per_step_field(oo, app_key, "opp_x", T_full)
        opp_y = extract_per_step_field(oo, app_key, "opp_y", T_full)
        fwd_x = extract_per_step_field(oo, face_key, "forward_x", T_full)
        fwd_y = extract_per_step_field(oo, face_key, "forward_y", T_full)
        phi = extract_per_step_field(oo, phi_key, "potential", T_full)

        if any(v is None for v in (
            l_cs, r_cs, left_elbow, right_elbow,
            left_hand_opp, right_hand_opp, self_x, opp_x, phi,
        )):
            for t in trajs:
                T = t.obs.shape[0]
                z = np.zeros(T, dtype=np.float32)
                ref = t.channels["r_fall"]
                for key in ("r_left_punch_speed", "r_right_punch_speed"):
                    t.channels[key] = ChannelData(
                        reward=z.copy(),
                        is_terminated=ref.is_terminated,
                        actor_weight=z.copy(),
                    )
            return trajs

        # Rebuild the same valid mask the parent used.
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
        arm_valid = (
            (dist <= self.D_ARM_GATE)
            & (cos_angle >= self.FACING_COS_THRESHOLD)
            & (phi_arr >= self.STAND_PHI_MIN)
        )

        # Per-arm completed-swing masks: reward is paid only on frames
        # of READY runs that end in a completed punch — waving without
        # committing to an extension crossing earns nothing.
        swing_l = _completed_swing_mask(
            np.asarray(left_elbow), arm_valid,
            self.extend_threshold, self.flex_threshold,
        )
        swing_r = _completed_swing_mask(
            np.asarray(right_elbow), arm_valid,
            self.extend_threshold, self.flex_threshold,
        )
        w_ls = (
            self.PUNCH_SPEED_WEIGHT * swing_l.astype(np.float32) * phi2
        ).astype(np.float32)
        w_rs = (
            self.PUNCH_SPEED_WEIGHT * swing_r.astype(np.float32) * phi2
        ).astype(np.float32)

        for t in trajs:
            T = t.obs.shape[0]
            ref = t.channels["r_fall"]
            t.channels["r_left_punch_speed"] = ChannelData(
                reward=np.maximum(
                    np.asarray(l_cs[:T], dtype=np.float32), 0.0),
                is_terminated=ref.is_terminated,
                actor_weight=w_ls[:T],
            )
            t.channels["r_right_punch_speed"] = ChannelData(
                reward=np.maximum(
                    np.asarray(r_cs[:T], dtype=np.float32), 0.0),
                is_terminated=ref.is_terminated,
                actor_weight=w_rs[:T],
            )
        return trajs

    # ------------------------------------------------------------------
    # Eval — attack_prep metrics + punch speed
    # ------------------------------------------------------------------

    def on_eval(self, episodes, update) -> Dict[str, Any]:
        result = super().on_eval(episodes, update)
        info = result["info"]

        speed_means: List[float] = []
        speed_peaks: List[float] = []
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

            l_cs = extract_per_step_field(
                oo, arm_key, "left_hand_speed_to_opp", T)
            r_cs = extract_per_step_field(
                oo, arm_key, "right_hand_speed_to_opp", T)
            l_elb = extract_per_step_field(
                oo, arm_key, "left_elbow_norm", T)
            r_elb = extract_per_step_field(
                oo, arm_key, "right_elbow_norm", T)
            self_x = extract_per_step_field(oo, app_key, "self_x", T)
            self_y = extract_per_step_field(oo, app_key, "self_y", T)
            opp_x = extract_per_step_field(oo, app_key, "opp_x", T)
            opp_y = extract_per_step_field(oo, app_key, "opp_y", T)
            fwd_x = extract_per_step_field(
                oo, face_key, "forward_x", T)
            fwd_y = extract_per_step_field(
                oo, face_key, "forward_y", T)
            phi = extract_per_step_field(oo, phi_key, "potential", T)
            if any(v is None for v in (
                l_cs, r_cs, l_elb, r_elb, self_x, opp_x, phi,
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
            valid = (
                (dist <= self.D_ARM_GATE)
                & (cos_angle >= self.FACING_COS_THRESHOLD)
                & standing
            )
            if not valid.any():
                continue
            # Measure speed exactly where we pay it: extending frames
            # of completed swings — undiluted by chambered/idle time.
            strike_l = _completed_swing_mask(
                np.asarray(l_elb), valid,
                self.extend_threshold, self.flex_threshold)
            strike_r = _completed_swing_mask(
                np.asarray(r_elb), valid,
                self.extend_threshold, self.flex_threshold)
            l_strike = np.asarray(l_cs, dtype=np.float64)[strike_l]
            r_strike = np.asarray(r_cs, dtype=np.float64)[strike_r]
            strike = np.concatenate([l_strike, r_strike])
            pos = strike[strike > 0]
            if pos.size:
                speed_means.append(float(pos.mean()))
                speed_peaks.append(float(pos.max()))

        spd = float(np.mean(speed_means)) if speed_means else 0.0
        spd_pk = float(np.mean(speed_peaks)) if speed_peaks else 0.0
        self._punch_speed_mean = spd

        quality = spd + 0.1 * self._punches_per_ep
        improved = (
            info.get("success", 0.0) >= 0.9
            and quality > self._pp_best_quality
        )
        if improved:
            self._pp_best_quality = quality
        if self._pp_quality_ema is None:
            self._pp_quality_ema = quality
        else:
            self._pp_quality_ema = (
                0.85 * self._pp_quality_ema + 0.15 * quality
            )
        trend_improved = (
            info.get("success", 0.0) >= 0.9
            and self._pp_quality_ema > self._pp_best_quality
        )
        if trend_improved:
            self._pp_best_quality = self._pp_quality_ema
        if improved or trend_improved:
            self._pp_last_best_update = update
        if self._pp_last_best_update < 0:
            self._pp_last_best_update = update

        no_improvement = update - self._pp_last_best_update
        stop_training = (
            no_improvement >= self._no_improvement_limit
            and update >= self._min_updates
        )

        info.update({
            "punch_speed": round(spd, 3),
            "punch_speed_pk": round(spd_pk, 3),
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
            "pp_best_quality": self._pp_best_quality,
            "pp_last_best_update": self._pp_last_best_update,
            "punch_speed_mean": self._punch_speed_mean,
        }

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self._pp_best_quality = -1.0
        self._pp_quality_ema = None
        self._pp_last_best_update = -1
        self._punch_speed_mean = 0.0


EXPERIMENT_CLASS = PunchPowerStep
