"""Energy-based (orbital-energy) standup potential rewarder — V1.

Design philosophy (single continuous potential, no stage machinery):

  Model the standup end-game as a pendulum swing-up about the support axis
  (the line through the two feet).  The body's center of mass (CoM) rotates
  about this axis; the "ideal" state is one where the current rotational
  kinetic energy is *just enough* to carry the CoM to directly above the
  axis and stop there.

  For CoM at perpendicular lever ``l`` from the axis, tilted ``theta`` from
  vertical, with toward-top tangential speed ``v_t``:

      E_need   = g * l * (1 - cos(theta))      # potential to climb to top
      E_have   = 0.5 * v_t * |v_t|             # signed kinetic energy
      E_deficit = E_need - E_have

  - E_deficit == 0 : exactly enough energy to reach the top and stop -> best
  - E_deficit  > 0 : not enough, will fall short ("half push-up")
  - E_deficit  < 0 : overshoot, will fly past the axis
  - v_t < 0 (rotating away from top) enlarges the deficit -> wrong direction
    is naturally penalized.

  E_score maps the deficit to [0, 1] via an asymmetric exponential kernel:

      E_deficit >= 0 : exp(-E_deficit      / e_scale_under)
      E_deficit  < 0 : exp(-|E_deficit|    / e_scale_over)

  Anti-hack gate (feet must stay near the ground, otherwise a robot could
  lie on its back and raise its feet above the CoM to fake E_deficit=0):

      foot_factor = clip((foot_h_max - h_foot) / (foot_h_max - foot_h_min), 0, 1)

  Support polygon factor (anti-hack: CoM projection must be between feet):

      t = normalized CoM projection along foot-to-foot axis (0=left, 1=right)
      support_factor = 0           if t < 0 or t > 1
      support_factor = t / rw      if 0 <= t < rw        (ramp up)
      support_factor = 1           if rw <= t <= 1-rw    (full credit)
      support_factor = (1-t) / rw  if 1-rw < t <= 1      (ramp down)
      where rw = support_ramp_width (default 0.15 = 85:15 split)

  Final potential:

      Phi = foot_factor * E_score * support_factor

Data sources (see envs/humanoid21/DATASPEC.md):
  - CoM position : mass-weighted ``body_xipos`` over all bodies
  - CoM velocity : mass-weighted ``body_linvel_world`` over all bodies
  - Support axis : ``body_xpos`` of foot_left / foot_right body centers,
                   EMA-smoothed to reduce axis jitter
  - Contacts     : parsed from ``get_derived_state(['contacts'])`` (for the
                   is_balanced diagnostic / future success detection)

V1 intentionally does NOT include: a height term, sideways-tilt angular
momentum penalty, exact rigid-body inertia, or truncation -Phi(s_T)
compensation.  Those are deferred to later versions.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class StandupEnergyRewarder(BaseObserverPlugin):
    """Single-potential orbital-energy rewarder for standup-and-balance."""

    def __init__(
        self,
        agent_id: str = "robot_a",
        g: float = 9.81,
        e_scale_under: float = 2.45,   # ~ g * 0.25 : energy shortfall tolerance
        e_scale_over: float = 1.23,    # ~ g * 0.125: overshoot tolerance (tighter)
        foot_h_min: float = 0.20,      # foot body-center height with full credit
        foot_h_max: float = 0.40,      # foot height where credit reaches zero
        foot_ema_alpha: float = 0.5,   # EMA weight for foot axis smoothing (new sample)
        support_ramp_width: float = 0.15, # CoM projection ramp: 0 at foot, 1 at this fraction inward
        lever_min: float = 0.30,       # CoM-to-axis distance below this -> lever_factor=0
        lever_max: float = 0.60,       # CoM-to-axis distance above this -> lever_factor=1
        balance_theta_deg: float = 20.0,  # CoM-above-axis tolerance for is_balanced
        balance_speed: float = 0.3,       # CoM speed tolerance (m/s) for is_balanced
    ) -> None:
        self.agent_id = str(agent_id)
        self.g = float(g)
        self.e_scale_under = float(e_scale_under)
        self.e_scale_over = float(e_scale_over)
        self.foot_h_min = float(foot_h_min)
        self.foot_h_max = float(foot_h_max)
        self.foot_ema_alpha = float(foot_ema_alpha)
        self.support_ramp_width = float(support_ramp_width)
        self.lever_min = float(lever_min)
        self.lever_max = float(lever_max)
        self.balance_theta_deg = float(balance_theta_deg)
        self.balance_speed = float(balance_speed)

        self._reset_state()

    def _reset_state(self) -> None:
        self._potential = 0.0
        self._e_score = 0.0
        self._foot_factor = 0.0
        self._e_deficit = 0.0
        self._e_need = 0.0
        self._e_have = 0.0
        self._lever = 0.0
        self._theta_deg = 0.0
        self._v_t = 0.0
        self._com_h = 0.0
        self._com_speed = 0.0
        self._h_foot = 0.0
        self._is_balanced = 0.0
        self._both_feet = 0.0
        self._support_factor = 0.0
        self._lever_factor = 0.0
        self._foot_l_ema: np.ndarray | None = None
        self._foot_r_ema: np.ndarray | None = None

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._reset_state()

    # ---- Core step --------------------------------------------------------

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        static_data = ctx.accessor.get_static_data()[self.agent_id]
        derived = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]

        body_names = static_data["body_names"]
        masses_by_name = static_data["body_masses_by_name"]
        keypoint_names = static_data["keypoint_body_names"]

        body_xipos = derived["body_xipos"]
        body_linvel = derived["body_linvel_world"]
        body_xpos = derived["body_xpos"]

        # ---- Full-body CoM position and velocity (mass-weighted) ----
        masses = np.array([masses_by_name[n] for n in body_names], dtype=np.float64)
        ipos = np.array([body_xipos[n] for n in body_names], dtype=np.float64)
        lvel = np.array([body_linvel[n] for n in body_names], dtype=np.float64)
        total_mass = float(masses.sum())
        com = (ipos * masses[:, None]).sum(axis=0) / total_mass
        com_vel = (lvel * masses[:, None]).sum(axis=0) / total_mass

        # ---- Support axis from foot body centers (EMA-smoothed) ----
        foot_l = np.asarray(body_xpos[keypoint_names["foot_left"]], dtype=np.float64)
        foot_r = np.asarray(body_xpos[keypoint_names["foot_right"]], dtype=np.float64)
        a = self.foot_ema_alpha
        if self._foot_l_ema is None:
            self._foot_l_ema = foot_l.copy()
            self._foot_r_ema = foot_r.copy()
        else:
            self._foot_l_ema = a * foot_l + (1.0 - a) * self._foot_l_ema
            self._foot_r_ema = a * foot_r + (1.0 - a) * self._foot_r_ema
        foot_l_s = self._foot_l_ema
        foot_r_s = self._foot_r_ema

        axis = foot_r_s - foot_l_s
        axis_len = float(np.linalg.norm(axis))
        a_hat = axis / axis_len if axis_len > 1e-6 else np.array([1.0, 0.0, 0.0])
        p0 = 0.5 * (foot_l_s + foot_r_s)

        # ---- Lever l and tilt theta of CoM about the axis ----
        w = com - p0
        w_perp = w - np.dot(w, a_hat) * a_hat
        lever = float(np.linalg.norm(w_perp))
        if lever < 1e-6:
            # CoM sits on the axis -> treat as "at top" (theta = 0)
            cos_theta = 1.0
            r_hat = np.array([0.0, 0.0, 1.0])
        else:
            r_hat = w_perp / lever
            cos_theta = float(np.clip(r_hat[2], -1.0, 1.0))
        theta = float(np.arccos(cos_theta))

        # ---- Energy accounting ----
        e_need = self.g * lever * (1.0 - cos_theta)

        # Toward-top tangential direction (perp to axis and radial, z-up)
        t_hat = np.cross(a_hat, r_hat)
        t_norm = float(np.linalg.norm(t_hat))
        if t_norm < 1e-6:
            v_t = 0.0
        else:
            t_hat = t_hat / t_norm
            if t_hat[2] < 0.0:
                t_hat = -t_hat
            v_t = float(np.dot(com_vel, t_hat))

        e_have_signed = 0.5 * v_t * abs(v_t)  # signed kinetic energy toward top
        e_deficit = e_need - e_have_signed

        if e_deficit >= 0.0:
            e_score = float(np.exp(-e_deficit / self.e_scale_under))
        else:
            e_score = float(np.exp(-abs(e_deficit) / self.e_scale_over))

        # ---- Anti-hack foot gate ----
        h_foot = float(max(foot_l[2], foot_r[2]))
        denom = max(self.foot_h_max - self.foot_h_min, 1e-6)
        foot_factor = float(np.clip((self.foot_h_max - h_foot) / denom, 0.0, 1.0))

        # ---- Support polygon factor: CoM projection must be between feet ----
        # t = normalized position of CoM projection along foot-to-foot axis
        # t=0 at foot_l, t=1 at foot_r; ramp from 0 to 1 over support_ramp_width
        proj = float(np.dot(w, a_hat))
        t = 0.5 + proj / axis_len if axis_len > 1e-6 else 0.5
        rw = self.support_ramp_width
        if t < 0.0 or t > 1.0:
            support_factor = 0.0
        elif t < rw:
            support_factor = t / rw if rw > 1e-6 else 1.0
        elif t > 1.0 - rw:
            support_factor = (1.0 - t) / rw if rw > 1e-6 else 1.0
        else:
            support_factor = 1.0

        # ---- Lever factor: CoM-to-axis distance must be large enough ----
        # lever = perpendicular distance from CoM to foot-to-foot line in 3D
        # Too small -> robot is collapsed/hacking, not standing
        l_lo = self.lever_min
        l_hi = self.lever_max
        if lever < l_lo:
            lever_factor = 0.0
        elif lever >= l_hi:
            lever_factor = 1.0
        else:
            lever_factor = (lever - l_lo) / (l_hi - l_lo)

        potential = foot_factor * e_score * support_factor * lever_factor

        # ---- Balance diagnostic (for success detection in a later step) ----
        contacts = self._get_ground_contacts(ctx)
        both_feet = contacts["foot_left"] and contacts["foot_right"]
        clean = not (
            contacts["hand_left"] or contacts["hand_right"]
            or contacts["has_other_contact"]
        )
        com_speed = float(np.linalg.norm(com_vel))
        is_balanced = bool(
            both_feet and clean
            and np.degrees(theta) <= self.balance_theta_deg
            and com_speed <= self.balance_speed
        )

        # ---- Store ----
        self._potential = potential
        self._e_score = e_score
        self._foot_factor = foot_factor
        self._e_deficit = e_deficit
        self._e_need = e_need
        self._e_have = e_have_signed
        self._lever = lever
        self._theta_deg = float(np.degrees(theta))
        self._v_t = v_t
        self._com_h = float(com[2])
        self._com_speed = com_speed
        self._h_foot = h_foot
        self._is_balanced = 1.0 if is_balanced else 0.0
        self._both_feet = 1.0 if both_feet else 0.0
        self._support_factor = support_factor
        self._lever_factor = lever_factor

    # ---- Ground contact parsing (feet / hands / other) --------------------

    def _get_ground_contacts(self, ctx: ReadOnlySimContext) -> Dict[str, bool]:
        """Detect which robot bodies touch the ground (feet/hands/other)."""
        cv = ctx.accessor.get_derived_state(['contacts']).get('contacts')

        static_data = ctx.accessor.get_static_data()[self.agent_id]
        keypoint_names = static_data["keypoint_body_names"]

        contacts = {
            "foot_left": False,
            "foot_right": False,
            "hand_left": False,
            "hand_right": False,
            "has_other_contact": False,
        }

        if cv is None or cv['ncon'] == 0:
            return contacts

        foot_left_body = keypoint_names["foot_left"]
        foot_right_body = keypoint_names["foot_right"]
        hand_left_body = keypoint_names["hand_left"]
        hand_right_body = keypoint_names["hand_right"]

        suffix = ""
        if foot_left_body.endswith("_a"):
            suffix = "_a"
        elif foot_left_body.endswith("_b"):
            suffix = "_b"
        lower_arm_left_body = f"lower_arm_left{suffix}"
        lower_arm_right_body = f"lower_arm_right{suffix}"

        static_all = ctx.accessor.get_static_data()
        body_id_to_name = static_all.get('body_id_to_name', {})
        geom_id_to_name = static_all.get('geom_id_to_name', {})

        robot_aff = 1 if self.agent_id == 'robot_a' else 2

        aff1 = cv['aff1']
        aff2 = cv['aff2']
        geom1 = cv['geom1']
        geom2 = cv['geom2']
        body1 = cv['body1']
        body2 = cv['body2']
        force_mag = cv['force_mag']

        for i in range(cv['ncon']):
            if aff1[i] == 0 and aff2[i] == robot_aff:
                geom_env = geom_id_to_name.get(int(geom1[i]), '')
                body_robot = body_id_to_name.get(int(body2[i]), '')
            elif aff2[i] == 0 and aff1[i] == robot_aff:
                geom_env = geom_id_to_name.get(int(geom2[i]), '')
                body_robot = body_id_to_name.get(int(body1[i]), '')
            else:
                continue

            if float(force_mag[i]) < 1.0:
                continue

            if geom_env == 'ground':
                if body_robot == foot_left_body:
                    contacts["foot_left"] = True
                elif body_robot == foot_right_body:
                    contacts["foot_right"] = True
                elif body_robot in (hand_left_body, lower_arm_left_body):
                    contacts["hand_left"] = True
                elif body_robot in (hand_right_body, lower_arm_right_body):
                    contacts["hand_right"] = True
                else:
                    contacts["has_other_contact"] = True

        return contacts

    # ---- Outputs ----------------------------------------------------------

    def get_output(self) -> Dict[str, float]:
        return {
            "potential": self._potential,
            "e_score": self._e_score,
            "foot_factor": self._foot_factor,
            "e_deficit": self._e_deficit,
            "e_need": self._e_need,
            "e_have": self._e_have,
            "lever": self._lever,
            "theta_deg": self._theta_deg,
            "v_tangential": self._v_t,
            "com_height": self._com_h,
            "com_speed": self._com_speed,
            "foot_height": self._h_foot,
            "is_balanced": self._is_balanced,
            "both_feet": self._both_feet,
            "support_factor": self._support_factor,
            "lever_factor": self._lever_factor,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "g": self.g,
            "e_scale_under": self.e_scale_under,
            "e_scale_over": self.e_scale_over,
            "foot_h_min": self.foot_h_min,
            "foot_h_max": self.foot_h_max,
            "foot_ema_alpha": self.foot_ema_alpha,
            "support_ramp_width": self.support_ramp_width,
            "lever_min": self.lever_min,
            "lever_max": self.lever_max,
            "balance_theta_deg": self.balance_theta_deg,
            "balance_speed": self.balance_speed,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandupEnergyRewarder":
        return cls(**config)
