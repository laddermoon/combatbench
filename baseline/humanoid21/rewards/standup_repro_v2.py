"""Parameterized standup potential rewarder for staged reproduction.

Supports V1 (smooth) and V2 (gapped, velocity-gated, wall-aware) modes
via constructor parameters, allowing stage transitions without code changes.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext


class StandupReproRewarder(BaseObserverPlugin):
    """Parameterized potential rewarder supporting V1 and V2 modes."""

    def __init__(
        self,
        agent_id: str = "robot_a",
        mode: str = "v2",
        vel_gate: bool = False,
        wall_aware: bool = False,
        s5_h_thr: float = 0.75,
        s5_u_thr: float = 0.85,
        s5_base: float = 0.85,
        s5_range: float = 0.15,
        s5_h_range: float = 0.15,
        s5_u_range: float = 0.15,
        s5_v_power: int = 1,
    ):
        self.agent_id = agent_id
        self.mode = mode
        self.vel_gate = vel_gate
        self.wall_aware = wall_aware
        self.s5_h_thr = s5_h_thr
        self.s5_u_thr = s5_u_thr
        self.s5_base = s5_base
        self.s5_range = s5_range
        self.s5_h_range = s5_h_range
        self.s5_u_range = s5_u_range
        self.s5_v_power = s5_v_power
        self._stage = 0
        self._potential = 0.0
        self._has_wall = False

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._stage = 0
        self._potential = 0.0
        self._has_wall = False

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        core_state = ctx.accessor.get_core_state()[self.agent_id]
        derived_state = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]

        h = float(core_state["root_pos"][2])
        u = float(np.asarray(derived_state["uprightness"], dtype=np.float32).reshape(-1)[0])

        sd = ctx.accessor.get_static_data()[self.agent_id]
        torso_bn = sd["keypoint_body_names"]["torso"]
        bxq = derived_state.get("body_xquat", {})
        q = bxq.get(torso_bn, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        w, x, y, z = q[0], q[1], q[2], q[3]
        f_down = -(2.0 * (x * z - w * y))

        jv = np.asarray(core_state["joint_vel_norm"], dtype=np.float32)
        mvel = float(np.mean(np.abs(jv)))

        ct = self._get_detailed_contacts(ctx)
        fl, fr = ct["foot_left"], ct["foot_right"]
        kl, kr = ct["shin_left"], ct["shin_right"]
        hl, hr = ct["hand_left"], ct["hand_right"]
        other = ct["has_other_contact"]
        has_wall = ct["has_wall_contact"]

        hh = hl or hr
        hf = fl or fr
        hk = kl or kr

        if self.mode == "v1":
            self._v1(h, u, f_down, mvel, fl, fr, hh, hf, other)
        else:
            self._v2(h, u, f_down, mvel, fl, fr, hk, hh, hf, other, has_wall)
        self._has_wall = has_wall

    def _v1(self, h, u, f_down, mvel, fl, fr, hh, hf, other):
        if fl and fr and not hh and u > 0.85 and h > 0.75 and not other:
            s = 5; hs = np.clip((h-0.75)/0.15, 0, 1); us = np.clip((u-0.85)/0.15, 0, 1)
            vs = np.exp(-mvel); p = 0.75 + 0.25*hs*us*vs
        elif fl and fr and not hh and not other:
            s = 4; hs = np.clip((h-0.40)/0.35, 0, 1); us = np.clip((u-0.30)/0.55, 0, 1)
            p = 0.60 + 0.15*hs*us
        elif hf and not hh and not other:
            s = 3; hs = np.clip((h-0.40)/0.35, 0, 1); us = np.clip((u-0.30)/0.55, 0, 1)
            p = 0.45 + 0.15*hs*us
        elif hh and hf:
            s = 2; hs = np.clip((h-0.20)/0.40, 0, 1); us = np.clip(u/0.80, 0, 1)
            p = 0.30 + 0.15*hs*us
        elif hh:
            s = 1; hs = np.clip((h-0.15)/0.30, 0, 1); us = np.clip(u/0.80, 0, 1)
            p = 0.20 + 0.10*hs*us
        else:
            s = 0; fs = np.clip((f_down+1.0)/2.0, 0, 1); p = 0.20*fs
        self._stage = s; self._potential = float(p)

    def _v2(self, h, u, f_down, mvel, fl, fr, hk, hh, hf, other, has_wall):
        s5h, s5u, s5b, s5r = self.s5_h_thr, self.s5_u_thr, self.s5_base, self.s5_range
        s5hr, s5ur, s5vp = self.s5_h_range, self.s5_u_range, self.s5_v_power
        vg, wa = self.vel_gate, self.wall_aware
        nw = not has_wall if wa else True
        s, p = 0, 0.0

        if (fl and fr and not hh and u > s5u and h > s5h and not other and nw
                and (not vg or mvel < 2.0)):
            s = 5
            hs = np.clip((h-s5h)/s5hr, 0, 1); us = np.clip((u-s5u)/s5ur, 0, 1)
            vs = np.exp(-mvel); p = s5b + s5r*hs*us*(vs**s5vp)
        elif wa and (fl and fr and not hh and u > s5u and h > s5h and not other
                and has_wall and (not vg or mvel < 2.0)):
            s = 3
            hs = np.clip((h-s5h)/s5hr, 0, 1); us = np.clip((u-s5u)/s5ur, 0, 1)
            vs = np.exp(-mvel); p = 0.40 + 0.08*hs*us*vs
        elif vg and (fl and fr and not hh and u > s5u and h > s5h and not other
                and nw and mvel >= 2.0):
            s = 4
            hs = np.clip((h-s5h)/s5hr, 0, 1); us = np.clip((u-s5u)/s5ur, 0, 1)
            vs = np.exp(-mvel); p = 0.80 + 0.05*hs*us*vs
        elif fl and fr and not hh and not other and nw:
            s = 4
            hs = np.clip((h-0.35)/0.40, 0, 1); us = np.clip((u-0.20)/0.65, 0, 1)
            p = 0.65 + 0.20*hs*us
        elif wa and fl and fr and not hh and not other and has_wall:
            s = 2
            hs = np.clip((h-0.25)/0.50, 0, 1); us = np.clip((u-0.10)/0.75, 0, 1)
            p = 0.30 + 0.08*hs*us
        elif fl and fr and hk and not hh and not other and nw:
            s = 4
            hs = np.clip((h-0.25)/0.50, 0, 1); us = np.clip((u-0.10)/0.75, 0, 1)
            p = 0.55 + 0.10*hs*us
        elif hf and not hh and not other and nw:
            s = 3
            hs = np.clip((h-0.35)/0.40, 0, 1); us = np.clip((u-0.20)/0.65, 0, 1)
            p = 0.40 + 0.15*hs*us
        elif hh and hf:
            s = 2
            hs = np.clip((h-0.20)/0.40, 0, 1); us = np.clip(u/0.80, 0, 1)
            p = 0.25 + 0.15*hs*us
        elif hh:
            s = 1
            hs = np.clip((h-0.15)/0.30, 0, 1); us = np.clip(u/0.80, 0, 1)
            p = 0.15 + 0.10*hs*us
        else:
            s = 0; fs = np.clip((f_down+1.0)/2.0, 0, 1); p = 0.15*fs
        self._stage = s; self._potential = float(p)

    def _get_detailed_contacts(self, ctx: ReadOnlySimContext) -> Dict[str, bool]:
        derived_state = ctx.accessor.get_derived_state(['contacts'])
        cv = derived_state.get('contacts')
        sd = ctx.accessor.get_static_data()[self.agent_id]
        kn = sd["keypoint_body_names"]

        contacts = {"foot_left": False, "foot_right": False, "hand_left": False,
                     "hand_right": False, "shin_left": False, "shin_right": False,
                     "has_other_contact": False, "has_wall_contact": False,
                     "wall_hand_contact": False}

        if cv is None or cv['ncon'] == 0:
            return contacts

        fl_b, fr_b = kn["foot_left"], kn["foot_right"]
        hl_b, hr_b = kn["hand_left"], kn["hand_right"]
        suffix = ""
        if fl_b.endswith("_a"): suffix = "_a"
        elif fl_b.endswith("_b"): suffix = "_b"
        sl_b = f"shin_left{suffix}"; sr_b = f"shin_right{suffix}"
        lal_b = f"lower_arm_left{suffix}"; lar_b = f"lower_arm_right{suffix}"

        sa = ctx.accessor.get_static_data()
        bid2n = sa.get('body_id_to_name', {})
        gid2n = sa.get('geom_id_to_name', {})
        robot_aff = 1 if self.agent_id == 'robot_a' else 2

        for i in range(cv['ncon']):
            if cv['aff1'][i] == 0 and cv['aff2'][i] == robot_aff:
                ge = gid2n.get(int(cv['geom1'][i]), '')
                br = bid2n.get(int(cv['body2'][i]), '')
            elif cv['aff2'][i] == 0 and cv['aff1'][i] == robot_aff:
                ge = gid2n.get(int(cv['geom2'][i]), '')
                br = bid2n.get(int(cv['body1'][i]), '')
            else:
                continue
            if float(cv['force_mag'][i]) < 1.0:
                continue
            if ge == 'ground':
                if br == fl_b: contacts["foot_left"] = True
                elif br == fr_b: contacts["foot_right"] = True
                elif br in (hl_b, lal_b): contacts["hand_left"] = True
                elif br in (hr_b, lar_b): contacts["hand_right"] = True
                elif br == sl_b: contacts["shin_left"] = True
                elif br == sr_b: contacts["shin_right"] = True
                else: contacts["has_other_contact"] = True
            else:
                contacts["has_wall_contact"] = True
                if br in (hl_b, lal_b, hr_b, lar_b):
                    contacts["wall_hand_contact"] = True
        return contacts

    def get_output(self) -> Dict[str, float]:
        return {
            "stage": float(self._stage),
            "potential": self._potential,
            "has_wall_contact": 1.0 if self._has_wall else 0.0,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id, "mode": self.mode, "vel_gate": self.vel_gate,
                "wall_aware": self.wall_aware, "s5_h_thr": self.s5_h_thr,
                "s5_u_thr": self.s5_u_thr, "s5_base": self.s5_base,
                "s5_range": self.s5_range, "s5_h_range": self.s5_h_range,
                "s5_u_range": self.s5_u_range, "s5_v_power": self.s5_v_power}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandupReproRewarder":
        return cls(**config)
