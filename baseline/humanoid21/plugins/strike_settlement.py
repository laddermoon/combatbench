"""Strike-settlement world plugin for the strike_step task.

Single-strike episode state machine (per learner, authoritative):

    PRE_HIT ──first hand→head/torso contact──▶ SETTLE
    SETTLE ──settle_steps action steps────────▶ RETRACT
    RETRACT ──retract combo held hold_steps───▶ terminate "strike_done"
    RETRACT ──retract_timeout_steps elapsed───▶ terminate "retract_timeout"

Semantics:

* A contact only *triggers* the event if its force >= ``min_hit_force``
  — a light graze never locks a hand and never starts SETTLE (s42c
  diagnosis: tap-and-retract farming produced strike_done episodes
  with Q as low as 0.02).  Once locked, ALL locked-hand contacts
  inside the window accrue damage regardless of force.
* The first legal hit locks the striking hand (left or right) — later
  contacts by the other hand or by feet do not extend the event and do
  not add to the strike score ``Q``.
* During SETTLE (``settle_steps`` action steps, starting with the hit
  step) damage is accumulated with the SAME formula as
  ``CombatScoringPlugin`` but ONLY for contacts where the locked hand
  is the attacker part.
* In RETRACT the score is frozen; the episode ends when the retract
  combo holds or when the retract timeout elapses (second half of the
  payment is withheld — see the experiment's r_strike channel).

Retract combo (all held for ``hold_steps`` consecutive action steps):

  1. locked hand 3D distance to the opponent's nearest {head, torso}
     body >= ``retract_dist``  — actual separation from the target;
  2. locked hand 3D distance to the learner's OWN torso <=
     ``hand_home_dist`` — the hand is back at the body (chambered
     guard OR relaxed hanging arm both count; a still-extended arm —
     toward the target, sideways or windmilled backward — does not);
  3. learner root height >= ``stand_height_min`` — upright enough that
     "retract" is not "fell over and rolled away".

The plugin is read-only (``require_mutator=False``): it never touches
physics, only contacts/derived state + ``ctx.metrics`` /
``ctx.request_termination``.  ``StrikeStateObserver`` mirrors the
metrics into per-step observer fields for trajectory building — the
metrics written here are the single source of truth.

Metrics written (per episode, keyed by learner agent id ``<aid>``):
    <aid>_strike_phase        int   0=PRE_HIT 1=SETTLE 2=RETRACT
    <aid>_strike_locked_hand  int   0=none 1=left 2=right
    <aid>_strike_step_damage  float locked-hand damage this action step
    <aid>_strike_damage_total float cumulative locked-hand damage (Q)
    <aid>_strike_hit_step     int   episode_step of first hit (-1 none)
    <aid>_strike_retract_done int   1 when retract combo completed
    <aid>_strike_gates        int   bitmask of retract gates on the last
                                    RETRACT check: bit0 upright,
                                    bit1 hand home, bit2 separated
    <aid>_strike_d_home       float locked hand → own torso distance (m),
                                    written once a hand is locked
    <aid>_strike_d_opp        float locked hand → opp {head,torso} min
                                    distance (m), written once locked

Termination reasons: ``strike_done`` / ``retract_timeout`` (issued for
ALL agents — the episode ends; neither equals "timeout", so learners
get true-termination bootstrap=0 semantics, which is intended).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.framework import BasePlugin, OBSERVER_DISPATCHER_PRIORITY
from envs.framework.context import SimContext
from envs.humanoid21.plugins import CombatScoringPlugin


PHASE_PRE_HIT: int = 0
PHASE_SETTLE: int = 1
PHASE_RETRACT: int = 2

HAND_NONE: int = 0
HAND_LEFT: int = 1
HAND_RIGHT: int = 2


class StrikeSettlementPlugin(BasePlugin):
    """First hand-hit → settlement window → retract → terminate."""

    def __init__(
        self,
        agent_id: str = "robot_a",
        settle_steps: int = 10,
        retract_dist: float = 0.5,
        hand_home_dist: float = 0.55,
        stand_height_min: float = 1.0,
        hold_steps: int = 3,
        retract_timeout_steps: int = 60,
        force_scale: float = 100.0,
        phy_step_dt: float = 0.002,
        min_hit_force: float = 250.0,
        end_after_settle: bool = False,
    ) -> None:
        if agent_id not in ("robot_a", "robot_b"):
            raise ValueError(f"agent_id must be robot_a/robot_b, got {agent_id!r}")
        self.agent_id = agent_id
        self.opp_id = "robot_b" if agent_id == "robot_a" else "robot_a"
        self._sfx = agent_id[-1]          # 'a' / 'b'
        self._osfx = self.opp_id[-1]
        self.settle_steps = int(settle_steps)
        self.retract_dist = float(retract_dist)
        self.hand_home_dist = float(hand_home_dist)
        self.stand_height_min = float(stand_height_min)
        self.hold_steps = int(hold_steps)
        self.retract_timeout_steps = int(retract_timeout_steps)
        self.force_scale = float(force_scale)
        self.phy_step_dt = float(phy_step_dt)
        self.min_hit_force = float(min_hit_force)
        # Simplified scoring: the SETTLE window itself is the whole
        # event — when it elapses, the episode ends "strike_done" with
        # the accumulated locked-hand damage as the score.  No RETRACT
        # phase, no hold, no retract timeout (user direction: drop the
        # retract stage entirely — it was the dominant failure mode).
        self.end_after_settle = bool(end_after_settle)

        self._phase = PHASE_PRE_HIT
        self._locked_hand = HAND_NONE
        self._settle_elapsed = 0
        self._retract_start_step: Optional[int] = None
        self._hold_count = 0
        self._step_damage = 0.0
        self._total_damage = 0.0

    # ------------------------------------------------------------------
    # Plugin contract
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"strike_settlement_{self.agent_id}"

    @property
    def priority(self) -> int:
        # Run before the observer dispatch so StrikeStateObserver reads
        # post-transition metrics of the current action step.
        return OBSERVER_DISPATCHER_PRIORITY + 1

    @property
    def require_mutator(self) -> bool:
        return False

    def to_blueprint(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "settle_steps": self.settle_steps,
            "retract_dist": self.retract_dist,
            "hand_home_dist": self.hand_home_dist,
            "stand_height_min": self.stand_height_min,
            "hold_steps": self.hold_steps,
            "retract_timeout_steps": self.retract_timeout_steps,
            "force_scale": self.force_scale,
            "phy_step_dt": self.phy_step_dt,
            "min_hit_force": self.min_hit_force,
            "end_after_settle": self.end_after_settle,
        }

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StrikeSettlementPlugin":
        return cls(**config)

    # ------------------------------------------------------------------
    # Metric keys
    # ------------------------------------------------------------------

    def _k(self, field: str) -> str:
        return f"{self.agent_id}_strike_{field}"

    def _write_metrics(self, ctx: SimContext, hit_step: int = -1) -> None:
        m = ctx.metrics
        m[self._k("phase")] = int(self._phase)
        m[self._k("locked_hand")] = int(self._locked_hand)
        m[self._k("step_damage")] = float(self._step_damage)
        m[self._k("damage_total")] = float(self._total_damage)
        if hit_step >= 0 or self._k("hit_step") not in m:
            m[self._k("hit_step")] = int(hit_step)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_pre_episode(self, ctx: SimContext) -> None:
        self._phase = PHASE_PRE_HIT
        self._locked_hand = HAND_NONE
        self._settle_elapsed = 0
        self._retract_start_step = None
        self._hold_count = 0
        self._step_damage = 0.0
        self._total_damage = 0.0
        self._write_metrics(ctx, hit_step=-1)
        ctx.metrics[self._k("retract_done")] = 0

    def on_pre_action_step(self, ctx: SimContext) -> None:
        self._step_damage = 0.0

    # ------------------------------------------------------------------
    # Contact resolution — hand-attacker variant of
    # CombatScoringPlugin._resolve_hit (ATTACK_PARTS restricted to hand,
    # plus which-hand attribution).
    # ------------------------------------------------------------------

    _part_category = staticmethod(CombatScoringPlugin._get_part_category)

    def _resolve_strike(
        self, body_a_name: str, body_b_name: str,
    ) -> Optional[int]:
        """Return HAND_LEFT/HAND_RIGHT if this contact is a learner
        hand striking an opponent damage part (head/torso), else None."""
        a_mine = body_a_name.endswith(f"_{self._sfx}")
        b_mine = body_b_name.endswith(f"_{self._sfx}")
        a_opp = body_a_name.endswith(f"_{self._osfx}")
        b_opp = body_b_name.endswith(f"_{self._osfx}")
        if not ((a_mine and b_opp) or (b_mine and a_opp)):
            return None

        cat_a = self._part_category(body_a_name)
        cat_b = self._part_category(body_b_name)

        attacker: Optional[str] = None
        if a_mine and cat_a == "hand" and cat_b in ("head", "torso"):
            attacker = body_a_name
        elif b_mine and cat_b == "hand" and cat_a in ("head", "torso"):
            attacker = body_b_name
        if attacker is None:
            return None

        base = attacker[:-2] if attacker.endswith(("_a", "_b")) else attacker
        if "hand_left" in base:
            return HAND_LEFT
        if "hand_right" in base:
            return HAND_RIGHT
        return None

    def _contact_damage(self, force: float, defender_part: str) -> float:
        part_weight = CombatScoringPlugin.DAMAGE_RULES.get(defender_part, 0.0)
        effective = min(force, CombatScoringPlugin.MAX_DAMAGE_FORCE)
        return (
            part_weight
            * (effective / self.force_scale) ** 2
            * self.phy_step_dt
        )

    def on_post_phy_step(self, ctx: SimContext) -> None:
        if self._phase == PHASE_RETRACT:
            return
        derived = ctx.accessor.get_derived_state(["contacts"])
        cv = derived.get("contacts")
        if cv is None or cv["ncon"] == 0:
            return

        static_data = ctx.accessor.get_static_data()
        body_id_to_name = static_data.get("body_id_to_name", {})

        rr_mask = (cv["aff1"] != 0) & (cv["aff2"] != 0) & (cv["aff1"] != cv["aff2"])
        for idx in np.where(rr_mask)[0]:
            force = float(cv["force_mag"][idx])
            if force <= 0:
                continue
            b1 = int(cv["body1"][idx])
            b2 = int(cv["body2"][idx])
            a1_is_a = int(cv["aff1"][idx]) == 1
            body_a = body_id_to_name.get(b1 if a1_is_a else b2, "")
            body_b = body_id_to_name.get(b2 if a1_is_a else b1, "")
            hand = self._resolve_strike(body_a, body_b)
            if hand is None:
                continue

            if self._phase == PHASE_PRE_HIT:
                if force < self.min_hit_force:
                    continue          # graze — not a strike event
                self._locked_hand = hand
                self._phase = PHASE_SETTLE
                self._settle_elapsed = 0
                self._write_metrics(ctx, hit_step=int(ctx.episode_step))
            elif hand != self._locked_hand:
                # Contacts by the other hand do not join the event.
                continue

            # Defender part: whatever _resolve_hit category the opponent
            # body was — recompute for the damage weight.
            a_mine = body_a.endswith(f"_{self._sfx}")
            opp_body = body_b if a_mine else body_a
            opp_cat = self._part_category(opp_body)
            if opp_cat not in ("head", "torso"):
                continue
            dmg = self._contact_damage(force, opp_cat)
            self._step_damage += dmg
            self._total_damage += dmg
            ctx.metrics[self._k("step_damage")] = float(self._step_damage)
            ctx.metrics[self._k("damage_total")] = float(self._total_damage)

    # ------------------------------------------------------------------
    # Action-step bookkeeping: settle window, retract combo, timeout
    # ------------------------------------------------------------------

    def _update_geometry(self, ctx: SimContext) -> int:
        """Compute retract-gate geometry; writes d_home / d_opp /
        gates metrics every call.  Returns the gates bitmask
        (bit0 upright, bit1 hand-home, bit2 separated)."""
        core = ctx.accessor.get_core_state()
        self_core = core.get(self.agent_id)
        if self_core is None:
            ctx.metrics[self._k("gates")] = 0
            return 0

        gates = 0
        # bit0: upright gate — root height
        root_pos = np.asarray(self_core.get("root_pos", [0, 0, 0]), dtype=np.float64)
        if float(root_pos[2]) >= self.stand_height_min:
            gates |= 1

        derived = ctx.accessor.get_derived_state([self.agent_id, self.opp_id])
        self_bx = derived.get(self.agent_id, {}).get("body_xpos", {})
        opp_bx = derived.get(self.opp_id, {}).get("body_xpos", {})
        side = "left" if self._locked_hand == HAND_LEFT else "right"
        hand_raw = self_bx.get(f"hand_{side}_{self._sfx}")
        torso_self_raw = self_bx.get(f"torso_{self._sfx}")
        if hand_raw is not None and torso_self_raw is not None:
            hand = np.asarray(hand_raw, dtype=np.float64)

            # hand → own torso, metric + bit1 gate
            d_home = float(np.linalg.norm(
                hand - np.asarray(torso_self_raw, dtype=np.float64)))
            ctx.metrics[self._k("d_home")] = d_home
            if d_home <= self.hand_home_dist:
                gates |= 2

            # hand → opp {head,torso} min, metric + bit2 gate
            targets = [
                opp_bx.get(f"{p}_{self._osfx}") for p in ("head", "torso")
            ]
            dists = [
                float(np.linalg.norm(hand - np.asarray(t, dtype=np.float64)))
                for t in targets if t is not None
            ]
            if dists:
                d_opp = min(dists)
                ctx.metrics[self._k("d_opp")] = d_opp
                if d_opp >= self.retract_dist:
                    gates |= 4

        ctx.metrics[self._k("gates")] = gates
        return gates

    def _retract_geometry_ok(self, ctx: SimContext) -> bool:
        return self._update_geometry(ctx) == 0b111

    def on_post_action_step(self, ctx: SimContext) -> None:
        # Once a hand is locked, keep the retract geometry metrics
        # (d_home / d_opp / gates) current every action step — the
        # experiment's r_retract channel consumes the series.
        if self._locked_hand != HAND_NONE:
            self._update_geometry(ctx)

        if self._phase == PHASE_SETTLE:
            self._settle_elapsed += 1
            if self._settle_elapsed >= self.settle_steps:
                if self.end_after_settle:
                    ctx.request_termination("strike_done")
                else:
                    self._phase = PHASE_RETRACT
                    self._retract_start_step = int(ctx.episode_step)
                    self._hold_count = 0

        elif self._phase == PHASE_RETRACT:
            start = self._retract_start_step or int(ctx.episode_step)
            if int(ctx.episode_step) - start >= self.retract_timeout_steps:
                ctx.request_termination("retract_timeout")
            elif (ctx.metrics.get(self._k("gates"), 0) == 0b111):
                self._hold_count += 1
                if self._hold_count == self.hold_steps:
                    ctx.metrics[self._k("retract_done")] = 1
                    ctx.request_termination("strike_done")
            else:
                self._hold_count = 0

        self._write_metrics(ctx)
