"""4-stage standing-balance potential rewarder (dead-zone-free design).

Stage Definitions
-----------------
Stage 1 (rollover):
    从非 Stage 2/3/4 的状态 → 到俯身趴在地上的状态。
    Entry gate: none (always starts here at episode reset).
    Target signal: f_score (torso face-down orientation).
    The robot starts in a random fallen state (often face-up); it must
    roll over to a prone position.

Stage 2 (establish support):
    从俯身趴在地 → 到只有手和脚接触地面的支撑状态。
    Entry gate: f_score ≥ F_ENTER (rollover achieved).
    Target signal: contact_score (hand/foot proximity × no-extra-contact).
    The robot must bring hands/feet toward the ground AND lift other body
    parts (knees, torso, etc.) off the ground.  Only when extra_contact_count
    == 0 (only hands/feet touching) can it advance to Stage 3.

Stage 3 (close hand-foot horizontal distance):
    从任意的手脚支撑状态 → 到双手双脚（水平方向）靠近的手脚支撑状态。
    Entry gate: only hands/feet on ground (extra_contact_count == 0),
                regardless of torso orientation.
    Target signal: d_score (hand-midpoint to foot-midpoint distance,
                   XY-PLANE ONLY — vertical separation is ignored).
    The robot must pull its feet underneath its body (equivalently, bring
    its hands back over its feet) while maintaining only-hands-feet support.

    Why XY-only: a 3D distance makes this condition geometrically
    INCOMPATIBLE with standing (standing puts hands ~1.3m above the feet,
    so a 3D d_score would be 0 while upright).  Since Stage 4 requires
    Stage 3's condition to stay satisfied, a 3D d_score creates a hard
    dead zone where standing up is punished.  Projecting onto XY keeps the
    intended "coil up / feet under body" semantics while remaining fully
    compatible with an upright posture.

Stage 4 (stand up on two feet):
    从手脚靠近的支撑状态 → 到双脚站立。
    Entry gate: only hands/feet on ground + d_score ≥ D_GATE.
    Target signal: p4 = w_foot * h_score.
    - w_foot = F_foot / (F_foot + F_hand): continuous load transfer from
      hands to feet.  This replaces the binary "hands off ground" gate
      with a smooth gradient, eliminating the discrete-jump dead zone.
      The robot can shift weight to feet WITHOUT lifting hands, getting
      continuous reward improvement.  Using hands as crutches to raise
      the torso increases F_hand → decreases w_foot → decreases p4,
      automatically blocking the crutch trap.
    - h_score: normalized torso height from crouch (H_CROUCH) to the
      validated standing height (H_STAND = 1.28 m).
    - The product (no additive constant) ensures the robot must FIRST
      transfer weight to feet, THEN rise, and that the full remaining
      potential range is spent on actually rising.  An additive constant
      here would hand out most of Stage 4's band for merely balancing on
      the feet, leaving too little gradient for the hard part (rising).

Potential Bands
---------------
  Stage 1 [0.00, 0.20):  potential = 0.20 * f_score
  Stage 2 [0.20, 0.40):  potential = 0.20 + 0.20 * contact_score
  Stage 3 [0.40, 0.60):  potential = 0.40 + 0.20 * d_score
  Stage 4 [0.60, 1.00]:  potential = 0.60 + 0.40 * p4

Stage 4 gets the widest band (0.40) because it is the hardest stage —
it is a reach-and-hold goal (standing balance), not reach-and-stable.
The extra resolution helps the critic distinguish small improvements
in weight transfer and height.

Stage is determined top-down each step: check Stage 4 first, then 3, 2, 1.
No hysteresis — Stage 3/4 do not depend on f_score, and the load-based
w_foot is continuous so there is no discrete jump to smooth.

Known Risk Points (待观察):
  1. 瞬时跳 Stage 3/4：短暂腾空时 extra_count == 0 可瞬间进入高 stage，
     下一步掉回。delta 模式下产生奖励尖峰。
  2. Stage 4 → Stage 1 跳变：在 Stage 4 时如果其它部位触地且 f_score 低，
     直接从 Stage 4 跳到 Stage 1，potential 从 ~0.6+ 暴跌到 ~0.2 以下。
     delta 模式下尤其显著。
  3. Stage 4 顶部信号坍缩：站立后 Δφ ≈ 0，delta 模式可能探索死锁。
     dense 模式（保持奖励）预期表现最好。
  4. 反复重入：站起来 → 摔 → 再站起来可反复刷 delta 进步奖励。
     若 max_potential >> final_potential 则在刷。
  5. 摆臂被惩罚：起身时张开手臂平衡会推大 XY 距离，若跌破 D_GATE 会被
     踢出 Stage 4（potential 掉 ~0.1+）。D_GATE 已放宽到 0.6（允许 XY
     距离达 0.52 m）留出余量，但极端摆臂仍可能触发。
  6. H_CROUCH 未标定：Stage 4 入场时的实际 torso 高度尚未实测。若实际
     入场高度低于 H_CROUCH，h_score 会在底部被 clip 到 0，形成新的死区。
     首轮训练需检查 max_h_torso / min h_torso 来标定。
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from envs.framework import BaseObserverPlugin, ReadOnlySimContext

# --- Tunable constants ---
H_HAND_MAX = 0.3   # hand height (m) at which proximity saturates to 0
H_FOOT_MAX = 0.3   # foot height (m) at which proximity saturates to 0
# NOTE: d_hf is an XY-plane (horizontal) distance — see Stage 3 docstring.
D_MAX = 1.0        # horizontal hand-foot distance (m) at which d_score saturates to 0
D_MIN = 0.2        # horizontal hand-foot distance (m) at which d_score saturates to 1.0
OTHER_PENALTY_K = 0.5  # soft penalty coefficient for extra body contacts

# --- Stage gates ---
F_ENTER = 0.8      # f_score needed to enter Stage 2
# D_GATE = 0.6 allows a horizontal hand-foot distance up to 0.52 m, leaving
# room for the arm swing that naturally accompanies rising.  (Reference:
# the standing keyframe holds a guard stance with hands 0.37 m forward,
# which scores d_score = 0.79; arms at the sides score ~1.0.)
D_GATE = 0.6       # d_score needed to enter Stage 4 from Stage 3

# --- Stage 4 constants ---
# Torso height is used (not COM): the 1.28 m standing value is validated.
H_CROUCH = 0.35    # torso height (m) at which h_score saturates to 0 (deep crouch)
H_STAND = 1.28     # torso height (m) at which h_score saturates to 1.0 (standing)
F_LOAD_MIN = 10.0  # minimum total contact force (N) to avoid w_foot division by zero


class StandingBalance4StageRewarder(BaseObserverPlugin):
    """Dead-zone-free 4-stage standing-balance potential function."""

    def __init__(self, agent_id: str = "robot_a"):
        self.agent_id = agent_id
        self._stage: int = 1
        self._potential: float = 0.0
        self._f_score: float = 0.0
        self._contact_score: float = 0.0
        self._d_score: float = 0.0
        self._d_hf: float = 0.0
        self._w_foot: float = 0.0
        self._h_score: float = 0.0
        self._h_torso: float = 0.0

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._stage = 1
        self._potential = 0.0
        self._f_score = 0.0
        self._contact_score = 0.0
        self._d_score = 0.0
        self._d_hf = 0.0
        self._w_foot = 0.0
        self._h_score = 0.0
        self._h_torso = 0.0

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        derived_state = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]
        static_data = ctx.accessor.get_static_data()[self.agent_id]

        # --- Signal 1: f_down (rollover orientation) ---
        torso_body_name = static_data["keypoint_body_names"]["torso"]
        body_xquat_dict = derived_state.get("body_xquat", {})
        q_torso = body_xquat_dict.get(
            torso_body_name, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        w, x, y, z = q_torso[0], q_torso[1], q_torso[2], q_torso[3]
        x_world_z = 2.0 * (x * z - w * y)
        f_down = -x_world_z
        f_score = float(np.clip((f_down + 1.0) / 2.0, 0.0, 1.0))

        # --- Signal 2: hand/foot proximity to ground + other-contact penalty ---
        body_xpos = derived_state.get("body_xpos", {})
        hand_l_name = static_data["keypoint_body_names"]["hand_left"]
        hand_r_name = static_data["keypoint_body_names"]["hand_right"]
        foot_l_name = static_data["keypoint_body_names"]["foot_left"]
        foot_r_name = static_data["keypoint_body_names"]["foot_right"]

        h_hand_l = float(body_xpos.get(hand_l_name, np.zeros(3, dtype=np.float32))[2])
        h_hand_r = float(body_xpos.get(hand_r_name, np.zeros(3, dtype=np.float32))[2])
        h_foot_l = float(body_xpos.get(foot_l_name, np.zeros(3, dtype=np.float32))[2])
        h_foot_r = float(body_xpos.get(foot_r_name, np.zeros(3, dtype=np.float32))[2])

        hand_proximity = float(np.clip(1.0 - (h_hand_l + h_hand_r) / (2.0 * H_HAND_MAX), 0.0, 1.0))
        foot_proximity = float(np.clip(1.0 - (h_foot_l + h_foot_r) / (2.0 * H_FOOT_MAX), 0.0, 1.0))
        support_score = (hand_proximity + foot_proximity) / 2.0

        contacts = self._get_detailed_contacts(ctx)
        extra_count = contacts["extra_contact_count"]
        other_penalty = 1.0 / (1.0 + OTHER_PENALTY_K * extra_count)

        contact_score = support_score * other_penalty

        # --- Signal 3: hand-foot midpoint distance (XY plane only) ---
        # Vertical separation is deliberately ignored so that this condition
        # stays satisfiable while standing upright (see Stage 3 docstring).
        hand_mid = (
            body_xpos.get(hand_l_name, np.zeros(3, dtype=np.float32))[:3]
            + body_xpos.get(hand_r_name, np.zeros(3, dtype=np.float32))[:3]
        ) / 2.0
        foot_mid = (
            body_xpos.get(foot_l_name, np.zeros(3, dtype=np.float32))[:3]
            + body_xpos.get(foot_r_name, np.zeros(3, dtype=np.float32))[:3]
        ) / 2.0
        d_hf = float(np.linalg.norm(hand_mid[:2] - foot_mid[:2]))
        d_score = float(np.clip((D_MAX - d_hf) / (D_MAX - D_MIN), 0.0, 1.0))

        # --- Signal 4: load transfer + torso height (Stage 4 target) ---
        f_hand = contacts["hand_force_total"]
        f_foot = contacts["foot_force_total"]
        total_load = f_hand + f_foot
        if total_load >= F_LOAD_MIN:
            w_foot = f_foot / total_load
        else:
            # Insufficient contact force — can't meaningfully transfer load.
            # Keep w_foot at 0 so Stage 4 potential stays low.
            w_foot = 0.0

        # Torso height (validated: 1.28 m when standing)
        h_torso = float(body_xpos.get(torso_body_name, np.zeros(3, dtype=np.float32))[2])
        h_score = float(np.clip((h_torso - H_CROUCH) / (H_STAND - H_CROUCH), 0.0, 1.0))

        # Stage 4 target: pure product, no additive constant.
        # First transfer weight to feet (w_foot), then rise (h_score).
        # The product blocks crutching (pushing up on the hands raises
        # F_hand → lowers w_foot → lowers p4), and using no additive
        # constant spends the whole Stage 4 band on actually rising.
        p4 = w_foot * h_score

        # --- Stage determination (top-down, before potential) ---
        # Stage 4: only hands/feet on ground + d_score ≥ D_GATE.
        # Stage 3: only hands/feet on ground (regardless of orientation).
        # Stage 2: rolled over (f_score ≥ F_ENTER) but still has extra contacts.
        # Stage 1: not yet rolled over.
        hf_contact = (
            contacts["hand_left"] or contacts["hand_right"]
            or contacts["foot_left"] or contacts["foot_right"]
        )
        only_hf_contact = (extra_count == 0 and hf_contact)

        if only_hf_contact and d_score >= D_GATE:
            stage = 4
        elif only_hf_contact:
            stage = 3
        elif f_score >= F_ENTER:
            stage = 2
        else:
            stage = 1

        # --- Stage-dependent potential ---
        # Each stage owns a [lo, hi] band and optimizes only its own target
        # signal.  Earlier stages are locked at their band ceiling, giving a
        # clean focused gradient with no interference between stages.
        # Stage 4 gets the widest band (0.40) for maximum resolution.
        if stage == 1:
            potential = 0.20 * f_score
        elif stage == 2:
            potential = 0.20 + 0.20 * contact_score
        elif stage == 3:
            potential = 0.40 + 0.20 * d_score
        else:
            potential = 0.60 + 0.40 * p4

        self._stage = stage
        self._potential = float(potential)
        self._f_score = f_score
        self._contact_score = contact_score
        self._d_score = d_score
        self._d_hf = d_hf
        self._w_foot = float(w_foot)
        self._h_score = h_score
        self._h_torso = h_torso

    def _get_detailed_contacts(self, ctx: ReadOnlySimContext) -> Dict[str, Any]:
        """Parse ground contacts for feet, hands, and other body parts.

        Also accumulates per-category contact forces (force_mag) for
        load-transfer computation in Stage 4.
        """
        derived_state = ctx.accessor.get_derived_state(['contacts'])
        cv = derived_state.get('contacts')

        static_data = ctx.accessor.get_static_data()[self.agent_id]
        keypoint_names = static_data["keypoint_body_names"]

        contacts = {
            "foot_left": False,
            "foot_right": False,
            "hand_left": False,
            "hand_right": False,
            "has_other_contact": False,
            "extra_contact_count": 0,
            "extra_contact_bodies": set(),
            "hand_force_total": 0.0,
            "foot_force_total": 0.0,
        }

        if cv is None or cv['ncon'] == 0:
            return contacts

        foot_left_body = keypoint_names["foot_left"]
        foot_right_body = keypoint_names["foot_right"]
        hand_left_body = keypoint_names["hand_left"]
        hand_right_body = keypoint_names["hand_right"]

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
                f = float(force_mag[i])
                if body_robot == foot_left_body:
                    contacts["foot_left"] = True
                    contacts["foot_force_total"] += f
                elif body_robot == foot_right_body:
                    contacts["foot_right"] = True
                    contacts["foot_force_total"] += f
                elif body_robot == hand_left_body:
                    contacts["hand_left"] = True
                    contacts["hand_force_total"] += f
                elif body_robot == hand_right_body:
                    contacts["hand_right"] = True
                    contacts["hand_force_total"] += f
                else:
                    contacts["has_other_contact"] = True
                    contacts["extra_contact_bodies"].add(body_robot)

        contacts["extra_contact_count"] = len(contacts["extra_contact_bodies"])

        return contacts

    def get_output(self) -> Dict[str, float]:
        return {
            "stage": float(self._stage),
            "potential": self._potential,
            "f_score": self._f_score,
            "contact_score": self._contact_score,
            "d_score": self._d_score,
            "d_hf": self._d_hf,
            "w_foot": self._w_foot,
            "h_score": self._h_score,
            "h_torso": self._h_torso,
        }

    def to_blueprint(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> "StandingBalance4StageRewarder":
        return cls(**config)
