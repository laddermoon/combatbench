"""Stage-three parameterized motion reference over a validated MainPath."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from .contracts import FOOT_NAMES, FootName, MainPathState, _vec3
from .gait_model import GaitModel, default_gait_model
from .trajectory import MainPath


@dataclass(frozen=True)
class MotionReferenceParams:
    """Parameterized coordination template for the motion-reference layer.

    All coordination rules are explicit numbers: gait posture uses a small
    set of gains mapping foot stride asymmetry into pelvis/torso twist and
    contralateral arm swing. ``arm_lift`` is the free regulation channel:
    it is derived from the current angular-momentum level and clipped by
    ``arm_lift_max``.
    """

    template: str = "walk"
    hip_twist_gain: float = 0.25
    hip_twist_max: float = 0.18
    torso_counter_twist_gain: float = 0.20
    torso_twist_max: float = 0.15
    arm_swing_gain: float = 0.80
    arm_swing_max: float = 0.55
    arm_lift_base: float = 0.0
    arm_lift_gain: float = 0.60
    arm_lift_max: float = 0.50
    elbow_angle: float = 0.45
    trunk_pitch: float = 0.08
    pelvis_com_offset_z: float = -0.02
    knee_flexion_base: float = 0.25
    knee_flexion_swing_extra: float = 0.55
    swing_foot_pitch_max: float = 0.30
    ankle_height: float = 0.08
    upper_leg_ratio: float = 0.55
    torso_ratio: float = 0.62
    head_ratio: float = 0.22
    shoulder_width_ratio: float = 1.60
    hip_width_ratio: float = 0.90
    upper_arm_ratio: float = 0.38
    forearm_ratio: float = 0.32

    def __post_init__(self) -> None:
        if self.template not in ("walk", "boxing_guard"):
            raise ValueError(f"unknown template {self.template!r}")
        for name in (
            "hip_twist_max", "torso_twist_max", "arm_swing_max",
            "arm_lift_max", "elbow_angle", "trunk_pitch",
            "knee_flexion_base", "knee_flexion_swing_extra",
            "swing_foot_pitch_max", "ankle_height",
            "upper_leg_ratio", "torso_ratio", "head_ratio",
            "shoulder_width_ratio", "hip_width_ratio",
            "upper_arm_ratio", "forearm_ratio",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")

    @classmethod
    def for_template(
        cls,
        template: str,
        gait: GaitModel,
    ) -> "MotionReferenceParams":
        """Return default coordination parameters for a named posture."""
        if template == "boxing_guard":
            return cls(
                template=template,
                hip_twist_gain=0.30,
                torso_counter_twist_gain=0.25,
                arm_swing_gain=0.20,
                arm_swing_max=0.25,
                arm_lift_base=0.55,
                arm_lift_gain=0.30,
                arm_lift_max=1.10,
                elbow_angle=1.70,
                trunk_pitch=0.14,
            )
        if template != "walk":
            raise ValueError(f"unknown template {template!r}")
        return cls(template=template)


@dataclass(frozen=True)
class MotionReferenceState:
    """Parameterized key-shape reference for one instant of a MainPath."""

    time: float
    heading_yaw: float
    pelvis_pos: np.ndarray
    pelvis_yaw_offset: float
    trunk_yaw_offset: float
    trunk_pitch: float
    left_foot_pos: np.ndarray
    right_foot_pos: np.ndarray
    left_foot_pitch: float
    right_foot_pitch: float
    left_cop_world: np.ndarray
    right_cop_world: np.ndarray
    left_support_active: bool
    right_support_active: bool
    left_knee_flexion: float
    right_knee_flexion: float
    left_arm_swing: float
    right_arm_swing: float
    left_arm_swing_rate: float
    right_arm_swing_rate: float
    left_arm_lift: float
    right_arm_lift: float
    left_elbow_angle: float
    right_elbow_angle: float
    left_hand_hint: np.ndarray
    right_hand_hint: np.ndarray
    com_pos: np.ndarray
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.time):
            raise ValueError("time must be finite")
        if not np.isfinite(self.heading_yaw):
            raise ValueError("heading_yaw must be finite")
        for name in (
            "pelvis_pos", "left_foot_pos", "right_foot_pos",
            "left_cop_world", "right_cop_world",
            "left_hand_hint", "right_hand_hint", "com_pos",
        ):
            object.__setattr__(self, name, _vec3(getattr(self, name), name))


@dataclass(frozen=True)
class SampledPose:
    """One representative complete skeleton consistent with the reference."""

    time: float
    keypoints: Dict[str, np.ndarray]
    com_pos: np.ndarray
    heading_yaw: float
    left_support_active: bool
    right_support_active: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.time):
            raise ValueError("time must be finite")
        for name, point in self.keypoints.items():
            if np.asarray(point).shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError(f"keypoint {name!r} must be a finite (3,) vector")


SKELETON_BONES: Tuple[Tuple[str, str], ...] = (
    ("pelvis", "chest"), ("chest", "head"),
    ("chest", "shoulder_left"), ("chest", "shoulder_right"),
    ("shoulder_left", "elbow_left"), ("elbow_left", "hand_left"),
    ("shoulder_right", "elbow_right"), ("elbow_right", "hand_right"),
    ("pelvis", "hip_left"), ("pelvis", "hip_right"),
    ("hip_left", "knee_left"), ("knee_left", "ankle_left"),
    ("hip_right", "knee_right"), ("knee_right", "ankle_right"),
    ("ankle_left", "toe_left"), ("ankle_right", "toe_right"),
)


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    kx, ky, kz = axis
    cross = np.array(
        [[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]],
        dtype=np.float64)
    c, s = np.cos(angle), np.sin(angle)
    return c * np.eye(3) + s * cross + (1.0 - c) * np.outer(axis, axis)


class MotionReference:
    """Pure-function key-shape reference over a validated ``MainPath``.

    ``state_at(t)`` returns a :class:`MotionReferenceState`; ``sample_pose(t)``
    instantiates one representative complete skeleton consistent with that
    reference so the whole path can be visualized and sanity-checked.
    """

    def __init__(
        self,
        path: MainPath,
        params: Optional[MotionReferenceParams] = None,
        gait: Optional[GaitModel] = None,
    ) -> None:
        self.path = path
        self.gait = gait or default_gait_model()
        self.params = params or MotionReferenceParams.for_template(
            "walk", self.gait)

    @property
    def start_time(self) -> float:
        return self.path.start_time

    @property
    def end_time(self) -> float:
        return self.path.end_time

    def states_at(
        self,
        times: Iterable[float],
    ) -> Tuple[MotionReferenceState, ...]:
        return tuple(self.state_at(float(t)) for t in times)

    def state_at(self, time: float) -> MotionReferenceState:
        t = float(time)
        state = self.path.state_at(t)
        p = self.params
        heading = state.heading_yaw
        forward = np.array([np.cos(heading), np.sin(heading)])

        left_foot = self._foot_center(state, "left")
        right_foot = self._foot_center(state, "right")
        stride_delta = float(
            np.dot(left_foot[:2] - right_foot[:2], forward))
        left_vel = self.path.foot_tracks["left"].velocity_at(t)
        right_vel = self.path.foot_tracks["right"].velocity_at(t)
        stride_rate = float(np.dot(
            (left_vel - right_vel)[:2], forward))

        pelvis_yaw = float(np.clip(
            p.hip_twist_gain * stride_delta,
            -p.hip_twist_max, p.hip_twist_max))
        trunk_yaw = float(np.clip(
            -p.torso_counter_twist_gain * stride_delta,
            -p.torso_twist_max, p.torso_twist_max))
        left_swing = float(np.clip(
            -p.arm_swing_gain * stride_delta,
            -p.arm_swing_max, p.arm_swing_max))
        right_swing = -left_swing
        left_swing_rate = float(np.clip(
            -p.arm_swing_gain * stride_rate,
            -4.0 * p.arm_swing_max, 4.0 * p.arm_swing_max))
        right_swing_rate = -left_swing_rate

        ang_scale = (
            self.path.model.mass * self.gait.leg_length
            * 1.0)
        lift_delta = float(np.clip(
            -p.arm_lift_gain * state.ang_mom[2] / max(ang_scale, 1e-9),
            -p.arm_lift_max, p.arm_lift_max))
        left_lift = float(np.clip(
            p.arm_lift_base + lift_delta, 0.0, p.arm_lift_max))
        right_lift = float(np.clip(
            p.arm_lift_base + lift_delta, 0.0, p.arm_lift_max))

        left_pitch = self._swing_pitch("left", t, state)
        right_pitch = self._swing_pitch("right", t, state)
        left_knee = self._knee_flexion("left", t, state)
        right_knee = self._knee_flexion("right", t, state)

        pelvis = state.com_pos.copy()
        pelvis[2] += p.pelvis_com_offset_z
        left_cop = state.left_point_world if state.left_support_active \
            else left_foot
        right_cop = state.right_point_world if state.right_support_active \
            else right_foot

        hands = self._hand_hints(
            pelvis, heading, pelvis_yaw, trunk_yaw,
            left_swing, right_swing, left_lift, right_lift)
        return MotionReferenceState(
            time=t,
            heading_yaw=heading,
            pelvis_pos=pelvis,
            pelvis_yaw_offset=pelvis_yaw,
            trunk_yaw_offset=trunk_yaw,
            trunk_pitch=p.trunk_pitch,
            left_foot_pos=left_foot,
            right_foot_pos=right_foot,
            left_foot_pitch=left_pitch,
            right_foot_pitch=right_pitch,
            left_cop_world=left_cop,
            right_cop_world=right_cop,
            left_support_active=state.left_support_active,
            right_support_active=state.right_support_active,
            left_knee_flexion=left_knee,
            right_knee_flexion=right_knee,
            left_arm_swing=left_swing,
            right_arm_swing=right_swing,
            left_arm_swing_rate=left_swing_rate,
            right_arm_swing_rate=right_swing_rate,
            left_arm_lift=left_lift,
            right_arm_lift=right_lift,
            left_elbow_angle=p.elbow_angle,
            right_elbow_angle=p.elbow_angle,
            left_hand_hint=hands["left"],
            right_hand_hint=hands["right"],
            com_pos=state.com_pos,
            details={
                "template": p.template,
                "stride_delta": stride_delta,
                "stride_rate": stride_rate,
                "arm_lift_channel": "ang_mom_regulation",
            },
        )

    def _foot_center(self, state: MainPathState, foot: FootName) -> np.ndarray:
        if getattr(state, f"{foot}_support_active"):
            point = self.path.support_schedule.active_point(foot, state.time)
            if point is not None:
                return point.copy()
        return self.path.foot_tracks[foot].point_at(state.time)

    def _swing_phase(self, foot: FootName, time: float) -> Optional[float]:
        track = self.path.foot_tracks[foot]
        for segment in track.swing_segments:
            if segment.start_time <= time < segment.end_time:
                duration = segment.end_time - segment.start_time
                return (time - segment.start_time) / duration
        return None

    def _swing_pitch(
        self,
        foot: FootName,
        time: float,
        state: MainPathState,
    ) -> float:
        u = self._swing_phase(foot, time)
        if u is None or getattr(state, f"{foot}_support_active"):
            return 0.0
        return float(self.params.swing_foot_pitch_max * np.sin(np.pi * u))

    def _knee_flexion(
        self,
        foot: FootName,
        time: float,
        state: MainPathState,
    ) -> float:
        p = self.params
        u = self._swing_phase(foot, time)
        if u is None or getattr(state, f"{foot}_support_active"):
            return p.knee_flexion_base
        return float(
            p.knee_flexion_base
            + p.knee_flexion_swing_extra * np.sin(np.pi * u))

    def _hand_hints(
        self,
        pelvis: np.ndarray,
        heading: float,
        pelvis_yaw: float,
        trunk_yaw: float,
        left_swing: float,
        right_swing: float,
        left_lift: float,
        right_lift: float,
    ) -> Dict[FootName, np.ndarray]:
        p = self.params
        frame = self._frames(heading)
        trunk_frame = self._frames(heading + pelvis_yaw + trunk_yaw)
        torso = p.torso_ratio * self.gait.leg_length
        spine = _rot(frame["left"], p.trunk_pitch) @ frame["up"]
        chest = pelvis + spine * (0.72 * torso)
        shoulder_half = 0.5 * p.shoulder_width_ratio * self.gait.stance_width
        upper = p.upper_arm_ratio * self.gait.leg_length
        forearm = p.forearm_ratio * self.gait.leg_length
        hands = {}
        for foot, swing, lift, side in (
            ("left", left_swing, left_lift, 1.0),
            ("right", right_swing, -right_lift, -1.0),
        ):
            shoulder = chest + trunk_frame["left"] * (side * shoulder_half)
            arm_dir = _rot(trunk_frame["left"], -swing) @ (-frame["up"])
            arm_dir = _rot(trunk_frame["forward"], side * lift) @ arm_dir
            elbow_pos = shoulder + arm_dir * upper
            fore_dir = _rot(trunk_frame["left"], -p.elbow_angle) @ arm_dir
            hands[foot] = elbow_pos + fore_dir * forearm
        return hands

    @staticmethod
    def _frames(yaw: float) -> Dict[str, np.ndarray]:
        return {
            "forward": np.array([np.cos(yaw), np.sin(yaw), 0.0]),
            "left": np.array([-np.sin(yaw), np.cos(yaw), 0.0]),
            "up": np.array([0.0, 0.0, 1.0]),
        }

    def sample_pose(self, time: float) -> SampledPose:
        """Instantiate one representative complete skeleton at ``time``."""
        ref = self.state_at(time)
        p = self.params
        leg = self.gait.leg_length
        frame = self._frames(ref.heading_yaw)
        pelvis_frame = self._frames(ref.heading_yaw + ref.pelvis_yaw_offset)
        trunk_frame = self._frames(
            ref.heading_yaw + ref.pelvis_yaw_offset + ref.trunk_yaw_offset)

        torso = p.torso_ratio * leg
        spine = _rot(frame["left"], ref.trunk_pitch) @ frame["up"]
        chest = ref.pelvis_pos + spine * (0.72 * torso)
        head = ref.pelvis_pos + spine * torso + frame["up"] * (
            p.head_ratio * leg)
        shoulder_half = 0.5 * p.shoulder_width_ratio * self.gait.stance_width
        hip_half = 0.5 * p.hip_width_ratio * self.gait.stance_width

        keypoints: Dict[str, np.ndarray] = {
            "pelvis": ref.pelvis_pos,
            "chest": chest,
            "head": head,
        }
        for foot in FOOT_NAMES:
            side = 1.0 if foot == "left" else -1.0
            keypoints[f"shoulder_{foot}"] = (
                chest + trunk_frame["left"] * (side * shoulder_half))
            keypoints[f"hip_{foot}"] = (
                ref.pelvis_pos + pelvis_frame["left"] * (side * hip_half))
            foot_pos = getattr(ref, f"{foot}_foot_pos")
            pitch = getattr(ref, f"{foot}_foot_pitch")
            ankle = foot_pos + frame["up"] * p.ankle_height
            keypoints[f"ankle_{foot}"] = ankle
            keypoints[f"toe_{foot}"] = (
                foot_pos + frame["forward"] * (0.40 * self.gait.foot_length)
                + frame["up"] * (p.ankle_height * np.sin(pitch)))
            keypoints[f"knee_{foot}"] = self._solve_knee(
                keypoints[f"hip_{foot}"], ankle, frame["forward"])
            swing = getattr(ref, f"{foot}_arm_swing")
            lift = getattr(ref, f"{foot}_arm_lift")
            arm_dir = _rot(trunk_frame["left"], -swing) @ (-frame["up"])
            arm_dir = _rot(trunk_frame["forward"], side * lift) @ arm_dir
            keypoints[f"elbow_{foot}"] = (
                keypoints[f"shoulder_{foot}"]
                + arm_dir * (p.upper_arm_ratio * leg))
            fore_dir = _rot(
                trunk_frame["left"], -p.elbow_angle) @ arm_dir
            keypoints[f"hand_{foot}"] = (
                keypoints[f"elbow_{foot}"]
                + fore_dir * (p.forearm_ratio * leg))
        return SampledPose(
            time=ref.time,
            keypoints=keypoints,
            com_pos=ref.com_pos,
            heading_yaw=ref.heading_yaw,
            left_support_active=ref.left_support_active,
            right_support_active=ref.right_support_active,
        )

    def sample_poses(self, times: Iterable[float]) -> Tuple[SampledPose, ...]:
        return tuple(self.sample_pose(float(t)) for t in times)

    def _solve_knee(
        self,
        hip: np.ndarray,
        ankle: np.ndarray,
        forward: np.ndarray,
    ) -> np.ndarray:
        """Two-link IK: knee bends toward the heading forward direction."""
        leg = self.gait.leg_length
        upper = self.params.upper_leg_ratio * leg
        lower = (1.0 - self.params.upper_leg_ratio) * leg
        leg_axis = ankle - hip
        d = min(float(np.linalg.norm(leg_axis)), (upper + lower) * 0.999)
        axis = leg_axis / max(np.linalg.norm(leg_axis), 1e-12)
        cos_a = np.clip(
            (upper * upper + d * d - lower * lower) / (2.0 * upper * d),
            -1.0, 1.0)
        bend = np.arccos(cos_a)
        perp = forward - axis * float(np.dot(forward, axis))
        norm = np.linalg.norm(perp)
        if norm < 1e-9:
            perp = np.cross(axis, np.array([0.0, 0.0, 1.0]))
            norm = np.linalg.norm(perp)
        perp = perp / max(norm, 1e-12)
        return hip + upper * (axis * np.cos(bend) + perp * np.sin(bend))
