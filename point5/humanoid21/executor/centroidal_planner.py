"""Observation decode machinery (shared by algorithm-1 tools).

The 96-dim policy obs carries no self keypoints — the local P5 must
be reconstructed by forward kinematics:

    q_rad = obs[0:21] * scale + ref            (ranges from FK model)
    root quat = minimal rot g_body -> -z       (tilt, arbitrary yaw)
    FK -> landmarks -> torso yaw -> local P5

``CentroidalPlanner`` is kept as the holder of this decode path;
algorithm-1 planning itself lives in the LandMark pipeline
(start_solver / support_planner).
"""
from __future__ import annotations

import numpy as np



def _level_rot(g_body: np.ndarray) -> np.ndarray:
    """Minimal rotation taking body-frame gravity to [0,0,-1]."""
    g = np.asarray(g_body, np.float64)
    g = g / (np.linalg.norm(g) + 1e-12)
    target = np.array([0.0, 0.0, -1.0])
    axis = np.cross(g, target)
    n = np.linalg.norm(axis)
    if n < 1e-9:                                     # (anti-)parallel
        return np.eye(3) if g[2] < 0 else np.diag([1.0, -1.0, -1.0])
    axis /= n
    ang = float(np.arccos(np.clip(np.dot(g, target), -1.0, 1.0)))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K



def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        return np.array([s / 4, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1) * 2
    q = np.empty(4)
    q[i + 1] = s / 4
    q[0] = (R[k, j] - R[j, k]) / s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


class ObsDecoder:
    """96-dim policy obs -> local P5 via FK (lazy Humanoid21FK)."""

    def __init__(self, robot_id: str = "robot_a"):
        self.robot_id = robot_id
        self._fk = None

    def fk(self):
        if self._fk is None:
            from point5.humanoid21.fk import Humanoid21FK
            self._fk = Humanoid21FK()
        return self._fk

    def decode_p5_local(self, obs: np.ndarray):
        """obs -> (local_p5 (5,3), R_local(3,3): body->P5 frame)."""
        from point5.contracts import ankle_mid_xy, to_local
        fk, rid = self.fk(), self.robot_id
        rng = fk.joint_ranges(rid).astype(np.float64)
        ref, sc = (rng[:, 0] + rng[:, 1]) / 2, (rng[:, 1] - rng[:, 0]) / 2
        R_tilt = _level_rot(obs[42:45])
        fk.set_pose(rid, np.array([0.0, 0.0, obs[45]]),
                    _quat_wxyz(R_tilt), obs[0:21] * sc + ref)
        lm = fk.landmark_positions(rid)
        poses = fk.body_pose(rid)
        torso = next(v for k, v in poses.items() if "torso" in k)
        fwd = torso[1][:, 0]
        yaw = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
        local = to_local(lm, ankle_mid_xy(lm), yaw)
        ang = np.radians(yaw)
        Rz = np.array([[np.cos(ang), np.sin(ang), 0],
                       [-np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        return local, Rz @ R_tilt



class CentroidalPlanner:
    """Algorithm 1 (bootstrip): (Observation, P5Clip) -> LandMark seq.

    Internal stages: 1) extract start LandMark from obs, 2) support
    planning sub-algorithm, 3) target design by priors,
    4) landmark planning.  Every stage marked TODO is a prior that
    still needs design.
    """

    def __init__(self, robot_id: str = "robot_a"):
        self.robot_id = robot_id
        self._decoder = ObsDecoder(robot_id)

    def decode_p5_local(self, obs: np.ndarray):
        return self._decoder.decode_p5_local(obs)

    def plan(self, observation: np.ndarray,
             clip) -> list:
        """(observation, remaining clip) -> ordered LandMark seq."""
        from .start_solver import extract_landmark
        from .support_planner import plan_footsteps
        from point5.contracts import compose_frame

        lm0 = extract_landmark(observation, self._decoder)
        steps = plan_footsteps(lm0, clip)
        lms: list = [lm0]
        poses = [(np.zeros(2), 0.0)]
        for k in range(1, len(clip.keyframes)):
            poses.append(compose_frame(
                *poses[-1], clip.keyframes[k].transfer))

        for st in steps:
            if st.kind == "corrective":
                lms.append(self._step_landmark(st, lms[-1]))
            elif st.kind == "recover":
                lms.append(self._step_landmark(st, lms[-1]))
            else:  # move
                lms.append(self._shift_landmark(st, lms[-1]))
                lms.append(self._land_landmark(st, lms[-1]))
        # TODO(algo1): terminal hold landmark from the clip's last
        # keyframe — COM target via tool1 on the kf P5 (frame0),
        # both feet loaded per the kf's support pattern
        return lms

    # -- per-footstep landmark constructors ---------------------------------
    def _shift_landmark(self, st, prev: "LandMark") -> "LandMark":
        """Weight transfer onto the support foot before the mover lifts."""
        from .contracts import FootSupport, LandMark
        other = "left" if st.foot == "right" else "right"
        sup_f = prev.feet[other]
        # TODO(90% prior): COM target = over the support foot's CoP,
        # smoothed from prev.com; z keep current height
        com_t = prev.com.copy()
        if sup_f.cop is not None:
            com_t[:2] = sup_f.cop
        return LandMark(
            com=com_t,
            feet={other: FootSupport(1.0, sup_f.cop,
                                     sup_f.foot_pos),
                  st.foot: FootSupport(0.0, None, st.target)},
            vel=np.zeros(3))                 # arrive settled

    def _land_landmark(self, st, prev: "LandMark") -> "LandMark":
        """Mover lands on the step target; double support resumes."""
        from .contracts import FootSupport, LandMark
        other = "left" if st.foot == "right" else "right"
        feet = dict(prev.feet)
        feet[st.foot] = FootSupport(0.5, st.target.copy(),
                                    st.target.copy())
        feet[other] = FootSupport(
            0.5, feet[other].cop, feet[other].foot_pos)
        # TODO(90% prior): COM target after landing — geometric
        # midpoint of the two CoPs is a placeholder; the clip kf's
        # com_estimate (frame0) is the real anchor, wire tool1 here
        cops = [f.cop for f in feet.values() if f.cop is not None]
        com_t = prev.com.copy()
        if cops:
            com_t[:2] = np.mean(cops, axis=0)
        return LandMark(com=com_t, feet=feet, vel=np.zeros(3))

    def _step_landmark(self, st, prev: "LandMark") -> "LandMark":
        """Corrective/recover step: foot placed at st.target."""
        from .contracts import FootSupport, LandMark
        feet = dict(prev.feet)
        feet[st.foot] = FootSupport(0.5, st.target.copy(),
                                    st.target.copy())
        # TODO: force split + COM during a brace step — currently a
        # 50/50 placeholder; real ratio should reflect the capture
        return LandMark(com=prev.com.copy(), feet=feet,
                        vel=np.zeros(3))
