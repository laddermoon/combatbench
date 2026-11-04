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



def design_next_target(next_step, following, prev: "LandMark"
                       ) -> "LandMark":
    """Bootstrip step 3: design the goal state at the NEXT support
    point, from its position and what follows.

    Target contents: post-event support state (load ratios + CoP)
    and a COM velocity constraint — ``vel=0`` when the plan must
    settle (corrective/recover, or the last needed step), ``None``
    (free) when the gait continues.

    Priors:
      - COM after landing -> mean of the new stance's CoPs (inside
        the fresh support hull by construction)
      - brace/recover always settle
      - a move followed by another move keeps momentum (v free)
    """
    from .contracts import FootSupport, LandMark
    other = "left" if next_step.foot == "right" else "right"
    feet = dict(prev.feet)
    feet[next_step.foot] = FootSupport(
        0.5, next_step.target.copy(), next_step.target.copy())
    fo = feet[other]
    feet[other] = FootSupport(0.5, fo.cop, fo.foot_pos)

    settle = (next_step.kind != "move"
              or not any(f.kind == "move" for f in following))
    cops = [f.cop for f in feet.values() if f.cop is not None]
    com_t = prev.com.copy()
    if cops:
        com_t[:2] = np.mean(cops, axis=0)
    return LandMark(com=com_t, feet=feet,
                    vel=np.zeros(3) if settle else None)



class CentroidalPlanner:
    """Algorithm 1 (bootstrip): (Observation, P5Clip) -> LandMark seq.

    Internal stages: 1) extract start LandMark from obs, 2) support
    planning sub-algorithm, 3) target design by priors,
    4) landmark planning.  Every stage marked TODO is a prior that
    still needs design.

    LandMark 序列的锚点组成 = 支撑事件锚点 + 姿态锚点 + 终点锚点。

    设计原则：锚点放在支撑状态变化处；相邻两个锚点之间，接触
    方式不变或只发生一次变化——算法2 每段只面对一种接触模式。
    锚点不带时间，先后顺序即可。

    1. 支撑事件锚点（每个 Footstep，含垫步/回位，各三个）：
       - 离地：支撑脚 ~1.0、摆动脚 0；质心移到支撑脚 CoP 区域
         附近；速度不强制为 0，只要求 capture point 指向落脚区。
       - 触地：摆动脚开始承重，进入双支撑；质心在两脚之间、
         朝新支撑方向；速度一般不限。
       - 稳定：载荷按下一阶段需要分配；质心在新支撑域内并留
         裕度；后面还有步则速度不限，最后一步或垫步后为 0。
    2. 姿态锚点：clip 中不换支撑、只改变质心的关键帧（蓄力下
       蹲、出拳前送、重心偏移）——工具1 由该帧 P5 得推荐质心，
       再投影进当时的支撑域。
    3. 终点锚点：clip 最后一帧的支撑方式与质心位置，速度为 0。

    规划范围：只需覆盖下一个和再下一个支撑事件；更远的步只用
    于判断当前步落地后是否要停住，由下一次重规划处理。
    """

    def __init__(self, robot_id: str = "robot_a"):
        self.robot_id = robot_id
        self._decoder = ObsDecoder(robot_id)

    def decode_p5_local(self, obs: np.ndarray):
        return self._decoder.decode_p5_local(obs)

    def plan(self, observation: np.ndarray,
             clip) -> list:
        """(observation, remaining clip) -> ordered LandMark seq.
        """
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

        for i, st in enumerate(steps):
            if st.kind != "move":
                lms.append(design_next_target(
                    st, steps[i + 1:], lms[-1]))
            else:
                lms.append(self._shift_landmark(st, lms[-1]))
                lms.append(design_next_target(
                    st, steps[i + 1:], lms[-1]))
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
