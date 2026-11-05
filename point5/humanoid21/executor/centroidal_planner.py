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


def gen_landmarks(lm0: "LandMark", clip, steps
                  ) -> "list":
    """LandMark sequence generation (bootstrip step 4).

    Inputs:
        lm0   — start LandMark, extracted from the observation
                (the sequence's first element)
        clip  — the remaining P5Clip; supplies posture anchors
                (per-kf recommended COM via tool1) and the
                terminal anchor
        steps — ordered Footstep list from the support-planning
                sub-algorithm (plan_footsteps); each produces a
                liftoff + touchdown anchor pair

    Output: ordered LandMark list — support-event anchors +
    posture anchors + terminal anchor, per the composition spec
    in CentroidalPlanner's docstring.
    """
    from .tools.com_solver import com_estimate
    from .contracts import FootSupport, LandMark
    from .tools.cop_solver import FLAT_Z as LIFT_Z
    from .tools.capability_solver import _PATCH, _hull2d
    from .support_planner import _inside
    from point5.contracts import compose_frame, to_world

    FIDX = {"right": 3, "left": 4}
    kfs = clip.keyframes
    poses = [(np.zeros(2), 0.0)]
    for k in range(1, len(kfs)):
        poses.append(compose_frame(*poses[-1], kfs[k].transfer))
    w_kfs = [to_world(kf.pos, *poses[k]) if k else kf.pos
             for k, kf in enumerate(kfs)]

    before = {k: [] for k in range(1, len(kfs))}
    for_k = {k: [] for k in range(1, len(kfs))}
    for st in steps:
        (for_k if st.for_frame else before)[
            st.for_frame or st.before_frame].append(st)

    # virtual state
    com = np.asarray(lm0.com, np.float64).copy()
    feet_xy = {s: (np.asarray(lm0.feet[s].foot_pos,
                              np.float64).copy()
                   if lm0.feet[s].foot_pos is not None else None)
               for s in FIDX}
    loaded = {s: lm0.feet[s].force_ratio > 0.05 for s in FIDX}
    lms = [lm0]

    def _feet():
        n = sum(loaded.values()) or 1
        return {s: FootSupport(
            force_ratio=(1.0 / n) if loaded[s] else 0.0,
            cop=(feet_xy[s].copy() if loaded[s] else None),
            foot_pos=(feet_xy[s].copy()
                      if feet_xy[s] is not None else None))
            for s in FIDX}

    def _liftoff(foot, frame=0):
        nonlocal com
        loaded[foot] = False
        com_t = com.copy()
        cops = [feet_xy[s] for s in FIDX if loaded[s]]
        if len(cops) == 1:                 # single support left
            com_t[:2] = cops[0]
        lms.append(LandMark(com=com_t, feet=_feet(), vel=None,
                            frame=frame))

    def _touchdown(foot, target, frame=0):
        nonlocal com
        feet_xy[foot] = np.asarray(target, np.float64).copy()
        loaded[foot] = True
        com_t = com.copy()
        cops = [feet_xy[s] for s in FIDX if loaded[s]]
        if cops:
            com_t[:2] = np.mean(cops, axis=0)
        lms.append(LandMark(com=com_t, feet=_feet(), vel=None,
                            frame=frame))

    def _project(pt, poly):
        if _inside(pt, poly):
            return pt
        best, bd = pt, np.inf
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            e = b - a
            t = np.clip(np.dot(pt - a, e)
                        / max(np.dot(e, e), 1e-12), 0.0, 1.0)
            p = a + t * e
            d = np.linalg.norm(pt - p)
            if d < bd:
                best, bd = p, d
        return best

    def _posture(k):
        """The frame's own requirement as an anchor."""
        nonlocal com
        w = np.asarray(w_kfs[k], np.float64)
        n_load = sum(kfs[k].pos[FIDX[s], 2] <= LIFT_Z
                     for s in FIDX) or 1
        feet = {}
        hull_pts = []
        for s, fi in FIDX.items():
            grounded = kfs[k].pos[fi, 2] <= LIFT_Z
            if grounded:
                hull_pts.append(w[fi, :2] + np.asarray(_PATCH))
            feet[s] = FootSupport(
                force_ratio=(1.0 / n_load) if grounded else 0.0,
                cop=w[fi, :2].copy() if grounded else None,
                foot_pos=w[fi, :2].copy())
            feet_xy[s] = w[fi, :2].copy()
            loaded[s] = grounded
        com_t = com_estimate(w)
        if hull_pts:
            com_t[:2] = _project(com_t[:2], _hull2d(np.vstack(hull_pts)))
        com = com_t.copy()
        settle = (k == len(kfs) - 1) or not for_k.get(k + 1)
        lms.append(LandMark(com=com_t, feet=feet,
                            vel=np.zeros(3) if settle else None,
                            frame=k))

    for k in range(1, len(kfs)):
        for st in before[k]:
            if loaded.get(st.foot):
                _liftoff(st.foot, k)
            _touchdown(st.foot, st.target, k)
        # frame foot-state diff: loaded -> airborne (no step covers)
        for s, fi in FIDX.items():
            was = kfs[k - 1].pos[fi, 2] <= LIFT_Z
            now = kfs[k].pos[fi, 2] <= LIFT_Z
            if was and not now and loaded[s]:
                _liftoff(s, k)
        for st in for_k[k]:
            if loaded.get(st.foot):
                _liftoff(st.foot, k)
            _touchdown(st.foot, st.target, k)
        _posture(k)
    return lms


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

    1. 支撑事件锚点：取某只脚离地和落地的一瞬间（含垫步、回
       位）。
       - 离地：支撑脚 ~1.0、摆动脚 0；质心移到支撑脚 CoP 区域
         附近；速度不强制为 0，只要求 capture point 指向落脚区。
       - 触地：摆动脚开始承重，进入双支撑；质心在两脚之间、
         朝新支撑方向；速度一般不限。
       （稳定锚点暂不做——落地即完成事件。）
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
        from .tools.start_solver import extract_landmark
        from .support_planner import plan_footsteps
        from point5.contracts import compose_frame

        lm0 = extract_landmark(observation, self._decoder)
        steps = plan_footsteps(lm0, clip)
        # 生成Landmark序列
        return gen_landmarks(lm0, clip, steps)

