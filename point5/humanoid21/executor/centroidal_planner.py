"""Algorithm 1: centroidal motion planning.

Input : Observation (96-dim policy obs) -> decode -> centroidal state
        z=(c,l,L) in the P5 local frame + the remaining P5Clip.
Output: CentroidalTrajectory.

Decode path (obs contains no self keypoints — must FK):
    q_rad = obs[0:21] * scale + ref            (ranges from FK model)
    root quat = minimal rot g_body -> -z       (tilt, arbitrary yaw 0)
    FK -> landmarks -> torso yaw -> local P5 -> tool1 com_estimate
    l = MASS * R_local @ v_body ;  L = I_est @ R_local @ w_body
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .centroidal import MASS_KG, CentroidalState, CentroidalTrajectory
from .com_solver import com_estimate, mass_estimate
from .contracts import Context, PlanError


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


class CentroidalPlanner:
    """Stateless centroidal planner used by Executor.gen_action."""

    def __init__(self, robot_id: str = "robot_a"):
        self.robot_id = robot_id
        self._fk = None

    def fk(self):
        if self._fk is None:
            from point5.humanoid21.fk import Humanoid21FK
            self._fk = Humanoid21FK()
        return self._fk

    def _decode_p5_local(self, obs: np.ndarray):
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

    def extract_centroidal_state(self, observation: np.ndarray
                                 ) -> CentroidalState:
        """96-dim policy obs -> z=(c,l,L) in the P5 local frame."""
        obs = np.asarray(observation, np.float64)
        local_p5, R_loc = self._decode_p5_local(obs)
        c = com_estimate(local_p5)
        v_loc = R_loc @ obs[46:49]                        # body->local
        w_obs = obs[49:52]
        w_body = np.sign(w_obs) * w_obs * np.abs(w_obs) * 2.0
        w_loc = R_loc @ w_body
        I_loc = mass_estimate(local_p5).inertia_per_mass * MASS_KG
        return CentroidalState(com=c, lin_mom=MASS_KG * v_loc,
                               ang_mom=I_loc @ w_loc)

    def plan(self, ctx: Context,
             c0: CentroidalState) -> CentroidalTrajectory:
        """Centroidal trajectory from current state + remaining clip.

        大逻辑（设计意图，先注释后实现）：

        1. 由果推因——直接从 P5Clip 设计质心轨迹这个"果"，
           不做支撑相位/落脚点级别的因果规划；物理上能不能
           实现这个果是算法2 要解决的问题。

        2. 轨迹由"业务约束"生成，而非动力学优化：
           - 锚点：每个关键帧经工具1 com_estimate 得到局部系
             推荐质心（世界系锚定由 clip frame 链给出）。
           - 曲线模板：按段类型套简单有效的经验曲线——
             e.g. 换步支撑段：质心横向滑移为 S 曲线，
             先加速离脚、中途恒速、落地前减速；
             单脚段：质心向支撑脚收敛后保持；
             出手段：z 维持，xy 微前送。
           - 速度约束：相邻段速度连续（或受控跳变），
             单段内 |v| 不超过该段能力预期。

        3. 能力预期（经验函数，P5->capability）：
           - capability(p5) -> 该姿态下质心可改变的速率/
             加速度包络（随姿态而异：宽站距 vs 窄站距、
             蹲姿 vs 立姿、单脚 vs 双脚）。
           - 用于定每段时长：dur = f(质心位移需求, capability)。
           - 初版可以是几个粗糙查表/分段线性经验值，
             90% 信息量即可。

        4. 输出 CentroidalTrajectory：
           states[k] = 第 k 个控制步之后的 z=(c,l,L)；
           l 由曲线速度*MASS 给出，L 初版可为 0
           （角动量后补 yaw 旋转需求）。
        """
        raise PlanError("algorithm 1 (centroidal planning) "
                        "not implemented")
