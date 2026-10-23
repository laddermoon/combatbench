"""Render sampled motion-reference poses inside the MuJoCo simulator."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import mujoco
import numpy as np

from .reference import (
    BodyProportions,
    MotionReference,
    MotionReferenceState,
    SampledPose,
    _rot,
)


@dataclass
class _Task:
    body: int
    kind: str                      # "pos" or "rot"
    target: np.ndarray
    weight: float


class ReferenceIKSolver:
    """Damped least-squares IK from sampled keypoints to full ``qpos``.

    The solver adjusts ``robot_id``'s free root plus all 21 hinge joints so
    that pelvis, feet, hands, and head land on the sampled skeleton targets.
    Foot and torso rotation tasks keep the pose oriented instead of letting
    the position-only residuals pick an arbitrary limb posture.
    """

    def __init__(self, sim, robot_id: str = "robot_a") -> None:
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        cache = sim._robot(robot_id)
        self.cache = cache
        self.kp = cache["keypoint_body_ids"]
        self.qpos_idx = np.asarray(cache["qpos_indices"], dtype=np.int64)
        self.qvel_idx = np.asarray(cache["qvel_indices"], dtype=np.int64)
        self.root_qpos = int(cache["root_qpos_adr"])
        self.root_qvel = int(cache["root_qvel_adr"])
        self.dofs = np.concatenate([
            self.root_qvel + np.arange(6), self.qvel_idx])
        self._dof_to_column = {int(d): i for i, d in enumerate(self.dofs)}
        self.hinge_jids = np.array([
            int(np.where(self.model.jnt_qposadr == adr)[0][0])
            for adr in self.qpos_idx])
        self.jnt_lo = self.model.jnt_range[self.hinge_jids, 0]
        self.jnt_hi = self.model.jnt_range[self.hinge_jids, 1]
        pose_cfg = sim.INITIAL_POSES["standing"]
        self.q_rest = pose_cfg["joint_pos"].astype(np.float64)
        mujoco.mj_forward(self.model, self.data)
        self._root_to_pelvis = (
            self.data.xpos[int(cache["root_body_id"])]
            - self.data.xpos[self.kp["pelvis"]]).copy()
        self.last_qpos: Optional[np.ndarray] = None

    def solve(
        self,
        pose: SampledPose,
        ref: MotionReferenceState,
        *,
        world_offset: np.ndarray,
        max_iters: int = 120,
        tol: float = 5e-3,
    ) -> Tuple[np.ndarray, float]:
        """Solve one sampled pose; returns ``(qpos, max_position_error)``."""
        if self.last_qpos is not None:
            q = self.last_qpos.copy()
        else:
            q = self.data.qpos.copy()
        q[self.qpos_idx] = self.q_rest
        pelvis_t = pose.keypoints["pelvis"] + world_offset
        q[self.root_qpos:self.root_qpos + 3] = pelvis_t + self._root_to_pelvis
        heading = ref.heading_yaw + ref.pelvis_yaw_offset + ref.trunk_yaw_offset
        q[self.root_qpos + 3:self.root_qpos + 7] = np.array(
            [np.cos(heading * 0.5), 0.0, 0.0, np.sin(heading * 0.5)])

        tasks = self._build_tasks(pose, ref, world_offset)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        max_err = np.inf
        for _ in range(max_iters):
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            rows: List[np.ndarray] = []
            errs: List[np.ndarray] = []
            max_err = 0.0
            for task in tasks:
                mujoco.mj_jacBody(
                    self.model, self.data, jacp, jacr, task.body)
                if task.kind == "pos":
                    err = task.target - self.data.xipos[task.body]
                    jac = jacp
                    if task.weight >= 1.0:
                        max_err = max(max_err, float(np.linalg.norm(err)))
                else:
                    err = self._orientation_error(
                        self.data.xmat[task.body].reshape(3, 3), task.target)
                    jac = jacr
                rows.append(jac[:, self.dofs] * task.weight)
                errs.append(err * task.weight)
            for i, adr in enumerate(self.qpos_idx):
                reg = np.zeros(len(self.dofs))
                reg[self._dof_to_column[int(self.qvel_idx[i])]] = 0.15
                rows.append(reg)
                errs.append(np.array([0.15 * (self.q_rest[i] - q[adr])]))
            a = np.vstack(rows)
            b = np.concatenate(errs)
            step = a.T @ np.linalg.solve(
                a @ a.T + 1e-4 * np.eye(len(b)), b)
            dq = np.zeros(self.model.nv)
            dq[self.dofs] = 0.6 * step
            mujoco.mj_integratePos(self.model, q, dq, 1.0)
            q[self.qpos_idx] = np.clip(
                q[self.qpos_idx], self.jnt_lo, self.jnt_hi)
            quat = q[self.root_qpos + 3:self.root_qpos + 7]
            q[self.root_qpos + 3:self.root_qpos + 7] = (
                quat / max(np.linalg.norm(quat), 1e-12))
            if max_err < tol:
                break
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        self.last_qpos = q.copy()
        return q.copy(), float(max_err)

    def _build_tasks(
        self,
        pose: SampledPose,
        ref: MotionReferenceState,
        offset: np.ndarray,
    ) -> List[_Task]:
        kp = pose.keypoints
        tasks = [
            _Task(self.kp["pelvis"], "pos",
                  kp["pelvis"] + offset, 1.0),
            _Task(self.kp["head"], "pos",
                  kp["head"] + offset, 0.3),
            _Task(self.kp["hand_left"], "pos",
                  kp["hand_left"] + offset, 0.6),
            _Task(self.kp["hand_right"], "pos",
                  kp["hand_right"] + offset, 0.6),
            _Task(self.kp["foot_left"], "pos",
                  kp["ankle_left"] + offset, 1.0),
            _Task(self.kp["foot_right"], "pos",
                  kp["ankle_right"] + offset, 1.0),
        ]
        frames = MotionReference._frames(ref.heading_yaw)
        trunk_yaw = (
            ref.heading_yaw + ref.pelvis_yaw_offset + ref.trunk_yaw_offset)
        trunk_frame = MotionReference._frames(trunk_yaw)
        torso_rot = np.column_stack([
            _rot(frames["left"], ref.trunk_pitch) @ trunk_frame["forward"],
            trunk_frame["left"],
            _rot(frames["left"], ref.trunk_pitch) @ frames["up"],
        ])
        tasks.append(_Task(
            self.cache["root_body_id"], "rot", torso_rot, 0.4))
        for foot in ("left", "right"):
            pitch = getattr(ref, f"{foot}_foot_pitch")
            foot_rot = np.column_stack([
                _rot(frames["left"], pitch) @ frames["forward"],
                frames["left"],
                _rot(frames["left"], pitch) @ frames["up"],
            ])
            tasks.append(_Task(
                self.kp[f"foot_{foot}"], "rot", foot_rot, 0.5))
        return tasks

    @staticmethod
    def _orientation_error(
        current: np.ndarray,
        desired: np.ndarray,
    ) -> np.ndarray:
        err = np.zeros(3)
        for i in range(3):
            err += np.cross(current[:, i], desired[:, i])
        return 0.5 * err


def render_reference_in_sim(
    reference: MotionReference,
    output_path: str | Path,
    *,
    fps: int = 20,
    sim=None,
    robot_id: str = "robot_a",
    max_ik_iters: int = 120,
    proportions: Optional[BodyProportions] = None,
) -> Path:
    """Drive the real Humanoid21 model through sampled poses and record MP4."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if sim is None:
        from envs.humanoid21.simulator import Humanoid21Simulator

        sim = Humanoid21Simulator(
            initial_pose_a="standing", initial_pose_b="standing")
        sim.reset(seed=0)
    prop = proportions or BodyProportions.from_simulator(sim, robot_id)
    if prop is not reference.proportions:
        reference = MotionReference(
            reference.path,
            params=reference.params,
            gait=reference.gait,
            proportions=prop,
        )
    solver = ReferenceIKSolver(sim, robot_id)

    cache = sim._robot(robot_id)
    stance_center = 0.5 * (
        sim.data.xipos[cache["keypoint_body_ids"]["foot_left"]][:2]
        + sim.data.xipos[cache["keypoint_body_ids"]["foot_right"]][:2])
    initial = reference.state_at(reference.start_time)
    ref_center = 0.5 * (
        initial.left_foot_pos[:2] + initial.right_foot_pos[:2])
    offset = np.array(
        [stance_center[0] - ref_center[0],
         stance_center[1] - ref_center[1], 0.0])

    duration = float(reference.end_time - reference.start_time)
    frame_count = max(2, int(np.floor(duration * fps)) + 1)
    times = reference.start_time + np.arange(frame_count) / float(fps)
    times = times[times <= reference.end_time + 1e-9]
    if abs(times[-1] - reference.end_time) > 1e-6:
        times = np.append(times, reference.end_time)

    frames = []
    residuals = []
    qpos_dim = sim.data.qpos.shape[0]
    for t in times:
        ref = reference.state_at(float(t))
        pose = reference.sample_pose(float(t))
        q, err = solver.solve(
            pose, ref, world_offset=offset, max_iters=max_ik_iters)
        sim.data.qpos[:] = q
        sim.data.qvel[:] = 0.0
        mujoco.mj_forward(sim.model, sim.data)
        image = sim.get_broadcastview_image()
        if image is None:
            raise RuntimeError("broadcast view render returned no image")
        frames.append(image)
        residuals.append(err)
    imageio.mimsave(output, frames, fps=int(fps), macro_block_size=16)
    return output
