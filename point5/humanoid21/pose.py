"""Pose — 精确的无位置机器人姿态表达（锚点用）。

    qpos   (21,)  关节角，弧度（真实单位，非归一化）
    root_z (1,)   躯干高度，世界系 z
    g_body (3,)   机体系重力向量 — 编码俯仰/侧倾，不含 yaw

位置缺失：水平位置与 yaw 不在此表达——由 ``anchor()`` 解析：
把命名点钉到目标 xy + 躯干朝向，得到锚定帧的 p5/foot_pos/com。
``com_height()`` 不需要锚定（z 是绝对的）。

    a = pa.anchor('left', xy=(0.3, 0.0), yaw=np.pi/4)
    a.p5()  a.foot_pos()  a.com()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np

from point5.contracts import POINT_NAMES

_JOINTS = (21,)
_G = (3,)
_FIDX = {"right": 3, "left": 4}     # landmark index of each foot


def _level_rot(g_body: np.ndarray) -> np.ndarray:
    """Minimal rotation taking body-frame gravity to [0,0,-1]."""
    g = np.asarray(g_body, np.float64)
    g = g / (np.linalg.norm(g) + 1e-12)
    target = np.array([0.0, 0.0, -1.0])
    axis = np.cross(g, target)
    n = np.linalg.norm(axis)
    if n < 1e-9:
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


from envs.humanoid21.meta import Humanoid21Meta
from envs.humanoid21.simulator import Humanoid21Simulator
from .mechanics import LANDMARKS

_POINT_RGBA = {
    "head": (224, 72, 62, 255),
    "hand_right": (59, 165, 59, 255),
    "hand_left": (59, 127, 212, 255),
    "foot_right": (224, 160, 42, 255),
    "foot_left": (181, 106, 224, 255),
}


def _project(cam, fovy: float, size: tuple, p3: np.ndarray):
    """Project world point to pixel using a free mjvCamera."""
    az = np.deg2rad(cam.azimuth)
    el = np.deg2rad(cam.elevation)
    fwd = np.array([-np.cos(el) * np.cos(az),
                    -np.cos(el) * np.sin(az), -np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(right, fwd)
    pos = cam.lookat - fwd * cam.distance
    rel = p3 - pos
    x, y = rel @ right, rel @ up
    z = -(rel @ fwd)
    if z <= 0:
        return None
    f = 0.5 * size[1] / np.tan(np.deg2rad(fovy) / 2)
    return (size[0] / 2 + f * x / z, size[1] / 2 - f * y / z)


@dataclass
class _Anchored:
    """Pose resolved into a frame — all outputs live here."""
    _p5: np.ndarray                 # (5,3) anchored local
    _feet: dict                     # side -> (2,) anchored local
    _com: np.ndarray                # (3,) anchored local
    R: np.ndarray = field(          # world -> anchored rotation
        default_factory=lambda: np.eye(3))

    def p5(self):
        """P5 — 局部 P5 姿态（点坐标集合语义）。"""
        from point5.contracts import P5
        return P5(pos=self._p5)

    def foot_pos(self) -> dict:
        return self._feet

    def com(self) -> np.ndarray:
        return self._com


@dataclass
class Pose:
    """关节角 + 高度 + 俯仰——机器人姿态的精确表达。"""
    qpos: np.ndarray               # (21,) rad
    root_z: float                  # 世界系高度
    g_body: np.ndarray             # (3,) 机体系重力向量
    robot_id: str = "robot_a"
    qvel: Optional[np.ndarray] = None        # (21,) rad/s（可选）
    root_linvel: Optional[np.ndarray] = None # (3,) 机体系线速度
    root_angvel: Optional[np.ndarray] = None # (3,) 机体系角速度

    _MODEL = None                  # class-level lazy kinematics
    _DATA = None
    _ROBOTS = None

    def __post_init__(self):
        self.qpos = np.array(self.qpos, np.float64)      # copy, not alias
        assert self.qpos.shape == _JOINTS, \
            f"qpos expected {_JOINTS}, got {self.qpos.shape}"
        self.g_body = np.array(self.g_body, np.float64)
        assert self.g_body.shape == _G, \
            f"g_body expected {_G}, got {self.g_body.shape}"
        self.root_z = float(self.root_z)
        for n, shp in (("qvel", _JOINTS), ("root_linvel", _G),
                       ("root_angvel", _G)):
            v = getattr(self, n)
            if v is not None:
                v = np.array(v, np.float64)
                assert v.shape == shp, \
                    f"{n} expected {shp}, got {v.shape}"
                setattr(self, n, v)

    @classmethod
    def _kinematics(cls):
        """Lazy shared (model, data, robots) — read-only kinematics."""
        if cls._MODEL is None:
            cls._MODEL = mujoco.MjSpec.from_file(
                Humanoid21Simulator.ARENA_XML).compile()
            cls._DATA = mujoco.MjData(cls._MODEL)
            cls._ROBOTS = Humanoid21Meta.build_runtime_tables(
                cls._MODEL)["robots"]
        return cls._MODEL, cls._DATA, cls._ROBOTS

    @classmethod
    def joint_ranges(cls, robot_id: str = "robot_a") -> np.ndarray:
        """(21,2) joint radian ranges (obs denormalization)."""
        _, _, robots = cls._kinematics()
        return np.asarray(robots[robot_id]["jnt_ranges"], dtype=np.float64)

    def _set_fk_pose(self):
        """Write this pose into the shared kinematics model."""
        model, data, robots = self._kinematics()
        r = robots[self.robot_id]
        adr = int(r["root_qpos_adr"])
        data.qpos[adr:adr + 3] = [0.0, 0.0, self.root_z]
        data.qpos[adr + 3:adr + 7] = _quat_wxyz(_level_rot(self.g_body))
        data.qpos[np.asarray(r["qpos_indices"])] = self.qpos
        mujoco.mj_kinematics(model, data)
        return model, data, r

    @classmethod
    def from_obs(cls, obs: np.ndarray,
                 joint_ranges: Optional[np.ndarray] = None,
                 robot_id: str = "robot_a"):
        """obs (96,) -> Pose（关节角反归一化）。

        joint_ranges: (21,2) 各关节 (lo,hi) 弧度；None = 从 fk 取。
        """
        if joint_ranges is None:
            joint_ranges = cls.joint_ranges(robot_id)
        rng = np.asarray(joint_ranges, np.float64)
        ref, sc = (rng[:, 0] + rng[:, 1]) / 2, (rng[:, 1] - rng[:, 0]) / 2
        ov = np.asarray(obs[21:42], np.float64)
        oa = np.asarray(obs[49:52], np.float64)
        return cls(qpos=np.asarray(obs[0:21], np.float64) * sc + ref,
                   root_z=float(obs[45]),
                   g_body=np.asarray(obs[42:45], np.float64),
                   robot_id=robot_id,
                   qvel=np.sign(ov) * (2.0 * ov) ** 2,
                   root_linvel=np.asarray(obs[46:49], np.float64),
                   root_angvel=np.sign(oa) * 2.0 * oa ** 2)

    @classmethod
    def from_p5(cls, p5, robot_id: str = "robot_a",
                tolerance: float = 0.02):
        """P5 局部姿态 -> Pose（solve_equilibrium 反解关节角）。

        p5: (5,3) 局部点阵或 P5 对象。g_body = 直立系重力
        [0,0,-9.81]——P5 局部帧本身是重力对齐的。
        """
        from point5.humanoid21.mechanics import (
            Skeleton, solve_equilibrium)
        pos = getattr(p5, "pos", p5)
        res = solve_equilibrium(np.asarray(pos, np.float64),
                                skeleton=Skeleton(), tolerance=tolerance)
        return cls(qpos=np.asarray(res["q_rad"]),
                   root_z=float(res["root_pos"][2]),
                   g_body=np.array([0.0, 0.0, -9.81]), robot_id=robot_id)

    def p5(self):
        """自身局部帧 P5（踝中点原点、躯干 +x）。"""
        return self.anchor('ankle_mid').p5()

    def support_landmark_on_p5_axis(self):
        """P5 轴系下的 SupportState。

        com 精确质量加权；foot pos 来自锚定帧；force_ratio 按
        落地脚均分（姿态本身不带接触力——这是支撑语义不是测量）。
        """
        from point5.humanoid21.executor.contracts import (
            FootSupport, SupportState)
        a = self.anchor('ankle_mid')
        fp, raw = a.foot_pos(), a.p5().pos
        # P5 脚点是踝中心——落地时 z≈0.087，不是 0
        planted = [s for s in ("right", "left")
                   if raw[_FIDX[s]][2] < 0.12]
        share = 1.0 / len(planted) if planted else 0.0
        feet = {s: FootSupport(force_ratio=share if s in planted
                               else 0.0, pos=fp[s])
                for s in ("right", "left")}
        return SupportState(com=a.com(), feet=feet)

    def tilt_angles(self) -> np.ndarray:
        """(2,) [roll, pitch] rad — 由 g_body 分解的倾斜角。"""
        gx, gy, gz = self.g_body / (np.linalg.norm(self.g_body) + 1e-12)
        return np.array([np.arctan2(-gy, -gz), np.arctan2(gx, -gz)])

    # ---------- anchored (position-resolved) outputs ----------

    def _fk_state(self):
        """Pose the shared kinematics model and read geometry.

        Returns (landmark positions (5,3), exact mass-weighted com (3,),
        torso yaw, R_tilt)."""
        model, data, r = self._set_fk_pose()
        lm = np.zeros((5, 3))
        for i, name in enumerate(POINT_NAMES):
            body, off = LANDMARKS[name]
            bid = r["keypoint_body_ids"].get(body) or                 mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                  f"{body}{r['suffix']}")
            lm[i] = (np.asarray(data.xpos[bid])
                     + np.asarray(data.xmat[bid]).reshape(3, 3) @ off)
        ids = np.asarray(r["body_ids_sorted"], int)
        ms = np.asarray(model.body_mass)[ids]
        xip = np.asarray(data.xipos)[ids]
        com = (ms[:, None] * xip).sum(0) / ms.sum()
        tbid = r["keypoint_body_ids"].get("torso") or             mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                              f"torso{r['suffix']}")
        fwd = np.asarray(data.xmat[tbid]).reshape(3, 3)[:, 0]
        torso_yaw = float(np.arctan2(fwd[1], fwd[0]))
        return lm, com, torso_yaw, _level_rot(self.g_body)

    def anchor(self, point: str = "ankle_mid",
               xy=(0.0, 0.0), yaw: float = 0.0) -> _Anchored:
        """Resolve the positionless pose into a frame.

        point: 'ankle_mid' | 'left' | 'right' | 'com'
        xy:    where the point lands
        yaw:   desired torso-heading direction (0 = +x, 沿用局部系约定)
        """
        lm, com, tyaw, R_tilt = self._fk_state()
        if point == "ankle_mid":
            raw = (lm[3, :2] + lm[4, :2]) / 2
        elif point == "com":
            raw = com[:2]
        elif point in ("left", "right"):
            raw = lm[_FIDX[point], :2]
        else:
            raise ValueError(f"unknown anchor point {point!r}")

        a = yaw - tyaw                    # rotation to put torso at yaw
        Rz = np.array([[np.cos(a), -np.sin(a), 0],
                       [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        T = (lambda v: Rz @ (np.asarray(v, np.float64)
                             - np.array([raw[0], raw[1], 0.0]))
             + np.array([xy[0], xy[1], 0.0]))
        feet = {s: T(lm[fi])[:2] for s, fi in _FIDX.items()}
        return _Anchored(_p5=np.array([T(p) for p in lm]),
                         _feet=feet, _com=T(com),
                         R=Rz @ R_tilt)

    def render(self, out_path,
               views=(("front", 0.0, 0.0), ("side", 90.0, 0.0),
                      ("oblique", 45.0, 18.0)),
               size=(640, 480), mark_points: bool = True,
               target_pos: Optional[np.ndarray] = None) -> list:
        """Render this pose to PNG(s) via MuJoCo offscreen.

        Needs an OpenGL platform (MUJOCO_GL/PYOPENGL_PLATFORM=egl).
        target_pos: optional (5,3) target drawn as hollow markers.
        """
        from pathlib import Path
        from PIL import Image, ImageDraw
        from point5.contracts import POINT_NAMES

        model, data, r = self._set_fk_pose()
        lm = np.zeros((5, 3))
        for i, name in enumerate(POINT_NAMES):
            body, off = LANDMARKS[name]
            bid = r["keypoint_body_ids"].get(body) or \
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                  f"{body}{r['suffix']}")
            lm[i] = (np.asarray(data.xpos[bid])
                     + np.asarray(data.xmat[bid]).reshape(3, 3) @ off)
        fovy = float(model.vis.global_.fovy) or 45.0
        outs = []
        for label, az, el in views:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = np.array([0., 0., self.root_z - 0.35])
            cam.distance, cam.azimuth, cam.elevation = 3.0, az, el
            with mujoco.Renderer(model, size[1], size[0]) as rd:
                rd.update_scene(data, camera=cam)
                img = rd.render()
            im = Image.fromarray(img)
            if mark_points:
                dr = ImageDraw.Draw(im)
                if target_pos is not None:
                    tp = np.asarray(target_pos, dtype=np.float64)
                    for i in range(len(POINT_NAMES)):
                        px = _project(cam, fovy, size, tp[i])
                        if px:
                            dr.ellipse([px[0]-8, px[1]-8, px[0]+8,
                                        px[1]+8], outline=(255, 255, 255),
                                       width=2)
                for i, n in enumerate(POINT_NAMES):
                    px = _project(cam, fovy, size, lm[i])
                    if px:
                        dr.ellipse([px[0]-6, px[1]-6, px[0]+6, px[1]+6],
                                   outline=(0, 0, 0),
                                   fill=_POINT_RGBA[n][:3])
            p = Path(str(out_path).format(label=label))
            im.save(p)
            outs.append(p)
        return outs

    def _momentum_state(self, point: str = "ankle_mid",
                        xy=(0.0, 0.0), yaw: float = 0.0):
        """(anchored, subtree_linvel, subtree_angmom) — 需速度字段。"""
        if self.qvel is None:
            raise ValueError("pose has no velocity (from_p5 is pose-only)")
        model, data, r = self._set_fk_pose()
        data.qvel[:] = 0.0
        adr = int(r["root_qvel_adr"])
        # freejoint: linvel WORLD, angvel BODY — obs linvel is body
        # frame, convert via the leveled quat (consistent w/ anchor)
        data.qvel[adr:adr + 3] = _level_rot(self.g_body) @ self.root_linvel
        data.qvel[adr + 3:adr + 6] = self.root_angvel
        data.qvel[np.asarray(r["qvel_indices"])] = self.qvel
        mujoco.mj_subtreeVel(model, data)
        a = self.anchor(point, xy, yaw)
        ids = np.asarray(r["body_ids_sorted"], int)
        lv = a.R @ np.asarray(data.subtree_linvel[int(ids[0])],
                              np.float64)
        am = a.R @ np.asarray(data.subtree_angmom[int(ids[0])],
                              np.float64)
        return a, lv, am

    def com_vel(self, **anchor_kw) -> np.ndarray:
        """Exact subtree COM velocity (3,) — anchored local frame.

        与 ``a.com()``/``a.p5()`` 同一坐标系（默认 ankle_mid 原点、
        躯干 +x、重力 -z）：com_vel 指哪，com 就往哪漂。
        mujoco subtree_linvel 原本在世界系，经 ``a.R`` 旋入锚定帧；
        旋转保模长——换锚点只改分量表达，不改 |v|。"""
        return self._momentum_state(**anchor_kw)[1]

    def ang_mom(self, **anchor_kw) -> np.ndarray:
        """Subtree angular momentum (3,) — anchored local frame."""
        return self._momentum_state(**anchor_kw)[2]

    # ---------- unanchored ----------

    def com_height(self) -> float:
        """Exact mass-weighted COM height — absolute, no anchor needed."""
        _, com, _, _ = self._fk_state()
        return float(com[2])
