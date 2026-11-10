"""Pose / RobotState — 机器人状态的静态/动态两级表达。

    Pose        静态姿态 = qpos + root_position + root_orientation（SE(3) 完备）
    RobotState  动态快照 = Pose + qvel + root_linvel + ang_mom（全量必带）

帧约定（leveled 帧族）：与坐标系有关的量可表达于任何
**z=0 为地面、+z 竖直向上的右手系**——帧间自由度只剩
xy 原点与 yaw（SE(2)）。所在帧由持有方约定；
from_obs 产 P5 坐标系的态。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_JOINTS = (21,)
_VEC3 = (3,)


def _vec(v, shape, name):
    v = np.array(v, np.float64)          # copy, never alias
    assert v.shape == shape, f"{name}: expected {shape}, got {v.shape}"
    return v


def tilt_angles(g_body: np.ndarray) -> np.ndarray:
    """(2,) [pitch, roll] rad — g_body 按 ZYX 约定解码躯干倾斜。

    与 yaw 无关（Rz 不改动竖直轴），g_body = g·[sinθ,
    -sinφ·cosθ, -cosφ·cosθ]（θ=pitch, φ=roll），故：
    pitch = atan2(gx, √(gy²+gz²))、roll = atan2(-gy, -gz)。
    obs 投影重力解码用。
    """
    g = np.asarray(g_body, np.float64)
    g = g / (np.linalg.norm(g) + 1e-12)
    return np.array([np.arctan2(g[0], np.hypot(g[1], g[2])),
                     np.arctan2(-g[1], -g[2])])


def joint_ranges() -> np.ndarray:
    """(21,2) 关节弧度范围 — 机器人形态元数据（懒加载）。"""
    return np.asarray(Pose._kinematics()[2]["jnt_ranges"], np.float64)


def _wrap(a: float) -> float:
    """角度归一到 (-pi, pi]。"""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _zyx_to_quat_wxyz(ori: np.ndarray) -> np.ndarray:
    """[pitch,roll,yaw] -> quat (wxyz)，R = Rz(yaw)·Ry(pitch)·Rx(roll)。"""
    p, r, y = ori
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    return np.array([cy * cp * cr + sy * sp * sr,
                     cy * cp * sr - sy * sp * cr,
                     cy * sp * cr + sy * cp * sr,
                     sy * cp * cr - cy * sp * sr])


@dataclass
class Pose:
    """关节角 + 根位置 + 根朝向——机器人姿态的完备静态表达。

    root_orientation = [pitch, roll, yaw] rad，ZYX 约定：
        R = Rz(yaw) · Ry(pitch) · Rx(roll)
    pitch>0 前倾（绕左方 +y）；roll>0 右倾（绕前方 +x）；
    yaw 为躯干朝向相对所在帧 +x 的夹角。
    奇异点：|pitch|≈π/2（躺平）时 yaw/roll 退化——下游
    勿在此区间判读 yaw。
    """
    qpos: np.ndarray                 # (21,) rad — 关节空间，与坐标系无关
    root_position: np.ndarray        # (3,) 根位置（所在帧）
    root_orientation: np.ndarray     # (3,) [pitch,roll,yaw] R=Rz·Ry·Rx

    def __post_init__(self):
        self.qpos = _vec(self.qpos, _JOINTS, "qpos")
        self.root_position = _vec(self.root_position, _VEC3,
                                  "root_position")
        self.root_orientation = _vec(self.root_orientation, _VEC3,
                                     "root_orientation")

    # ---------- constructors ----------

    @classmethod
    def infer_from_p5(cls, p5, tolerance: float = 0.02) -> "Pose":
        """P5 局部姿态 -> Pose（solve_equilibrium 反解关节角）。

        infer_* = 近似推断（源里没有 qpos/position，靠 solver 重建），
        区别于 from_* 的无损解码。产 P5 坐标系的姿态。
        """
        raise NotImplementedError


    def _se2(self, dyaw: float, dxy) -> "Pose":
        """SE(2) 平面变换：root_position' = Rz(dyaw)·pos + (dxy,0)，
        orientation 仅 yaw += dyaw——姿态形状不变。"""
        c, s = np.cos(dyaw), np.sin(dyaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        o = self.root_orientation
        return Pose(qpos=self.qpos.copy(),
                    root_position=R @ self.root_position
                    + np.array([dxy[0], dxy[1], 0.0]),
                    root_orientation=np.array(
                        [o[0], o[1], _wrap(o[2] + dyaw)]))

    def standardize(self) -> "Pose":
        """转成自身 P5 系（踝中点原点、躯干 +x、yaw→0）。"""
        p5 = self.to_p5()
        mid = (p5.left_ankle[:2] + p5.right_ankle[:2]) / 2.0
        yaw = float(self.root_orientation[2])
        c, s = np.cos(-yaw), np.sin(-yaw)
        t = -np.array([[c, -s], [s, c]]) @ mid      # -Rz(-yaw)·mid
        return self._se2(-yaw, t)

    def reanchor(self, xy=(0.0, 0.0),
                 yaw: float = 0.0) -> "Pose":
        """重新锚定到指定位置/朝向（standardize 的逆操作）。

        输出保证踝中点在 xy、躯干朝向 yaw，其余不变——
        内部先 standardize 再做 SE(2) 平面变换，任意帧输入
        都合法，不动姿态形状。
        """
        return self.standardize()._se2(float(yaw), np.asarray(xy, float))

    def to_p5(self) -> "P5":
        """-> P5 点集（所在帧表达）——FK 提取 5 landmark + torso_yaw。

        头/手取 body xpos；踝 = foot xpos + xmat·踝偏移
        （ankle_y/x 关节锚点中点）。
        """
        import mujoco
        from point5 import P5
        model, data, r, off = self._fk_data()
        kw = {}
        for body, p5name, o in (
                ("head", "head", np.zeros(3)),
                ("hand_left", "left_hand", np.zeros(3)),
                ("hand_right", "right_hand", np.zeros(3)),
                ("foot_left", "left_ankle", off["left"]),
                ("foot_right", "right_ankle", off["right"])):
            bid = r["keypoint_body_ids"].get(body)
            if bid is None:
                bid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{body}{r['suffix']}")
            kw[p5name] = (np.asarray(data.xpos[bid])
                          + np.asarray(data.xmat[bid]).reshape(3, 3) @ o)
        return P5(**kw, torso_yaw=float(self.root_orientation[2]))
    # ---------- MuJoCo FK（simulator 集成，私有） ----------

    _FK = None      # 懒加载共享 (model, data, robot_a 表, 踝偏移)

    @classmethod
    def _kinematics(cls):
        """懒加载共享 FK 模型 + 踝点偏移（ankle_y/x 锚点中点）。

        注意：**不支持多线程**——_FK 里的 MjData 是全部实例共享的
        可变缓冲区，并行调用 FK 方法会互相覆盖中间结果。
        """
        if cls._FK is None:
            import mujoco
            from envs.humanoid21.meta import Humanoid21Meta
            from envs.humanoid21.simulator import Humanoid21Simulator
            model = mujoco.MjSpec.from_file(
                Humanoid21Simulator.ARENA_XML).compile()
            data = mujoco.MjData(model)
            r = Humanoid21Meta.build_runtime_tables(model)["robots"][
                "robot_a"]
            ankle_off = {}
            for side in ("left", "right"):
                jy = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT,
                    f"ankle_y_{side}{r['suffix']}")
                jx = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT,
                    f"ankle_x_{side}{r['suffix']}")
                ankle_off[side] = (np.asarray(model.jnt_pos[jy])
                                   + np.asarray(model.jnt_pos[jx])) / 2
            cls._FK = (model, data, r, ankle_off)
        return cls._FK

    def _fk_data(self):
        """把本姿态写入共享 FK 模型并跑 mj_kinematics。"""
        import mujoco
        model, data, r, off = self._kinematics()
        adr = int(r["root_qpos_adr"])
        data.qpos[adr:adr + 3] = self.root_position
        data.qpos[adr + 3:adr + 7] = _zyx_to_quat_wxyz(
            self.root_orientation)
        data.qpos[np.asarray(r["qpos_indices"])] = self.qpos
        mujoco.mj_kinematics(model, data)
        return model, data, r, off

    def to_support_geometry(self) -> "SupportGeometry":
        """-> SupportGeometry：MuJoCo FK 提取 com/脚位/足高/yaw。

        踝位 = foot body xpos + xmat·踝偏移（ankle_y/x 关节锚点
        中点，_kinematics 预提取）；com = body_mass 加权 xipos；
        foot_height = 踝高 − ANKLE_Z_PLANTED；yaw 取 orientation[2]。
        """
        import mujoco
        from .support_state import SupportGeometry, ANKLE_Z_PLANTED
        model, data, r, off = self._fk_data()

        def _ankle(side):
            bid = r["keypoint_body_ids"].get(f"foot_{side}")
            if bid is None:
                bid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY,
                    f"foot_{side}{r['suffix']}")
            return (np.asarray(data.xpos[bid])
                    + np.asarray(data.xmat[bid]).reshape(3, 3)
                    @ off[side])

        la, ra = _ankle("left"), _ankle("right")
        ids = np.asarray(r["body_ids_sorted"], int)
        ms = np.asarray(model.body_mass)[ids]
        com = (ms[:, None] * np.asarray(data.xipos)[ids]).sum(0) / ms.sum()
        return SupportGeometry(
            com=com,
            left_foot=la[:2].copy(),
            right_foot=ra[:2].copy(),
            torso_yaw=float(self.root_orientation[2]),
            left_foot_height=float(la[2] - ANKLE_Z_PLANTED),
            right_foot_height=float(ra[2] - ANKLE_Z_PLANTED))

    def render(self, out_path, **kw) -> list:
        """渲染为 PNG（需 OpenGL platform）。"""
        raise NotImplementedError


@dataclass
class RobotState(Pose):
    """动态快照 = 姿态 + 速度——全部必带，from_obs 唯一入口。

    与坐标系有关的量可在任何 z=0 为地面、+z 竖直向上的右手系
    中定义（帧族约定见模块头）；无动态用 Pose，不构造残缺
    RobotState。
    """
    qvel: np.ndarray                 # (21,) rad/s — 关节空间，与坐标系无关
    root_linvel: np.ndarray          # (3,) 根线速度（所在帧）
    root_angvel: np.ndarray          # (3,) 根角速度（所在帧）——FK 速度通道 raw 量
    ang_mom: np.ndarray              # (3,) 全身角动量，关于质心——换帧平移不变，仅随 yaw 旋转分量

    def __post_init__(self):
        super().__post_init__()
        self.qvel = _vec(self.qvel, _JOINTS, "qvel")
        self.root_linvel = _vec(self.root_linvel, _VEC3, "root_linvel")
        self.root_angvel = _vec(self.root_angvel, _VEC3, "root_angvel")
        self.ang_mom = _vec(self.ang_mom, _VEC3, "ang_mom")

    @classmethod
    def from_obs(cls, obs: np.ndarray) -> "RobotState":
        """obs (96,) -> RobotState，产 **P5 坐标系**的态。

        obs 输入由机体系定义——机体系非 leveled 帧族成员，
        故解码产物统一表达为 P5 系（leveled）：

          qpos  = obs[0:21]·scale + ref（线性反归一化）
          qvel  = sign(o)·(2o)²·scale（sqrt 解压 + 归一尺度）
          pitch/roll = tilt_angles(obs[42:45])；yaw≡0（帧定义）
          root  = [0,0,obs[45]] 经 standardize 解析踝中点原点
          linvel/angvel = R @ obs 机体系值（R = 躯干 xmat）
          ang_mom = mj_subtreeVel 子树角动量（关于质心）
        """
        obs = np.asarray(obs, np.float64)
        rng = joint_ranges()
        ref, sc = (rng[:, 0] + rng[:, 1]) / 2, (rng[:, 1] - rng[:, 0]) / 2
        qpos = obs[0:21] * sc + ref
        qvel = np.sign(obs[21:42]) * (2.0 * obs[21:42]) ** 2 * sc
        pit, rol = tilt_angles(obs[42:45])
        pose = Pose(qpos=qpos,
                    root_position=np.array([0.0, 0.0, float(obs[45])]),
                    root_orientation=np.array([pit, rol, 0.0])
                    ).standardize()
        _, data, r, _ = pose._fk_data()
        R = np.asarray(data.xmat[r["root_body_id"]]).reshape(3, 3)
        linvel = R @ obs[46:49]
        angvel = R @ (np.sign(obs[49:52]) * 2.0 * obs[49:52] ** 2)
        tmp = cls(qpos=qpos, root_position=pose.root_position,
                  root_orientation=pose.root_orientation, qvel=qvel,
                  root_linvel=linvel, root_angvel=angvel,
                  ang_mom=np.zeros(3))
        _, vdata, vr = tmp._fk_vel_data()
        return cls(qpos=tmp.qpos, root_position=tmp.root_position,
                   root_orientation=tmp.root_orientation, qvel=tmp.qvel,
                   root_linvel=tmp.root_linvel, root_angvel=tmp.root_angvel,
                   ang_mom=np.asarray(
                       vdata.subtree_angmom[vr["root_body_id"]]).copy())

    def _se2(self, dyaw: float, dxy) -> "RobotState":
        """SE(2) 变换：姿态走父类，动态量乘 Rz(dyaw)（qvel 关节空间不变）。"""
        p = super()._se2(dyaw, dxy)
        c, s = np.cos(dyaw), np.sin(dyaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return RobotState(
            qpos=p.qpos, root_position=p.root_position,
            root_orientation=p.root_orientation,
            qvel=self.qvel.copy(),
            root_linvel=R @ self.root_linvel,
            root_angvel=R @ self.root_angvel,
            ang_mom=R @ self.ang_mom)

    def standardize(self) -> "RobotState":
        """几何归一 + 动态量随帧旋转——保持动态类型与字段。

        qpos/qvel 关节空间不变；root_position 转 SE(2)；
        orientation 仅 yaw 加 Δ；linvel/angvel/ang_mom 乘 Rz(Δyaw)。
        """
        return super().standardize()

    def reanchor(self, xy=(0.0, 0.0),
                 yaw: float = 0.0) -> "RobotState":
        """同 standardize——linvel/angvel/ang_mom 随 Δyaw 旋转。"""
        return super().reanchor(xy, yaw)

    def pose(self) -> Pose:
        """剥动态 → Pose（对应 SupportState.geometry）。"""
        return Pose(qpos=self.qpos.copy(),
                    root_position=self.root_position.copy(),
                    root_orientation=self.root_orientation.copy())

    def _fk_vel_data(self):
        """写完整速度（含 ω）跑前馈 + mj_subtreeVel——FK 速度量。

        MuJoCo freejoint qvel[3:6] 是机体系角速度——存储按所在帧
        约定，喂模型前用躯干 xmat 转一道。subtree_linvel/angmom
        依赖速度段管线，须先 mj_forward 再 subtreeVel。
        """
        import mujoco
        model, data, r, _ = self._fk_data()
        adr = int(r["root_qvel_adr"])
        data.qvel[adr:adr + 3] = self.root_linvel
        data.qvel[adr + 3:adr + 6] = (
            np.asarray(data.xmat[r["root_body_id"]]).reshape(3, 3).T
            @ self.root_angvel)
        data.qvel[np.asarray(r["qvel_indices"])] = self.qvel
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        return model, data, r

    def to_support_state(self):
        """-> SupportState：FK 提取几何 + 速度量（实现住此）。

        com_vel 取子树质心线速度（subtree_linvel[torso]）；
        ang_mom 直取本态语义量（应与 subtree_angmom 一致）。
        """
        from .support_state import SupportState
        g = self.to_support_geometry()
        model, data, r = self._fk_vel_data()
        return SupportState(
            com=g.com, left_foot=g.left_foot, right_foot=g.right_foot,
            torso_yaw=g.torso_yaw,
            left_foot_height=g.left_foot_height,
            right_foot_height=g.right_foot_height,
            com_vel=np.asarray(
                data.subtree_linvel[r["root_body_id"]]).copy(),
            ang_mom=self.ang_mom.copy())
