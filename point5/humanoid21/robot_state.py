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


    def reanchor(self, xy=(0.0, 0.0),
                 yaw: float = 0.0) -> "Pose":
        """重新锚定到指定位置/朝向（standardize 的逆操作）。

        输出保证踝中点在 xy、躯干朝向 yaw，其余不变——
        内部先 standardize 再做 SE(2) 平面变换，任意帧输入
        都合法，不动姿态形状。
        """
        raise NotImplementedError

    def standardize(self) -> "Pose":
        """转成自身 P5 系（踝中点原点、躯干 +x、yaw→0）。"""
        raise NotImplementedError

    def to_p5(self) -> "P5":
        """-> P5 点集（所在帧表达）——FK 提取 5 landmark + torso_yaw。"""
        import mujoco
        from point5 import P5
        from point5.humanoid21.mechanics import LANDMARKS
        model, data, r = self._fk_data()
        kw = {}
        for body, p5name in (("head", "head"),
                             ("hand_left", "left_hand"),
                             ("hand_right", "right_hand"),
                             ("foot_left", "left_ankle"),
                             ("foot_right", "right_ankle")):
            bid = r["keypoint_body_ids"].get(body)
            if bid is None:
                bid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{body}{r['suffix']}")
            kw[p5name] = (np.asarray(data.xpos[bid])
                          + np.asarray(data.xmat[bid]).reshape(3, 3)
                          @ LANDMARKS[body][1])
        return P5(**kw, torso_yaw=float(self.root_orientation[2]))
    # ---------- MuJoCo FK（simulator 集成，私有） ----------

    _FK = None      # 懒加载共享 (model, data, robot_a 表)

    @classmethod
    def _kinematics(cls):
        """懒加载共享 FK 模型——只读使用。"""
        if cls._FK is None:
            import mujoco
            from envs.humanoid21.meta import Humanoid21Meta
            from envs.humanoid21.simulator import Humanoid21Simulator
            model = mujoco.MjSpec.from_file(
                Humanoid21Simulator.ARENA_XML).compile()
            data = mujoco.MjData(model)
            r = Humanoid21Meta.build_runtime_tables(model)["robots"][
                "robot_a"]
            cls._FK = (model, data, r)
        return cls._FK

    def _fk_data(self):
        """把本姿态写入共享 FK 模型并跑 mj_kinematics。"""
        import mujoco
        model, data, r = self._kinematics()
        adr = int(r["root_qpos_adr"])
        data.qpos[adr:adr + 3] = self.root_position
        data.qpos[adr + 3:adr + 7] = _zyx_to_quat_wxyz(
            self.root_orientation)
        data.qpos[np.asarray(r["qpos_indices"])] = self.qpos
        mujoco.mj_kinematics(model, data)
        return model, data, r

    def to_support_geometry(self) -> "SupportGeometry":
        """-> SupportGeometry：MuJoCo FK 提取 com/脚位/足高/yaw。

        踝位 = foot body xpos + xmat·(0,0,0.06)（踝中心，复用
        mechanics.LANDMARKS 偏移）；com = body_mass 加权 xipos；
        foot_height = 踝高 − ANKLE_Z_PLANTED；yaw 取 orientation[2]。
        """
        import mujoco
        from point5.humanoid21.mechanics import LANDMARKS
        from .support_state import SupportGeometry, ANKLE_Z_PLANTED
        model, data, r = self._fk_data()

        def _ankle(side):
            bid = r["keypoint_body_ids"].get(f"foot_{side}")
            if bid is None:
                bid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY,
                    f"foot_{side}{r['suffix']}")
            return (np.asarray(data.xpos[bid])
                    + np.asarray(data.xmat[bid]).reshape(3, 3)
                    @ LANDMARKS[f"foot_{side}"][1])

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
    ang_mom: np.ndarray              # (3,) 全身角动量，关于质心——换帧平移不变，仅随 yaw 旋转分量

    def __post_init__(self):
        super().__post_init__()
        self.qvel = _vec(self.qvel, _JOINTS, "qvel")
        self.root_linvel = _vec(self.root_linvel, _VEC3, "root_linvel")
        self.ang_mom = _vec(self.ang_mom, _VEC3, "ang_mom")

    @classmethod
    def from_obs(cls, obs: np.ndarray) -> "RobotState":
        """obs (96,) -> RobotState，产 **P5 坐标系**的态。

        obs 输入由机体系定义——机体系非 leveled 帧族成员，
        故解码产物统一表达为 P5 系（leveled）。
        """
        raise NotImplementedError

    def standardize(self) -> "RobotState":
        """几何归一 + 动态量随帧旋转——保持动态类型与字段。

        qpos/qvel 关节空间不变；root_position 转 SE(2)；
        orientation 仅 yaw 加 Δ；root_linvel/ang_mom 乘 Rz(Δyaw)。
        """
        raise NotImplementedError

    def reanchor(self, xy=(0.0, 0.0),
                 yaw: float = 0.0) -> "RobotState":
        """同 standardize——帧变换时 root_linvel/ang_mom 随 Δyaw 旋转。"""
        raise NotImplementedError

    def pose(self) -> Pose:
        """剥动态 → Pose（对应 SupportState.geometry）。"""
        return Pose(qpos=self.qpos.copy(),
                    root_position=self.root_position.copy(),
                    root_orientation=self.root_orientation.copy())

    def to_support_state(self):
        """-> SupportState：委托 SupportState.from_robot_state(self)。

        实现住 target 层（X.from_y 惯例），此方法是源侧语法糖。
        """
        from .support_state import SupportState
        return SupportState.from_robot_state(self)
