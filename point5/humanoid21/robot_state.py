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
    if v.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError(f"{name}: non-finite values")
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


def _quat_wxyz_to_zyx(q: np.ndarray) -> np.ndarray:
    """quat (wxyz) -> [pitch,roll,yaw]，_zyx_to_quat_wxyz 之逆。"""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        np.arctan2(-2.0 * (x * z - y * w),
                   np.hypot(1.0 - 2.0 * (y * y + z * z),
                            2.0 * (x * y + z * w))),
        np.arctan2(2.0 * (y * z + x * w),
                   1.0 - 2.0 * (x * x + y * y)),
        np.arctan2(2.0 * (x * y + z * w),
                   1.0 - 2.0 * (y * y + z * z))])


def _canonical_zyx(ori: np.ndarray) -> np.ndarray:
    """[pitch,roll,yaw] -> 标准 ZYX 分支（cos(pitch)>=0）的等价角。

    非标准分支（cos(pitch)<0）的同一旋转写成
    [π-pitch, roll+π, yaw+π]；奇异点 cos(pitch)=0 原样保留。
    规范化后 yaw 分量恒等于躯干前向的水平朝向。
    """
    p, r, y = ori
    if np.cos(p) < 0.0:
        p, r, y = np.pi - p, r + np.pi, y + np.pi
    return np.array([_wrap(p), _wrap(r), _wrap(y)])


def _torso_yaw(ori: np.ndarray) -> float:
    """躯干前向的水平朝向 rad——与欧拉角分支无关。

    = atan2 对 R@[1,0,0] 的 xy 分量（= [cos(yaw)cos(pitch),
    sin(yaw)cos(pitch)]，roll 不影响）。躯干前向近竖直
    （|cos(pitch)|≈0，gimbal lock）时退回存储 yaw 分量——
    退化区的约定值，下游勿在此区间判读。
    """
    cp = np.cos(ori[0])
    if abs(cp) < 1e-12:
        return float(ori[2])
    return float(np.arctan2(np.sin(ori[2]) * cp,
                            np.cos(ori[2]) * cp))


@dataclass
class Pose:
    """关节角 + 根位置 + 根朝向——机器人姿态的完备静态表达。

    root_orientation = [pitch, roll, yaw] rad，ZYX 约定：
        R = Rz(yaw) · Ry(pitch) · Rx(roll)
    pitch>0 前倾（绕左方 +y）；roll>0 右倾（绕前方 +x）；
    yaw 为躯干朝向相对所在帧 +x 的夹角。
    奇异点：|pitch|≈π/2（躺平）时 yaw/roll 退化——下游
    勿在此区间判读 yaw。

    构造时 orientation 规范化到标准分支（cos(pitch)>=0，
    见 _canonical_zyx）——非标准分支输入被改写为同旋转的
    等价角，使 yaw 分量恒等于躯干前向的水平朝向。
    """
    qpos: np.ndarray                 # (21,) rad — 关节空间，与坐标系无关
    root_position: np.ndarray        # (3,) 根位置（所在帧）
    root_orientation: np.ndarray     # (3,) [pitch,roll,yaw] R=Rz·Ry·Rx（标准分支）

    def __post_init__(self):
        self.qpos = _vec(self.qpos, _JOINTS, "qpos")
        self.root_position = _vec(self.root_position, _VEC3,
                                  "root_position")
        self.root_orientation = _canonical_zyx(
            _vec(self.root_orientation, _VEC3, "root_orientation"))

    # ---------- constructors ----------

    @classmethod
    def infer_from_p5(cls, p5, tolerance: float = 0.02) -> "Pose":
        """P5 -> Pose（三阶段反解，能量最小+硬位置约束）。

        infer_* = 近似推断（源里没有 qpos/position，靠 solver 重建），
        区别于 from_* 的无损解码。产 P5 坐标系的姿态。

        流程（infer_pose_from_p5.md）：standardize → 阶段一 13 变量
        （pitch/roll+腰3+腿8）min 弹性能 s.t. 踝心−头相对向量 6 eq →
        阶段二冻结躯干闭式臂 IK（不可达放宽）→ 阶段三踝=弹簧中心 →
        root 由头绝对位置恢复 → _se2 反变换回原帧。

        阶段一几何不可达（多初值后残差仍超 tolerance）时 ValueError。
        """
        from .mechanics import solve_p5_pose
        std = p5.standardize()
        target = np.stack([std.head, std.right_hand, std.left_hand,
                           std.right_ankle, std.left_ankle])
        res = solve_p5_pose(target, tolerance=tolerance)
        if not res["satisfied"]:
            raise ValueError(
                "infer_from_p5: 阶段一不可达 "
                f"(max_resid={res['stage1']['max_resid']:.4f} m)")
        pose_std = cls(qpos=res["q_rad"],
                       root_position=res["root_pos"],
                       root_orientation=np.array(
                           [res["pitch"], res["roll"], 0.0]))
        mid_xy = (p5.left_ankle[:2] + p5.right_ankle[:2]) / 2.0
        return pose_std._se2(p5.torso_yaw, mid_xy)

    @classmethod
    def infer_from_cpose(cls, cpose) -> "Pose":
        """CPose -> Pose——本机对 CPose 语义的闭式解释。

        infer_* = 近似推断（CPose 不表达的自由度由先验补全）。
        映射约定见 mechanics.py "CPose 适配" 段：torso_pry 1:1 →
        腹部三铰链；肢向量 ×LMax → 3-DOF 闭式 IK；root 由 pelvis
        位姿反推。所在帧 = CPose 所在帧；不可达端点按最近可达。
        """
        from .mechanics import cpose_to_pose
        res = cpose_to_pose(cpose)
        return cls(qpos=res["qpos"], root_position=res["root_pos"],
                   root_orientation=res["root_zyx"])

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
        yaw = _torso_yaw(self.root_orientation)
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

        头/手取 body xpos；踝 = 踝关节中心 xanchor[ankle_y]
        （脚绕其旋转的锚点，脚踝铰链运动下不动，随小腿走）。
        """
        import mujoco
        from point5 import P5
        model, data, r, jid = self._fk_data()
        kw = {}
        for body, p5name in (("head", "head"),
                             ("hand_left", "left_hand"),
                             ("hand_right", "right_hand")):
            bid = r["keypoint_body_ids"].get(body)
            if bid is None:
                bid = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"{body}{r['suffix']}")
            kw[p5name] = np.asarray(data.xpos[bid]).copy()
        kw["left_ankle"] = np.asarray(data.xanchor[jid["left"]]).copy()
        kw["right_ankle"] = np.asarray(data.xanchor[jid["right"]]).copy()
        return P5(**kw, torso_yaw=_torso_yaw(self.root_orientation))


    def to_cpose(self) -> "CPose":
        """-> CPose：FK 提取（所在帧表达）——infer_from_cpose 逆读。

        纯骨架 FK（Skeleton，无 MuJoCo）；映射约定见 mechanics.py
        "CPose 适配" 段。
        """
        from .mechanics import pose_to_cpose
        return pose_to_cpose(self.qpos, self.root_position,
                             _zyx_to_quat_wxyz(self.root_orientation))
    # ---------- MuJoCo FK（simulator 集成，私有） ----------

    _FK = None      # 懒加载共享 (model, data, robot_a 表, 踝铰 joint id)

    @classmethod
    def _kinematics(cls):
        """懒加载共享 FK 模型 + 踝铰（ankle_y）joint id。

        踝点 = 踝关节中心 = ankle_y 铰链锚点世界坐标
        (xanchor)：它是脚绕其旋转的点——脚踝屈伸/侧滚下
        不动（上游铰链的旋转中心，不受下游 ankle_x 影响），
        随小腿刚体走，不是脚上的物质点。

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
            ankle_jid = {}
            for side in ("left", "right"):
                ankle_jid[side] = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT,
                    f"ankle_y_{side}{r['suffix']}")
            cls._FK = (model, data, r, ankle_jid)
        return cls._FK

    def _fk_data(self):
        """把本姿态写入共享 FK 模型并跑 mj_kinematics。"""
        import mujoco
        model, data, r, jid = self._kinematics()
        adr = int(r["root_qpos_adr"])
        data.qpos[adr:adr + 3] = self.root_position
        data.qpos[adr + 3:adr + 7] = _zyx_to_quat_wxyz(
            self.root_orientation)
        data.qpos[np.asarray(r["qpos_indices"])] = self.qpos
        mujoco.mj_kinematics(model, data)
        return model, data, r, jid

    def to_support_geometry(self) -> "SupportGeometry":
        """-> SupportGeometry：MuJoCo FK 提取 com/脚位/足高/yaw。

        脚位 = 踝关节中心 xanchor[ankle_y]（脚绕其旋转的锚点，
        脚踝铰链运动下不动，随小腿走）；com = body_mass 加权
        xipos；foot_height = 踝心高 − ANKLE_Z_PLANTED；
        yaw 取躯干前向水平朝向（_torso_yaw）。
        """
        from .support_state import SupportGeometry, ANKLE_Z_PLANTED
        model, data, r, jid = self._fk_data()

        def _ankle(side):
            return np.asarray(data.xanchor[jid[side]])

        la, ra = _ankle("left"), _ankle("right")
        ids = np.asarray(r["body_ids_sorted"], int)
        ms = np.asarray(model.body_mass)[ids]
        com = (ms[:, None] * np.asarray(data.xipos)[ids]).sum(0) / ms.sum()
        return SupportGeometry(
            com=com,
            left_foot=la[:2].copy(),
            right_foot=ra[:2].copy(),
            torso_yaw=_torso_yaw(self.root_orientation),
            left_foot_height=float(la[2] - ANKLE_Z_PLANTED),
            right_foot_height=float(ra[2] - ANKLE_Z_PLANTED))

    _RENDERERS = {}   # (height,width) -> mujoco.Renderer 懒缓存
    _FK_RB_ADR = None  # robot_b root_qpos_adr（render 停置用）

    def render(self, out_path=None, width: int = 1280,
               height: int = 720, camera=None,
               distance: float = 2.5, elevation: float = -20.0,
               azimuth: float = 135.0) -> "PIL.Image.Image":
        """-> PIL.Image：当前姿态离屏渲染（参照 simulator
        broadcast-view：mjv_defaultCamera 自定位 + 尺寸缓存
        Renderer + update_scene/render）。

        camera=None 时用自由相机：lookat=root_position（z
        clamp [0.2,0.9]），distance/elevation/azimuth 可调；
        camera=<xml 相机名>（如 "cam_diag_0"/"ceilingcamera"）
        改用场景相机。robot_b 停置 (10,10) 不入画；out_path
        给定时另存 PNG。

        需 OpenGL platform——本机 egl/glfw 不可用，未设
        MUJOCO_GL 时默认 "osmesa"（预设值优先）。共享渲染
        资源与 FK 缓冲区**非线程安全**（同 _FK 约定）。
        """
        import os
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        import mujoco
        from PIL import Image
        model, data, r, jid = self._kinematics()
        adr = int(r["root_qpos_adr"])
        data.qpos[adr:adr + 3] = self.root_position
        data.qpos[adr + 3:adr + 7] = _zyx_to_quat_wxyz(
            self.root_orientation)
        data.qpos[np.asarray(r["qpos_indices"])] = self.qpos
        if Pose._FK_RB_ADR is None:
            from envs.humanoid21.meta import Humanoid21Meta
            Pose._FK_RB_ADR = int(Humanoid21Meta
                .build_runtime_tables(model)["robots"]
                ["robot_b"]["root_qpos_adr"])
        data.qpos[Pose._FK_RB_ADR:Pose._FK_RB_ADR + 3] = (
            10.0, 10.0, 0.5)                           # robot_b 停置
        mujoco.mj_kinematics(model, data)

        if camera is None:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            lookat = self.root_position.copy()
            lookat[2] = np.clip(lookat[2], 0.2, 0.9)
            cam.lookat[:] = lookat
            cam.distance, cam.elevation, cam.azimuth = (
                float(distance), float(elevation), float(azimuth))
            camera = cam
        key = (int(height), int(width))
        renderer = Pose._RENDERERS.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(model, height=key[0],
                                       width=key[1])
            Pose._RENDERERS[key] = renderer
        renderer.update_scene(data, camera=camera)
        img = Image.fromarray(renderer.render())
        if out_path is not None:
            img.save(out_path)
        return img


    def inertia_matrix(self) -> np.ndarray:
        """-> (3,3) 全身转动惯量矩阵——**关于质心**，所在帧表达。

        J_com = Σ_b [ R_b·R_i·diag(I_b)·(R_b·R_i)ᵀ
                      + m_b·(|r_b|²I − r_b·r_bᵀ) ]
        I_b = body_inertia——**主惯量系**主值，须经 body_iquat
        旋回 body 系（R_i）；r_b = xipos_b − com。
        刚体锁定时 H_com = J_com·ω——与 ang_mom 的"关于质心"
        约定同点；lip_analyze 绕支点惯量经平行轴外推。
        """
        import mujoco
        model, data, r, _ = self._fk_data()
        ids = np.asarray(r["body_ids_sorted"], int)
        ms = np.asarray(model.body_mass)[ids]
        com = (ms[:, None] * np.asarray(data.xipos)[ids]).sum(0) / ms.sum()
        J = np.zeros((3, 3))
        for m_b, b in zip(ms, ids):
            Rb = np.asarray(data.xmat[b]).reshape(3, 3)
            Ri = np.empty(9)
            mujoco.mju_quat2Mat(Ri, np.asarray(model.body_iquat[b]))
            Ri = Ri.reshape(3, 3)
            Rp = Rb @ Ri                      # 主惯量系 -> 所在帧
            Ib = Rp @ np.diag(np.asarray(model.body_inertia[b])) @ Rp.T
            d = np.asarray(data.xipos[b]) - com
            J += Ib + m_b * ((d @ d) * np.eye(3) - np.outer(d, d))
        return J

    def corresponding_action(self) -> np.ndarray:
        """-> (21,) 归一化 action：本姿态关节角对应的位置目标。

        rad_to_norm 闭式逆映射 + [-1,1] 截断（qpos 越限部分
        无法表达为目标）。即 action_for_pose 的闭式解——PD
        伺服负责跟踪。区别于 INITIAL_POSES['action']：那是
        全体姿态共用的复位扶持目标，非各姿态自身映射。
        """
        from .mechanics import rad_to_norm
        return np.clip(rad_to_norm(self.qpos), -1.0, 1.0)

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


    # ---------- 标准态工厂：快速生成 RobotState ----------
    # standing/squat/stand_on_left_leg/prone/supine 取
    # Humanoid21Meta.INITIAL_POSES（sim reset 权威表）；
    # ready 为自建格斗准备姿（常量见模块尾 _READY_*）。
    # 约定：**标准姿态不带速度**——qvel/root_linvel/
    # root_angvel/ang_mom 恒零；受扰/随机态走 random()。

    STANDARD_POSES = ("standing", "squat", "stand_on_left_leg",
                      "prone", "supine", "ready")

    @classmethod
    def standard(cls, name: str) -> "RobotState":
        """-> RobotState：按名取标准姿态（全零速度静态量）。

        INITIAL_POSES 源数据 = joint_pos rad + root_pos +
        root_quat(wxyz) 直搬（quat -> [pitch,roll,yaw] 经
        _quat_wxyz_to_zyx）。root_pos 不含 reset 的双人
        x_offset——那是占位而非姿态属性；面朝 +x。
        """
        name = str(name)
        if name == "ready":
            qpos = _READY_QPOS
            root, ori = _READY_ROOT_POS, _READY_ROOT_ORI
        else:
            from envs.humanoid21.meta import Humanoid21Meta
            cfg = Humanoid21Meta.INITIAL_POSES.get(name)
            if cfg is None:
                raise ValueError(
                    f"standard: unknown pose {name!r} "
                    f"(expected one of {cls.STANDARD_POSES})")
            qpos = np.asarray(cfg["joint_pos"], np.float64)
            root = np.asarray(cfg["root_pos"], np.float64)
            ori = _quat_wxyz_to_zyx(
                np.asarray(cfg["root_quat"], np.float64))
        z3, z21 = np.zeros(3), np.zeros(21)
        return cls(qpos=qpos, root_position=root,
                   root_orientation=ori, qvel=z21,
                   root_linvel=z3, root_angvel=z3.copy(),
                   ang_mom=z3.copy())

    @classmethod
    def standing(cls) -> "RobotState":
        """站姿——standard("standing")。"""
        return cls.standard("standing")

    @classmethod
    def squat(cls) -> "RobotState":
        """深蹲——standard("squat")。"""
        return cls.standard("squat")

    @classmethod
    def stand_on_left_leg(cls) -> "RobotState":
        """左腿独立——standard("stand_on_left_leg")。"""
        return cls.standard("stand_on_left_leg")

    @classmethod
    def prone(cls) -> "RobotState":
        """俯卧——standard("prone")。"""
        return cls.standard("prone")

    @classmethod
    def supine(cls) -> "RobotState":
        """仰卧——standard("supine")。"""
        return cls.standard("supine")

    @classmethod
    def ready(cls) -> "RobotState":
        """运动准备姿——standard("ready")。

        双腿张开、膝微屈、手臂腹部屈收；P5 意图经
        infer_from_p5 反解后烘焙常量（带前倾 bias——
        对本姿态语义无害）。
        """
        return cls.standard("ready")

    @classmethod
    def random(cls, seed=None, qpos_scale: float = 0.3,
               vel_scale: float = 0.0) -> "RobotState":
        """测试用随机态：关节 = 站姿 ± scale·限位半幅均匀
        采样；root 在站姿附近小扰动（xy/z ±0.3/±0.2 m，
        pitch/roll ±0.2 rad，yaw ±π）。vel_scale>0 时各速度
        维 ±vel_scale 均匀采样，ang_mom 经 FK 真值回填（不
        是独立噪声）；=0 时与标准姿态同约定全零。

        非物理可达姿态——供 solver/编解码 round-trip 采样，
        勿当可立姿态用。
        """
        rng = np.random.default_rng(seed)
        st = cls.standing()
        r = joint_ranges()
        half = (r[:, 1] - r[:, 0]) / 2.0
        tmp = cls(
            qpos=st.qpos + rng.uniform(-1.0, 1.0, 21)
            * float(qpos_scale) * half,
            root_position=st.root_position + np.array([
                rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3),
                rng.uniform(-0.2, 0.2)]),
            root_orientation=np.array([
                rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2),
                rng.uniform(-np.pi, np.pi)]),
            qvel=rng.uniform(-vel_scale, vel_scale, 21),
            root_linvel=rng.uniform(-vel_scale, vel_scale, 3),
            root_angvel=rng.uniform(-vel_scale, vel_scale, 3),
            ang_mom=np.zeros(3))
        if vel_scale > 0.0:
            _, vdata, vr = tmp._fk_vel_data()
            tmp.ang_mom = np.asarray(
                vdata.subtree_angmom[vr["root_body_id"]]).copy()
        return tmp

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
        if obs.shape != (96,):
            raise ValueError(f"obs: expected (96,), got {obs.shape}")
        if not np.isfinite(obs).all():
            raise ValueError("obs: non-finite values")
        rng = joint_ranges()
        ref, sc = (rng[:, 0] + rng[:, 1]) / 2, (rng[:, 1] - rng[:, 0]) / 2
        qpos = obs[0:21] * sc + ref
        qvel = np.sign(obs[21:42]) * (2.0 * obs[21:42]) ** 2 * sc
        g = obs[42:45]
        if np.linalg.norm(g) < 1e-3:
            raise ValueError(
                f"obs 投影重力范数≈0 ({np.linalg.norm(g):.2e})"
                "——姿态不可解码（传感器失效/自由落体）")
        pit, rol = tilt_angles(g)
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

    _SIM = None     # 懒加载共享 (simulator, robot_a 表, robot_b 表)

    @classmethod
    def _sim(cls):
        """懒加载共享模拟器 + 双机器人运行时表——take_action
        的动力学 oracle。

        注意：**不支持多线程**——sim.data 是全部调用共享的
        可变缓冲区（与 _FK 同约定）。
        """
        if cls._SIM is None:
            from envs.humanoid21.meta import Humanoid21Meta
            from envs.humanoid21.simulator import Humanoid21Simulator
            sim = Humanoid21Simulator()
            rb = Humanoid21Meta.build_runtime_tables(
                sim.model)["robots"]
            cls._SIM = (sim, rb["robot_a"], rb["robot_b"])
        return cls._SIM


    # 根据当前状态，在一个控制周期内（50MS），各个关节所能到的范围。每个节关一对范围值。
    # 假定关节以最大力矩朝那个方向工作，至于如何让关节有哪个力矩，是PD反解的问题。
    # 不需要做关节关联的分析
    

    def take_action(self, action, dt: float = 0.05) -> "RobotState":
        """21 维归一化动作 -> 新 RobotState（模拟器单控制周期
        真值推演，**返回新对象**——本态保持不变，供其它候选
        动作在同一底本上评估）。

        action 与 env action 空间一致：[-1,1] 归一化关节目标，
        PD 伺服逐物理子步驱动（DT=0.002s）。dt = 控制周期，
        折合 round(dt/DT) 个子步；dt<=0 报 ValueError。

        状态原样写入 sim 世界系（输入属 leveled 帧族、z=0 地面
        与 sim 一致，输出即同帧表达、免帧簿记）；robot_b 停置
        (10,10) 不参与。arena 墙体在世界坐标固定位置——输入态
        xy 远离场心时撞墙是诚实物理而非误差。共享 sim 状态
        **非线程安全**；推演确定性（同态同动作同输出）。
        """
        import mujoco
        if dt <= 0:
            raise ValueError(f"take_action: dt<=0 ({dt})")
        action = np.asarray(action, np.float64)
        if action.shape != _JOINTS:
            raise ValueError(
                f"take_action: action shape {action.shape} != {_JOINTS}")
        sim, ra, rb = self._sim()
        model, data = sim.model, sim.data
        mujoco.mj_resetData(model, data)   # 清 warmstart/act/接触
                                          # 等易变态——跨调用确定性

        adr, vadr = int(ra["root_qpos_adr"]), int(ra["root_qvel_adr"])
        quat = _zyx_to_quat_wxyz(self.root_orientation)
        data.qpos[adr:adr + 3] = self.root_position
        data.qpos[adr + 3:adr + 7] = quat
        data.qpos[np.asarray(ra["qpos_indices"])] = self.qpos
        data.qvel[vadr:vadr + 3] = self.root_linvel       # 世界系线速度
        Rm = np.empty(9)
        mujoco.mju_quat2Mat(Rm, quat)
        data.qvel[vadr + 3:vadr + 6] = (
            Rm.reshape(3, 3).T @ self.root_angvel)        # 存机体系角速度
        data.qvel[np.asarray(ra["qvel_indices"])] = self.qvel
        badr, bvadr = int(rb["root_qpos_adr"]), int(rb["root_qvel_adr"])
        data.qpos[badr:badr + 3] = (10.0, 10.0, 0.5)      # robot_b 停置远处
        data.qvel[bvadr:bvadr + 6] = 0.0
        sim._data_cache.clear()
        mujoco.mj_forward(model, data)

        sim.set_action({"robot_a": action, "robot_b": np.zeros(21)})
        for _ in range(max(1, int(round(dt / sim.dt)))):
            sim.physical_step()

        mujoco.mj_subtreeVel(model, data)
        quat_o = np.asarray(data.qpos[adr + 3:adr + 7], np.float64)
        Rm_o = np.empty(9)
        mujoco.mju_quat2Mat(Rm_o, quat_o)
        Rm_o = Rm_o.reshape(3, 3)
        zyx = np.array([
            np.arctan2(-Rm_o[2, 0], np.hypot(Rm_o[0, 0], Rm_o[1, 0])),
            np.arctan2(Rm_o[2, 1], Rm_o[2, 2]),
            np.arctan2(Rm_o[1, 0], Rm_o[0, 0])])
        return RobotState(
            qpos=data.qpos[np.asarray(ra["qpos_indices"])].copy(),
            root_position=data.qpos[adr:adr + 3].copy(),
            root_orientation=zyx,
            qvel=data.qvel[np.asarray(ra["qvel_indices"])].copy(),
            root_linvel=data.qvel[vadr:vadr + 3].copy(),
            root_angvel=Rm_o @ data.qvel[vadr + 3:vadr + 6],
            ang_mom=np.asarray(
                data.subtree_angmom[ra["root_body_id"]]).copy())

    # ---------- 能力查询：单步可达域 ----------

    _REACH = None   # 懒加载 (names, jid, jb, sub_of, ra, kp, kd, tmax, lim)

    @classmethod
    def _reach_tables(cls):
        """懒加载可达域所需的执行器/拓扑表（结构常量，与位形无关）。"""
        if cls._REACH is None:
            from envs.humanoid21.meta import Humanoid21Meta
            sim, ra, _ = cls._sim()
            m = sim.model
            pd = sim._pd_tables["robot_a"]
            names = list(Humanoid21Meta.CONTROLLED_JOINTS)
            parent = list(m.body_parentid)

            def _sub(root):
                out, grow = {root}, True
                while grow:
                    grow = False
                    for b in range(m.nbody):
                        if parent[b] in out and b not in out:
                            out.add(b)
                            grow = True
                return out

            bodies_a = _sub(int(ra["root_body_id"]))
            jid = {names[k]: int(m.actuator_trnid[a, 0])
                   for k, a in enumerate(np.asarray(pd["actuator_ids"]))}
            jb = {n: int(m.jnt_bodyid[j]) for n, j in jid.items()}
            sub_of = {n: _sub(b) & bodies_a for n, b in jb.items()}
            tmax = (np.maximum(np.abs(pd["ctrl_lo"]),
                               np.abs(pd["ctrl_hi"]))
                    * np.abs(pd["gear"]))
            lim = np.array([m.jnt_range[jid[n]] for n in names])
            cls._REACH = dict(names=names, jid=jid, jb=jb,
                              sub_of=sub_of, bodies_a=bodies_a,
                              kp=np.asarray(Humanoid21Meta.KP, float),
                              kd=np.asarray(Humanoid21Meta.KD, float),
                              tmax=np.asarray(tmax, float), lim=lim)
        return cls._REACH

    def joint_reach_range(self, dt: float = 0.05,
                          clamp_pd: bool = True) -> np.ndarray:
        """-> (21,2) rad：本态起 dt 内各关节**独立**可达角区间。

        模型（"冻住他人"近似，无关节关联）：关节 i 以最大力矩
        推 ± 方向、其余关节视为 PD 锁死，单关节二阶积分

            θ(T) = θ0 + ω0·T + ½·α·T²
            α± = (±τ_eff − τ_g) 换向即 (∓τ_eff + τ_g)/J_eff

        - J_eff：动侧绕关节轴惯量。下游侧（腰/臂/摆动腿）= M
          对角元精确值；上游侧（触地腿驱动上身）= 补集 bodies
          绕轴惯量（轴惯量+平行轴）。
        - τ_g：动侧重力矩的关节广义力（上游侧作用反号——
          关节角 = 子侧相对父侧）；触地腿为上游负载，双腿
          触地按对称载减半（静不定近似）。
        - τ_eff：clamp_pd 时用 PD 可发力矩上界
          min(τ_act, |KP·(θ_t−θ0)−KD·ω0| 端点极值)——θ_t 最多
          打到关节限位，贴边/反向速度时真实上限收窄；
          =False 时用裸执行器上限 τ_act（"假设力矩能给满"）。
        - stance 判定：foot_height < 0.05 m 视为触地
          （to_support_geometry 的踝心高）。

        输出顺序 = CONTROLLED_JOINTS（qpos 序）；区间 = 时刻 T
        可达集合（恒力界下即该区间），再与关节限位求交。
        语义边界：这是单关节可达域，非全身联合可达域——多关节
        联动的真实包络要 take_action 采样式刻画。
        """
        import mujoco
        if dt <= 0:
            raise ValueError(f"joint_reach_range: dt<=0 ({dt})")
        T = self._reach_tables()
        model, data, r, _ = self._fk_data()
        mujoco.mj_crb(model, data)
        Mmat = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, Mmat, data.qM)
        Mdiag = np.diag(Mmat)[np.asarray(r["qvel_indices"])]

        g = self.to_support_geometry()
        planted = {"right": g.right_foot_height < 0.05,
                   "left": g.left_foot_height < 0.05}
        both = planted["right"] and planted["left"]

        # 全体 body 的世界系惯量张量/质心（遍历一遍，各关节复用）
        Iw, mb_, pb = {}, {}, {}
        for b in T["bodies_a"]:
            Rb = np.asarray(data.xmat[b]).reshape(3, 3)
            Ri = np.empty(9)
            mujoco.mju_quat2Mat(Ri, np.asarray(model.body_iquat[b]))
            Rp = Rb @ Ri.reshape(3, 3)
            Iw[b] = Rp @ np.diag(np.asarray(model.body_inertia[b])) @ Rp.T
            mb_[b] = model.body_mass[b]
            pb[b] = np.asarray(data.xipos[b])

        def side_stats(bodies, anchor, e):
            """-> (J_axis, tau_g)：bodies 绕轴 e@anchor 的轴惯量
            与重力矩（带符号，+ 方向沿 e）。"""
            J, com, ml = 0.0, np.zeros(3), 0.0
            for b in bodies:
                d = pb[b] - anchor
                dp = d - (d @ e) * e
                J += float(e @ Iw[b] @ e) + mb_[b] * (dp @ dp)
                com += mb_[b] * pb[b]
                ml += mb_[b]
            com /= ml
            tau_g = float(e @ np.cross(com - anchor,
                                       np.array([0., 0., -ml * 9.81])))
            return J, tau_g

        out = np.empty((21, 2))
        leg = {"right": T["names"][3:9], "left": T["names"][9:15]}
        for i, n in enumerate(T["names"]):
            ax_w = (np.asarray(data.xmat[T["jb"][n]]).reshape(3, 3)
                    @ np.asarray(model.jnt_axis[T["jid"][n]]))
            ax_w /= np.linalg.norm(ax_w) + 1e-12
            anchor = np.asarray(data.xanchor[T["jid"][n]])
            side = ("right" if n in leg["right"]
                    else "left" if n in leg["left"] else None)
            use_complement = side is not None and planted[side]
            bodies = (T["bodies_a"] - T["sub_of"][n]
                      if use_complement else T["sub_of"][n])
            J, tau_g = side_stats(bodies, anchor, ax_w)
            # 重力矩对关节角的广义力：子侧旋转=+θ，父侧（补集）
            # 旋转=−θ——上游负载侧作用反号。
            q_g = -tau_g if use_complement else tau_g
            if use_complement and both:
                q_g *= 0.5                  # 双腿分摊重力负载
            if not use_complement:
                J = float(Mdiag[i])         # 下游侧精确惯量
            J = max(J, 1e-9)

            th0, w0 = self.qpos[i], self.qvel[i]
            lo_lim, hi_lim = T["lim"][i]
            tp = tn = T["tmax"][i]
            if clamp_pd:
                kp, kd = T["kp"][i], T["kd"][i]
                tp = min(tp, max(0.0, kp * (hi_lim - th0) - kd * w0))
                tn = min(tn, max(0.0, kp * (th0 - lo_lim) + kd * w0))
            a_hi = (tp + q_g) / J
            a_lo = (-tn + q_g) / J
            out[i, 0] = max(lo_lim, th0 + w0 * dt + 0.5 * a_lo * dt * dt)
            out[i, 1] = min(hi_lim, th0 + w0 * dt + 0.5 * a_hi * dt * dt)
        return out

    # ---------- action_for_*：take_action 的逆向查询族 ----------

    _BASE_ACT = None  # 懒缓存站姿对应 action——踝策略非踝目标锚点

    @classmethod
    def _base_action(cls) -> np.ndarray:
        """-> (21,) 站姿 corresponding_action 快照（懒缓存）。

        踝策略族的非踝目标基准：每次输出 = 站姿 Action +
        踝改动——把躯干钉回站姿，而不是锚每步当前态（那
        会棘轮式锁进漂移体态）。
        """
        if cls._BASE_ACT is None:
            cls._BASE_ACT = cls.standing().corresponding_action()
        return cls._BASE_ACT.copy()          # 返回拷贝，调用方会改踝位

    def action_for_balance_with_ankle_policy(
            self, base: np.ndarray = None) -> np.ndarray:
        """-> (21,) action：双支撑踝调节闭环——阻尼律稳站。

        base = 非踝通道基准（None=站姿快照；非站姿起步传
        起步姿的 corresponding_action——钉住的是起步体态
        而非站姿）。

        双支撑铰链模的可控失稳自由度只有一维：垂直双踝连
        线 e_perp（沿轴侧向 = 载重分摊通道、非踝职——
        ankle_policy docstring 的侧向边界）。每步取期望
        加速度

            d = −ω²·(com−a_mid) − 2ω·com_vel   沿 e_perp 的
            PD 调节，ω=√(g/l) 摆频、临界阻尼

        调 ankle_policy——理想 LIP 下 ẍ=d，即把 COM 当二
        阶系统直接镇定到双踝中点。**不用 margin 触发**
        （核内 hold、出核才拉）：LIP 在核内也指数发散，
        越界介入时速度已超踝能力——sim 实测证伪。非双支
        撑 NotImplementedError，走各形态自己的策略族。
        """
        from .support_state import (ANKLE_Z_PLANTED, LIP_G,
                                    SupportClass)
        st = self.to_support_state()
        if st.support_class() != SupportClass.DOUBLE:
            raise NotImplementedError(
                "action_for_balance_with_ankle_policy: "
                f"{st.support_class().name} 未实现——v1 只做双脚")
        l = float(st.com[2]) - ANKLE_Z_PLANTED
        if l <= 0:
            raise ValueError(
                "action_for_balance_with_ankle_policy: "
                f"com.z {st.com[2]:.3f} ≤ 踝心——LIP 失效")
        p0, p1 = np.asarray(st.left_foot), np.asarray(st.right_foot)
        e = p1 - p0
        if np.linalg.norm(e) > 1e-3:
            e = e / np.linalg.norm(e)
        else:                                   # 双踝重合退化——取躯干侧向
            cy, sy = np.cos(st.torso_yaw), np.sin(st.torso_yaw)
            e = np.array([-sy, cy])
        e_perp = np.array([-e[1], e[0]])
        a_mid = 0.5 * (p0 + p1)
        x_err = float((st.com[:2] - a_mid) @ e_perp)
        v = float(st.com_vel[:2] @ e_perp)
        om2 = LIP_G / l
        d_perp = -(om2 * x_err + 2.0 * np.sqrt(om2) * v)
        return self.ankle_policy(d_perp * e_perp, base=base)

    def action_for_pose(self, aim_pose: "Pose") -> np.ndarray:
        """-> (21,) 归一化 action：朝 aim 关节构型推进。

        aim_pose.qpos 给出关节层目标——action 空间本身即
        归一化关节目标，答案近似闭式 rad_to_norm(aim.qpos)，
        PD 伺服负责跟踪；aim 的 root_position/orientation 只
        表达意图（root 不可直接命令，经动力学涌现）。候选
        实现 = 闭式转换 + take_action 验证/修正。
        单步语义同上。
        """
        raise NotImplementedError

    def action_for_support_state(
            self, aim_support_state: "SupportState") -> np.ndarray:
        """-> (21,) 归一化 action：朝 aim 支撑态推进。

        三档中最重——aim 在质心层（com/脚位/动量），须消费
        step-transition 栈（drift 预测、can_balance 判核、
        落点可行集、接力映射）再做"支撑层计划 -> 21 维全身
        action"的全身映射。aim 为 SupportGeometry 时 = 纯
        几何目标（不含动量）。单步语义同上。
        """
        raise NotImplementedError

    def ankle_policy(self, direction,
                     base: np.ndarray = None) -> np.ndarray:
        """-> (21,) action：双支撑踝策略原语——朝 direction 推重心。

        base = 非踝通道的钉住基准 (21,) action——None 用站
        姿快照；非站姿起步应传起步姿的 corresponding_action
        （钉住的是"初始姿态"而非"站姿"）。
        direction = 期望重心水平加速度 (2,) m/s²（world xy，
        带幅值；超出踝能力逐轴饱和）。

        机械链（双踝摆，双脚均摊法向力 N_i = mg/2）：
          l  = com.z − 踝心平面高     —— LIP 摆长
          c* = com_xy − l·d/g         —— 目标系统 CoP：想重心
               朝 +d 加速，CoP 须落在 −d 一侧
          Δ  = c* − 双踝心中点         —— 各脚 CoP 相对己踝心
               的同步偏移需求
          τ_wy = +N_i·Δx，τ_wx = −N_i·Δy
                                      —— 各脚踝心所需世界系
               力矩（踝矩平衡）
          τ_jy = τ_wy；τ_jx = τ_wx/kx  —— ankle_y 纯俯仰轴直
               通；ankle_x 斜轴 kx=±0.8944（附带 ∓0.4472·τ_jx
               偏航矩，双脚同侧向需求下相加、靠扭转摩擦吸收，
               v1 不补偿）
          |τ_w| ≤ min(τ_act, N_i·margin_dir) 逐轴 clamp——饱和
               即本原语的能力边界（前后 ±0.1m 级，侧向 ±0.047m）
          target = qpos + (τ_j + KD·qvel)/KP → rad_to_norm——
               action 是归一化 PD 位置目标，非力矩指令；
               踝目标锚当前 qpos（力矩精确、脚板贴地、踝角
               随体转自由浮动）

        非踝关节 = _base_action（站姿 action 快照）——钉回
        站姿体态；若锚每步当前态则棘轮式锁进漂移。

        侧向边界：双脚踩平时沿双踝连线方向踝无可控力
        ——ankle_x 只能在己脚板内挪 ±4.7cm 压心，对 COM
        侧向加速度无控制权；该方向是 hip_x/腹部的载重分
        摊通道（实测 ankle_x 全关与全开侧向发散无差别）。
        """
        from .mechanics import _HALF, _JIDX, _MID
        from .support_state import (ANKLE_Z_PLANTED, LIP_G,
                                    LIP_MARGIN_HEEL,
                                    LIP_MARGIN_SIDE,
                                    LIP_MARGIN_TOE, LIP_MASS,
                                    LIP_TAU_ACT, SupportClass)
        d = np.asarray(direction, np.float64).ravel()
        if d.shape != (2,) or not np.isfinite(d).all():
            raise ValueError(
                "ankle_policy: direction 期望有限 (2,) "
                f"m/s²——got {d}")
        g = self.to_support_geometry()
        if g.support_class() != SupportClass.DOUBLE:
            raise NotImplementedError(
                f"ankle_policy: {g.support_class().name} 未实现"
                "——v1 只做双脚")
        l = float(g.com[2]) - ANKLE_Z_PLANTED
        if l <= 0:
            raise ValueError(f"ankle_policy: com.z {g.com[2]:.3f} "
                             "≤ 踝心——LIP 失效")
        c_star = g.com[:2] - (l / LIP_G) * d
        a_mid = 0.5 * (np.asarray(g.left_foot)
                       + np.asarray(g.right_foot))
        dc = c_star - a_mid
        ni = 0.5 * LIP_MASS * LIP_G
        dx = float(np.clip(dc[0],
                           -min(LIP_TAU_ACT, ni * LIP_MARGIN_HEEL)
                           / ni,
                           min(LIP_TAU_ACT, ni * LIP_MARGIN_TOE)
                           / ni))
        dy = float(np.clip(dc[1],
                           -min(LIP_TAU_ACT, ni * LIP_MARGIN_SIDE)
                           / ni,
                           min(LIP_TAU_ACT, ni * LIP_MARGIN_SIDE)
                           / ni))
        action = (np.asarray(base, dtype=np.float64).copy()
                  if base is not None else self._base_action())
        tau_y, tau_x = ni * dx, -ni * dy
        for side, kx in (("right", _ANKLE_AX), ("left", -_ANKLE_AX)):
            for j, t in ((_JIDX[f"ankle_y_{side}"], tau_y),
                         (_JIDX[f"ankle_x_{side}"], tau_x / kx)):
                tgt = (self.qpos[j]
                       + (t + _ANKLE_KD * self.qvel[j]) / _ANKLE_KP)
                action[j] = float(np.clip(
                    (tgt - _MID[j]) / _HALF[j], -1.0, 1.0))
        return action

    def arm_as_reaction_wheel_policy(self, direction: float,
                                     base: np.ndarray = None,
                                     straight: bool = None
                                     ) -> np.ndarray:
        """-> (21,) action：双臂当**有界反应轮**——矢状面内满
        扭矩旋转，给本体反向角动量。

        base = 非臂通道的钉住基准 (21,) action——None 用站
        姿快照 _base_action()；从非站姿起步（如 ready）应
        传该起步姿的对应 action（corresponding_action），
        否则 15 个非臂通道被拽向站姿基准、起步即崩。

        direction = 期望身体绕铰链轴 e 获得的力矩（N·m，
        带符号）：身体前倒需反向刹时 direction<0，手臂朝
        前加速把角动量抽进手臂（臂 L 增长方向 = 倾倒方
        向，反作用给身体反向矩）。

        实现：矢状摆角目标 = 当前摆角 + lead。位置 PD 下
        关节误差 ≈lead − 臂在控制窗内的行程 ⇒ 平均力矩
        ≈ KP·(lead − δ_travel)；lead 取
        −direction/(2·KP_shoulder)·GAIN（GAIN≈3 补回被
        行程吃掉的部分，oracle 标定），即"位置超前量=力
        矩指令"的近似线性映射。|lead| 截断 _WHEEL_LEAD
        （饱和输出 gear 上限的最大角加速度——实测
        bang-bang 满扭矩反冲直接踹翻站姿，必须按比例
        给）。双臂同相镜像；肘保持（|v| 不变 ⇒ 惯量不
        变）。非臂通道 = _base_action（站姿快照，含踝）。

        本质是有界飞轮——方向矢量在矢状面可达弧仅 ~130°
        （FK 实测：可达方向是前方扇区，无完整圆可塞入，
        **连续风车不可行**），容量一次性 ~5-10 kg·m²/s，
        饱和后无法续力。非双支撑 NotImplementedError。

        straight：None 取模块档 _WHEEL_STRAIGHT——True=直
        臂圆摆（肘锁全伸、经下方转向，ready 实证 6s 站住）；
        False=保肘角摆（standing 上更稳——直臂前悬的静态
        前偏在站姿上反伤稳定，实测 com 漂出 0.23m）。
        """
        from .mechanics import _HALF, _JIDX, _MID, Skeleton
        from .support_state import SupportClass
        d = float(np.asarray(direction).ravel()[0])
        if not np.isfinite(d):
            raise ValueError(
                "arm_as_reaction_wheel_policy: direction 期望"
                f"有限标量 N·m——{d}")
        g = self.to_support_geometry()
        if g.support_class() != SupportClass.DOUBLE:
            raise NotImplementedError(
                "arm_as_reaction_wheel_policy: "
                f"{g.support_class().name} 未实现——v1 只做双脚")
        p0, p1 = np.asarray(g.left_foot), np.asarray(g.right_foot)
        e = p1 - p0
        if np.linalg.norm(e) > 1e-3:
            e = e / np.linalg.norm(e)
        else:
            cy, sy = np.cos(g.torso_yaw), np.sin(g.torso_yaw)
            e = np.array([-sy, cy])
        e3 = np.array([e[0], e[1], 0.0])
        ep3 = np.array([-e[1], e[0], 0.0])
        sk = Skeleton()
        th = self._swing_theta_now(e3, ep3, sk)
        lead = float(np.clip(-d / (2.0 * _WHEEL_KP)
                             * _WHEEL_GAIN,
                             -_WHEEL_LEAD, _WHEEL_LEAD))
        th_cmd = th + lead
        ls = _WHEEL_LH_SCALE
        st_flag = _WHEEL_STRAIGHT if straight is None else straight
        if not self._swing_theta_feasible(th_cmd, e3, ep3, sk, ls,
                                          st_flag):
            lo, hi = 0.0, 1.0                     # θ=th+t·lead 上二分
            for _ in range(18):
                m = 0.5 * (lo + hi)
                if self._swing_theta_feasible(th + m * lead, e3,
                                              ep3, sk, ls, st_flag):
                    lo = m
                else:
                    hi = m
            th_cmd = th + 0.5 * (lo + hi) * lead
        q = self._arm_swing_qpos(th_cmd, e3, ep3, sk, ls,
                                 st_flag)
        action = (np.asarray(base, dtype=np.float64).copy()
                  if base is not None else self._base_action())
        for i in _ARM_J:
            action[i] = float(np.clip(
                (q[i] - _MID[i]) / _HALF[i], -1.0, 1.0))
        return action

    def _swing_theta_now(self, e3, ep3, sk) -> float:
        """当前臂摆角（"肩→手"矢量在 {ep3,ez} 面内从下垂
        量起的角度），双臂取平均。"""
        xf = sk.body_transforms(
            self.root_position,
            _zyx_to_quat_wxyz(self.root_orientation), self.qpos)
        t_pos, t_mat = xf["torso"]
        ez3 = np.array([0.0, 0.0, 1.0])
        acc = 0.0
        from .mechanics import _ARM
        for side in ("left", "right"):
            sp = _ARM[side]
            v = (xf[f"hand_{side}"][0]
                 - (t_pos + t_mat @ sp["s_off"]))
            acc += float(np.arctan2(v @ ep3, -(v @ ez3)))
        return 0.5 * acc

    def action_for_balance_with_wheel_policy(
            self, base: np.ndarray = None,
            straight: bool = None) -> np.ndarray:
        """-> (21,) action：有界反应轮闭环——bang-bang 抡摆
        律：前倒向后摆、后倒向前摆，其余一切钉站姿（含踝）。

        base = 非臂通道基准（None=站姿快照；非站姿起步传
        起步姿的 corresponding_action）。

        ξ = com + com_vel·τ 对 a_mid 的偏移 = 倾倒强度 ⇒
        d = clip(err·K, ±BANG_TAU) 比例律——小误差小摆
        （抑制 bang-bang 满冲量在近平衡处的超调振荡），
        大误差自然饱和到满弧。物理即"前倒向后摆、后倒
        向前摆"：臂摆到位的姿势恰是反向刹车的起始位，
        自充电。sim 验证：v0=0 站姿偏置下 5s 站稳、com
        收敛 ~0；v0≥0.05 扰动仍超出臂容量。轮模式肘
        趋伸直（lh_scale>1 ⇒ 惯量 ~scale²）。非双支撑
        NotImplementedError。
        """
        from .support_state import (ANKLE_Z_PLANTED, LIP_G,
                                    SupportClass)
        st = self.to_support_state()
        if st.support_class() != SupportClass.DOUBLE:
            raise NotImplementedError(
                "action_for_balance_with_wheel_policy: "
                f"{st.support_class().name} 未实现——v1 只做双脚")
        l = float(st.com[2]) - ANKLE_Z_PLANTED
        if l <= 0:
            raise ValueError(
                "action_for_balance_with_wheel_policy: "
                f"com.z {st.com[2]:.3f} ≤ 踝心——LIP 失效")
        p0, p1 = np.asarray(st.left_foot), np.asarray(st.right_foot)
        e = p1 - p0
        if np.linalg.norm(e) > 1e-3:
            e = e / np.linalg.norm(e)
        else:
            cy, sy = np.cos(st.torso_yaw), np.sin(st.torso_yaw)
            e = np.array([-sy, cy])
        e_perp = np.array([-e[1], e[0]])
        a_mid = 0.5 * (p0 + p1)
        tau = np.sqrt(l / LIP_G)
        xi = st.com[:2] + st.com_vel[:2] * tau
        err = float((xi - a_mid) @ e_perp)
        d = float(np.clip(err * _WHEEL_GAIN_ERR,
                          -_WHEEL_BANG_TAU, _WHEEL_BANG_TAU))
        return self.arm_as_reaction_wheel_policy(
            d, base=base, straight=straight)

    def arm_swing_policy(self, delta_x: float,
                         base: np.ndarray = None) -> np.ndarray:
        """-> (21,) action：双臂同步前后摆原语——矢状面搬移 com。

        只双脚支撑可用（support_class()!=DOUBLE ->
        NotImplementedError）。delta_x = 期望 com 沿铰链垂向
        e_perp（垂直双踝连线、水平面内）的搬移量 m——正值
        朝 +e_perp 搬。

        摆动语义（v2）：手臂保持**全伸直杆**（肘→−100°，
        |v|=0.6214m 最大伸展），肩→手向量在含 e_perp 的
        竖直面内沿半径 L 的圆周摆动，θ 从竖直下垂量起
        （θ>0 前摆、θ<0 后摆）——经最低点转向前后，两臂
        同相对称。直臂语义三收益：伸展半径最大（com 搬移
        域）、下垂段重力助转（肩不供矩→身体不吃反踢）、
        臂惯量最大（动量缓冲容量）。可达摆角不对称：后向
        ~−0.5rad（手向后 ~0.30m）、前向 +1.2rad+。肘支
        选按 vio+|elb−伸直限位|+弹簧能排序。θ -> com_e
        FK 求值二分反解；dx 截到有效域端点。

        输出 = _base_action（站姿快照，**含踝**）+ 肩肘 6 通
        道 IK 目标——只挥手臂、其余钉 base（None=站姿快
        照；非站姿起步传起步姿的 corresponding_action，
        语义同 action_for_balance_with_ankle_policy.base）。
        搬移是准静态通道：直接改写 com 相对支点的水平位
        置；速度只能经闭环侧 ξ 超前项间接压制（动量效应
        交给动力学真值——摆臂途中的反踢/制动踢 sim 可见）。

        实测能力边界（sim oracle，2026-05）：固定任何摆角
        都挂不住站姿——双踝 PD 被动刚度 ~200 N·m/rad <
        重力反刚度 ~340 N·m/rad，平衡点是鞍点；闭环搬移
        方向正确（后摆→后倒、前摆→前倒）但速率 ~mm/s
        追不上指数加速的倾翻。臂摆只能做踝/躯干通道的
        增强，不能单独站桩。
        """
        from .mechanics import _HALF, _JIDX, _MID
        from .support_state import SupportClass
        dx = float(np.asarray(delta_x).ravel()[0])
        if not np.isfinite(dx):
            raise ValueError(
                f"arm_swing_policy: delta_x 期望有限标量 m——{dx}")
        g = self.to_support_geometry()
        if g.support_class() != SupportClass.DOUBLE:
            raise NotImplementedError(
                f"arm_swing_policy: {g.support_class().name} 未实现"
                "——v1 只做双脚")
        p0, p1 = np.asarray(g.left_foot), np.asarray(g.right_foot)
        e = p1 - p0
        if np.linalg.norm(e) > 1e-3:
            e = e / np.linalg.norm(e)
        else:                                   # 双踝重合退化——取躯干侧向
            cy, sy = np.cos(g.torso_yaw), np.sin(g.torso_yaw)
            e = np.array([-sy, cy])
        e_perp = np.array([-e[1], e[0]])
        e3 = np.array([e[0], e[1], 0.0])
        ep3 = np.array([-e[1], e[0], 0.0])
        from .mechanics import Skeleton
        sk = Skeleton()
        com, *_ = sk.support_jac(
            self.root_position,
            _zyx_to_quat_wxyz(self.root_orientation), self.qpos,
            ("ankle_y_right", "ankle_y_left"))
        x_now = float(com[:2] @ e_perp)                # 当前 com 基线
        lo, hi = (self._swing_com_e(-_SWING_PHI_MAX, e3, ep3, e_perp,
                                    sk, True) - x_now,
                  self._swing_com_e(_SWING_PHI_MAX, e3, ep3, e_perp,
                                    sk, True) - x_now)
        tgt = float(np.clip(dx, lo, hi))
        a, b = -_SWING_PHI_MAX, _SWING_PHI_MAX
        for _ in range(40):                            # com_e 对 θ 单调
            m = 0.5 * (a + b)
            if self._swing_com_e(m, e3, ep3, e_perp, sk,
                                 True) - x_now < tgt:
                a = m
            else:
                b = m
        q = self._arm_swing_qpos(0.5 * (a + b), e3, ep3, sk,
                                 straight=True)
        action = (self._base_action() if base is None
                  else np.asarray(base, dtype=float).copy())
        for i in _ARM_J:
            action[i] = float(np.clip(
                (q[i] - _MID[i]) / _HALF[i], -1.0, 1.0))
        return action

    def _swing_com_e(self, theta, e3, ep3, e_perp, sk,
                     straight=False) -> float:
        """com 沿 e_perp 的 FK 求值——θ->com_e 单调映射供二分。"""
        q = self._arm_swing_qpos(theta, e3, ep3, sk,
                                 straight=straight)
        com, *_ = sk.support_jac(
            self.root_position,
            _zyx_to_quat_wxyz(self.root_orientation), q,
            ("ankle_y_right", "ankle_y_left"))
        return float(com[:2] @ e_perp)

    def _swing_arm_ik(self, theta, e3, ep3, sk, lh_scale=1.0,
                      straight=False):
        """θ 摆角下各臂 IK 最优候选 {side: pick|None}——公共
        求解层，_arm_swing_qpos 与可行性检查共用。

        lh_scale 缩放"肩→手"矢状模长（>1 肘趋伸直、惯量
        ~scale² 增大——反应轮模式用，搬移模式保持 1 保
        持当前肘角）。

        straight=True：直臂圆摆（v2 语义）——|v| 直接取
        最大伸展 _ARM_LEN（肘被 |d| 反推到全伸限位）、侧
        向分量归零（手在过肩锚的矢状面内）；候选按
        (vio, |elb−全伸|, 弹簧能) 排序锁定全伸支。"""
        from .mechanics import _ARM, _arm_ik
        xf = sk.body_transforms(
            self.root_position,
            _zyx_to_quat_wxyz(self.root_orientation), self.qpos)
        t_pos, t_mat = xf["torso"]
        ez3 = np.array([0.0, 0.0, 1.0])
        picks = {}
        for side in ("left", "right"):
            sp = _ARM[side]
            v = (xf[f"hand_{side}"][0]
                 - (t_pos + t_mat @ sp["s_off"]))       # 肩→手 世界向量
            v_p, v_e, v_z = (float(v @ ep3), float(v @ e3),
                             float(v @ ez3))
            if straight:
                lh, v_e = _ARM_LEN, 0.0
            else:
                lh = float(np.hypot(v_p, v_z)) * lh_scale
            vt = (lh * np.sin(theta) * ep3 + v_e * e3
                  - lh * np.cos(theta) * ez3)
            cand = _arm_ik(vt, t_mat, sp)
            key = ((lambda c: (c["vio"],
                               abs(c["rad"][2] - _ELB_STRAIGHT),
                               c["e"]))
                   if straight else
                   (lambda c: (c["vio"], c["e"])))
            picks[side] = min(cand, key=key) if cand else None
        return picks

    def _swing_theta_feasible(self, theta, e3, ep3, sk,
                              lh_scale=1.0, straight=False) -> bool:
        """θ 摆角双臂 IK 均有解（vio<1e-3）则可达。"""
        picks = self._swing_arm_ik(theta, e3, ep3, sk, lh_scale,
                                   straight)
        return all(p is not None and p["vio"] < 1e-3
                   for p in picks.values())

    def _arm_swing_qpos(self, theta, e3, ep3, sk, lh_scale=1.0,
                        straight=False):
        """摆角 theta 下替换肩肘 6 通道后的 qpos(21)。

        摆动 = "肩→手"向量在 {ê_perp, ê, ẑ} 系的矢状-竖直面
        内摆：v(θ) = L_h·sinθ·ê_perp + v_e·ê − L_h·cosθ·ẑ，
        L_h = |v 的 (e_perp,z) 分量|×lh_scale（straight 时
        =_ARM_LEN 且 v_e=0——直臂圆周），θ=0 竖直下垂、
        θ>0 前摆。θ 从下垂量起——com_e(θ) 单调（绕轴转整
        圈的参数化会抡过头顶折回、非单调，已弃）。
        """
        from .mechanics import _ARM
        picks = self._swing_arm_ik(theta, e3, ep3, sk, lh_scale,
                                   straight)
        q = self.qpos.copy()
        for side, sp in _ARM.items():
            pick = picks[side]
            if pick is not None:
                rad = np.clip(pick["rad"], sp["rng"][:, 0],
                              sp["rng"][:, 1])
                for i, val in zip(sp["ji"], rad):
                    q[i] = float(val)
        return q

    def action_for_balance_with_arm_policy(
            self, base: np.ndarray = None) -> np.ndarray:
        """-> (21,) action：双臂摆动闭环——搬移 com 维持在两踝
        支点上方（不用踝策略）。

        base = 非臂通道基准（None=站姿快照；非站姿起步传
        起步姿的 corresponding_action——如 ready 上平衡）。

        铰链模只调可控自由度 e_perp：搬移指令
            Δx = clip((a_mid − ξ)·e_perp, ±0.03)
            ξ = com + com_vel·τ（捕获点，τ=√(l/g)）
        ξ 偏前则手臂后摆、把 com 往支点拽——静态语义"com 对
        准 a_mid"；速度经 ξ 超前项折算成更深的搬移。能力边
        界：双臂 ~±3cm 搬移域，只能接小扰动——躯干/髋载重
        分摊通道待补。非双支撑 NotImplementedError。
        """
        from .support_state import (ANKLE_Z_PLANTED, LIP_G,
                                    SupportClass)
        st = self.to_support_state()
        if st.support_class() != SupportClass.DOUBLE:
            raise NotImplementedError(
                "action_for_balance_with_arm_policy: "
                f"{st.support_class().name} 未实现——v1 只做双脚")
        l = float(st.com[2]) - ANKLE_Z_PLANTED
        if l <= 0:
            raise ValueError(
                "action_for_balance_with_arm_policy: "
                f"com.z {st.com[2]:.3f} ≤ 踝心——LIP 失效")
        p0, p1 = np.asarray(st.left_foot), np.asarray(st.right_foot)
        e = p1 - p0
        if np.linalg.norm(e) > 1e-3:
            e = e / np.linalg.norm(e)
        else:
            cy, sy = np.cos(st.torso_yaw), np.sin(st.torso_yaw)
            e = np.array([-sy, cy])
        e_perp = np.array([-e[1], e[0]])
        a_mid = 0.5 * (p0 + p1)
        tau = np.sqrt(l / LIP_G)
        xi = st.com[:2] + st.com_vel[:2] * tau
        dx = float(np.clip(float((a_mid - xi) @ e_perp),
                           -0.03, 0.03))
        return self.arm_swing_policy(dx, base=base)



# ---------------- 标准姿态烘焙常量 ----------------
# ready 运动准备姿：P5 意图（踝 ±0.15 m 张宽、双手收至腹部
# 高 [0.28,±0.18,1.05]、头降 0.03 m）经 Pose.infer_from_p5 反解，
# root_z 校准至踝心贴地后烘焙。前倾 ~14° 是反解 v1 已知
# bias，对"准备"语义无害（渲染验证过：腿张膝屈、手臂
# 腹部屈收）。v2：绕踝轴前倾 +2.7°，com 落在踝心前
# +0.02m（原姿态 com 在踝心后 ~1.3cm）；v3：踝牛顿解
# 把脚底解到完全水平贴地（z_axis=[0,0,1]，保留外八
# ~16°），脚原点 z=0.027 与 standing 一致。
_READY_QPOS = np.array([
    0.0, -0.3598, 0.0,
    -0.1000, -0.3027, -0.1709, -0.4474, -0.1802, 0.0828,
    -0.1000, -0.3027, -0.1709, -0.4474, -0.1802, 0.0828,
    0.3608, -0.2878, -0.1958,
    0.3608, -0.2878, -0.1958])
_READY_ROOT_POS = np.array([0.0083, 0.0, 1.2504])
_READY_ROOT_ORI = np.array([0.2890, 0.0, 0.0])


# 踝策略执行器参数——Humanoid21Meta.KP/KD 踝列与 ankle_x
# 斜轴投影（力矩上限 τ_act=20 已在 support_state.LIP_TAU_ACT）。
_ANKLE_KP = 100.0     # N·m/rad — 踝 PD 位置增益
_ANKLE_KD = 10.0      # N·m·s/rad — 踝 PD 阻尼
_ANKLE_AX = 0.8944    # ankle_x 轴世界 x 分量（(±.8944,0,±.4472)）

# 臂摆策略：肩肘 6 通道索引与摆角上限
from .mechanics import _JIDX as _JIDX_
_ARM_J = frozenset(_JIDX_[n] for n in (
    "shoulder1_right", "shoulder2_right", "elbow_right",
    "shoulder1_left", "shoulder2_left", "elbow_left"))
del _JIDX_
_SWING_PHI_MAX = 1.2  # rad — 双臂摆角搜索界（±~69°）
_ARM_LEN = 0.6214     # m — 直臂最大伸展 |u+R·w|（elb=-100°）
_ELB_STRAIGHT = -1.7453  # rad — 肘全伸限位（IK 选支参照）
_WHEEL_KP = 150.0     # N·m/rad — 肩 PD 位置增益（lead↔矩）
_WHEEL_GAIN = 3.0     # — lead→平均力矩标定增益（补控制窗
                      # 内臂行程吃掉的误差；oracle 标定）
_WHEEL_LEAD = 2.6     # rad — 位置目标超前角上限（≈矢状可
                      # 达弧全长；大指令经此截断+IK 二分落锥界）
_WHEEL_BANG_TAU = 400.0  # N·m — 力矩指令饱和值（对应满弧
                       # 抡摆；lead 超界由可达二分限幅）
_WHEEL_GAIN_ERR = 8000.0  # N·m/m — ξ 误差→指令增益：比例
                          # 律实测 8000/12000/16000 均 5s 站
                          # 稳（取最平滑档）；bang-bang 版
                          # 近平衡超调 ~3.4s 死，已弃
_WHEEL_LH_SCALE = 1.0    # — 轮模式臂矢状模长倍数。1.45 伸
                         # 直臂实测反而更快倒（伸展臂前悬的
                         # 静态前偏加重 + IK 构型变化），回
                         # 退 1.0；惯量增益的代价不划算
_WHEEL_STRAIGHT = True   # — 轮模式直臂圆摆（v2 语义：肘锁
                         # 全伸、侧向归零、经下方转向）。覆盖
                         # lh_scale；standing 实证需复测——
                         # 此前 lh_scale=1.45 偏置更快倒



