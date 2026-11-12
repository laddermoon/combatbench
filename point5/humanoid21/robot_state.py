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

        双腿张开、膝微屈、手臂胸前屈收；P5 意图经
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
        
    # ---------- action_for_*：take_action 的逆向查询族 ----------

    def action_for_balance(self, dt: float = 0.05,
                           w_hold: float = 0.01,
                           max_evals: int = 100,
                           eps: float = 0.02,
                           band: float = 1.0,
                           horizon: int = 1) -> np.ndarray:
        """-> (21,) 归一化 action：无点目标，只求下一步留在
        平衡核内（"站桩"档）——最优化问题 v1：

            决策变量   a ∈ [-1,1]²¹（归一化关节目标）
            oracle    s'(a) = take_action(s, a)（重复 horizon
                      次 = 开环保持该 action 至 horizon·dt），
                      ξ' 取末端 support_state
            目标       J = max(0, M̃(s') + band)²
                       + w_hold·‖a−a_hold‖²
                       M̃ = balance_margin / reach_ref（无量纲，
                       reach_ref = τ_act/(mg) ≈ 0.05m）；
                       band = 深度带——目标不是贴边入核而是
                       margin ≤ −band（核内浅层即启动纠正，
                       纯 hinge 会让核内深度无差别、贴边滑出）；
                       a_hold = rad_to_norm(qpos)（不动先验）
            求解       L-BFGS-B 盒界 + 有限差分梯度（eps=0.02
                       action 单位），黑盒 ~22 evals/grad，
                       maxfun 封顶

        边界与限制：margin 按末端态贴地脚评估（摆动期以实际
        stance 计）；horizon=1 时 50ms 单步只见踝/姿态策略——
        迈步恢复的 payoff 在落地才兑现，不属本档（走
        action_for_support_state）；horizon>1 评估的是
        "保持该 action 的开环推演"，比闭环保守但能看到
        发散/收敛趋势。oracle 内部异常按 +inf 罚回推。
        """
        from scipy.optimize import minimize
        from .mechanics import rad_to_norm
        from .support_state import LIP_G, LIP_MASS, LIP_TAU_ACT
        a_hold = np.clip(rad_to_norm(self.qpos), -1.0, 1.0)
        ref = LIP_TAU_ACT / (LIP_MASS * LIP_G)

        def margin_of(a):
            s2 = self
            for _ in range(int(horizon)):
                s2 = s2.take_action(a, dt=dt)
            return s2.to_support_state().balance_margin() / ref

        # 快路径：a_hold 推演后 margin ≤ −band → J=0 已是全局
        # 下界（两项非负且同时为 0），深度安全的核内态直接返回。
        try:
            if margin_of(a_hold) <= -band:
                return a_hold
        except Exception:
            pass

        def J(a):
            try:
                m = margin_of(a)
            except Exception:
                return 1e6
            if not np.isfinite(m):    # 腾空/LIP 失效——
                return 1e6            # 大罚而非 inf²（会污染 FD 梯度）
            return max(0.0, m + band) ** 2 \
                + w_hold * float(np.sum((a - a_hold) ** 2))

        res = minimize(J, a_hold, method="L-BFGS-B",
                       bounds=[(-1.0, 1.0)] * 21,
                       options={"maxfun": int(max_evals),
                                "eps": float(eps)})
        return np.clip(np.asarray(res.x, np.float64), -1.0, 1.0)

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


# ---------------- 标准姿态烘焙常量 ----------------
# ready 运动准备姿：P5 意图（踝 ±0.15 m 张宽、双手抬至胸前
# [0.28,±0.18,1.32]、头降 0.03 m）经 Pose.infer_from_p5 反解，
# root_z 校准至踝心贴地后烘焙。前倾 ~14° 是反解 v1 已知
# bias，对"准备"语义无害（渲染验证过：腿张膝屈、手臂
# 胸前屈收）。
_READY_QPOS = np.array([
    0.0, -0.3598, 0.0,
    -0.1000, -0.3027, -0.1709, -0.4474, 0.0, 0.0,
    -0.1000, -0.3027, -0.1709, -0.4474, 0.0, 0.0,
    -0.4494, 0.0296, 0.1852,
    -0.4494, 0.0296, 0.1852])
_READY_ROOT_POS = np.array([-0.0455, 0.0, 1.2500])
_READY_ROOT_ORI = np.array([0.2419, 0.0, 0.0])



