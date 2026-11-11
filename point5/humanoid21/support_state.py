"""SupportGeometry / SupportState — 支撑语义类型。

两种粒度，均**帧无关**——与坐标系有关的量可表达于任何
**z=0 为地面、+z 竖直向上的右手系**（leveled 帧族），
所在帧由持有方约定：

    SupportGeometry  静态几何：com + 双脚位/足高 + 躯干朝向
    SupportState     动态支撑态 = 几何 + 动量

来源决定内容：obs 解码态带实测动量；规划态给设计值；
P5 反解无动量——只能构造 SupportGeometry。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

_COM = (3,)
_FOOT = (2,)
_VEL = (3,)
_MOM = (3,)

# 平脚贴地时踝关节中心的基准高度（模型形态常数，实测 0.107）
ANKLE_Z_PLANTED = 0.107

# ---- lip_analyze 常量（踝摆能力）----
# 来源：ankle_pendulum/REPORT.md + battle_circular_v2.xml 脚 geom
# 参数；模型改动需同步。
LIP_MASS = 40.84            # kg — 全身质量（body_mass.sum）
LIP_G = 9.81
LIP_TAU_ACT = 20.0          # N·m — 踝执行器力矩限（gear=20 直驱，单轴）
LIP_MARGIN_TOE = 0.167      # m — 踝心到脚尖侧支撑域前缘
LIP_MARGIN_HEEL = 0.097     # m — 踝心到脚跟侧支撑域后缘
LIP_MARGIN_SIDE = 0.047     # m — 踝心到侧向支撑域边缘
LIP_MU_SLIDE = 1.0          # 脚-地滑动摩擦（脚 geom friction[0]）
LIP_MU_TORSION = 0.02       # m — 扭转摩擦系数（friction[1]），τ_yaw≤μ_t·N
# 另一脚竖直推力上限：全脚支撑按全重；离地脚按前掌着地满踝矩
# 的杠杆对地压力 τ_act/margin_toe ≈ 120N（实现里分段取）。
LIP_F_PUSH_PLANTED = LIP_MASS * LIP_G


def _wrap(a: float) -> float:
    """角度归一到 (-pi, pi]。"""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _vec(v, shape, name):
    v = np.array(v, np.float64)          # copy, never alias
    if v.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {v.shape}")
    if not np.isfinite(v).all():
        raise ValueError(f"{name}: non-finite values")
    return v


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """平面点集凸包顶点 CCW（monotone chain）。"""
    pts = np.unique(np.asarray(pts, np.float64), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


@dataclass
class SupportGeometry:
    """静态几何锚点 — leveled 帧族，所在帧由持有方约定
    （SupportPlan 统一首帧 P5 系）。足高记发力潜力——贴地才有
    推力权威，不记实测分力。

    脚位/足高的"踝"一律指**踝关节中心**（脚绕其旋转的关节
    锚点，脚踝铰链运动下不动，随小腿刚体走）。foot_height
    = 踝心实际高度 − 平脚贴地基准踝心高，≈0 表示落地。
    """
    com: np.ndarray                    # (3,) 质心
    left_foot: np.ndarray              # (2,) 踝关节中心 xy
    right_foot: np.ndarray             # (2,) 踝关节中心 xy
    torso_yaw: float                   # 躯干朝向 rad
    left_foot_height: float = 0.0      # 左踝心抬升高度（≈0 落地）
    right_foot_height: float = 0.0     # 右踝心抬升高度

    def __post_init__(self):
        self.com = _vec(self.com, _COM, "geometry.com")
        self.left_foot = _vec(self.left_foot, _FOOT,
                              "geometry.left_foot")
        self.right_foot = _vec(self.right_foot, _FOOT,
                               "geometry.right_foot")
        for n in ("torso_yaw", "left_foot_height", "right_foot_height"):
            if not np.isfinite(getattr(self, n)):
                raise ValueError(f"geometry.{n}: non-finite")
        self.torso_yaw = float(self.torso_yaw)
        self.left_foot_height = float(self.left_foot_height)
        self.right_foot_height = float(self.right_foot_height)

    @classmethod
    def from_pose(cls, pose) -> "SupportGeometry":
        """Pose -> 几何锚点（薄委托：提取实现住 Pose.to_support_geometry）。"""
        return pose.to_support_geometry()

    def _se2(self, dyaw: float, dxy) -> "SupportGeometry":
        """SE(2) 平面变换：点乘 Rz+平移，yaw += dyaw，足高不变。"""
        c, s = np.cos(dyaw), np.sin(dyaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        R2 = np.array([[c, -s], [s, c]])
        return SupportGeometry(
            com=R @ self.com + np.array([dxy[0], dxy[1], 0.0]),
            left_foot=R2 @ self.left_foot + np.asarray(dxy, float),
            right_foot=R2 @ self.right_foot + np.asarray(dxy, float),
            torso_yaw=_wrap(self.torso_yaw + dyaw),
            left_foot_height=self.left_foot_height,
            right_foot_height=self.right_foot_height)

    def standardize(self) -> "SupportGeometry":
        """转成自身 P5 系（踝中点→原点、torso_yaw→0）。"""
        mid = (self.left_foot + self.right_foot) / 2.0
        yaw = float(self.torso_yaw)
        c, s = np.cos(-yaw), np.sin(-yaw)
        t = -np.array([[c, -s], [s, c]]) @ mid
        return self._se2(-yaw, t)

    def reanchor(self, xy=(0.0, 0.0),
                 yaw: float = 0.0) -> "SupportGeometry":
        """重新锚定到指定位置/朝向（standardize 的逆操作）。

        输出保证踝中点在 xy、躯干朝向 yaw，几何形状不变——
        内部先 standardize 再做 SE(2) 平面变换。
        """
        return self.standardize()._se2(float(yaw), np.asarray(xy, float))

    def support_polygon(self, foot_height_eps: float = 0.02
                        ) -> np.ndarray:
        """支撑多边形 (N,2) 顶点 CCW——贴地脚支撑域的凸包。

        每贴地脚（foot_height < eps）贡献脚板支撑域四角
        （踝心局部系 x∈[−heel,+toe] 前后、y∈±side，脚近似与
        躯干朝向对齐）；双脚时取八点凸包。无贴地脚返回 (0,2)。
        """
        c, s = np.cos(self.torso_yaw), np.sin(self.torso_yaw)
        R = np.array([[c, -s], [s, c]])
        corners = np.array([
            [LIP_MARGIN_TOE, LIP_MARGIN_SIDE],
            [LIP_MARGIN_TOE, -LIP_MARGIN_SIDE],
            [-LIP_MARGIN_HEEL, -LIP_MARGIN_SIDE],
            [-LIP_MARGIN_HEEL, LIP_MARGIN_SIDE]]) @ R.T
        pts = [np.asarray(fxy) + corners
               for fxy, fh in ((self.left_foot, self.left_foot_height),
                               (self.right_foot, self.right_foot_height))
               if fh < foot_height_eps]
        return _convex_hull(np.vstack(pts)) if pts else np.empty((0, 2))

    def lip_analyze(self, foot_height_eps: float = 0.02,
                    inertia_com: np.ndarray = None
                    ) -> Tuple["PivotAnalysis", ...]:
        """倒立摆支点分析——每个 foot_height < eps 的脚一份。

        每支点均按**此脚单支撑**分析：支点 = 踝关节中心，
        N = mg；重力项假定另一脚未支撑。三方向分解（轴按
        躯干朝向取，踝铰实际轴在脚系——脚近似与朝向对齐）：

          前后摆    绕侧向轴，踝 τ∈[−min(τ_act,N·margin_heel),
                   +min(τ_act,N·margin_toe)]
          侧向摆    绕朝向轴，踝 τ∈±min(τ_act,N·margin_side)
          绕轴转    绕竖直轴，地面扭转摩擦 τ∈±μ_t·N

        每方向分列三来源：支点踝力矩角加速度区间 / 重力项
        定值 / 另一脚可给区间。另一脚按其当前位置、假定
        可触地计——全脚支撑时竖直推力上限 mg；有离地高度
        时按前掌着地满踝矩对地压力 τ_act/margin_toe，触点
        取踝心 + margin_toe·朝向；水平摩擦 |F_xy|≤μ·F_z。

        inertia_com：关于质心的全身惯量 (3,3) 可选——参考级
        即可（Geometry→P5→Pose→inertia_matrix 链），缺省退化
        为质点惯量 m·d⊥²。
        """
        feet = (("left", np.asarray(self.left_foot),
                 self.left_foot_height),
                ("right", np.asarray(self.right_foot),
                 self.right_foot_height))
        cy, sy = np.cos(self.torso_yaw), np.sin(self.torso_yaw)
        fwd = np.array([cy, sy])
        lat = np.array([-sy, cy])
        N = LIP_MASS * LIP_G
        mg = np.array([0.0, 0.0, -N])
        out = []
        for name, fxy, fh in feet:
            if fh >= foot_height_eps:
                continue
            pivot = np.array([fxy[0], fxy[1],
                              ANKLE_Z_PLANTED + fh])
            d = self.com - pivot
            # 另一脚：全脚支撑→推力上限 mg、触点踝心；
            # 离地→前掌满踝矩压力、触点踝心+margin_toe·朝向
            oname = "right" if name == "left" else "left"
            oxy = np.asarray(
                self.right_foot if name == "left"
                else self.left_foot)
            oh = (self.right_foot_height if name == "left"
                  else self.left_foot_height)
            if oh < foot_height_eps:
                f_max, app_xy = LIP_F_PUSH_PLANTED, oxy
            else:
                f_max = LIP_TAU_ACT / LIP_MARGIN_TOE
                app_xy = oxy + LIP_MARGIN_TOE * fwd
            rxy = app_xy - pivot[:2]
            rz = -pivot[2]                     # 触点在地面

            axes = {
                "fore_aft": np.array([lat[0], lat[1], 0.0]),
                "lateral": np.array([fwd[0], fwd[1], 0.0]),
                "yaw": np.array([0.0, 0.0, 1.0]),
            }
            t_ankle = {
                "fore_aft": (-min(LIP_TAU_ACT, N * LIP_MARGIN_HEEL),
                             min(LIP_TAU_ACT, N * LIP_MARGIN_TOE)),
                "lateral": (-min(LIP_TAU_ACT, N * LIP_MARGIN_SIDE),
                            min(LIP_TAU_ACT, N * LIP_MARGIN_SIDE)),
                "yaw": (-LIP_MU_TORSION * N, LIP_MU_TORSION * N),
            }
            dirs = {}
            for dname, e in axes.items():
                I = LIP_MASS * (d @ d - (d @ e) ** 2)
                if inertia_com is not None:
                    I += e @ np.asarray(inertia_com) @ e
                lo, hi = t_ankle[dname]
                g_alpha = float(e @ np.cross(d, mg) / I)
                tz = f_max * (e[0] * rxy[1] - e[1] * rxy[0])
                if dname == "yaw":          # F_z 无 yaw 分量
                    txy = LIP_MU_SLIDE * f_max * np.hypot(*rxy)
                    of_rng = (-txy / I, txy / I)
                else:
                    txy = LIP_MU_SLIDE * f_max * abs(rz)
                    of_rng = ((min(tz, 0.0) - txy) / I,
                              (max(tz, 0.0) + txy) / I)
                dirs[dname] = DirectionAlpha(
                    ankle=(lo / I, hi / I), gravity=g_alpha,
                    other_foot=of_rng)
            out.append(PivotAnalysis(foot=name, pivot=pivot, **dirs))
        return tuple(out)

    def reference_pose(self, tolerance: float = 0.02,
                       **solver_kw) -> "Pose":
        """SupportGeometry -> 参考 Pose（单阶段反解，非精确还原）。

        reference_pose_for_support_geometry.md：标准帧 23 变量
        （pitch/roll + 全 21 关节）对 6 硬约束（踝心−com 相对
        向量），目标 max|头−com| + 弹性能 + 臂强先验；root 平移
        由 com 恢复，yaw/锚点经 _se2 反变换还原。
        阶段一几何不可达（多初值后残差仍超 tolerance）时 ValueError。

        solver_kw 透传 solve_support_pose（w_head / w_arm /
        arm_ref / max_evals / restarts 等）。
        """
        from .mechanics import solve_support_pose
        from .robot_state import Pose
        std = self.standardize()
        a_l = np.array([std.left_foot[0], std.left_foot[1],
                        ANKLE_Z_PLANTED + std.left_foot_height])
        a_r = np.array([std.right_foot[0], std.right_foot[1],
                        ANKLE_Z_PLANTED + std.right_foot_height])
        res = solve_support_pose(std.com, a_r, a_l,
                                 tolerance=tolerance, **solver_kw)
        if not res["satisfied"]:
            raise ValueError(
                "reference_pose: 几何不可达 "
                f"(max_resid={res['stage1']['max_resid']:.4f} m)")
        pose_std = Pose(qpos=res["q_rad"],
                        root_position=res["root_pos"],
                        root_orientation=np.array(
                            [res["pitch"], res["roll"], 0.0]))
        mid_xy = (self.left_foot + self.right_foot) / 2.0
        return pose_std._se2(self.torso_yaw, mid_xy)


@dataclass
class DirectionAlpha:
    """单方向角加速度分解 rad/s²——三来源分列，不代数和。

    消费方一般只读 total；分量留给诊断/预算——注意 ankle
    是可控项、gravity 是不可控负载、other_foot 是反事实
    项（用了它支撑构型即改变）。
    """
    ankle: Tuple[float, float]        # 支点踝力矩可给区间
    gravity: float                    # 重力项（定值）
    other_foot: Tuple[float, float]   # 另一脚推力+摩擦可给区间

    @property
    def total(self) -> Tuple[float, float]:
        """三来源合并的可行角加速度区间（Minkowski 和）。"""
        return (self.ankle[0] + self.other_foot[0] + self.gravity,
                self.ankle[1] + self.other_foot[1] + self.gravity)


@dataclass
class PivotAnalysis:
    """单支点倒立摆分析——支点 = 踝关节中心。"""
    foot: str                         # "left" / "right"
    pivot: np.ndarray                 # (3,) 踝心
    fore_aft: DirectionAlpha          # 朝向方向前后摆（绕侧向轴）
    lateral: DirectionAlpha           # 侧向摆（绕朝向轴）
    yaw: DirectionAlpha               # 绕竖直轴（摩擦驱动）


@dataclass
class SupportState(SupportGeometry):
    """动态支撑态 = 几何 + 动量 — 帧无关。

    com_vel/ang_mom 必带——动量缺席的场合用 SupportGeometry，
    不构造残缺的 SupportState。
    """
    com_vel: np.ndarray = field(kw_only=True)  # (3,) 质心速度
    ang_mom: np.ndarray = field(kw_only=True)  # (3,) 全身角动量，关于质心——换帧平移不变，仅随 yaw 旋转分量

    def __post_init__(self):
        super().__post_init__()
        self.com_vel = _vec(self.com_vel, _VEL, "com_vel")
        self.ang_mom = _vec(self.ang_mom, _MOM, "ang_mom")

    @classmethod
    def from_robot_state(cls, robot_state) -> "SupportState":
        """RobotState -> SupportState（薄委托：提取实现住
        RobotState.to_support_state）。"""
        return robot_state.to_support_state()

    @classmethod
    def from_pose(cls, pose):
        """禁用——Pose 无动态字段，构造不出 SupportState；
        走 from_robot_state。"""
        raise TypeError(
            "SupportState.from_pose 不能提供 com_vel/ang_mom；"
            "用 SupportState.from_robot_state(robot_state)")

    def _se2(self, dyaw: float, dxy) -> "SupportState":
        """SE(2) 变换：几何走父类，com_vel/ang_mom 乘 Rz(dyaw)。"""
        g = super()._se2(dyaw, dxy)
        c, s = np.cos(dyaw), np.sin(dyaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return SupportState(
            com=g.com, left_foot=g.left_foot, right_foot=g.right_foot,
            torso_yaw=g.torso_yaw,
            left_foot_height=g.left_foot_height,
            right_foot_height=g.right_foot_height,
            com_vel=R @ self.com_vel, ang_mom=R @ self.ang_mom)

    def standardize(self) -> "SupportState":
        """几何归一 + 动量随帧旋转——保持动态字段。"""
        return super().standardize()

    def reanchor(self, xy=(0.0, 0.0),
                 yaw: float = 0.0) -> "SupportState":
        """同 standardize——帧变换时 com_vel/ang_mom 随 yaw 旋转。"""
        return super().reanchor(xy, yaw)

    def geometry(self) -> SupportGeometry:
        """获取静态几何锚点（剥掉动量）。"""
        return SupportGeometry(
            com=self.com.copy(),
            left_foot=self.left_foot.copy(),
            right_foot=self.right_foot.copy(),
            left_foot_height=self.left_foot_height,
            right_foot_height=self.right_foot_height,
            torso_yaw=self.torso_yaw)

    def _cp_terms(self, fxy, fh):
        """该支点的 CP 公共项 (ξ_xy, τ=√(l/g), 踝心高)。"""
        pz = ANKLE_Z_PLANTED + fh                   # 支点踝心高
        l = self.com[2] - pz
        if l <= 0:
            raise ValueError(
                f"capture_point: com.z {self.com[2]:.3f}"
                f" ≤ 踝心 {pz:.3f}——LIP 模型失效")
        tau = np.sqrt(l / LIP_G)                    # 1/ω
        return self.com[:2] + self.com_vel[:2] * tau, tau, pz

    def capture_point(self, foot_height_eps: float = 0.02
                      ) -> Dict[str, np.ndarray]:
        """捕获点 {foot: (3,)}——每个支点一个（沿用 lip_analyze
        支点约定：foot_height < eps 的脚）。

        **瞬时量**：ξ = com_xy + com_vel_xy·τ——状态特征/
        失衡诊断，非落脚目标。迈步期 ξ 继续绕支点发散，
        落点预测用 capture_point_at(dt)。返回 z = 踝心高
        （与 pivot 同层）。com_vel.z 不用（LIP 恒高假定）。
        """
        out = {}
        for name, fxy, fh in (("left", self.left_foot,
                               self.left_foot_height),
                              ("right", self.right_foot,
                               self.right_foot_height)):
            if fh >= foot_height_eps:
                continue
            xi, _, pz = self._cp_terms(fxy, fh)
            out[name] = np.array([xi[0], xi[1], pz])
        return out

    def capture_point_at(self, dt: float,
                         foot_height_eps: float = 0.02
                         ) -> Dict[str, np.ndarray]:
        """dt 秒后的捕获点预测 {foot: (3,)}——迈步落点语义。

        ξ 以 ω=√(g/l) 绕当前支点 u 持续发散：ξ(t) =
        u + (ξ−u)·e^{t/τ}，u = 该支点踝心 xy。dt=0 等价
        capture_point。
        """
        if dt < 0:
            raise ValueError(f"capture_point_at: dt<0 ({dt})")
        out = {}
        for name, fxy, fh in (("left", self.left_foot,
                               self.left_foot_height),
                              ("right", self.right_foot,
                               self.right_foot_height)):
            if fh >= foot_height_eps:
                continue
            xi, tau, pz = self._cp_terms(np.asarray(fxy), fh)
            pred = np.asarray(fxy) + (xi - np.asarray(fxy)) * np.exp(dt / tau)
            out[name] = np.array([pred[0], pred[1], pz])
        return out

    def can_balance(self, foot_height_eps: float = 0.02) -> bool:
        """踝关节能力可达平衡——不挪脚，靠踝矩/载重分摊能否
        把发散的质心救回平衡支撑态。

        判据（LIP 级，0 步可捕获性）：发散坐标 ξ = com + v·τ
        满足 ξ̇ = ω(ξ − c)，c = 可控 COP 位置——ξ 超出 COP 可
        达域则 ξ̇>0 恒成立、必发散，任何踝矩都救不回。按模式：

          0 贴地脚 → False（无 COP 可控通道）；
          1 脚 pivot：单脚踝矩搬 COP 限 ±min(τ_act, N·margin)/N
             米——τ_act=20 折 ~0.050m，比鞋底几何缘还紧
             （执行器限；侧向 0.047 是鞋底限）。ξ 相对支点在
             前后（fwd）/侧向（lat）两方向分别入界才算；
          2 脚 hinge：双脚载重分摊可把 COP 放到支撑多边形内
             任意点，不占踝矩——可达域 = support_polygon()
             垂直踝轴方向的投影全长；沿轴自由度被两触点锁死，
             不查（com 沿轴偏出端点 ≠ 失衡，只是载荷分布偏）。
             摆脚悬空与否不参与——"自身努力"不含迈步。

        边界 ξ = c_max 判 True（COP 顶在边缘可持——临界平衡）。
        限制：LIP 恒高近似；com_vel.z / ang_mom 不入判据；
        com 低于踝心（LIP 失效）按 False 处理。
        """
        feet = (("left", np.asarray(self.left_foot),
                 self.left_foot_height),
                ("right", np.asarray(self.right_foot),
                 self.right_foot_height))
        planted = [(n, f, h) for n, f, h in feet
                   if h < foot_height_eps]
        if not planted:
            return False
        try:
            xi, _, _ = self._cp_terms(planted[0][1], planted[0][2])
        except ValueError:
            return False
        cy, sy = np.cos(self.torso_yaw), np.sin(self.torso_yaw)
        fwd = np.array([cy, sy])
        lat = np.array([-sy, cy])
        N = LIP_MASS * LIP_G
        if len(planted) == 1:
            # pivot：踝矩搬 COP，执行器/鞋底缘取小
            off = xi - planted[0][1]
            reach = {
                "fwd+": min(LIP_TAU_ACT, N * LIP_MARGIN_TOE) / N,
                "fwd-": min(LIP_TAU_ACT, N * LIP_MARGIN_HEEL) / N,
                "lat": min(LIP_TAU_ACT, N * LIP_MARGIN_SIDE) / N}
            o_fwd, o_lat = float(off @ fwd), float(off @ lat)
            return (-reach["fwd-"] <= o_fwd <= reach["fwd+"]
                    and abs(o_lat) <= reach["lat"])
        # hinge：载重分摊，COP 可达域 = 支撑多边形垂直轴投影
        p0, p1 = planted[0][1], planted[1][1]
        e = p1 - p0
        e = e / np.linalg.norm(e) if np.linalg.norm(e) > 1e-3 \
            else lat
        e_perp = np.array([-e[1], e[0]])
        proj = self.support_polygon(foot_height_eps) @ e_perp
        return bool(proj.size
                    and proj.min() <= xi @ e_perp <= proj.max())
        

    def drift(self, dt: float = 0.05,
              inertia_com: np.ndarray = None,
              alpha: np.ndarray = None,
              foot_height_eps: float = 0.02,
              substeps: int = 25,
              v_tol: float = 0.2) -> "DriftReport":
        """重力 + 接触约束下的支撑态演化——一个控制周期的漂移。

        刚体化近似：内部构型/惯量冻结，踝 = 无矩铰（alpha=None，
        即被动漂移）或按给定角加速度发力的可控铰。外力 = 重力
        + 接触约束 + 踝矩贡献。按贴地脚数（foot_height < eps）
        分模式：

          0 脚 → "ballistic"：纯弹道，绕 com 角动量守恒。无支点
                 踝矩无意义——alpha 若给，唯一物理解是绕 com
                 外力矩：有惯量时 H += J·alpha·dt，com 轨迹不变；
          1 脚 → "pivot"：绕踝心旋转——动量形式积分
                 Ḣ_p = d×mg + I_p·alpha_eff，ω = I_p⁻¹H_p，
                 I_p = J_com + m(|d|²I−ddᵀ)。inertia_com 缺省时
                 退化为点球摆（lip_analyze 同约定，此时 ang_mom
                 无法更新、原样保留）；
          2 脚 → "hinge"：绕双踝连线 1-DOF 铰链——两固定点只剩
                 沿轴旋转自由度，侧向锁死；φ̈ += e·alpha。

        alpha：世界系下绕支点的附加角加速度 (3,) rad/s²——踝矩
        通道的控制输入，单位与 lip_analyze 的 DirectionAlpha.
        ankle 一致。投影规则：pivot 砍掉 ẑ 分量（踝 2-DOF 铰
        无 yaw 矩通道，只有前后/侧向两轴）；hinge 只取沿轴
        分量 e·alpha。本方法**不裁剪幅度**——预算归 lip_analyze
        管，调用方自行限幅；超预算的 alpha 会经约束力体现在
        体检（min_normal / max_friction_ratio）里。

        **规则**：模式在 t=0 由贴地脚数一次定死，演化期内不切换；
        摆动脚的再触地**不被考虑**（轨迹不入模型、输出脚高原样
        保留）。作为补偿性诊断，pivot 模式顺带给出摆动脚踝心
        轨迹的最低离地间隙（swing_min_clearance < 0 ⇒ 被忽略
        的触地本会发生）——纯报告，不改动力学、不进
        assumption_ok。ballistic / hinge 下该字段为 NaN。

        旋转细节（pitch/roll）超出 SupportState 表达域，只经
        com/com_vel/ang_mom/torso_yaw 反映（yaw += ω_z·dt）。

        assumption_ok 三项体检：初速与支点约束一致（径向分量
        ≤ v_tol）、演化中约束法向力 F_z > 0（脚不被拉起）、
        |F_xy| ≤ μ_slide·F_z（不打滑；hinge 只验合力，分到
        每脚不定）。为假时"脚保持贴地"的演化结果不可信。
        """
        if dt < 0:
            raise ValueError(f"drift: dt<0 ({dt})")
        J = (None if inertia_com is None
             else _vec(inertia_com, (3, 3), "inertia_com"))
        a_ext = (None if alpha is None
                 else _vec(alpha, (3,), "alpha"))
        m = LIP_MASS
        g = np.array([0.0, 0.0, -LIP_G])
        n = max(1, int(substeps))
        h = dt / n if n else 0.0
        min_nz, max_mu, dyaw = np.inf, 0.0, 0.0

        def track(F):
            nonlocal min_nz, max_mu
            nz = F[2]
            min_nz = min(min_nz, nz)
            if nz > 1e-9:
                max_mu = max(max_mu, float(np.hypot(*F[:2])) / nz)
            else:
                max_mu = np.inf

        feet = ((np.append(self.left_foot,
                           ANKLE_Z_PLANTED + self.left_foot_height),
                 self.left_foot_height),
                (np.append(self.right_foot,
                           ANKLE_Z_PLANTED + self.right_foot_height),
                 self.right_foot_height))
        pivots = [p for p, fh in feet if fh < foot_height_eps]
        swing = [p for p, fh in feet if fh >= foot_height_eps]
        swing_min = np.inf               # 摆脚踝心轨迹最低离地间隙
        p0, v0, H0 = self.com.copy(), self.com_vel.copy(), \
            self.ang_mom.copy()

        if len(pivots) == 0:                       # ---- ballistic ----
            mode = "ballistic"
            radial = 0.0
            p_end = p0 + v0 * dt + 0.5 * g * dt * dt
            v_end = v0 + g * dt
            H_end = (H0 + J @ a_ext * dt
                     if J is not None and a_ext is not None else H0)
            min_nz = 0.0

        elif len(pivots) == 1:                     # ---- pivot ----
            mode = "pivot"
            c = pivots[0]
            d = p0 - c
            r0 = float(np.linalg.norm(d))
            a_eff = (None if a_ext is None else
                     np.array([a_ext[0], a_ext[1], 0.0]))  # 踝无 yaw 通道
            if r0 < 1e-3:                          # com 压在支点上：退化
                radial = float(np.linalg.norm(v0))
                p_end = p0 + v0 * dt + 0.5 * g * dt * dt
                v_end = v0 + g * dt
                H_end = (H0 + J @ a_ext * dt
                         if J is not None and a_ext is not None
                         else H0)
            elif J is None:                        # 点球摆
                from .mechanics import _skew
                d_hat = d / r0
                radial = abs(float(v0 @ d_hat))
                v = v0 - (v0 @ d_hat) * d_hat      # 投影到切向
                R_b = np.eye(3)
                sd = swing[0] - c if swing else None
                for _ in range(n):
                    d_hat = d / np.linalg.norm(d)
                    beta = float(g @ d_hat + (v @ v) / r0)
                    a = g - beta * d_hat
                    if a_eff is not None:
                        a = a + r0 * np.cross(a_eff, d_hat)
                    track(m * (a - g))
                    v = v + a * h
                    d = d + v * h
                    d *= r0 / np.linalg.norm(d)    # 约束投影
                    if sd is not None:             # 摆脚踝心轨迹
                        w = np.cross(d, v) / (r0 * r0)
                        R_b += _skew(w) @ R_b * h  # 近似：与 com 同角速
                        swing_min = min(
                            swing_min,
                            (c + R_b @ sd)[2] - ANKLE_Z_PLANTED)
                p_end = c + d
                v_end = v
                H_end = H0                         # 无惯量不可更新
            else:                                  # 刚体定点转动（动量式）
                from .mechanics import _skew
                d_hat = d / r0
                radial = abs(float(v0 @ d_hat))
                H_p = H0 + np.cross(d, m * v0)     # 关于支点的角动量
                omega = np.zeros(3)
                R_b = np.eye(3)
                sd = swing[0] - c if swing else None
                for _ in range(n):
                    I_p = J + m * ((d @ d) * np.eye(3) - np.outer(d, d))
                    omega = np.linalg.solve(I_p, H_p)
                    tau = np.cross(d, m * g) + (
                        I_p @ a_eff if a_eff is not None else 0.0)
                    alpha_dyn = np.linalg.solve(I_p, tau)
                    acc = (np.cross(alpha_dyn, d)
                           + np.cross(omega, np.cross(omega, d)))
                    track(m * (acc - g))
                    H_p = H_p + tau * h
                    d = d + np.cross(omega, d) * h
                    d *= r0 / np.linalg.norm(d)
                    dyaw += omega[2] * h
                    if sd is not None:
                        R_b += _skew(omega) @ R_b * h
                        swing_min = min(
                            swing_min,
                            (c + R_b @ sd)[2] - ANKLE_Z_PLANTED)
                p_end = c + d
                v_end = np.cross(omega, d)
                H_end = J @ omega                  # 刚体 H_com = J·ω

        else:                                      # ---- hinge ----
            mode = "hinge"
            e = pivots[1] - pivots[0]
            if np.linalg.norm(e) < 1e-3:
                e = np.array([0.0, 1.0, 0.0])      # 脚几乎重合：退化轴向
            else:
                e = e / np.linalg.norm(e)
            c = pivots[0]
            d_par = (p0 - c) @ e * e
            d_perp = (p0 - c) - d_par
            r = float(np.linalg.norm(d_perp))
            ed = np.cross(e, p0 - c)
            v_rot = float(v0 @ ed / (ed @ ed)) if ed @ ed > 1e-12 else 0.0
            radial = float(np.linalg.norm(v0 - v_rot * ed))
            I_e = float(e @ (J if J is not None
                             else np.zeros((3, 3))) @ e) + m * r * r
            L_e = float(e @ (H0 + np.cross(p0 - c, m * v0)))
            phi, phid = 0.0, (L_e / I_e if I_e > 1e-9 else 0.0)
            e_cross_perp = np.cross(e, d_perp)
            a_e = float(e @ a_ext) if a_ext is not None else 0.0
            for _ in range(n):
                d_rot = d_par + d_perp * np.cos(phi) \
                    + e_cross_perp * np.sin(phi)
                tau_e = float(e @ np.cross(d_rot, m * g))
                phidd = (tau_e / I_e if I_e > 1e-9 else 0.0) + a_e
                acc = (phidd * np.cross(e, d_rot)
                       + phid * phid * np.cross(e, np.cross(e, d_rot)))
                track(m * (acc - g))
                phid += phidd * h
                phi += phid * h
                dyaw += phid * e[2] * h
            d_end = d_par + d_perp * np.cos(phi) \
                + e_cross_perp * np.sin(phi)
            p_end = c + d_end
            v_end = phid * np.cross(e, d_end)
            H_end = (J @ (phid * e) if J is not None else H0)

        ok = (radial <= v_tol and min_nz > 0.0
              and max_mu <= LIP_MU_SLIDE)
        if mode == "ballistic":
            ok = True                            # 无接触可违反
        st = SupportState(
            com=p_end,
            left_foot=self.left_foot.copy(),
            right_foot=self.right_foot.copy(),
            torso_yaw=_wrap(self.torso_yaw + dyaw),
            left_foot_height=self.left_foot_height,
            right_foot_height=self.right_foot_height,
            com_vel=v_end,
            ang_mom=H_end)
        return DriftReport(
            state=st, mode=mode, assumption_ok=ok,
            radial_violation=radial,
            min_normal=0.0 if mode == "ballistic" else float(min_nz),
            max_friction_ratio=(0.0 if mode == "ballistic"
                                else float(max_mu)),
            swing_min_clearance=(float(swing_min)
                                 if mode == "pivot" and swing
                                 and np.isfinite(swing_min)
                                 else float("nan")))


    def path_to(self, aim: SupportGeometry,
                step_radius: float = 0.6,
                pos_tol: float = 0.01,
                max_moves: int = 60
                ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """双脚 XY 站位序列——纯几何脚步规划（不评稳定性/动力学）。

        每项 (left_xy, right_xy)；相邻项只允许一只脚移动，且
        落点须在**不动脚** step_radius 水平半径内（腿长包络）。
        首项 = 当前双脚位，末项 = aim 双脚位；只差一脚且可达
        时自然只有两项。

        贪心交替：每步动"距各自 aim 更远"的脚；aim 落点可达则
        直落，否则朝 aim 方向落 0.95·step_radius 处（保证每步
        单调逼近、必终止）。同脚连动合法（抬-落-抬）。
        aim 自身双脚距超 step_radius ⇒ 终点站位不可达，ValueError；
        max_moves 未收敛 ⇒ 规划失败，ValueError。

        边界：点模型无碰撞/越腿检查——双脚换位这类交叉步在
        本模型下合法；yaw/足高不参与。
        """
        al = np.asarray(aim.left_foot, dtype=np.float64)
        ar = np.asarray(aim.right_foot, dtype=np.float64)
        if np.linalg.norm(al - ar) > step_radius:
            raise ValueError(
                "path_to: aim 双脚距 "
                f"{np.linalg.norm(al - ar):.3f}m > step_radius "
                f"{step_radius}——终点站位不可达")
        cur = [self.left_foot.copy(), self.right_foot.copy()]
        tgt = [al.copy(), ar.copy()]
        seq = [(cur[0].copy(), cur[1].copy())]
        for _ in range(max_moves):
            diffs = [float(np.linalg.norm(cur[i] - tgt[i]))
                     for i in range(2)]
            if max(diffs) <= pos_tol:
                return seq
            for i in sorted(range(2), key=lambda i: -diffs[i]):
                pivot = cur[1 - i]
                d = float(np.linalg.norm(tgt[i] - pivot))
                land = (tgt[i].copy() if d <= step_radius
                        else pivot + (tgt[i] - pivot) / d
                        * step_radius * 0.95)
                cur[i] = land
                seq.append((cur[0].copy(), cur[1].copy()))
                break
        raise ValueError(
            f"path_to: {max_moves} 步内未收敛——aim 不可达")

    def transition_to(self, anchor: SupportGeometry
                      ) -> Tuple[SupportGeometry, "SupportState"]:
        """向目标锚点迈一步。

        返回 (next_geometry, next_state)：转移语义待 F4 落地。
        """
        raise NotImplementedError


@dataclass
class DriftReport:
    """drift 报告——演化结果 + 接触假定体检。"""
    state: "SupportState"             # 演化后的支撑态
    mode: str                         # "ballistic" / "pivot" / "hinge"
    assumption_ok: bool               # 贴地/不打滑/切向一致假定成立
    radial_violation: float           # 初速与支点约束的不一致分量 m/s
    min_normal: float                 # 演化中最小法向约束力 N（ballistic=0）
    max_friction_ratio: float         # |F_xy|/F_z 峰值（ballistic=0）
    swing_min_clearance: float = float("nan")  # pivot 摆脚踝心最低离地
                                      # 间隙（<0 = 本会触地）；非 pivot NaN
