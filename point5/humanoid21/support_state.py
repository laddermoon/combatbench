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
from typing import Tuple

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
    assert v.shape == shape, f"{name}: expected {shape}, got {v.shape}"
    return v


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

    def reference_p5(self):
        """由几何合成参考 P5 意图——非精确还原。

        几何只含 com/脚/yaw，头与手按标称站姿生成；返回的是
        可用参照点阵，不做信息保真承诺（故名 reference 非 to_）。
        """
        raise NotImplementedError


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



    def reference_p5(self):
        """由几何合成参考 P5 意图——非精确还原。

        几何只含 com/脚/yaw，头与手按标称站姿生成；返回的是
        可用参照点阵，不做信息保真承诺（故名 reference 非 to_）。
        """
        raise NotImplementedError


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

    def capture_point(self) -> np.ndarray:
        """捕获点（3,) — com + com_vel·√(z/g)。"""
        raise NotImplementedError

    def transition_to(self, anchor: SupportGeometry
                      ) -> Tuple[SupportGeometry, "SupportState"]:
        """向目标锚点迈一步。

        返回 (next_geometry, next_state)：转移语义待 F4 落地。
        """
        raise NotImplementedError
