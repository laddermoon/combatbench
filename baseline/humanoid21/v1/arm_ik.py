"""手臂反解器：CPose 式描述符 → 三关节角（D6 指令表的生成端）。

描述符（每臂各自约定系，torso 局部系 X+身前/Y+向左/Z+向头）：
    azimuth θ   : 矢状面扫掠角——手向量在 ⊥极轴平面（x−z 面）内
                  相对身体前向的夹角 atan2(z,x)：0=前水平、+90°=向天、
                  −90°=向地、±180°=正后。整圆连续（jab→上勾→后摆
                  是同一参数单向走）。两臂同式无 σ。
    polar φ     : 摆开度——手−肩连线与**外展极轴**的夹角（极轴=肩心
                  沿外侧向：右臂 −y、左臂 +y）：0=纯外展、90°=前平面、
                  180°=穿中线（物理近不可达），恒正∈[0,180°]。
                  同一描述符在两臂产出镜像动作。注：θ 在 φ≈0/180°退化。
    extension ρ : |手−肩|/LMax ∈ (0,1]（LMax=该臂最大伸展距离）
    hint        : 肘分支选择器（3 值描述符在 3-DOF 臂上至多两支解，
                  非连续冗余）：None=自然肘（肘 z 最低支，默认）、
                  "tuck"=肘向内下（贴肋）/"flare"=肘向外上（张开）

运动学（battle_circular_v2.xml，torso 系，左臂为右臂对 y 镜像）：
    肩中心 S=(0,∓.17,.06)；上臂偏移 E=(.18,∓.18,-.18)；拳偏移 H=(.18,±.18,.18)
    shoulder1 轴 (±2,1,∓1)/√6、shoulder2 轴 (0,-1,∓1)/√2、elbow 轴 (0,-1,∓1)/√2
    range rad：s1/s2 [-85°,60°]，e [-100°,50°]（e 负=伸展、正=屈曲）

求解分解（3 参数=3 DOF，确定性，无通用 IK）：
    1. e：|手−肩| 与肩部双轴无关（肩绕肩心旋转保距）→ d(e)=ρ·LMax
       一维求根；d(e) 非单调时枚举全部根。
    2. (s1,s2)：R_a1(s1)·R_a2(s2)·p_e = w 的双轴指向闭式解
       ——两圆交点 ≤2 支；选支：prev（序列连续）> 显式 hint > 自然肘。
    3. 可达性分两层（用户拍板）：闭式解在限位内 → 精确命中零开销；
       不可达 → 可达集内**笛卡尔最近点** min|FK(q)−P|²（3-DOF 箱
       约束 LM，多起点确定性），非"解出再钳"——最近点可能换分支/
       换 e。clipped 三标志（extension/direction/limits）保留作诊断，
       pos_err=|hand−P| 报真残差。

用法：
    ik = ArmIK()
    sol = ik.solve("right", ArmPose(azimuth=0.1, polar=1.5, extension=0.9))
    sol.norm      # (s1,s2,e) 归一化目标（=action/obs 空间值）
    sol.angles    # 弧度
    sol.clipped   # None | "extension" | "direction" | "limits"
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# ------------------------------------------------------------ 模型常数 ----

def _u(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


@dataclass(frozen=True)
class ArmSpec:
    side: str
    S: np.ndarray    # 肩旋转中心（torso 系）
    a1: np.ndarray   # shoulder1 轴（torso 系，单位向量）
    a2: np.ndarray   # shoulder2 轴
    ae: np.ndarray   # elbow 轴（lower_arm 局部系=上臂系）
    E: np.ndarray    # 肘在上臂系中的偏移
    H: np.ndarray    # 拳心在肘系中的偏移
    out_sigma: float  # 外展极轴符号：+1=+y 左臂；−1=−y 右臂
    s_range: Tuple[float, float] = (math.radians(-85), math.radians(60))
    e_range: Tuple[float, float] = (math.radians(-100), math.radians(50))


_SPECS = {
    "right": ArmSpec("right",
                     S=np.array([0, -0.17, 0.06]),
                     a1=_u([2, 1, 1]), a2=_u([0, -1, 1]), ae=_u([0, -1, 1]),
                     E=np.array([0.18, -0.18, -0.18]),
                     H=np.array([0.18, 0.18, 0.18]),
                     out_sigma=-1.0),
    "left": ArmSpec("left",
                    S=np.array([0, 0.17, 0.06]),
                    a1=_u([-2, 1, -1]), a2=_u([0, -1, -1]),
                    ae=_u([0, -1, -1]),
                    E=np.array([0.18, 0.18, -0.18]),
                    H=np.array([0.18, -0.18, 0.18]),
                    out_sigma=+1.0),
}

# 肘 Hint 预设方向（torso 系，肘偏离肩→拳连线的期望方向）
_HINT_DIRS = {
    "right": {"tuck": _u([0, 0.8, -0.6]), "flare": _u([0, -0.8, 0.6])},
    "left": {"tuck": _u([0, -0.8, -0.6]), "flare": _u([0, 0.8, 0.6])},
}

# norm 空间映射（rad = norm*half + mid），与 point5 mechanics 一致
_S_MID, _S_HALF = (math.radians(-85) + math.radians(60)) / 2, \
                  (math.radians(60) - math.radians(-85)) / 2
_E_MID, _E_HALF = (math.radians(-100) + math.radians(50)) / 2, \
                  (math.radians(50) - math.radians(-100)) / 2


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) * c + K * s + np.outer(axis, axis) * (1 - c)


def _atan2_rot(axis: np.ndarray, p: np.ndarray, x: np.ndarray) -> float:
    """绕 axis 把 p 转到 x 的角度（两者在同轴圆上时精确）。"""
    return math.atan2(axis @ np.cross(p, x),
                      p @ x - (axis @ p) * (axis @ x))


# ------------------------------------------------------------ 解结构 ------

@dataclass
class ArmPose:
    """单臂姿态描述符（按臂约定：θ 矢状扫掠两臂同式、φ 按臂外展）。

    重表意对象：构造即校验合法性，域外值响亮拒绝（调用方 bug）。
    域内合法但物理不可达不在此判定——solve() 投影到最近可达
    并在 ArmSolution.clipped/dir_err/dist_err 上报残差。"""
    azimuth: float             # rad，atan2(z,x)：0=前、+90°=天、−90°=地、±180°=后
    polar: float               # rad，与外展轴夹角：0=纯外展、90°=前平面、180°=穿中线
    extension: float           # |手−肩|/LMax ∈(0,1]
    hint: Optional[str] = None  # None=自然肘；"tuck"|"flare" 显式选支

    def __post_init__(self) -> None:
        vals = (self.azimuth, self.polar, self.extension)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"ArmPose 非有限值: {self}")
        # azimuth 是周期量：wrap 归一到 (−π, π]（非拒绝项）
        self.azimuth = (self.azimuth + math.pi) % (2 * math.pi) - math.pi
        if not (0.0 <= self.polar <= math.pi):
            raise ValueError(
                f"polar 越域（须∈[0°,180°]）: {math.degrees(self.polar):.1f}°")
        if self.extension <= 0.0 or self.extension > 1.0 + 1e-3:
            raise ValueError(
                f"extension 越域（须∈(0,1]）: {self.extension}")
        self.extension = min(self.extension, 1.0)   # describe 浮点噪声容差
        if self.hint not in (None, "tuck", "flare"):
            raise ValueError(f"hint 非法: {self.hint!r}")

    def dir_vec(self) -> np.ndarray:
        """约定系方向（+y=外展分量、+z=向上）；solve 内按 out_sigma 映射。

        dir = cosφ·外展轴 + sinφ·(cosθ·前 + sinθ·上)"""
        s = math.sin(self.polar)
        return np.array([s * math.cos(self.azimuth),
                         math.cos(self.polar),
                         s * math.sin(self.azimuth)])

    def mirrored(self) -> "ArmPose":
        """镜像 = 同一描述符（按臂约定天然镜像）。恒等，仅 API 兼容。"""
        return ArmPose(self.azimuth, self.polar, self.extension, self.hint)

    @classmethod
    def deg(cls, azimuth: float, polar: float, extension: float,
            hint: Optional[str] = None) -> "ArmPose":
        """度数控构造：ArmPose.deg(-5, 92, 0.95)（锚点/生成端人读面）。"""
        return cls(math.radians(azimuth), math.radians(polar),
                   extension, hint)


@dataclass
class ArmTarget:
    """躯干系观测点（如 obs 对手关键点）的描述符表征——给上层策略。

    与 ArmPose 的区别：ArmPose 是"我臂要摆成的形状"（ext∈(0,1] 表意
    合法）；ArmTarget 是"外界有个点在哪"——ext 可 >1（超臂展），
    属观测事实不是格式错。to_pose() 截到臂展内即得可解描述符：
    `ik.solve(side, ik.describe_point(side, p).to_pose())`。"""
    azimuth: float         # rad，同 ArmPose 约定
    polar: float           # rad，同 ArmPose 约定
    dist: float            # |点−肩心|（m）
    extension: float       # dist/LMax（>1 = 超臂展）
    point: np.ndarray      # 肩心系原始向量

    @property
    def reachable(self) -> bool:
        """粗略可及：仅距离判据（不含方向可行性）。"""
        return self.extension <= 1.0 + 1e-3

    def to_pose(self, ext_cap: float = 1.0,
                hint: Optional[str] = None) -> ArmPose:
        """方向照搬、距离截臂展 → 合法 ArmPose（喂 solve 打向该点）。"""
        return ArmPose(self.azimuth, self.polar,
                       min(self.extension, ext_cap), hint)


@dataclass
class ArmSolution:
    angles: np.ndarray          # (3,) rad (s1,s2,e)
    norm: np.ndarray            # (3,) 归一化目标 = action/obs 空间值
    hand: np.ndarray            # 拳心 torso 系位置
    elbow: np.ndarray           # 肘 torso 系位置
    candidates: int             # 本描述符的可行解支数
    clipped: Optional[str]      # None|extension|direction|limits（诊断）
    dir_err_deg: float          # 实际拳方向与目标方向的角误差
    dist_err: float             # 实际伸展距离与目标距离之差（m）
    pos_err: float = 0.0        # |hand−目标点P|（m，真残差——0=精确命中）


# ------------------------------------------------------------ 求解器 ------

class ArmIK:
    """描述符 → 关节角。纯 numpy；FK 与 MuJoCo 对拍验证过（渲染脚本附）。"""

    def __init__(self) -> None:
        # d(e) 形状探查：e 范围内采样，求 LMax/距离极值
        e_grid = np.linspace(*_SPECS["right"].e_range, 400)
        d_right = [self._dist_at_e(_SPECS["right"], e) for e in e_grid]
        self._e_grid, self._d_grid = e_grid, np.asarray(d_right)
        self.LMax: float = float(self._d_grid.max())
        self.d_min: float = float(self._d_grid.min())
        # 左右臂 |E|,|H|,轴对称 ⇒ d(e) 相同，网格共用

    # ---- FK ----
    def _dist_at_e(self, spec: ArmSpec, e: float) -> float:
        return float(np.linalg.norm(spec.E + _rodrigues(spec.ae, e) @ spec.H))

    def fk(self, side: str, s1: float, s2: float, e: float
           ) -> Tuple[np.ndarray, np.ndarray]:
        """(hand_pos, elbow_pos)，肩中心为原点的 torso 系。"""
        sp = _SPECS[side]
        R = _rodrigues(sp.a1, s1) @ _rodrigues(sp.a2, s2)
        elbow = R @ sp.E
        hand = elbow + R @ (_rodrigues(sp.ae, e) @ sp.H)
        return hand, elbow

    # ---- 支解：R_a1(s1)·R_a2(s2)·p = w ----
    @staticmethod
    def _pointing(a1, a2, p, w) -> List[Tuple[float, float]]:
        """两圆交点闭式解。x 满足 x·a2=p·a2 ∧ x·a1=w·a1 ∧ |x|=|w|。"""
        d = np.linalg.norm(w)
        c = a1 @ a2
        # 两平面交线：x0 = αa1+βa2，解 [[1,c],[c,1]]·[α,β]=[w·a1, p·a2]
        det = 1 - c * c
        if det < 1e-12:
            return []
        h1, h2 = w @ a1, p @ a2
        alpha = (h1 - c * h2) / det
        beta = (h2 - c * h1) / det
        x0 = alpha * a1 + beta * a2
        r2 = d * d - x0 @ x0
        if r2 < -1e-9:
            return []                      # 两圆不交——方向不可达
        u = np.cross(a1, a2) / math.sqrt(det)
        t = math.sqrt(max(r2, 0.0))
        out = []
        for x in (x0 + t * u, x0 - t * u):
            if abs(np.linalg.norm(x) - d) > 1e-6:
                continue
            s2 = _atan2_rot(a2, p, x)
            s1 = _atan2_rot(a1, x, w)
            out.append((s1, s2))
        return out

    # ---- 不可达方向的最小二乘兜底 ----
    def _pointing_ls(self, sp, p, d_hat, seeds) -> Tuple[float, float]:
        """方向不可达时：最小化 |R·p̂ − d̂|²，多起点选最优。"""
        best, best_err = None, np.inf
        p_hat = p / np.linalg.norm(p)
        for s1_0, s2_0 in seeds:
            s = np.array([s1_0, s2_0])
            for _ in range(60):
                R = _rodrigues(sp.a1, s[0]) @ _rodrigues(sp.a2, s[1])
                r = R @ p_hat - d_hat
                err = r @ r
                if err < 1e-12:
                    break
                # 数值雅可比（2 列，便宜）
                J = np.empty((3, 2))
                for j in range(2):
                    ds = s.copy(); ds[j] += 1e-6
                    Rj = _rodrigues(sp.a1, ds[0]) @ _rodrigues(sp.a2, ds[1])
                    J[:, j] = (Rj @ p_hat - d_hat) / 1e-6 - r / 1e-6
                # LM 阻尼牛顿
                A = J.T @ J + 1e-4 * np.eye(2)
                step = np.linalg.solve(A, -J.T @ r)
                s += step
                if np.linalg.norm(step) < 1e-10:
                    break
            if err < best_err:
                best, best_err = s.copy(), err
        return best[0], best[1]

    # ---- 不可达点：可达集内最近点（3-DOF 箱约束 LM，确定性多起点）----
    @staticmethod
    def _fk_hand(sp, s: np.ndarray) -> np.ndarray:
        R = _rodrigues(sp.a1, s[0]) @ _rodrigues(sp.a2, s[1])
        return R @ (sp.E + _rodrigues(sp.ae, s[2]) @ sp.H)

    def _nearest_point(self, sp, P: np.ndarray,
                       seeds: List[np.ndarray]
                       ) -> List[Tuple[np.ndarray, float]]:
        """min |FK(q)−P|² s.t. q∈joint box。返回 [(angles, 残差m)]。"""
        lo = np.array([sp.s_range[0], sp.s_range[0], sp.e_range[0]])
        hi = np.array([sp.s_range[1], sp.s_range[1], sp.e_range[1]])
        out = []
        for s0 in seeds:
            s = np.clip(np.asarray(s0, np.float64), lo, hi)
            hand = self._fk_hand(sp, s)
            err = float(((hand - P) ** 2).sum())
            lam = 1e-4
            for _ in range(60):
                if err < 1e-14:
                    break
                J = np.empty((3, 3))
                for j in range(3):
                    ds = s.copy()
                    ds[j] += 1e-6
                    J[:, j] = (self._fk_hand(sp, ds) - hand) / 1e-6
                A = J.T @ J + lam * np.eye(3)
                step = np.linalg.solve(A, -J.T @ (hand - P))
                if np.linalg.norm(step) < 1e-11:
                    break
                s_new = np.clip(s + step, lo, hi)
                hand_new = self._fk_hand(sp, s_new)
                err_new = float(((hand_new - P) ** 2).sum())
                if err_new < err:          # 接受：降阻尼加速
                    s, hand, err = s_new, hand_new, err_new
                    lam = max(lam / 3.0, 1e-8)
                else:                      # 拒绝：升阻尼重试
                    lam *= 4.0
                    if lam > 1e2:
                        break
            out.append((s.copy(), math.sqrt(err)))
        return out

    # ---- 主入口 ----
    def solve(self, side: str, pose: ArmPose,
              prev: Optional[np.ndarray] = None,
              hint_dir: Optional[np.ndarray] = None) -> ArmSolution:
        sp = _SPECS[side]
        d = pose.dir_vec()
        d_hat = np.array([d[0], sp.out_sigma * d[1], d[2]])  # 约定系→torso系
        clipped: List[str] = []

        # --- 目标点 P = ρ·LMax·d̂；ext∈(0,1] 已校验 ---
        dist_req = pose.extension * self.LMax
        dist = min(max(dist_req, self.d_min), self.LMax)
        if dist != dist_req:
            clipped.append("extension")
        P = dist_req * d_hat                       # 残差基准（原始请求）
        e_roots = self._solve_e(sp, dist)
        w = dist * d_hat

        # 分支评分模式：显式 hint_dir > hint 预设 > 自然肘（肘 z 最低）
        if hint_dir is not None:
            hd: Optional[np.ndarray] = hint_dir
        elif pose.hint is not None:
            hd = _HINT_DIRS[side][pose.hint]
        else:
            hd = None

        def _score(elbow: np.ndarray) -> float:
            if hd is None:
                return -float(elbow[2])
            off = elbow - (elbow @ d_hat) * d_hat
            return float(off @ hd / (np.linalg.norm(off) + 1e-9))

        # --- 闭式候选：(s1,s2) 指向解 × e 根（LS 兜底仅作种子） ---
        cands: List[np.ndarray] = []
        exact = True
        for e in e_roots:
            p = sp.E + _rodrigues(sp.ae, e) @ sp.H
            sols = self._pointing(sp.a1, sp.a2, p, w)
            if not sols:
                exact = False
                sols = [self._pointing_ls(
                    sp, p, d_hat,
                    [(0.0, 0.0), (-0.8, -0.8), (-0.8, 0.8),
                     (0.8, -0.8), (0.8, 0.8)])]
            for s1, s2 in sols:
                cands.append(np.array([s1, s2, e]))
        if not exact:
            clipped.append("direction")

        # --- 快路径：闭式解在限位内 = 精确命中/边界命中（即最优） ---
        lo_s, hi_s = sp.s_range
        lo_e, hi_e = sp.e_range

        def _in_lim(a: np.ndarray) -> bool:
            return (lo_s - 1e-9 <= a[0] <= hi_s + 1e-9 and
                    lo_s - 1e-9 <= a[1] <= hi_s + 1e-9 and
                    lo_e - 1e-9 <= a[2] <= hi_e + 1e-9)

        if exact and any(_in_lim(c) for c in cands):
            pool = [c for c in cands if _in_lim(c)]
            if prev is not None:
                i = int(np.argmin([np.linalg.norm(c - prev) for c in pool]))
            else:
                i = int(np.argmax([_score(self.fk(side, *c)[1])
                                   for c in pool]))
            ang = pool[i]
        else:
            # --- 投影：可达集内笛卡尔最近点 min|FK(q)−P|² ---
            if exact:
                clipped.append("limits")       # 有解但全超限位
            seeds = list(cands)
            seeds += [np.zeros(3),
                      np.array([sp.s_range[1] * 0.8] * 2 + [0.0]),
                      np.array([sp.s_range[0] * 0.8] * 2 + [0.0])]
            if prev is not None:
                seeds.append(np.asarray(prev, np.float64))
            opt = self._nearest_point(sp, P, seeds)
            res_best = min(r for _, r in opt)
            pool = [s for s, r in opt if r <= res_best + 1e-3]
            if prev is not None:
                i = int(np.argmin([np.linalg.norm(c - prev) for c in pool]))
            else:
                i = int(np.argmax([_score(self.fk(side, *c)[1])
                                   for c in pool]))
            ang = pool[i]

        hand, elbow = self.fk(side, *ang)
        achieved = hand / np.linalg.norm(hand)
        dir_err = math.degrees(math.acos(
            float(np.clip(achieved @ d_hat, -1, 1))))
        norm = np.array([(ang[0] - _S_MID) / _S_HALF,
                         (ang[1] - _S_MID) / _S_HALF,
                         (ang[2] - _E_MID) / _E_HALF])
        return ArmSolution(
            angles=ang, norm=norm, hand=hand, elbow=elbow,
            candidates=len(cands),
            clipped="+".join(clipped) if clipped else None,
            dir_err_deg=dir_err,
            dist_err=float(np.linalg.norm(hand) - dist_req),
            pos_err=float(np.linalg.norm(hand - P)))

    # d(e)=dist 全部根（采样+二分）
    def _solve_e(self, sp: ArmSpec, dist: float) -> List[float]:
        f = self._d_grid - dist
        roots = []
        lo, hi = sp.e_range
        for i in range(len(self._e_grid) - 1):
            if f[i] == 0 or f[i] * f[i + 1] < 0:
                a, b = self._e_grid[i], self._e_grid[i + 1]
                for _ in range(50):
                    m = 0.5 * (a + b)
                    fm = self._dist_at_e(sp, m) - dist
                    fa = self._dist_at_e(sp, a) - dist
                    if fa * fm <= 0:
                        b = m
                    else:
                        a = m
                roots.append(0.5 * (a + b))
        if not roots:  # 距离恰在极值点上（数值边界）
            roots = [float(self._e_grid[np.argmin(np.abs(f))])]
        return roots

    def describe(self, side: str, angles_rad: np.ndarray) -> ArmPose:
        """关节角 → 描述符（反变换/校验用；azimuth 按臂约定）。"""
        sp = _SPECS[side]
        hand, _ = self.fk(side, *angles_rad)
        r = np.linalg.norm(hand)
        x, y, z = hand / r
        return ArmPose(azimuth=math.atan2(z, x),
                       polar=math.acos(float(np.clip(sp.out_sigma * y, -1, 1))),
                       extension=r / self.LMax)

    # ---- 观测点 → 描述符（上层策略工具）----
    def describe_point(self, side: str, p_torso: np.ndarray) -> ArmTarget:
        """躯干系点（obs 对手关键点即此系）→ 描述符表征。

        原点=肩心（spec.S 同系）；ext 如实可>1——对手常在臂展外。"""
        sp = _SPECS[side]
        rel = np.asarray(p_torso, np.float64) - sp.S
        r = float(np.linalg.norm(rel))
        if r < 1e-9:
            raise ValueError("观测点与肩心重合，方向无定义")
        x, y, z = rel / r
        return ArmTarget(
            azimuth=math.atan2(z, x),
            polar=math.acos(float(np.clip(sp.out_sigma * y, -1, 1))),
            dist=r, extension=r / self.LMax, point=rel)


__all__ = ["ArmIK", "ArmPose", "ArmSolution", "ArmTarget"]
