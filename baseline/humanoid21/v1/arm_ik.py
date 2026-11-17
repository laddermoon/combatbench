"""手臂反解器：CPose 式描述符 → 三关节角（D6 指令表的生成端）。

描述符（每臂，torso 局部系，X+身前/Y+向左/Z+向头，与 point5/cpose.py 一致）：
    azimuth θ   : 拳方向方位角 atan2(y,x)：0=正前、+90°=向左、±180°=正后
    polar φ     : 拳方向极角 arccos(z/r)：0=向头、90°=水平、180°=下垂
    extension ρ : |手−肩|/LMax ∈ (0,1]（LMax=该臂最大伸展距离）
    hint        : 肘位置偏好——(s1,s2) 双支解的离散选择器
                  "tuck"=肘向内下（贴肋）/"flare"=肘向外上（张开）

运动学（battle_circular_v2.xml，torso 系，左臂为右臂对 y 镜像）：
    肩中心 S=(0,∓.17,.06)；上臂偏移 E=(.18,∓.18,-.18)；拳偏移 H=(.18,±.18,.18)
    shoulder1 轴 (±2,1,∓1)/√6、shoulder2 轴 (0,-1,∓1)/√2、elbow 轴 (0,-1,∓1)/√2
    range rad：s1/s2 [-85°,60°]，e [-100°,50°]（e 负=伸展、正=屈曲）

求解分解（3 参数=3 DOF，确定性，无通用 IK）：
    1. e：|手−肩| 与肩部双轴无关（肩绕肩心旋转保距）→ d(e)=ρ·LMax
       一维求根；d(e) 非单调时枚举全部根。
    2. (s1,s2)：R_a1(s1)·R_a2(s2)·p_e = w 的双轴指向闭式解
       ——两圆交点 ≤2 支；hint 按肘位置偏好选支，prev 提供序列连续性。
    3. 不可达：方向不可达/距离超限 → 投影到最近可达 + clipped 标记，
       残差随返回值上报，绝不静默。

用法：
    ik = ArmIK()
    sol = ik.solve("right", azimuth=0.1, polar=1.5, extension=0.9, hint="flare")
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
    s_range: Tuple[float, float] = (math.radians(-85), math.radians(60))
    e_range: Tuple[float, float] = (math.radians(-100), math.radians(50))


_SPECS = {
    "right": ArmSpec("right",
                     S=np.array([0, -0.17, 0.06]),
                     a1=_u([2, 1, 1]), a2=_u([0, -1, 1]), ae=_u([0, -1, 1]),
                     E=np.array([0.18, -0.18, -0.18]),
                     H=np.array([0.18, 0.18, 0.18])),
    "left": ArmSpec("left",
                    S=np.array([0, 0.17, 0.06]),
                    a1=_u([-2, 1, -1]), a2=_u([0, -1, -1]),
                    ae=_u([0, -1, -1]),
                    E=np.array([0.18, 0.18, -0.18]),
                    H=np.array([0.18, -0.18, 0.18])),
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
    """单臂姿态描述符。"""
    azimuth: float      # rad，0=正前、+90°=向左
    polar: float        # rad，0=向头、90°=水平、180°=下垂
    extension: float    # |手−肩|/LMax
    hint: str = "tuck"

    def dir_vec(self) -> np.ndarray:
        s = math.sin(self.polar)
        return np.array([s * math.cos(self.azimuth),
                         s * math.sin(self.azimuth),
                         math.cos(self.polar)])

    def mirrored(self) -> "ArmPose":
        """右臂描述符 → 镜像左臂（方位角对 y 取反）。"""
        return ArmPose(-self.azimuth, self.polar, self.extension, self.hint)


@dataclass
class ArmSolution:
    angles: np.ndarray          # (3,) rad (s1,s2,e)
    norm: np.ndarray            # (3,) 归一化目标 = action/obs 空间值
    hand: np.ndarray            # 拳心 torso 系位置
    elbow: np.ndarray           # 肘 torso 系位置
    candidates: int             # 本描述符的可行解支数
    clipped: Optional[str]      # None|extension|direction|limits
    dir_err_deg: float          # 实际拳方向与目标方向的角误差
    dist_err: float             # 实际伸展距离与目标距离之差（m）


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

    # ---- 主入口 ----
    def solve(self, side: str, pose: ArmPose,
              prev: Optional[np.ndarray] = None,
              hint_dir: Optional[np.ndarray] = None) -> ArmSolution:
        sp = _SPECS[side]
        d_hat = pose.dir_vec()
        clipped: List[str] = []

        # --- 1. 距离 → e ---
        dist = float(np.clip(pose.extension, 1e-6, 1.5)) * self.LMax
        if dist > self.LMax:
            dist = self.LMax
            clipped.append("extension")
        if dist < self.d_min:
            dist = self.d_min
            clipped.append("extension")
        e_roots = self._solve_e(sp, dist)

        w = dist * d_hat
        hd = hint_dir if hint_dir is not None else _HINT_DIRS[side][pose.hint]

        # --- 2&3. (s1,s2) 指向解 × e 根 → 候选集 ---
        cands: List[Tuple[np.ndarray, np.ndarray, float]] = []
        exact = True
        for e in e_roots:
            p = sp.E + _rodrigues(sp.ae, e) @ sp.H
            sols = self._pointing(sp.a1, sp.a2, p, w)
            if not sols:
                exact = False
                seeds = [(0.0, 0.0), (-0.8, -0.8), (-0.8, 0.8),
                         (0.8, -0.8), (0.8, 0.8)]
                sols = [self._pointing_ls(sp, p, d_hat, seeds)]
            for s1, s2 in sols:
                hand, elbow = self.fk(side, s1, s2, e)
                # 肘偏离 肩→目标线 的垂距方向 = 支的特征
                off = elbow - (elbow @ d_hat) * d_hat
                score = off @ hd / (np.linalg.norm(off) + 1e-9)
                cands.append((np.array([s1, s2, e]), elbow, score))
        if not exact:
            clipped.append("direction")

        # --- 选支：prev 连续优先，其次 hint 分 ---
        if prev is not None:
            i = int(np.argmin([np.linalg.norm(c[0] - prev) for c in cands]))
        else:
            i = int(np.argmax([c[2] for c in cands]))
        ang, elbow, _ = cands[i]

        # --- 限位钳制 ---
        lo_s, hi_s = sp.s_range
        lo_e, hi_e = sp.e_range
        ang_c = np.array([np.clip(ang[0], lo_s, hi_s),
                          np.clip(ang[1], lo_s, hi_s),
                          np.clip(ang[2], lo_e, hi_e)])
        if np.abs(ang_c - ang).max() > 1e-3:
            clipped.append("limits")
            ang = ang_c

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
            dist_err=float(np.linalg.norm(hand) - dist))

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
        """关节角 → 描述符（反变换/校验用）。"""
        hand, _ = self.fk(side, *angles_rad)
        r = np.linalg.norm(hand)
        x, y, z = hand / r
        return ArmPose(azimuth=math.atan2(y, x),
                       polar=math.acos(float(np.clip(z, -1, 1))),
                       extension=r / self.LMax)


__all__ = ["ArmIK", "ArmPose", "ArmSolution"]
