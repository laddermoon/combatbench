"""算法1 功能点 f1-f4 —— alg1_infoflow.md / algo1_meta.py 的实现层。

    f1(obs)          -> I1   当前支撑状态
    f2(obs)          -> I2   当前质心动量
    f3(clip)         -> I3   逐帧目标支撑状态
    f4(i1, i2, i3)   -> R    支撑状态序列

物理可行+先验系统，非最优化。当前全部为空壳——仅落签名。
"""
from __future__ import annotations

import numpy as np

from point5.contracts import P5Clip
from .algo1_meta import I1, I2, I3, R
from typing import Tuple
from .contracts import Observation


def f12(obs: Observation) -> Tuple[I1, I2]:
    """f12(Observation) -> (I1, I2)：一次 Pose 解码出支撑态+动量。

    I1.lm 几何经 Pose.support_state_on_p5_axis()（脚位+精确质心），
    force_ratio 覆盖为 obs[52:54] 实测分配（当前态是测量非设计）；
    I2 经 mj_subtreeVel 出精确子树量转锚定局部帧。
    """
    from point5.humanoid21.pose import Pose

    obs = np.asarray(obs, np.float64)
    p = Pose.from_obs(obs)
    lm = p.support_state_on_p5_axis()
    fr = obs[52:54]
    lm.feet["right"].force_ratio = float(fr[0])
    lm.feet["left"].force_ratio = float(fr[1])
    return I1(lm=lm), I2(com_vel=p.com_vel(), ang_mom=p.ang_mom())


def f3(clip: P5Clip) -> I3:
    """f3(Clip) -> I3：每帧 P5 → 对应支撑状态，统一 Base 帧。

    转换链：clip.p5[i] -> Pose.from_p5 -> support_state_on_p5_axis
    （同一套 com/feet/ratio 语义）-> 按帧位姿 (mid_i, yaw_i)
    旋转+平移到首帧坐标系。SupportState 的语义来源与 f12 一致。
    """
    from point5.humanoid21.pose import Pose
    from .contracts import FootSupport, SupportState

    def _to_base(lm, mid, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        feet = {sd: FootSupport(force_ratio=f.force_ratio,
                                pos=R @ f.pos + mid)
                for sd, f in lm.feet.items()}
        com = lm.com.copy()
        com[:2] = R @ com[:2] + mid
        return SupportState(com=com, feet=feet,
                            torso_yaw=lm.torso_yaw, for_frame=0)

    landmarks = []
    for i in range(len(clip.p5)):
        raw = clip.Base.pos if i == 0 else \
            clip.Base.pos + clip.Transfers[i - 1].delta_pos
        yaw = 0.0 if i == 0 else float(clip.Transfers[i - 1].delta_yaw)
        mid = 0.5 * (raw[3, :2] + raw[4, :2])
        lm = Pose.from_p5(clip.p5[i]).support_state_on_p5_axis()
        st = _to_base(lm, mid, yaw)
        # ratio 取**声明**的落地事件（clip raw 脚 z），不是解算
        # 残差——f12 用实测力是同一原则：状态语义由来源定。
        grounded = {sd: raw[k, 2] < 0.12
                    for sd, k in (("right", 3), ("left", 4))}
        n = sum(grounded.values())
        for sd in ("right", "left"):
            st.feet[sd].force_ratio = \
                (1.0 / n) if (n and grounded[sd]) else 0.0
        st.for_frame = i
        landmarks.append(st)
    return I3(landmarks=landmarks)


def f4(i1: I1, i2: I2, i3: I3) -> R:
    """f4(I1, I2, I3) -> R：逐帧目标 → 稀疏锚点序列。

    把逐帧目标压缩成锚点：哪里必须换脚（支撑事件锚点）、
    哪里只需姿态锚点；com 转移须物理可行且有裕度（I2 动量
    决定转移方向——离地锚点要求 capture point 指向落脚区）。
    """
    raise NotImplementedError
