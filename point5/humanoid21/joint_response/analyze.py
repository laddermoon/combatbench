#!/usr/bin/env python3
"""关节响应能力分析——action 能否真的到达目标角度。

对每个受控关节输出两类能力：

  1. 杠杆权威（静态）：关节在支撑相可对抗的最大"上游 com
     水平偏移" = tau_max / (m_load·g)。m_load = 该关节当前
     负载侧质量——腿关节在 stance 为上游（除该腿子树的全部），
     摆动态为下游子树；腰/臂恒为下游子树。
  2. 动态响应：PD 二阶近似 ω_n=√(KP/I_diag)、ζ、settle 时间
     （M 对角元取自当前位形的 mj_fullM）。

负载侧规则：
  - 腰/臂关节：负载 = 关节子树（下游所悬挂质量），摆动/支撑皆然；
  - 腿关节  stance：负载 = robot 全体 − 该腿子树（上游体重）；
           swing： 负载 = 该腿子树本身；
  - 双腿分摊：静不定，按对称载各半（表中单腿值为保守上界）。

用法：
    PYTHONPATH=. python3 point5/humanoid21/joint_response/analyze.py
"""
from __future__ import annotations

import sys
import numpy as np
import mujoco

sys.path.insert(0, ".")
from envs.humanoid21.simulator import Humanoid21Simulator
from envs.humanoid21.meta import Humanoid21Meta

L1, L2 = 0.42, 0.31          # 髋→膝 / 膝→踝心 杆长（mechanics 几何）


def _tables():
    sim = Humanoid21Simulator()
    sim.reset()
    m, d = sim.model, sim.data
    r = Humanoid21Meta.build_runtime_tables(m)["robots"]["robot_a"]
    names = Humanoid21Meta.CONTROLLED_JOINTS
    pd = sim._pd_tables["robot_a"]
    tmax = (np.maximum(np.abs(pd["ctrl_lo"]), np.abs(pd["ctrl_hi"]))
            * np.abs(pd["gear"]))
    qi = np.asarray(r["qpos_indices"])
    adr = int(r["root_qpos_adr"])
    root_body = int(r["root_body_id"])
    nrm = sim._norm_params["robot_a"]
    scale, ref = nrm["scale"], nrm["reference"]
    return (sim, m, d, r, names, tmax, qi, adr, root_body, scale, ref)


def _subtrees(m, root_body, names, pd):
    parent = list(m.body_parentid)

    def sub(root):
        out, grow = {root}, True
        while grow:
            grow = False
            for b in range(m.nbody):
                if parent[b] in out and b not in out:
                    out.add(b)
                    grow = True
        return out

    ra = sub(root_body)
    jid = {names[k]: int(m.actuator_trnid[a, 0])
           for k, a in enumerate(np.asarray(pd["actuator_ids"]))}
    jb = {n: int(m.jnt_bodyid[j]) for n, j in jid.items()}
    return ra, jid, jb, {n: sub(b) & ra for n, b in jb.items()}


LEG = {"hip_x_right", "hip_z_right", "hip_y_right", "knee_right",
       "ankle_y_right", "ankle_x_right",
       "hip_x_left", "hip_z_left", "hip_y_left", "knee_left",
       "ankle_y_left", "ankle_x_left"}


def _com_side(m, d, bodies):
    ms = np.array([m.body_mass[b] for b in bodies])
    ps = np.array([d.xipos[b] for b in bodies])
    return (ms @ ps) / ms.sum(), float(ms.sum())


def main():
    (sim, m, d, r, names, tmax, qi, adr, root_body,
     scale, ref) = _tables()
    ra, jid, jb, sub_of = _subtrees(m, root_body, names, sim._pd_tables["robot_a"])
    mujoco.mj_forward(m, d)
    Mmat = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, Mmat, d.qM)
    Mdiag = np.diag(Mmat)[np.asarray(r["qvel_indices"])]
    Mtot = sum(m.body_mass[b] for b in ra)

    def to_rad(x): return np.asarray(x) * scale + ref
    def to_norm(x): return (np.asarray(x) - ref) / scale
    base_rad = to_rad(d.qpos[qi].copy())
    base_root = d.qpos[adr:adr + 3].copy()

    # ---------------- 表 1：杠杆权威 ----------------
    print("=" * 78)
    print("表1  杠杆权威：关节负载侧 com 相对关节轴的可持最大水平偏移")
    print("=" * 78)
    print(f"{'joint':<16}{'axis':>5}{'tau_max':>8}{'m_stance':>9}"
          f"{'m_swing':>8}{'lever_st':>9}{'lever_sw':>9}")
    for i, n in enumerate(names):
        ax = np.asarray(m.jnt_axis[jid[n]])
        axs = ("x" if abs(ax[0]) > .7 else
               "y" if abs(ax[1]) > .7 else "z" if abs(ax[2]) > .7 else "?")
        m_sw = float(m.body_subtreemass[jb[n]])
        if n in LEG:
            m_st = Mtot - m_sw
        else:
            m_st = m_sw
        lev_st = tmax[i] / (m_st * 9.81)
        lev_sw = tmax[i] / (m_sw * 9.81)
        print(f"{n:<16}{axs:>5}{tmax[i]:>8.1f}{m_st:>9.2f}"
              f"{m_sw:>8.2f}{lev_st:>9.3f}{lev_sw:>9.3f}")

    # ---------------- 表 2：动态响应 ----------------
    print()
    print("=" * 78)
    print("表2  动态响应（站位位形 M 对角元近似；欠阻尼ζ<1 会振铃）")
    print("=" * 78)
    print(f"{'joint':<16}{'KP':>6}{'KD':>5}{'I_diag':>8}{'w_n':>7}"
          f"{'zeta':>6}{'t_settle':>9}")
    for i, n in enumerate(names):
        I = Mdiag[i]
        wn = np.sqrt(Humanoid21Meta.KP[i] / I)
        zeta = Humanoid21Meta.KD[i] / (2 * np.sqrt(Humanoid21Meta.KP[i] * I))
        if zeta < 1:
            ts = 4.0 / (zeta * wn)          # 2% 欠阻尼估计
        else:
            ts = 3.0 * Humanoid21Meta.KD[i] / Humanoid21Meta.KP[i]
        print(f"{n:<16}{Humanoid21Meta.KP[i]:>6.0f}"
              f"{Humanoid21Meta.KD[i]:>5.0f}{I:>8.3f}{wn:>7.1f}"
              f"{zeta:>6.2f}{ts:>9.2f}s")

    # ---------------- 表 3：工况需求力矩 ----------------
    def set_pose(rad_off, dx=0.0, dz=0.0):
        rad = base_rad.copy()
        for k, v in rad_off.items():
            rad[names.index(k)] += v
        d.qpos[qi] = to_norm(rad)
        d.qpos[adr + 2] = base_root[2] + dz
        d.qpos[adr] = base_root[0] + dx
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

    def req(n, stance=True):
        """单腿负载侧重力矩需求；swing 用子树侧。

        lever = 负载侧 com→关节锚点在**垂直于关节轴水平
        分量**方向的距离（重力矩 = m·g·lever；与轴平行的
        偏移不产生该轴力矩）。竖直轴关节直立时无重力负载。
        """
        if stance and n in LEG:
            bodies = ra - sub_of[n]
        else:
            bodies = sub_of[n]
        c, ml = _com_side(m, d, bodies)
        anchor = np.asarray(d.xanchor[jid[n]])
        off = c[:2] - anchor[:2]
        e3 = np.asarray(d.xmat[jb[n]]).reshape(3, 3) \
            @ np.asarray(m.jnt_axis[jid[n]])          # 世界系轴
        e = e3[:2]
        ne = np.linalg.norm(e)
        if ne < 1e-6:
            return 0.0, 0.0, ml
        e /= ne
        lever = float(abs(off @ np.array([-e[1], e[0]])))
        return ml * 9.81 * lever, lever, ml

    sel = ["abdomen_y", "abdomen_x", "hip_y_right", "knee_right",
           "ankle_y_right", "hip_x_left", "shoulder1_right", "elbow_right"]

    def report(label):
        print(f"\n== {label} ==  (单腿负载=保守上界，双腿对称载≈减半)")
        print(f"{'joint':<16}{'m_load':>7}{'lever':>7}{'tau_req':>8}"
              f"{'tau_max':>8}  单腿判定")
        for n in sel:
            rq, lev, ml = req(n)
            ok = "X不足" if rq > tmax[names.index(n)] else "ok"
            print(f"{n:<16}{ml:>7.1f}{lev:>7.3f}{rq:>8.1f}"
                  f"{tmax[names.index(n)]:>8.1f}  {ok}")

    print()
    print("=" * 78)
    print("表3  工况需求力矩（负载侧重力矩 = m_load·g·lever）")
    print("=" * 78)
    mujoco.mj_forward(m, d)
    report("站立")
    for ts, tt in [(0.4, 0.4), (0.8, 0.8), (1.0, 1.0)]:
        rad_off = {}
        for side in ("right", "left"):
            rad_off[f"knee_{side}"] = -(ts + tt)
            rad_off[f"hip_y_{side}"] = -tt
            rad_off[f"ankle_y_{side}"] = ts
        dz = (L1 + L2) - (L1 * np.cos(tt) + L2 * np.cos(ts))
        dx = -(L2 * np.sin(ts) - L1 * np.sin(tt))
        set_pose(rad_off, dx=dx, dz=-dz)
        report(f"躯干竖直深蹲 θs={ts} θt={tt}（膝弯 {ts + tt:.1f}rad）")
    set_pose({"abdomen_y": -0.6})
    report("腰前弯 0.6 rad")
    set_pose({"abdomen_y": -1.0})
    report("腰前弯 1.0 rad")
    set_pose({"abdomen_x": 0.5})
    report("腰侧弯 0.5 rad")


if __name__ == "__main__":
    main()
