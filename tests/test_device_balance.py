"""E5 W2 — basic_balance 设备单元与 CPU 版的同输入回放对照。

模式对齐 ``test_device_standup.py`` Layer A：相同注入态（contacts/
xpos/xquat/qpos/xanchor）分别喂 CPU 单元（伪 accessor）与设备单元
（ContactFlatNamespace + SimNamespace），逐字段对拍。

玩具拓扑与 standup 测试共享：geom0=ground(aff0)、geom1=wall(aff0)、
geom2..7→body2..7(aff1=robot_a)、geom8..13→body8..13(aff2=robot_b)。
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

project_root = Path(__file__).resolve().parent.parent

from envs.batchframework.device_state import (  # noqa: E402
    ContactFlatNamespace, DeviceBatchState, IoNamespace,
    RngNamespace, SimNamespace, alloc_episode_namespace,
)
from envs.batchframework.device_plugin import DeviceCtx  # noqa: E402
from envs.batchframework.device_balance import (  # noqa: E402
    DeviceCrossSupportObserver, DeviceDualImbalancePlugin,
    DeviceHeightPhiObserver, DevicePostureObserver,
)
from baseline.humanoid21.plugins.imbalance_termination import (  # noqa: E402
    DualImbalanceTerminationPlugin,
)
from baseline.humanoid21.plugins.height_phi_observer import (  # noqa: E402
    HeightPhiObserver,
)
from baseline.humanoid21.rewards.cross_support import (  # noqa: E402
    CrossSupportBalanceRewarder,
)
from baseline.humanoid21.rewards.posture_reward import (  # noqa: E402
    PostureRewarder,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 玩具拓扑（与 test_device_standup 同布局）
# ---------------------------------------------------------------------------
NGEOM, NBODY = 14, 14
GROUND_GID, WALL_GID = 0, 1
TORSO, HAND_L, HAND_R, FOOT_L, FOOT_R, OTHER_A, OTHER_A2 = 1, 2, 3, 4, 5, 6, 7
TORSO_B, HAND_L_B, FOOT_L_B, OTHER_B = 8, 9, 10, 11
FOOT_IDS = {0: (FOOT_L, FOOT_R), 1: (FOOT_L_B, 13)}
NAMES_A = {TORSO: "torso_a", HAND_L: "hand_left_a", HAND_R: "hand_right_a",
           FOOT_L: "foot_left_a", FOOT_R: "foot_right_a",
           OTHER_A: "knee_left_a", OTHER_A2: "elbow_right_a"}
NAMES_B = {TORSO_B: "torso_b", HAND_L_B: "hand_left_b",
           FOOT_L_B: "foot_left_b", OTHER_B: "head_b"}

GEOM_BODY = np.zeros(NGEOM, dtype=np.int64)
GEOM_BODY[2:8] = [HAND_L, HAND_R, FOOT_L, FOOT_R, OTHER_A, OTHER_A2]
GEOM_BODY[8:14] = [HAND_L_B, TORSO_B, FOOT_L_B, OTHER_B, 12, 13]
GEOM_AFF = np.zeros(NGEOM, dtype=np.int64)
GEOM_AFF[2:8] = 1
GEOM_AFF[8:14] = 2
GEOM_NAME = {0: "ground", 1: "wall"}

# 玩具 robot 布局：qpos 40 宽（root 自由关节 robot_a 占 0..6、robot_b
# 占 8..14），关节 norm 维度=21（PostureRewarder.STANDING_JOINT_POS 是
# 真模型 21 维——测试直接引用该常量）。
# nj=8（0=root, 1=hip_x_left, 2=ankle_x_left, 3=ankle_x_right, ...）
NQ, NJ = 40, 8
ROOT_ADR = {0: 0, 1: 8}           # robot_b 的 root 在 qpos 8..14（玩具）
QPOS_IDX = np.arange(0, 21)       # 玩具 norm 维度 21
QVEL_IDX = np.arange(0, 21)
ANKLE_L_JID, ANKLE_R_JID = 2, 3


def _geom_of_body(body):
    return int(np.nonzero(GEOM_BODY == body)[0][0])


def _contacts_vec(rows):
    """rows: [(geom1, geom2, force)] → CPU contacts SoA dict。"""
    n = len(rows)
    cv = dict(ncon=n,
              geom1=np.zeros(n, np.int32), geom2=np.zeros(n, np.int32),
              body1=np.zeros(n, np.int32), body2=np.zeros(n, np.int32),
              aff1=np.zeros(n, np.int8), aff2=np.zeros(n, np.int8),
              force_mag=np.zeros(n, np.float32))
    for i, (g1, g2, f) in enumerate(rows):
        cv["geom1"][i], cv["geom2"][i] = g1, g2
        cv["body1"][i], cv["body2"][i] = GEOM_BODY[g1], GEOM_BODY[g2]
        cv["aff1"][i], cv["aff2"][i] = GEOM_AFF[g1], GEOM_AFF[g2]
        cv["force_mag"][i] = f
    return cv


def _efc_rows(fz):
    """force_world=(0,0,fz) 的 condim=3 四行 efc（见 standup 测试）。"""
    return np.array([fz / 4, fz / 4, fz / 4, fz / 4], dtype=np.float32)


def make_device_state(B, contacts_rows=None, qpos=None, qvel=None,
                      xpos=None, xquat=None, xanchor=None):
    """contacts_rows: [(world, geom1, geom2, force)]；其余张量缺省置零。"""
    t = lambda a, dt=torch.float32: torch.as_tensor(  # noqa: E731
        np.asarray(a), dtype=dt, device=DEV)
    rows = contacts_rows or []
    M = len(rows)
    worldid = np.array([r[0] for r in rows] or [0], np.int32)
    geom = np.array([[r[1], r[2]] for r in rows] or [[0, 0]], np.int32)
    dist = np.zeros(max(M, 1), np.float32)
    dist[M:] = np.inf
    efc_force = np.zeros((B, max(M, 1) * 4), np.float32)
    for i, r in enumerate(rows):
        efc_force[r[0], i * 4:i * 4 + 4] = _efc_rows(r[3])
    frame = np.zeros((max(M, 1), 3, 3), np.float32)
    frame[:, 0, 0] = frame[:, 1, 1] = frame[:, 2, 2] = 1.0
    contacts = ContactFlatNamespace(
        worldid=t(worldid, torch.int32), geom=t(geom, torch.int32),
        dist=t(dist), pos=t(np.zeros((max(M, 1), 3), np.float32)),
        frame=t(frame), dim=t(np.full(max(M, 1), 3, np.int32),
                              torch.int32),
        efc_address=t(np.arange(max(M, 1)) * 4, torch.int32),
        efc_force=t(efc_force),
        n_active=t(np.int32(M), torch.int32), cap_per_world=64)
    sim_ns = SimNamespace(
        qpos=t(np.zeros((B, NQ), np.float32) if qpos is None else qpos),
        qvel=t(np.zeros((B, NQ), np.float32) if qvel is None else qvel),
        ctrl=t(np.zeros((B, 4), np.float32)),
        xpos=t(np.zeros((B, NBODY, 3), np.float32)
               if xpos is None else xpos),
        xquat=t(np.tile(np.array([1., 0, 0, 0], np.float32),
                        (B, NBODY, 1)) if xquat is None else xquat),
        xipos=t(np.zeros((B, NBODY, 3), np.float32)),
        xanchor=t(np.zeros((B, NJ, 3), np.float32)
                  if xanchor is None else xanchor),
        cvel=t(np.zeros((B, NBODY, 6), np.float32)),
        xfrc_applied=t(np.zeros((B, NBODY, 6), np.float32)),
        act_target=t(np.zeros((B, 4), np.float32)),
        contacts_flat=contacts)
    ep = alloc_episode_namespace(B, DEV)
    io = IoNamespace(action_a=torch.zeros(B, 21, device=DEV),
                     action_b=torch.zeros(B, 21, device=DEV))
    rng = RngNamespace(
        seed_offsets=torch.zeros(B, dtype=torch.int64, device=DEV),
        step_counter=torch.zeros((), dtype=torch.int64, device=DEV))
    return DeviceBatchState(B, sim_ns, ep, io, rng)


def _contacts_ns(B, rows):
    """[(world, g1, g2, force)] → ContactFlatNamespace。"""
    t = lambda a, dt=torch.float32: torch.as_tensor(  # noqa: E731
        np.asarray(a), dtype=dt, device=DEV)
    M = len(rows)
    worldid = np.array([r[0] for r in rows] or [0], np.int32)
    geom = np.array([[r[1], r[2]] for r in rows] or [[0, 0]], np.int32)
    dist = np.zeros(max(M, 1), np.float32)
    dist[M:] = np.inf
    efc_force = np.zeros((B, max(M, 1) * 4), np.float32)
    for i, r in enumerate(rows):
        efc_force[r[0], i * 4:i * 4 + 4] = _efc_rows(r[3])
    frame = np.zeros((max(M, 1), 3, 3), np.float32)
    frame[:, 0, 0] = frame[:, 1, 1] = frame[:, 2, 2] = 1.0
    return ContactFlatNamespace(
        worldid=t(worldid, torch.int32), geom=t(geom, torch.int32),
        dist=t(dist), pos=t(np.zeros((max(M, 1), 3), np.float32)),
        frame=t(frame), dim=t(np.full(max(M, 1), 3, np.int32),
                              torch.int32),
        efc_address=t(np.arange(max(M, 1)) * 4, torch.int32),
        efc_force=t(efc_force),
        n_active=t(np.int32(M), torch.int32), cap_per_world=64)


def set_step_state(st, contacts_rows=None, qpos=None, xpos=None,
                   xquat=None, xanchor=None):
    """原位更新已有 state 的可变输入（插件 pool/episode 保留）。"""
    if contacts_rows is not None:
        st.sim.contacts_flat = _contacts_ns(st.batch_size, contacts_rows)
    for name, arr in (("qpos", qpos), ("xpos", xpos), ("xquat", xquat),
                      ("xanchor", xanchor)):
        if arr is not None:
            getattr(st.sim, name).copy_(
                torch.as_tensor(np.asarray(arr), dtype=torch.float32,
                                device=st.sim.qpos.device))


def _geom_tables(agent_idx):
    """设备单元的 from_sim 手工表（绕开真 sim，注入玩具 id）。"""
    rid = "robot_a" if agent_idx == 0 else "robot_b"
    fl, fr = FOOT_IDS[agent_idx]
    return {
        "ground_gid": GROUND_GID,
        "geom_bodyid": torch.as_tensor(GEOM_BODY, device=DEV),
        "geom_aff": torch.as_tensor(GEOM_AFF, device=DEV),
        "robot_aff": agent_idx + 1,
        "foot_l": fl, "foot_r": fr,
        "foot_ids": (fl, fr),
        "root_adr": ROOT_ADR[agent_idx],
        "ankle_l_jid": ANKLE_L_JID, "ankle_r_jid": ANKLE_R_JID,
    }


class _CpuCtx(SimpleNamespace):
    """CPU 侧伪 SimContext——带 accessor + 终止收集。"""

    def __init__(self, accessor, terminated):
        super().__init__(accessor=accessor)
        self.requests = []                    # (reason, agent_id)
        self._terminated = terminated         # {"robot_a":bool,"robot_b":bool}

    def request_termination(self, reason, agent_id=None):
        self.requests.append((reason, agent_id))
        if agent_id is None:
            for k in self._terminated:
                self._terminated[k] = True
        else:
            self._terminated[agent_id] = True

    def is_agent_terminated(self, agent_id):
        return self._terminated[agent_id]


# ===========================================================================
# DeviceDualImbalancePlugin
# ===========================================================================
class TestDualImbalance:
    """逐 agent 终止：reason/时序/目标 agent 三要素对拍。"""

    def _run_cpu(self, seqs, tolerance=2, force_threshold=1.0,
                 min_height=0.0, heights=None):
        """seqs: List[steps][env] of contact rows。heights: (steps,B,2)。"""
        T = len(seqs)
        B = len(seqs[0])
        terminated = [{"robot_a": False, "robot_b": False}
                      for _ in range(B)]
        reqs = [[] for _ in range(B)]
        for e in range(B):
            plug = DualImbalanceTerminationPlugin(
                force_threshold=force_threshold, tolerance=tolerance,
                min_height=min_height)          # CPU 语义：每 env 一实例
            ctx0 = _CpuCtx(SimpleNamespace(get_static_data=lambda: {
                "ground_geom_name": "ground"}), terminated[e])
            plug.on_pre_episode(ctx0)
            for t in range(T):
                cv = _contacts_vec(seqs[t][e])

                class _Acc:
                    def get_derived_state(self_, fields=None):
                        return {"contacts": cv}

                    def get_static_data(self_):
                        return {"body_id_to_name": {
                                    b: n for b, n in
                                    {**NAMES_A, **NAMES_B}.items()},
                                "geom_id_to_name": GEOM_NAME}

                    def get_core_state(self_):
                        h = heights[t][e] if heights is not None else None
                        if h is None:
                            return {"robot_a": {"root_pos": np.zeros(3)},
                                    "robot_b": {"root_pos": np.zeros(3)}}
                        return {"robot_a": {
                                    "root_pos": np.array([0, 0, h[0]])},
                                "robot_b": {
                                    "root_pos": np.array([0, 0, h[1]])}}

                ctx = _CpuCtx(_Acc(), terminated[e])
                plug.on_post_action_step(ctx)
                reqs[e].extend(ctx.requests)
        return reqs, terminated

    def _run_device(self, seqs, tolerance=2, force_threshold=1.0,
                    min_height=0.0, heights=None):
        T = len(seqs)
        B = len(seqs[0])
        plug = DeviceDualImbalancePlugin(
            force_threshold=force_threshold, tolerance=tolerance,
            min_height=min_height,
            tables={
                "ground_gid": GROUND_GID,
                "geom_bodyid": torch.as_tensor(GEOM_BODY, device=DEV),
                "geom_aff": torch.as_tensor(GEOM_AFF, device=DEV),
                "foot_ids": [FOOT_IDS[0], FOOT_IDS[1]],
                "root_adr": [ROOT_ADR[0], ROOT_ADR[1]],
            })
        st = make_device_state(B)
        plug.declare_state(st)
        ep = st.episode
        ep.episode_steps.zero_()
        ep.world_running.fill_(True)
        for t in range(T):
            rows = []
            for e in range(B):
                rows += [(e, g1, g2, f) for g1, g2, f in seqs[t][e]]
            qpos = np.zeros((B, NQ), np.float32)
            for e in range(B):
                if heights is not None:
                    qpos[e, ROOT_ADR[0] + 2] = heights[t][e][0]
                    qpos[e, ROOT_ADR[1] + 2] = heights[t][e][1]
            set_step_state(st, contacts_rows=rows, qpos=qpos)
            ctx = DeviceCtx(st, plugin_name=plug.name)
            plug.on_post_action_step(ctx)
            ep.episode_steps.add_(1)
        return ep

    def test_asymmetric_termination(self):
        """env0 的 robot_a 连续失衡 → 只终 A；env 仍 RUNNING（B 未终）。"""
        tol = 2
        g_other = _geom_of_body(OTHER_A)
        # B=1：env0 robot_a 每步都有非足接触；robot_b 无接触
        seqs = [[[(GROUND_GID, g_other, 50.0)]] for _ in range(4)]
        reqs, term = self._run_cpu(seqs, tolerance=tol)
        ep = self._run_device(seqs, tolerance=tol)
        assert reqs[0] == [("imbalance_robot_a", "robot_a")]
        assert ep.agent_done[0].tolist() == [True, False]
        assert ep.agent_term_reason[0, 0].item() == \
            ep.reason_registry["imbalance_robot_a"]

    def test_counter_decay(self):
        """命中-空-命中-命中：counter 衰减语义与 CPU 一致（tol=2）。"""
        g = _geom_of_body(OTHER_A)
        hit = [(GROUND_GID, g, 50.0)]
        seqs = [[hit], [[]], [hit], [hit]]       # B=1, T=4
        reqs, _ = self._run_cpu(seqs, tolerance=2)
        ep = self._run_device(seqs, tolerance=2)
        assert reqs[0] == [("imbalance_robot_a", "robot_a")]
        assert ep.agent_done[0].tolist() == [True, False]

    def test_min_height_gate(self):
        """双机均低于 min_height → 不计数（CPU all_below 早退）。"""
        g = _geom_of_body(OTHER_A)
        hit = [(GROUND_GID, g, 50.0)]
        seqs = [[hit] for _ in range(3)]
        heights = np.zeros((3, 1, 2), np.float32)   # 两机 h=0 < gate
        reqs, _ = self._run_cpu(seqs, tolerance=1, min_height=0.5,
                                heights=heights)
        ep = self._run_device(seqs, tolerance=1, min_height=0.5,
                              heights=heights)
        assert reqs[0] == []
        assert not ep.agent_done.any()

    def test_force_threshold(self):
        """force < threshold 的接触不计（=1.0 恰好计入）。"""
        g = _geom_of_body(OTHER_A)
        seqs = [[[(GROUND_GID, g, 0.99)]],
                [[(GROUND_GID, g, 1.0)]],
                [[(GROUND_GID, g, 1.0)]]]
        reqs, _ = self._run_cpu(seqs, tolerance=2, force_threshold=1.0)
        ep = self._run_device(seqs, tolerance=2, force_threshold=1.0)
        assert reqs[0] == [("imbalance_robot_a", "robot_a")]
        assert ep.agent_done[0, 0].item() is True

    def test_other_robot_contact_ignored(self):
        """robot_b 的失衡接触不影响 robot_a 的判定。"""
        g_b = _geom_of_body(OTHER_B)
        seqs = [[[(GROUND_GID, g_b, 50.0)], [(GROUND_GID, g_b, 50.0)]]]
        reqs, _ = self._run_cpu(seqs, tolerance=1)
        ep = self._run_device(seqs, tolerance=1)
        assert reqs[0] == [("imbalance_robot_b", "robot_b")]
        assert ep.agent_done[0].tolist() == [False, True]


# ===========================================================================
# DeviceCrossSupportObserver
# ===========================================================================
class TestCrossSupport:
    """状态机逐步对拍：相同接触序列 → 两侧每步 reward 一致。"""

    THR = 0.067 + 0.06      # STANDING_FOOT_HEIGHT + foot_lift_min_height

    def _cpu_step(self, plugin, lc, rc, ankle_l, ankle_r, root_h=1.0):
        rows = []
        if lc:
            rows.append((GROUND_GID, _geom_of_body(FOOT_L), 50.0))
        if rc:
            rows.append((GROUND_GID, _geom_of_body(FOOT_R), 50.0))
        cv = _contacts_vec(rows)

        class _Acc:
            def get_derived_state(self_, fields=None):
                fields = fields or []
                out = {}
                if "contacts" in fields or not fields:
                    out["contacts"] = cv
                jwa = {"ankle_x_left_a": np.array([0, 0, ankle_l],
                                                  np.float32),
                       "ankle_x_right_a": np.array([0, 0, ankle_r],
                                                   np.float32)}
                out["robot_a"] = {"joint_world_anchor": jwa}
                return out

            def get_static_data(self_):
                return {"body_id_to_name": {
                            b: n for b, n in NAMES_A.items()},
                        "geom_id_to_name": GEOM_NAME,
                        "ground_geom_name": "ground"}

            def get_core_state(self_):
                return {"robot_a": {
                    "root_pos": np.array([0, 0, root_h], np.float32)}}

        ctx = SimpleNamespace(accessor=_Acc())
        plugin.on_post_action_step(ctx)
        return plugin.get_output()

    def _dev_step(self, obs, lc, rc, ankle_l, ankle_r, root_h=1.0):
        rows = []
        if lc:
            rows.append((0, GROUND_GID, _geom_of_body(FOOT_L), 50.0))
        if rc:
            rows.append((0, GROUND_GID, _geom_of_body(FOOT_R), 50.0))
        xanchor = np.zeros((1, NJ, 3), np.float32)
        xanchor[0, ANKLE_L_JID, 2] = ankle_l
        xanchor[0, ANKLE_R_JID, 2] = ankle_r
        qpos = np.zeros((1, NQ), np.float32)
        qpos[0, ROOT_ADR[0] + 2] = root_h
        st = make_device_state(1, contacts_rows=rows, qpos=qpos,
                               xanchor=xanchor)
        obs.on_post_action_step(DeviceCtx(st, plugin_name="xsup"))
        return float(obs.get_output()["reward"][0].item())

    def _run_both(self, seq):
        """seq: [(lc, rc, ankle_l, ankle_r)] 逐步对拍。"""
        cpu = CrossSupportBalanceRewarder(agent_id="robot_a")
        cpu.on_pre_episode(SimpleNamespace(accessor=SimpleNamespace(
            get_static_data=lambda: {"ground_geom_name": "ground"})))
        obs = DeviceCrossSupportObserver(
            0, tables=dict(_geom_tables(0)))
        st0 = make_device_state(1)
        obs.on_pre_episode(DeviceCtx(st0, plugin_name="xsup"))
        got, exp = [], []
        for lc, rc, al, ar in seq:
            exp.append(self._cpu_step(cpu, lc, rc, al, ar))
            got.append(self._dev_step(obs, lc, rc, al, ar))
        return np.array(got), np.array(exp)

    HI, LO = 0.5, 0.0       # ankle 高/低（LO < THR → 视为仍接触）

    def test_grace_period_overrun(self):
        """开局双足支撑超过 grace → 线性 initial 惩罚（状态机 WAIT）。"""
        seq = [(True, True, self.LO, self.LO)] * 35
        got, exp = self._run_both(seq)
        np.testing.assert_allclose(got, exp, atol=1e-6)
        assert exp[-1] < 0                     # 已开始惩罚

    def test_first_single_support_entry(self):
        """双足 → 左单脚（右脚抬起且踝高过阈）→ 进入 tracking。"""
        seq = ([(True, True, self.LO, self.LO)] * 5
               + [(True, False, self.LO, self.HI)] * 6)
        got, exp = self._run_both(seq)
        np.testing.assert_allclose(got, exp, atol=1e-6)
        assert np.all(exp == 0)                # grace 内 + tracking 无罚

    def test_short_segment_penalty(self):
        """左单脚 < foot_lift_min_steps 段结束 → 过短惩罚按 deficit 计。"""
        seq = ([(True, True, self.LO, self.LO)] * 2
               + [(True, False, self.LO, self.HI)] * 2      # 段长 2 < 4
               + [(True, True, self.LO, self.LO)] * 2       # 段结束
               + [(False, True, self.HI, self.LO)] * 5)     # 换脚
        got, exp = self._run_both(seq)
        np.testing.assert_allclose(got, exp, atol=1e-6)

    def test_switch_interval_penalty(self):
        """左单脚长段后无对侧出现 → 换脚间隔超 max 时惩罚（右侧首现结算）。"""
        seq = ([(True, False, self.LO, self.HI)] * 20       # 左段 20>18
               + [(False, True, self.HI, self.LO)] * 3)     # 对侧首现→结算
        got, exp = self._run_both(seq)
        np.testing.assert_allclose(got, exp, atol=1e-6)
        assert exp[-3] < 0                                 # 结算步有罚

    def test_micro_lift_still_contact(self):
        """脚抬起但踝高 < 阈值 → 仍算接触（micro-lift 不算抬脚）。"""
        seq = [(True, False, self.LO, self.LO)] * 8        # 右踝低→仍算双足
        got, exp = self._run_both(seq)
        np.testing.assert_allclose(got, exp, atol=1e-6)


# ===========================================================================
# DevicePostureObserver
# ===========================================================================
class TestPosture:
    """四叶输出与 CPU 逐字段一致（norm/tilt/foot_height）。"""

    def _tables(self, agent_idx=0):
        kp = dict(_geom_tables(agent_idx))
        kp.update({
            "torso": TORSO,
            "qpos_idx": torch.as_tensor(QPOS_IDX, dtype=torch.long,
                                        device=DEV),
            "qvel_idx": torch.as_tensor(QVEL_IDX, dtype=torch.long,
                                        device=DEV),
            "norm_ref": torch.full((21,), 0.1, device=DEV),
            "norm_scale": torch.full((21,), 2.0, device=DEV),
        })
        return kp

    def test_outputs_match_cpu(self):
        B = 2
        qpos = np.zeros((B, NQ), np.float32)
        qvel = np.zeros((B, NQ), np.float32)
        xpos = np.zeros((B, NBODY, 3), np.float64)
        xquat = np.tile(np.array([1., 0, 0, 0], np.float32),
                        (B, NBODY, 1))
        qpos[0, QPOS_IDX] = np.linspace(-0.4, 0.6, 21)
        qvel[0, QVEL_IDX] = np.linspace(-0.5, 0.9, 21)
        xpos[0, FOOT_L, 2] = 0.12
        xpos[0, FOOT_R, 2] = 0.34
        # env1: 躯干侧倾（xquat 绕 y 转 60°）
        ang = np.deg2rad(30.0)
        xquat[1, TORSO] = [np.cos(ang), 0, np.sin(ang), 0]
        xpos[1, FOOT_L, 2] = 0.05
        xpos[1, FOOT_R, 2] = 0.05

        st = make_device_state(B, qpos=qpos, qvel=qvel, xpos=xpos,
                               xquat=xquat)
        obs = DevicePostureObserver(0, tables=self._tables(0))
        obs.on_post_action_step(DeviceCtx(st, plugin_name="posture"))
        dev_out = {k: v.cpu().numpy() for k, v in obs.get_output().items()}

        # CPU 侧：喂相同数值（norm 已由公式得到 → 直接给归一化值）
        norm = self._tables()["norm_ref"].cpu().numpy(), \
            self._tables()["norm_scale"].cpu().numpy()
        jpn = (qpos[:, QPOS_IDX] - norm[0]) / norm[1]
        jvn = qvel[:, QVEL_IDX] / norm[1]

        class _Acc:
            def __init__(self, e):
                self.e = e

            def get_core_state(self_):
                return {"robot_a": {
                    "joint_pos_norm": jpn[self_.e],
                    "joint_vel_norm": jvn[self_.e]}}

            def get_derived_state(self_, fields=None):
                up = np.array([xquat_to_r22(xquat[self_.e, TORSO])],
                              np.float32)
                return {"robot_a": {
                    "uprightness": up,
                    "body_xpos": {
                        NAMES_A[FOOT_L]: xpos[self_.e, FOOT_L],
                        NAMES_A[FOOT_R]: xpos[self_.e, FOOT_R]}}}

            def get_static_data(self_):
                return {"robot_a": {"keypoint_body_names": {
                    "foot_left": NAMES_A[FOOT_L],
                    "foot_right": NAMES_A[FOOT_R]}}}

        for e in range(B):
            cpu = PostureRewarder(agent_id="robot_a")
            cpu.on_pre_episode(SimpleNamespace())
            cpu.on_post_action_step(SimpleNamespace(accessor=_Acc(e)))
            out = cpu.get_output()
            for k in ("joint_deviation", "joint_vel", "torso_tilt",
                      "foot_height"):
                assert dev_out[k][e] == pytest.approx(out[k], abs=1e-5), k


def xquat_to_r22(q):
    w, x, y, z = q
    return 1.0 - 2.0 * (x * x + y * y)


# ===========================================================================
# DeviceHeightPhiObserver
# ===========================================================================
class TestHeightPhi:
    """phi/initial_phi 与 CPU 一致；partial reset 重捕获 initial_phi。"""

    def _tables(self, agent_idx=0):
        return {"torso": TORSO, "root_adr": ROOT_ADR[agent_idx]}

    def _state(self, heights, tilts_deg):
        B = len(heights)
        qpos = np.zeros((B, NQ), np.float32)
        xquat = np.tile(np.array([1., 0, 0, 0], np.float32),
                        (B, NBODY, 1))
        for e in range(B):
            qpos[e, ROOT_ADR[0] + 2] = heights[e]
            a = np.deg2rad(tilts_deg[e] / 2)
            xquat[e, TORSO] = [np.cos(a), 0, np.sin(a), 0]
        return make_device_state(B, qpos=qpos, xquat=xquat)

    def test_phi_and_initial(self):
        cpu = HeightPhiObserver(agent_id="robot_a", standing_height=1.28)
        dev = DeviceHeightPhiObserver(0, standing_height=1.28,
                                      tables=self._tables())
        # episode 起点：h=1.28 直立
        st0 = self._state([1.28], [0.0])
        ctx0 = DeviceCtx(st0, plugin_name="phi")

        class _Acc0:
            def get_core_state(self_):
                return {"robot_a": {
                    "root_pos": np.array([0, 0, 1.28], np.float32)}}

            def get_derived_state(self_, fields=None):
                return {"robot_a": {"uprightness": np.array([1.0],
                                                            np.float32)}}

        cpu.on_pre_episode(SimpleNamespace(accessor=_Acc0()))
        dev.on_pre_episode(ctx0)

        # 步进：h=0.9，tilt=30°
        st1 = self._state([0.9], [60.0])   # tilt角=2*30=60° → R22=cos60
        class _Acc1:
            def get_core_state(self_):
                return {"robot_a": {
                    "root_pos": np.array([0, 0, 0.9], np.float32)}}

            def get_derived_state(self_, fields=None):
                return {"robot_a": {"uprightness": np.array(
                    [np.cos(np.deg2rad(60.0))], np.float32)}}

        cpu.on_post_action_step(SimpleNamespace(accessor=_Acc1()))
        dev.on_post_action_step(DeviceCtx(st1, plugin_name="phi"))
        out = cpu.get_output()
        for k in ("height", "uprightness", "phi", "initial_phi"):
            got = float(dev.get_output()[k][0].item())
            assert got == pytest.approx(out[k], abs=1e-5), (
                k, got, out[k])

    def test_envs_reset_recapture(self):
        """partial reset 行重捕获 initial_phi（CPU on_pre_episode 同义）。"""
        dev = DeviceHeightPhiObserver(0, tables=self._tables())
        st0 = self._state([1.28, 1.28], [0.0, 0.0])
        dev.on_pre_episode(DeviceCtx(st0, plugin_name="phi"))
        # env1 reset 到 h=0.5/tilt90 的姿态
        st1 = self._state([1.28, 0.5], [0.0, 90.0])
        ctx = DeviceCtx(st1, plugin_name="phi")
        ctx.reset_env_ids = torch.tensor([1], device=DEV)
        dev.on_envs_reset(ctx)
        dev.on_post_action_step(ctx)
        out = dev.get_output()
        assert out["initial_phi"][0].item() == pytest.approx(1.0)
        assert out["initial_phi"][1].item() == pytest.approx(
            0.0 * (0.5 / 1.28), abs=1e-6)
