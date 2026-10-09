"""step 实验设备单元对照测试（GaitClockSimulator 绑定 + FootStateObserver）。

三层验证（对齐 M4_PLAN / test_device_standup.py 的验收哲学）：

- **公式/波契约（无 GPU）**：``_gait_clock_cols`` 与 CPU
  ``GaitClockSimulator.gait_clock`` 逐帧同值（含半窗边界、奇数周期、
  per-env 相位分叉）；``BatchRuntime`` + 真 obs_builder 验证
  ``episode_steps`` 簿记耦合（步尾自增→obs 构造读到本步帧索引，
  部分 reset 行归零分叉）。

- **Layer A（纯逻辑）**：相同注入张量态（xipos/xpos/xquat/contacts），
  CPU ``FootStateObserver``（伪 accessor 喂相同数值）vs
  ``DeviceFootStateObserver``。无物理参与 → 期望 fp32 舍入级一致。
  覆盖：站立/抬足/绕趾摇摆（midpoint 升但 sole 净空≈0）、有/无接触、
  wall/robot-robot/他机足接触排除、bool 输出叶。

- **Layer B（真机，CUDA）**：相同 qpos/qvel 注入 CPU
  ``Humanoid21Simulator`` 与 ``WarpHumanoid21Simulator``，各步进
  1 子步后对照；``WarpGaitClockSimulator`` 真机装配检查 obs_dim=99。
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

project_root = Path(__file__).resolve().parent.parent

from envs.batchframework.device_state import (  # noqa: E402
    ContactFlatNamespace, SimNamespace, compose_state)
from envs.batchframework.device_plugin import DeviceCtx  # noqa: E402
from envs.batchframework.device_runtime import BatchRuntime  # noqa: E402
from envs.batchframework.device_step import (  # noqa: E402
    DeviceFootStateObserver, GaitClockObsBuilder, _gait_clock_cols)
from baseline.humanoid21.end2end.foot_state_observer import (  # noqa: E402
    FOOT_ENDPOINTS_LOCAL, FOOT_GEOM_RADIUS, STANDING_FOOT_Z,
    FootStateObserver)
from baseline.humanoid21.end2end.gait_clock_simulator import (  # noqa: E402
    GaitClockSimulator)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 玩具场景拓扑（两实现共享同一套 id/name 表）
# ---------------------------------------------------------------------------
# geoms: 0=ground(aff0)  1=wall(aff0)  2..7→body(aff1)  8..13→body(aff2)
NGEOM, NBODY = 14, 14
GROUND_GID, WALL_GID = 0, 1
TORSO, HAND_L, HAND_R, FOOT_L, FOOT_R, OTHER_A, OTHER_A2 = 1, 2, 3, 4, 5, 6, 7
TORSO_B, HAND_L_B, FOOT_L_B, FOOT_R_B = 8, 9, 10, 11

NAMES_A = {TORSO: "torso_a", HAND_L: "hand_left_a", HAND_R: "hand_right_a",
           FOOT_L: "foot_left_a", FOOT_R: "foot_right_a",
           OTHER_A: "knee_left_a", OTHER_A2: "elbow_right_a"}
NAMES_B = {TORSO_B: "torso_b", HAND_L_B: "hand_left_b",
           FOOT_L_B: "foot_left_b", FOOT_R_B: "foot_right_b"}

GEOM_BODY = np.zeros(NGEOM, dtype=np.int64)
GEOM_BODY[2:8] = [HAND_L, HAND_R, FOOT_L, FOOT_R, OTHER_A, OTHER_A2]
GEOM_BODY[8:14] = [TORSO_B, HAND_L_B, FOOT_L_B, FOOT_R_B, 12, 13]
GEOM_AFF = np.zeros(NGEOM, dtype=np.int64)
GEOM_AFF[2:8] = 1
GEOM_AFF[8:14] = 2
GEOM_NAME = {0: "ground", 1: "wall"}


def _geom_of_body(body):
    return int(np.nonzero(GEOM_BODY == body)[0][0])


# ---------------------------------------------------------------------------
# CPU 侧伪上下文（Feed FootStateObserver 相同数值）
# ---------------------------------------------------------------------------
class _FakeAccessor:
    def __init__(self, xipos, xpos, xquat, contacts_vec):
        self._xipos, self._xpos, self._xquat, self._cv = \
            xipos, xpos, xquat, contacts_vec

    def get_derived_state(self, fields=None):
        fields = list(fields) if fields else []
        out = {}
        for f in fields:
            if f == "contacts":
                out["contacts"] = self._cv
            elif f in ("robot_a", "robot_b"):
                names = NAMES_A if f == "robot_a" else NAMES_B
                out[f] = {
                    "body_xipos": {n: self._xipos[b].astype(np.float32)
                                   for b, n in names.items()},
                    "body_xpos": {n: self._xpos[b].astype(np.float32)
                                  for b, n in names.items()},
                    "body_xquat": {n: self._xquat[b].astype(np.float32)
                                   for b, n in names.items()},
                }
        return out

    def get_static_data(self):
        body_id_to_name = {b: n for b, n in
                           {**NAMES_A, **NAMES_B}.items()}
        return {
            "ground_geom_name": "ground",
            "body_id_to_name": body_id_to_name,
            "geom_id_to_name": GEOM_NAME,
        }


def _contacts_vec(rows):
    """rows: list of (geom1, geom2) → CPU SoA dict。"""
    n = len(rows)
    cv = dict(
        ncon=n,
        geom1=np.zeros(n, np.int32), geom2=np.zeros(n, np.int32),
        body1=np.zeros(n, np.int32), body2=np.zeros(n, np.int32),
        aff1=np.zeros(n, np.int8), aff2=np.zeros(n, np.int8),
    )
    for i, (g1, g2) in enumerate(rows):
        cv["geom1"][i], cv["geom2"][i] = g1, g2
        cv["body1"][i], cv["body2"][i] = GEOM_BODY[g1], GEOM_BODY[g2]
        cv["aff1"][i], cv["aff2"][i] = GEOM_AFF[g1], GEOM_AFF[g2]
    return cv


def run_cpu_observer(xipos, xpos, xquat, cv, agent_id):
    obs = FootStateObserver(agent_id=agent_id)
    ctx = SimpleNamespace(accessor=_FakeAccessor(xipos, xpos, xquat, cv))
    obs.on_pre_episode(ctx)
    obs.on_post_action_step(ctx)
    return obs.get_output()


# ---------------------------------------------------------------------------
# 设备侧注入：ContactFlatNamespace + SimNamespace
# ---------------------------------------------------------------------------
def _contacts_flat(contacts_per_env, B, device):
    """contacts_per_env: List[List[(geom1, geom2)]] → ContactFlatNamespace。"""
    rows = []
    for w, contacts in enumerate(contacts_per_env):
        for g1, g2 in contacts:
            rows.append((w, g1, g2))
    M = len(rows)
    worldid = np.array([r[0] for r in rows] or [0], np.int32)
    geom = np.array([[r[1], r[2]] for r in rows] or [[0, 0]], np.int32)
    dist = np.zeros(max(M, 1), np.float32)
    dist[M:] = np.inf                      # 非活跃槽位
    frame = np.zeros((max(M, 1), 3, 3), np.float32)
    frame[:, 0, 0] = frame[:, 1, 1] = frame[:, 2, 2] = 1.0

    t = lambda a, dt=torch.float32: torch.as_tensor(  # noqa: E731
        np.asarray(a), dtype=dt, device=device)
    return ContactFlatNamespace(
        worldid=t(worldid, torch.int32), geom=t(geom, torch.int32),
        dist=t(dist), pos=t(np.zeros((max(M, 1), 3), np.float32)),
        frame=t(frame), dim=t(np.full(max(M, 1), 3, np.int32), torch.int32),
        efc_address=t(np.arange(max(M, 1)) * 4, torch.int32),
        efc_force=t(np.zeros((B, max(M, 1) * 4), np.float32)),
        n_active=t(np.int32(M), torch.int32), cap_per_world=64)


def make_foot_state(contacts_per_env, xipos, xpos, xquat, device):
    """注入态 → DeviceBatchState（经 compose_state，与 runtime 同组装）。"""
    B = len(contacts_per_env)
    t = lambda a, dt=torch.float32: torch.as_tensor(  # noqa: E731
        np.asarray(a), dtype=dt, device=device)
    sim_ns = SimNamespace(
        qpos=t(np.zeros((B, 4), np.float32)),
        qvel=t(np.zeros((B, 4), np.float32)),
        ctrl=t(np.zeros((B, 4), np.float32)),
        xpos=t(np.asarray(xpos, np.float32)),
        xquat=t(np.asarray(xquat, np.float32)),
        xipos=t(np.asarray(xipos, np.float32)),
        xanchor=t(np.zeros((B, 4, 3), np.float32)),
        cvel=t(np.zeros((B, NBODY, 6), np.float32)),
        xfrc_applied=t(np.zeros((B, NBODY, 6), np.float32)),
        act_target=t(np.zeros((B, 4), np.float32)),
        contacts_flat=_contacts_flat(contacts_per_env, B, device),
    )
    return compose_state(B, sim_ns, torch.device(device),
                         action_dim=4, obs_dim=99)


def _foot_tables(agent_idx, device):
    if agent_idx == 0:
        fl, fr, aff = FOOT_L, FOOT_R, 1
    else:
        fl, fr, aff = FOOT_L_B, FOOT_R_B, 2
    return dict(
        foot_l=fl, foot_r=fr, ground_gid=GROUND_GID, agent_aff=aff,
        geom_bodyid=torch.as_tensor(GEOM_BODY, device=device),
        geom_aff=torch.as_tensor(GEOM_AFF, device=device))


def run_device_observer(state, agent_idx, device):
    obs = DeviceFootStateObserver(
        agent_idx, tables=_foot_tables(agent_idx, device))
    ctx = DeviceCtx(state, plugin_name="foot_state")
    obs.on_pre_episode(ctx)
    obs.on_post_action_step(ctx)
    return {k: v.detach().cpu().numpy()
            for k, v in obs.get_output().items()}


# ---------------------------------------------------------------------------
# 姿态构造
# ---------------------------------------------------------------------------
def _quat_pitch(theta):
    """绕 y 轴的 [w,x,y,z] 四元数。"""
    return np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])


def _base_scene(n_env):
    """站立位姿：双足体心平 z=STANDING_FOOT_Z、单位姿态（sole≈0）。"""
    xipos = np.zeros((n_env, NBODY, 3), np.float64)
    xpos = np.zeros((n_env, NBODY, 3), np.float64)
    xquat = np.zeros((n_env, NBODY, 4), np.float64)
    xquat[..., 0] = 1.0
    for foot in (FOOT_L, FOOT_R, FOOT_L_B, FOOT_R_B):
        xipos[:, foot, 2] = STANDING_FOOT_Z
        xpos[:, foot, 2] = STANDING_FOOT_Z
    return xipos, xpos, xquat


def _cpu_per_env(xipos, xpos, xquat, contacts_per_env, agent_id):
    return [run_cpu_observer(xipos[e], xpos[e], xquat[e],
                             _contacts_vec(c), agent_id)
            for e, c in enumerate(contacts_per_env)]


def _run_both(xipos, xpos, xquat, contacts_per_env, agent=0):
    aid = "robot_a" if agent == 0 else "robot_b"
    cpu_outs = _cpu_per_env(xipos, xpos, xquat, contacts_per_env, aid)
    state = make_foot_state(contacts_per_env, xipos, xpos, xquat, DEV)
    return cpu_outs, run_device_observer(state, agent, DEV)


def _assert_leaves(cpu_outs, dev_out, atol=1e-6):
    for k in ("h_left_foot", "h_right_foot",
              "sole_clear_left", "sole_clear_right"):
        exp = np.array([o[k] for o in cpu_outs], dtype=np.float32)
        np.testing.assert_allclose(
            dev_out[k], exp, rtol=1e-5, atol=atol,
            err_msg=f"{k}: dev={dev_out[k]} cpu={exp}")
    for k in ("left_foot_contact", "right_foot_contact"):
        exp = np.array([o[k] for o in cpu_outs], dtype=bool)
        np.testing.assert_array_equal(dev_out[k].astype(bool), exp)


# ===========================================================================
# 步态时钟：公式 + 波契约
# ===========================================================================
def _cpu_clock(period, t):
    """CPU gait_clock() 的免实例化调用（self 只需两字段）。"""
    sim = SimpleNamespace(gait_period=period, _action_step=int(t))
    return GaitClockSimulator.gait_clock(sim)


class TestGaitClockFormula:
    @pytest.mark.parametrize("period", [40, 7])
    def test_sweep_matches_cpu(self, period):
        steps = np.arange(0, 3 * period + 5, dtype=np.int64)
        got = _gait_clock_cols(
            torch.as_tensor(steps), period).numpy()
        exp = np.stack([_cpu_clock(period, t) for t in steps])
        np.testing.assert_allclose(got, exp, rtol=0, atol=1e-7)

    def test_half_boundary(self):
        period = 40
        half = period // 2
        got = _gait_clock_cols(
            torch.tensor([half - 1, half, half + 1]), period).numpy()
        # pos=half-1 → 左窗末帧；pos=half → 右窗第 0 帧
        np.testing.assert_allclose(got[0], [1, 0, (half - 1) / half],
                                   atol=1e-7)
        np.testing.assert_allclose(got[1], [0, 1, 0], atol=1e-7)
        np.testing.assert_allclose(got[2], [0, 1, 1 / half], atol=1e-7)

    def test_per_env_divergence(self):
        """同一批内行可处于不同相位（部分 reset 后的常态）。"""
        steps = torch.tensor([0, 5, 20, 39, 40, 60, 79, 80])
        got = _gait_clock_cols(steps, 40).numpy()
        exp = np.stack([_cpu_clock(40, t) for t in steps.numpy()])
        np.testing.assert_allclose(got, exp, atol=1e-7)
        # 步 0/40/80 在左窗（pos<half），步 60/79 在右窗
        assert got[0, 0] == 1 and got[4, 0] == 1 and got[5, 0] == 0 \
            and got[6, 1] == 1 and got[7, 0] == 1


# ---------------------------------------------------------------------------
# 波契约：BatchRuntime 簿记耦合（最小 sim 壳，真实 obs builder）
# ---------------------------------------------------------------------------
# 假表（使 _robot_obs 恰产 96 维）：21 jpos + 21 jvel + 3+1+3+3+2 + 3+3+3+3
# + 15 kp_pos + 3 kp_head_vel + 12 kp_vel = 96。
_QI, _VI, _NB2 = 21, 21, 16
NQ, NV, NBODY_F, NJNT = 56, 54, _NB2, 8
_KP_A = {"head": 2, "hand_right": 3, "hand_left": 4,
         "foot_right": 5, "foot_left": 6}
_KP_B = {"head": 9, "hand_right": 10, "hand_left": 11,
         "foot_right": 12, "foot_left": 13}


def _fake_tables(device):
    def robot(root, qva, qidx, vidx, kp):
        return dict(root_body_id=root, root_qvel_adr=qva,
                    qpos_indices=torch.as_tensor(qidx, device=device),
                    qvel_indices=torch.as_tensor(vidx, device=device),
                    norm_ref=torch.zeros(_QI, device=device),
                    norm_scale=torch.ones(_QI, device=device),
                    keypoint_body_ids=kp, body_weight=1.0)
    return SimpleNamespace(
        device=torch.device(device), ground_geom_id=GROUND_GID,
        geom_bodyid=torch.as_tensor(GEOM_BODY, device=device),
        geom_aff=torch.as_tensor(GEOM_AFF, device=device),
        robots={
            "robot_a": robot(1, 0, np.arange(7, 28), np.arange(6, 27),
                             _KP_A),
            "robot_b": robot(8, 27, np.arange(35, 56), np.arange(33, 54),
                             _KP_B),
        })


def _fake_ns(B, device):
    t = lambda shape, dt=torch.float32: torch.zeros(  # noqa: E731
        *shape, dtype=dt, device=device)
    cf = ContactFlatNamespace(
        worldid=t((1,), torch.int32).fill_(-1),
        geom=t((1, 2), torch.int32), dist=t((1,)).fill_(np.inf),
        pos=t((1, 3)), frame=t((1, 3, 3)), dim=t((1,), torch.int32),
        efc_address=t((1,), torch.int32), efc_force=t((B, 4)),
        n_active=torch.zeros((), dtype=torch.int32, device=device),
        cap_per_world=64)
    xquat = t((B, NBODY_F, 4))
    xquat[..., 0] = 1.0
    return SimNamespace(
        qpos=t((B, NQ)), qvel=t((B, NV)), ctrl=t((B, 4)),
        xpos=t((B, NBODY_F, 3)), xquat=xquat,
        xipos=t((B, NBODY_F, 3)), xanchor=t((B, NJNT, 3)),
        cvel=t((B, NBODY_F, 6)), xfrc_applied=t((B, NBODY_F, 6)),
        act_target=t((B, 4)), contacts_flat=cf)


class _WaveSim:
    """BatchRuntime 契约的最小 sim 壳（恒等物理——本测试只关心
    obs 构造时读到的 episode_steps 簿记语义，不在物理正确性）。"""

    DT = 0.02
    ACTION_DIM = 4
    _FIELDS = ("qpos", "qvel", "ctrl", "xpos", "xquat", "xipos",
               "xanchor", "cvel", "xfrc_applied", "act_target")

    def __init__(self, ns: SimNamespace, B: int, dev: torch.device):
        self._ns, self._B, self._dev = ns, int(B), dev
        self._st = None

    @property
    def batch_size(self):
        return self._B

    @property
    def device(self):
        return self._dev

    def build_sim_namespace(self):
        return self._ns

    def attach_state(self, st):
        self._st = st

    def reset(self, seeds=None, options=None):
        for f in self._FIELDS:
            getattr(self._ns, f).zero_()

    def physical_step(self, n_steps=1, keep_history=False,
                      pre_step=None, post_step=None):
        for i in range(int(n_steps)):
            if pre_step is not None:
                pre_step(i)
            if post_step is not None:
                post_step(i)

    def dev_set_action(self, a, b):
        self._st.io.action_a.copy_(a)
        self._st.io.action_b.copy_(b)

    def dev_reset_rows(self, env_ids, options=None):
        ids = env_ids.to(torch.long)
        for f in self._FIELDS:
            getattr(self._ns, f)[ids] = 0

    def dev_set_integration_rows(self, env_ids, qpos, qvel):
        ids = env_ids.to(torch.long)
        self._ns.qpos[ids] = qpos
        self._ns.qvel[ids] = qvel

    def dev_add_ext_force(self, body_id, force, torque=None):
        pass

    def dev_upload_force_schedule(self, sched):
        pass

    def capture(self, mask, level=None):
        return {f: getattr(self._ns, f)[mask].clone()
                for f in self._FIELDS}

    def restore(self, mask, snapshot):
        for k, v in snapshot.items():
            getattr(self._ns, k)[mask] = v

    def get_physical_frequency(self):
        return 1.0 / self.DT


def _exp_clock(steps, period):
    return np.stack([_cpu_clock(period, int(t)) for t in steps])


class TestGaitClockWaveContract:
    """BatchRuntime + GaitClockObsBuilder：obs 读到的帧索引 ==
    CPU ``_action_step``（步尾自增后构造；reset/部分 reset 归零）。"""

    def _make(self, B=4, period=40, device=DEV):
        tables = _fake_tables(device)
        ns = _fake_ns(B, device)
        sim = _WaveSim(ns, B, torch.device(device))
        builder = GaitClockObsBuilder(tables, B, gait_period=period)
        rt = BatchRuntime(sim, obs_builder=builder, phy_substeps=1)
        return sim, rt, builder

    def _frame_clock(self, st):
        return st.io.obs_a[:, -3:].cpu().numpy()

    def test_frame_index_alignment(self):
        B, T = 4, 45
        _, rt, builder = self._make(B=B)
        rt.reset()
        st = rt.state
        builder.build(st)                    # obs_0（帧 0）
        np.testing.assert_allclose(self._frame_clock(st),
                                   _exp_clock(np.zeros(B), 40), atol=1e-7)
        a = torch.zeros(B, 4)
        for t in range(1, T):
            rt.step((a, a))
            got = self._frame_clock(st)
            np.testing.assert_allclose(
                got, _exp_clock(np.full(B, t), 40), atol=1e-7,
                err_msg=f"frame {t} gait clock mismatch")
            # 双 agent 共钟（CPU 同语义）
            np.testing.assert_allclose(
                st.io.obs_b[:, -3:].cpu().numpy(), got, atol=0)

    def test_partial_reset_divergence(self):
        """部分行 reset 后 episode_steps 归零 → 相位分叉。"""
        B = 4
        _, rt, builder = self._make(B=B)
        rt.reset()
        st = rt.state
        builder.build(st)
        a = torch.zeros(B, 4)
        for _ in range(10):
            rt.step((a, a))
        # abandon + reset_rows 行 {1,3}：时钟归零，行 0/2 继续前进
        ids = torch.tensor([1, 3])
        rt.abandon(ids)
        rt.reset_rows(ids)
        rt.step((a, a))                       # 全局第 11 步 / 新 ep 第 1 步
        exp = np.array([11, 1, 11, 1])
        np.testing.assert_allclose(
            st.episode.episode_steps.cpu().numpy(), exp, atol=0)
        np.testing.assert_allclose(self._frame_clock(st),
                                   _exp_clock(exp, 40), atol=1e-7)

    def test_obs_dim_99(self):
        _, rt, builder = self._make(B=2)
        assert builder.obs_dim() == 99
        assert rt.state.io.obs_a.shape[-1] == 99


# ===========================================================================
# Layer A：FootStateObserver 纯逻辑对照
# ===========================================================================
class TestFootStateLogic:
    def test_standing_flat(self):
        xipos, xpos, xquat = _base_scene(4)
        cv = [[(GROUND_GID, _geom_of_body(FOOT_L)),
               (GROUND_GID, _geom_of_body(FOOT_R))]] * 4
        cpu, dev = _run_both(xipos, xpos, xquat, cv)
        _assert_leaves(cpu, dev)
        assert dev["h_left_foot"].max() < 1e-5
        np.testing.assert_allclose(dev["sole_clear_left"], 0, atol=1e-5)

    def test_lifted_foot(self):
        xipos, xpos, xquat = _base_scene(4)
        lift = np.array([0.05, 0.02, 0.10, 0.0])
        for e, h in enumerate(lift):
            xipos[e, FOOT_L, 2] += h
            xpos[e, FOOT_L, 2] += h
        cv = [[(GROUND_GID, _geom_of_body(FOOT_R))]] * 4
        cpu, dev = _run_both(xipos, xpos, xquat, cv)
        _assert_leaves(cpu, dev)
        np.testing.assert_allclose(dev["h_left_foot"], lift, atol=1e-5)
        np.testing.assert_allclose(dev["sole_clear_left"], lift, atol=1e-5)
        np.testing.assert_array_equal(dev["left_foot_contact"], False)
        np.testing.assert_array_equal(dev["right_foot_contact"], True)

    def test_pivot_rock_sole_stays(self):
        """绕趾摇摆：中点升高但 sole 净空仍 ≈0——反刷分关键区分。"""
        xipos, xpos, xquat = _base_scene(4)
        theta = 0.35
        # y-pitch θ：端点旋转后世界 z = px·sinθ（pz=0）——最低端点是
        # 趾端 (px=-0.07)。把 body 抬到使最低端点恰在地：sole≈0。
        pts = np.asarray(FOOT_ENDPOINTS_LOCAL)
        zrot = pts[:, 0] * np.sin(theta) + pts[:, 2] * np.cos(theta)
        xpos[:, FOOT_L, 2] = FOOT_GEOM_RADIUS - zrot.min()
        xquat[:, FOOT_L] = _quat_pitch(theta)
        # 趾端触地 → xipos（体心）上升
        xipos[:, FOOT_L, 2] = xpos[:, FOOT_L, 2]
        cv = [[(GROUND_GID, _geom_of_body(FOOT_L)),
               (GROUND_GID, _geom_of_body(FOOT_R))]] * 4
        cpu, dev = _run_both(xipos, xpos, xquat, cv)
        _assert_leaves(cpu, dev)
        assert dev["h_left_foot"][0] > 0.01       # midpoint 明显升高
        np.testing.assert_allclose(dev["sole_clear_left"], 0, atol=1e-5)

    def test_squat_negative_height(self):
        xipos, xpos, xquat = _base_scene(4)
        xipos[:, FOOT_L, 2] -= 0.02
        xpos[:, FOOT_L, 2] -= 0.02
        cpu, dev = _run_both(xipos, xpos, xquat, [[]] * 4)
        _assert_leaves(cpu, dev)
        assert dev["h_left_foot"].max() < -0.01
        assert dev["sole_clear_left"].max() < -0.01

    def test_contact_exclusions(self):
        """wall/robot-robot/他机足接触不算该足触地。"""
        xipos, xpos, xquat = _base_scene(4)
        cv = [
            [(WALL_GID, _geom_of_body(FOOT_L))],            # wall×foot
            [(GROUND_GID, _geom_of_body(FOOT_L_B))],        # ground×他机足
            [(_geom_of_body(FOOT_L), _geom_of_body(FOOT_L_B))],  # 足×足
            [(GROUND_GID, _geom_of_body(FOOT_L))],          # 真触地
        ]
        cpu, dev = _run_both(xipos, xpos, xquat, cv)
        _assert_leaves(cpu, dev)
        np.testing.assert_array_equal(
            dev["left_foot_contact"], [False, False, False, True])

    def test_agent_b_mirrored(self):
        xipos, xpos, xquat = _base_scene(4)
        xipos[:, FOOT_R_B, 2] += 0.08
        xpos[:, FOOT_R_B, 2] += 0.08
        cv = [[(GROUND_GID, _geom_of_body(FOOT_L_B))]] * 4
        cpu, dev = _run_both(xipos, xpos, xquat, cv, agent=1)
        _assert_leaves(cpu, dev)
        np.testing.assert_allclose(dev["h_right_foot"], 0.08, atol=1e-5)
        np.testing.assert_array_equal(dev["left_foot_contact"], True)
        np.testing.assert_array_equal(dev["right_foot_contact"], False)

    def test_output_dtypes(self):
        xipos, xpos, xquat = _base_scene(2)
        _, dev = _run_both(xipos, xpos, xquat, [[]] * 2)
        assert dev["left_foot_contact"].dtype == bool
        assert dev["h_left_foot"].dtype == np.float32
        for k in ("h_left_foot", "h_right_foot", "sole_clear_left",
                  "sole_clear_right", "left_foot_contact",
                  "right_foot_contact"):
            assert dev[k].shape == (2,)


# ===========================================================================
# Layer B：真机 warp vs CPU（CUDA-gated）
# ===========================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
class TestStepDeviceWarp:
    """相同 qpos/qvel 注入两后端 → 步进 1 子步 → FootStateObserver 对照；
    WarpGaitClockSimulator 装配契约（obs_dim=99 + gait 列）。"""

    @pytest.fixture(scope="class")
    def sims(self):
        from envs.batchframework.warp_simulator import (
            WarpHumanoid21Simulator)
        from envs.humanoid21.simulator import Humanoid21Simulator
        cpu = Humanoid21Simulator()
        warp = WarpHumanoid21Simulator(batch_size=4)
        cpu.reset(seed=0)
        warp.reset(seeds=np.zeros(4, np.int64))
        yield cpu, warp
        warp.close()

    def test_foot_state_warp_vs_cpu(self, sims):
        cpu, warp = sims
        B = warp.batch_size
        states = []
        rng = np.random.default_rng(7)
        for e in range(B):
            cpu.reset(seed=int(rng.integers(1e6)))
            for _ in range(5 * e):
                cpu.physical_step()
            states.append(cpu.get_core_state())
        batched = {rid: {k: np.stack([s[rid][k] for s in states])
                         for k in states[0][rid]}
                   for rid in ("robot_a", "robot_b")}
        warp.set_core_state(batched)

        warp.physical_step(1)
        warp_state = warp.build_device_state()
        warp_outs = {}
        for a in (0, 1):
            obs = DeviceFootStateObserver.from_sim(warp, a)
            ctx = DeviceCtx(warp_state, plugin_name="foot")
            obs.on_post_action_step(ctx)
            warp_outs[a] = {k: v.detach().cpu().numpy()
                            for k, v in obs.get_output().items()}

        cpu_outs = []
        for e in range(B):
            cpu.set_core_state(states[e])
            cpu.physical_step()
            ctx = SimpleNamespace(accessor=cpu)
            row = {}
            for a, aid in ((0, "robot_a"), (1, "robot_b")):
                o = FootStateObserver(agent_id=aid)
                o.on_pre_episode(ctx)
                o.on_post_action_step(ctx)
                row[a] = o.get_output()
            cpu_outs.append(row)

        for a in (0, 1):
            got = warp_outs[a]
            for k in ("h_left_foot", "h_right_foot",
                      "sole_clear_left", "sole_clear_right"):
                exp = np.array([cpu_outs[e][a][k] for e in range(B)],
                               dtype=np.float32)
                np.testing.assert_allclose(
                    got[k], exp, rtol=1e-2, atol=2e-2,
                    err_msg=f"agent{a} {k}: warp={got[k]} cpu={exp}")
            for k in ("left_foot_contact", "right_foot_contact"):
                exp = np.array([cpu_outs[e][a][k] for e in range(B)],
                               dtype=bool)
                np.testing.assert_array_equal(got[k].astype(bool), exp)

    def test_gait_sim_assembly(self):
        """真机装配：binding → 99 维 io schema + reset 后 clock(0)。"""
        from envs.batchframework.binding_registry import resolve_binding
        binding = resolve_binding(
            "baseline.humanoid21.end2end.gait_clock_simulator"
            ":GaitClockSimulator")
        sim = binding.make_sim(
            4, "cuda", sim_config={"gait_period": 40,
                                   "initial_distance": 2.0})
        sim.reset()
        schema = binding.io_schema(sim)
        assert schema.obs_dim == 99 and schema.action_dim == 21
        rt = BatchRuntime(
            sim, obs_builder=sim.device_obs_builder(), phy_substeps=25)
        st = rt.state
        rt.obs_builder.build(st)             # obs_0
        got = st.io.obs_a[:, 96:99].cpu().numpy()
        np.testing.assert_allclose(
            got, np.tile(np.array([1.0, 0.0, 0.0], np.float32), (4, 1)),
            atol=1e-6)
        rt.step()
        got1 = st.io.obs_a[:, 96:99].cpu().numpy()
        np.testing.assert_allclose(
            got1, np.tile(np.array([1.0, 0.0, 0.05], np.float32), (4, 1)),
            atol=1e-6)
        sim.close()


# ===========================================================================
# L3：DeviceRollouter / MultiDeviceRollouter collect 契约（CUDA-gated）
# ===========================================================================
_T_COLLECT = 48   # >40：帧序列跨过右窗起点（t=20）与周期回卷（t=40）


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
class TestStepDeviceCollect:
    """collect(jobs) -> List[Episode] 对 step 蓝图（obs_dim=99）的
    完整契约：trajectory/终止记录/bootstrap/observer 输出/gait 列值。"""

    @pytest.fixture(scope="class")
    def env_bp(self):
        from envs.framework.parameterized_blueprint import (
            ParameterizedEnvBlueprint)
        pb = ParameterizedEnvBlueprint.load(
            project_root / "baseline/humanoid21/end2end/step_env.yaml")
        return pb.materialize(max_steps=_T_COLLECT)

    @pytest.fixture(scope="class")
    def policy_bp(self, tmp_path_factory):
        from baseline.framework.ppo.policies.truncated_normal_mlp import (
            TruncatedNormalPolicy)
        pol = TruncatedNormalPolicy(obs_dim=99, action_dim=21,
                                    hidden_dim=256, device="cpu")
        dest = tmp_path_factory.mktemp("pol_step") / "export"
        return pol.to_blueprint(str(dest))

    @staticmethod
    def _jobs(env_bp, policy_bp, n, seed0=0):
        """与 Step.build_jobs 同构：声明式 ef 程序 + initial_distance。"""
        from baseline.framework.rollout.job import Job, SamplingSpec
        from baseline.experiments_ppo.exp_step import _PHASE_EF_PROGRAM
        sampling = SamplingSpec(explore_factor=dict(_PHASE_EF_PROGRAM))
        return [
            Job(policy_a_bp=policy_bp, policy_b_bp=policy_bp,
                env_bp=env_bp, seed=seed0 + i,
                episode_options={"initial_distance": 2.0 + 0.05 * i},
                sampling_a=sampling, sampling_b=sampling,
                stochastic=True)
            for i in range(n)
        ]

    @staticmethod
    def _assert_episode(ep, T=_T_COLLECT):
        from baseline.framework.rollout.episode import Episode
        assert isinstance(ep, Episode)
        assert ep.num_frames == T
        clock_exp = _exp_clock(np.arange(T), 40)
        for rid in ("robot_a", "robot_b"):
            assert ep.observations[rid].shape == (T, 99)
            assert ep.actions[rid].shape == (T, 21)
            assert ep.final_observation[rid].shape == (99,)
            assert ep.explore_factors[rid].shape == (T,)
            # gait-clock 列逐帧匹配 CPU 公式（帧索引对齐的最强检验）
            np.testing.assert_allclose(
                ep.observations[rid][:, 96:99], clock_exp, atol=1e-6,
                err_msg=f"{rid} gait-clock cols")
            ex = ep.action_extras[rid]
            for k in ("log_prob", "explore_factor",
                      "sctx__delta_factor"):
                assert k in ex and ex[k].shape == (T,)
            assert np.isfinite(ex["log_prob"]).all()
            assert ep.agent_termination_proposal_records[rid] == (
                ("timeout", T),)
        # observer 输出：standup 势场 + foot_state 六叶
        for name in ("standing_balance_a", "standing_balance_b"):
            out = ep.observer_outputs[name]
            for k in ("potential", "stage", "h_torso", "f_score",
                      "contact_score", "d_score", "w_foot"):
                v = np.asarray(out[k], dtype=np.float64)
                assert v.shape == (T,)
                assert np.isfinite(v).all()
        for name in ("foot_state_a", "foot_state_b"):
            out = ep.observer_outputs[name]
            for k in ("h_left_foot", "h_right_foot",
                      "sole_clear_left", "sole_clear_right"):
                v = np.asarray(out[k], dtype=np.float64)
                assert v.shape == (T,)
                assert np.isfinite(v).all()
            for k in ("left_foot_contact", "right_foot_contact"):
                v = np.asarray(out[k], dtype=bool)
                assert v.shape == (T,)

    def test_collect_episode_contract(self, env_bp, policy_bp):
        from envs.batchframework.device_rollouter import DeviceRollouter
        with DeviceRollouter(batch_size=4, device="cuda") as dr:
            eps = dr.collect(self._jobs(env_bp, policy_bp, 4))
        assert len(eps) == 4
        for i, ep in enumerate(eps):
            assert ep.episode_index == i and ep.base_seed == i
            assert ep.episode_options["initial_distance"] == 2.0 + 0.05 * i
            self._assert_episode(ep)

    def test_ppo_pipeline_compat(self, env_bp, policy_bp):
        """device episode 无差别流经 Step.build_trajectories（全部
        foot_state/clock 字段被真实消费——缺失字段会静默产出零通道，
        故显式断言各通道有非零信号或按数据规则校验）。"""
        from baseline.experiments_ppo.exp_step import Step
        from baseline.framework.ppo.trainer import PPOBuffer
        from envs.batchframework.device_rollouter import DeviceRollouter

        with DeviceRollouter(batch_size=4, device="cuda") as dr:
            eps = dr.collect(self._jobs(env_bp, policy_bp, 4))
            actor = dr._policy(policy_bp.to_dict())

        trajs = Step().build_trajectories(eps)
        assert len(trajs) == 8
        for t in trajs:
            assert t.obs.shape == (_T_COLLECT, 99)
            assert t.actions.shape == (_T_COLLECT, 21)
            assert t.last_obs.shape == (99,)
            for ch in ("r_potential", "r_left_foot", "r_right_foot",
                       "r_torso", "r_fall"):
                assert t.channels[ch].reward.shape == (_T_COLLECT,)
                assert np.isfinite(t.channels[ch].reward).all()
                assert t.channels[ch].actor_weight.shape == (_T_COLLECT,)
            assert sorted(t.sampling_ctx) == [
                "delta_factor", "explore_factor"]
        buf = PPOBuffer(trajectories=trajs, actor=actor, device="cuda",
                        reward_keys=list(Step._channel_names))
        rec = np.concatenate([
            np.asarray(e.action_extras[r]["log_prob"], dtype=np.float32)
            for e in eps for r in ("robot_a", "robot_b")])
        assert np.abs(buf.log_probs - rec).max() < 1e-4
        assert np.isfinite(buf.log_probs).all()

    @pytest.mark.skipif(torch.cuda.device_count() < 2,
                        reason="needs >=2 CUDA devices")
    def test_multi_device_collect(self, env_bp, policy_bp):
        """MultiDeviceRollouter 分片收集：job 顺序保持，契约同构。"""
        from envs.batchframework.multi_rollouter import (
            MultiDeviceRollouter)
        with MultiDeviceRollouter(devices=[0, 1],
                                  batch_size_per_worker=4) as mdr:
            eps = mdr.collect(self._jobs(env_bp, policy_bp, 8))
        assert len(eps) == 8
        for i, ep in enumerate(eps):
            assert ep.episode_index == i and ep.base_seed == i
            self._assert_episode(ep)
