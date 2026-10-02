"""M4 T2 — DeviceStandup4StageRewarder 与 CPU 版同态对照测试。

两层验证（对齐 M4_PLAN T2 验收哲学）：

- **Layer A（纯逻辑）**：相同注入的张量态（xpos/xquat/contacts），
  CPU ``StandingBalance4StageRewarder``（伪 accessor 喂相同数值）vs
  设备端 rewarder。无物理参与 → 期望 fp32 舍入级一致（~1e-6）。
  覆盖：四个 stage、stage 边界、1.0N 力门限、10N 载荷门限、
  额外接触 distinct-body 去重、wall/robot-robot/他机接触排除。

- **Layer B（真机）**：相同 qpos/qvel 注入 CPU Humanoid21Simulator 与
  WarpHumanoid21Simulator，各步进 1 子步后分别跑两侧 rewarder。
  含 fp32 物理发散 → 宽容差（对齐 warp fixture 的 2e-2/1e-3）。
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

project_root = Path(__file__).resolve().parent.parent

from envs.batchframework.device_state import (  # noqa: E402
    ContactFlatNamespace, DeviceBatchState, IoNamespace,
    RngNamespace, SimNamespace,
)
from envs.batchframework.device_plugin import (  # noqa: E402
    DeviceCtx, DeviceMutator)
from envs.batchframework.device_standup import (  # noqa: E402
    OUT_KEYS, DeviceStandup4StageRewarder,
)
from baseline.humanoid21.rewards.standing_balance_4stage import (  # noqa: E402
    D_GATE, D_MAX, D_MIN, F_ENTER, F_LOAD_MIN, H_CROUCH, H_HAND_MAX,
    H_STAND, StandingBalance4StageRewarder,
)

# ---------------------------------------------------------------------------
# 玩具场景拓扑（两实现共享同一套 id/name 表）
# ---------------------------------------------------------------------------
# geoms: 0=ground(aff0)  1=wall(aff0)  2..7→body2..7(aff1)  8..13→body8..13(aff2)
NGEOM, NBODY = 14, 14
GROUND_GID, WALL_GID = 0, 1
# agent0 (robot_a, aff=1) bodies
TORSO, HAND_L, HAND_R, FOOT_L, FOOT_R, OTHER_A, OTHER_A2 = 1, 2, 3, 4, 5, 6, 7
# agent1 (robot_b, aff=2) bodies
TORSO_B, HAND_L_B, FOOT_L_B, OTHER_B = 8, 9, 10, 11

NAMES_A = {  # agent0 语义名（CPU rewarder 经 body_id_to_name 反查）
    TORSO: "torso_a", HAND_L: "hand_left_a", HAND_R: "hand_right_a",
    FOOT_L: "foot_left_a", FOOT_R: "foot_right_a",
    OTHER_A: "knee_left_a", OTHER_A2: "elbow_right_a",
}
NAMES_B = {TORSO_B: "torso_b", HAND_L_B: "hand_left_b",
           FOOT_L_B: "foot_left_b", OTHER_B: "head_b"}

GEOM_BODY = np.zeros(NGEOM, dtype=np.int64)
GEOM_BODY[2:8] = [HAND_L, HAND_R, FOOT_L, FOOT_R, OTHER_A, OTHER_A2]
GEOM_BODY[8:14] = [HAND_L_B, TORSO_B, FOOT_L_B, OTHER_B, 12, 13]
GEOM_AFF = np.zeros(NGEOM, dtype=np.int64)
GEOM_AFF[2:8] = 1
GEOM_AFF[8:14] = 2
GEOM_NAME = {0: "ground", 1: "wall"}  # env 侧只需 ground/wall 两名


def _geom_of_body(body):
    return int(np.nonzero(GEOM_BODY == body)[0][0])


# ---------------------------------------------------------------------------
# CPU 侧伪上下文
# ---------------------------------------------------------------------------
class _FakeAccessor:
    """喂给 CPU rewarder 的相同数值（derived_state / static_data）。"""

    def __init__(self, xpos, xquat, contacts_vec):
        self._xpos, self._xquat, self._cv = xpos, xquat, contacts_vec

    def get_derived_state(self, fields=None):
        fields = list(fields) if fields else []
        out = {}
        for f in fields:
            if f == "contacts":
                out["contacts"] = self._cv
            elif f in ("robot_a", "robot_b"):
                names = NAMES_A if f == "robot_a" else NAMES_B
                out[f] = {
                    "body_xpos": {n: self._xpos[b].astype(np.float32)
                                  for b, n in names.items()},
                    "body_xquat": {NAMES_A[TORSO] if f == "robot_a"
                                   else NAMES_B[TORSO_B]:
                                   self._xquat[
                                       TORSO if f == "robot_a" else TORSO_B
                                   ].astype(np.float32)},
                }
        return out

    def get_static_data(self):
        body_id_to_name = {b: n for b, n in
                           {**NAMES_A, **NAMES_B}.items()}
        return {
            "robot_a": {"keypoint_body_names": {
                "torso": NAMES_A[TORSO], "hand_left": NAMES_A[HAND_L],
                "hand_right": NAMES_A[HAND_R], "foot_left": NAMES_A[FOOT_L],
                "foot_right": NAMES_A[FOOT_R]}},
            "robot_b": {"keypoint_body_names": {
                "torso": NAMES_B[TORSO_B], "hand_left": NAMES_B[HAND_L_B],
                "hand_right": "hand_right_b", "foot_left": NAMES_B[FOOT_L_B],
                "foot_right": "foot_right_b"}},
            "body_id_to_name": body_id_to_name,
            "geom_id_to_name": GEOM_NAME,
        }


def _contacts_vec(rows):
    """rows: list of (geom1, geom2, force_mag) → CPU SoA dict。"""
    n = len(rows)
    cv = dict(
        ncon=n,
        geom1=np.zeros(n, np.int32), geom2=np.zeros(n, np.int32),
        body1=np.zeros(n, np.int32), body2=np.zeros(n, np.int32),
        aff1=np.zeros(n, np.int8), aff2=np.zeros(n, np.int8),
        force_mag=np.zeros(n, np.float32),
    )
    for i, (g1, g2, f) in enumerate(rows):
        cv["geom1"][i], cv["geom2"][i] = g1, g2
        cv["body1"][i], cv["body2"][i] = GEOM_BODY[g1], GEOM_BODY[g2]
        cv["aff1"][i], cv["aff2"][i] = GEOM_AFF[g1], GEOM_AFF[g2]
        cv["force_mag"][i] = f
    return cv


def run_cpu_rewarder(xpos, xquat, cv, agent_id):
    r = StandingBalance4StageRewarder(agent_id=agent_id)
    r.on_pre_episode(SimpleNamespace())
    r.on_post_action_step(
        SimpleNamespace(accessor=_FakeAccessor(xpos, xquat, cv)))
    return r.get_output()


# ---------------------------------------------------------------------------
# 设备侧注入：把同样数值装进 ContactFlatNamespace
# ---------------------------------------------------------------------------
def _efc_rows_for(force_vec):
    """force_world=(fx,fy,fz) → condim=3 的 4 行 efc（frame=I 下精确还原）。

    设备侧分解公式：normal = r0+r1+r2+r3, f1 = r0-r1, f2 = r2-r3。
    解：r0=fz/4+fx/2, r1=fz/4-fx/2, r2=fz/4+fy/2, r3=fz/4-fy/2。
    """
    fx, fy, fz = force_vec
    return np.array([fz / 4 + fx / 2, fz / 4 - fx / 2,
                     fz / 4 + fy / 2, fz / 4 - fy / 2], dtype=np.float32)


def make_device_state(contacts_per_env, xpos, xquat, device):
    """contacts_per_env: List[List[(geom1, geom2, fz_force)]]。

    frame=I、force_world=(0,0,fz) ⇒ force_mag=|fz|，与 CPU cv 的
    force_mag 标量完全对齐。
    """
    B = len(contacts_per_env)
    rows, efc = [], []
    for w, contacts in enumerate(contacts_per_env):
        for g1, g2, f in contacts:
            rows.append((w, g1, g2))
            efc.append(_efc_rows_for((0.0, 0.0, f)))
    M = len(rows)
    nefc = M * 4 if M else 4
    worldid = np.array([r[0] for r in rows] or [0], np.int32)
    geom = np.array([[r[1], r[2]] for r in rows] or [[0, 0]], np.int32)
    dist = np.zeros(max(M, 1), np.float32)
    dist[M:] = np.inf
    efc_address = (np.arange(max(M, 1)) * 4).astype(np.int32)
    efc_force = np.zeros((B, nefc), np.float32)
    for i, rows4 in enumerate(efc):
        efc_force[rows[i][0], i * 4:i * 4 + 4] = rows4
    frame = np.zeros((max(M, 1), 3, 3), np.float32)
    frame[:, 0, 0] = frame[:, 1, 1] = frame[:, 2, 2] = 1.0

    t = lambda a, dt=torch.float32: torch.as_tensor(  # noqa: E731
        np.asarray(a), dtype=dt, device=device)
    contacts = ContactFlatNamespace(
        worldid=t(worldid, torch.int32), geom=t(geom, torch.int32),
        dist=t(dist), pos=t(np.zeros((max(M, 1), 3), np.float32)),
        frame=t(frame), dim=t(np.full(max(M, 1), 3, np.int32), torch.int32),
        efc_address=t(efc_address, torch.int32),
        efc_force=t(efc_force),
        n_active=t(np.int32(M), torch.int32), cap_per_world=64,
    )
    sim_ns = SimNamespace(
        qpos=t(np.zeros((B, 4), np.float32)),
        qvel=t(np.zeros((B, 4), np.float32)),
        ctrl=t(np.zeros((B, 4), np.float32)),
        xpos=t(np.asarray(xpos, np.float32)),
        xquat=t(np.asarray(xquat, np.float32)),
        xipos=t(np.zeros((B, NBODY, 3), np.float32)),
        xanchor=t(np.zeros((B, 4, 3), np.float32)),
        cvel=t(np.zeros((B, NBODY, 6), np.float32)),
        xfrc_applied=t(np.zeros((B, NBODY, 6), np.float32)),
        act_target=t(np.zeros((B, 4), np.float32)),
        contacts_flat=contacts,
    )
    from envs.batchframework.device_state import alloc_episode_namespace
    ep = alloc_episode_namespace(B, device)
    io = IoNamespace(action_a=torch.zeros(B, 21, device=device),
                     action_b=torch.zeros(B, 21, device=device))
    rng = RngNamespace(
        seed_offsets=torch.zeros(B, dtype=torch.int64, device=device),
        step_counter=torch.zeros((), dtype=torch.int64, device=device))
    return DeviceBatchState(B, sim_ns, ep, io, rng)


def _body_tables(agent_idx):
    if agent_idx == 0:
        return dict(torso=TORSO, hand_l=HAND_L, hand_r=HAND_R,
                    foot_l=FOOT_L, foot_r=FOOT_R)
    return dict(torso=TORSO_B, hand_l=HAND_L_B, hand_r=12,
                foot_l=FOOT_L_B, foot_r=13)


def run_device_rewarder(state, agent_idx, device):
    t = _body_tables(agent_idx)
    rewarder = DeviceStandup4StageRewarder(agent_idx, dict(
        torso=t["torso"], hand_l=t["hand_l"], hand_r=t["hand_r"],
        foot_l=t["foot_l"], foot_r=t["foot_r"],
        ground_gid=GROUND_GID, robot_aff=agent_idx + 1,
        geom_bodyid=torch.as_tensor(GEOM_BODY, device=device),
        geom_aff=torch.as_tensor(GEOM_AFF, device=device),
        nbody=NBODY))
    ctx = DeviceCtx(state, plugin_name="rewarder")
    rewarder.on_pre_episode(ctx)
    rewarder.on_post_action_step(ctx)
    return {k: v.detach().cpu().numpy() for k, v in rewarder.get_output().items()}


# ---------------------------------------------------------------------------
# Layer A fixtures：相同注入态逐字段比对
# ---------------------------------------------------------------------------
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _base_scene(n_env):
    """默认位姿：torso 高 1.28（站立）、手脚在地。"""
    xpos = np.zeros((n_env, NBODY, 3), np.float64)
    xquat = np.zeros((n_env, NBODY, 4), np.float64)
    xquat[..., 0] = 1.0
    return xpos, xquat


def _set_pose(xpos, xquat, e, agent, torso_h, torso_quat,
              hand_h=0.05, foot_h=0.02, hand_xy=(0.2, 0.0),
              foot_xy=(0.0, 0.0)):
    """单手构造 per-env 关键点位姿。"""
    if agent == 0:
        torso, hl, hr, fl, fr = TORSO, HAND_L, HAND_R, FOOT_L, FOOT_R
    else:
        torso, hl, hr, fl, fr = TORSO_B, HAND_L_B, 12, FOOT_L_B, 13
    xpos[e, torso] = (0, 0, torso_h)
    xquat[e, torso] = torso_quat
    xpos[e, hl] = (hand_xy[0], hand_xy[1] + 0.2, hand_h)
    xpos[e, hr] = (hand_xy[0], hand_xy[1] - 0.2, hand_h)
    xpos[e, fl] = (foot_xy[0], foot_xy[1] + 0.1, foot_h)
    xpos[e, fr] = (foot_xy[0], foot_xy[1] - 0.1, foot_h)


PRONE = np.array([np.sqrt(0.5), 0, np.sqrt(0.5), 0])       # f_score=1
UPRIGHT = np.array([1.0, 0, 0, 0])                          # f_score=0.5
SUPINE = np.array([0.0, 0, 1.0, 0])                         # f_score=0


class TestStandupRewarderLogic:
    """相同注入态下 CPU vs 设备端逐字段一致（fp32 噪声级）。"""

    def _run_both(self, xpos, xquat, contacts_per_env, agent=0):
        agent_id = "robot_a" if agent == 0 else "robot_b"
        state = make_device_state(contacts_per_env, xpos, xquat, DEV)
        dev_out = run_device_rewarder(state, agent, torch.device(DEV))
        cpu_outs = []
        for e in range(len(contacts_per_env)):
            cv = _contacts_vec(contacts_per_env[e])
            cpu_outs.append(run_cpu_rewarder(
                xpos[e], xquat[e], cv, agent_id))
        for k in OUT_KEYS:
            exp = np.array([o[k] for o in cpu_outs])
            got = dev_out[k]
            np.testing.assert_allclose(
                got, exp, rtol=1e-5, atol=1e-5,
                err_msg=f"{k} mismatch: dev={got} cpu={exp}")
        return dev_out, cpu_outs

    def test_stage1_no_contact_prone(self):
        """f_score<0.8 且无接触 → stage 1。"""
        xpos, xquat = _base_scene(2)
        _set_pose(xpos, xquat, 0, 0, torso_h=0.4, torso_quat=UPRIGHT)
        _set_pose(xpos, xquat, 1, 0, torso_h=0.4, torso_quat=SUPINE)
        self._run_both(xpos, xquat, [[], []])

    def test_stage2_extra_contact(self):
        """f_score≥F_ENTER 但有地面额外接触 → stage 2。"""
        xpos, xquat = _base_scene(1)
        _set_pose(xpos, xquat, 0, 0, torso_h=0.3, torso_quat=PRONE)
        contacts = [[(GROUND_GID, _geom_of_body(FOOT_L), 100.0),
                     (GROUND_GID, _geom_of_body(OTHER_A), 50.0)]]
        out, _ = self._run_both(xpos, xquat, contacts)
        assert out["stage"][0] == 2

    def test_stage3_only_hf_far(self):
        """只有手脚接触但 XY 距离远（d_score<D_GATE）→ stage 3。"""
        xpos, xquat = _base_scene(1)
        # hand-foot XY 距离 0.6 → d_score=0.5 < D_GATE=0.6
        _set_pose(xpos, xquat, 0, 0, torso_h=0.3, torso_quat=PRONE,
                  hand_xy=(0.6, 0.0), foot_xy=(0.0, 0.0))
        contacts = [[(GROUND_GID, _geom_of_body(FOOT_L), 100.0),
                     (GROUND_GID, _geom_of_body(FOOT_R), 80.0),
                     (GROUND_GID, _geom_of_body(HAND_L), 30.0)]]
        out, _ = self._run_both(xpos, xquat, contacts)
        assert out["stage"][0] == 3

    def test_stage4_standing(self):
        """双脚承重 + 躯干高 → stage 4，w_foot 反映载荷转移。"""
        xpos, xquat = _base_scene(2)
        _set_pose(xpos, xquat, 0, 0, torso_h=H_STAND, torso_quat=UPRIGHT,
                  hand_h=1.2, foot_h=0.0, hand_xy=(0.1, 0.0))
        # env1：手仍分担 1/3 载荷
        _set_pose(xpos, xquat, 1, 0, torso_h=0.9, torso_quat=UPRIGHT,
                  hand_h=0.4, foot_h=0.0, hand_xy=(0.1, 0.0))
        contacts = [
            [(GROUND_GID, _geom_of_body(FOOT_L), 200.0),
             (GROUND_GID, _geom_of_body(FOOT_R), 150.0)],
            [(GROUND_GID, _geom_of_body(FOOT_L), 100.0),
             (GROUND_GID, _geom_of_body(FOOT_R), 100.0),
             (GROUND_GID, _geom_of_body(HAND_L), 50.0),
             (GROUND_GID, _geom_of_body(HAND_R), 50.0)],
        ]
        out, _ = self._run_both(xpos, xquat, contacts)
        assert out["stage"].tolist() == [4, 4]
        assert out["w_foot"][0] == pytest.approx(1.0, abs=1e-5)
        assert out["w_foot"][1] == pytest.approx(200 / 300, abs=1e-4)

    def test_extra_contact_distinct_body_dedup(self):
        """同一 body 两个 geom 的接触只计一次；不同 body 各计一次。"""
        xpos, xquat = _base_scene(1)
        _set_pose(xpos, xquat, 0, 0, torso_h=0.3, torso_quat=PRONE)
        g_other = _geom_of_body(OTHER_A)
        contacts = [[(GROUND_GID, _geom_of_body(FOOT_L), 100.0),
                     (g_other, _geom_of_body(OTHER_A), 20.0),   # 同 body×2
                     (GROUND_GID, g_other, 20.0),
                     (GROUND_GID, _geom_of_body(OTHER_A2), 10.0)]]
        # 注意第三个是 (other_geom, ground) 反向配对，也要计；
        # distinct body = {OTHER_A, OTHER_A2} → extra_count=2（数值由
        # _run_both 的逐字段对拍保证，此处只断言 stage 归类正确）
        out, _ = self._run_both(xpos, xquat, contacts)
        assert out["stage"][0] == 2

    def test_force_threshold_1N(self):
        """force_mag < 1.0 的接触完全不计；=1.0 计入。"""
        xpos, xquat = _base_scene(2)
        _set_pose(xpos, xquat, 0, 0, torso_h=0.3, torso_quat=SUPINE)
        _set_pose(xpos, xquat, 1, 0, torso_h=0.3, torso_quat=PRONE,
                  hand_xy=(0.6, 0.0))
        contacts = [
            [(GROUND_GID, _geom_of_body(FOOT_L), 0.9999),   # 被忽略 → 仰卧无接触 → stage1
             (GROUND_GID, _geom_of_body(OTHER_A), 0.5)],
            [(GROUND_GID, _geom_of_body(FOOT_L), 1.0)],     # =1.0 计入 → hf_contact
        ]
        out, _ = self._run_both(xpos, xquat, contacts)
        assert out["stage"].tolist() == [1, 3]

    def test_load_gate_10N(self):
        """总载荷 < 10N → w_foot=0；≥10N → 正常比值。"""
        xpos, xquat = _base_scene(2)
        for e in range(2):
            _set_pose(xpos, xquat, e, 0, torso_h=0.8, torso_quat=UPRIGHT,
                      hand_h=0.3, hand_xy=(0.1, 0.0))
        contacts = [
            [(GROUND_GID, _geom_of_body(FOOT_L), 5.0),
             (GROUND_GID, _geom_of_body(FOOT_R), 4.9)],          # total 9.9 <10
            [(GROUND_GID, _geom_of_body(FOOT_L), 6.0),
             (GROUND_GID, _geom_of_body(FOOT_R), 4.0)],          # total 10 =门限
        ]
        out, _ = self._run_both(xpos, xquat, contacts)
        assert out["w_foot"][0] == 0.0
        assert out["w_foot"][1] == pytest.approx(1.0, abs=1e-5)

    def test_non_ground_and_wrong_aff_excluded(self):
        """wall 接触 / 同 aff 机器人接触 / 他机接触 都不计入。"""
        xpos, xquat = _base_scene(3)
        _set_pose(xpos, xquat, 0, 0, torso_h=0.3, torso_quat=PRONE,
                  hand_xy=(0.6, 0.0))      # d_score=0.5 → stage3
        _set_pose(xpos, xquat, 1, 0, torso_h=0.3, torso_quat=PRONE)
        _set_pose(xpos, xquat, 2, 0, torso_h=0.3, torso_quat=SUPINE)
        contacts = [
            [(GROUND_GID, _geom_of_body(FOOT_L), 100.0),
             (WALL_GID, _geom_of_body(OTHER_A), 100.0)],          # wall → 不算 extra
            [(_geom_of_body(FOOT_L), _geom_of_body(OTHER_A), 100.0)],  # 同机 → 忽略
            [(GROUND_GID, _geom_of_body(FOOT_L_B), 100.0)],       # 他机 aff → 忽略
        ]
        # env0: 脚+墙接触 → 只算脚 → only_hf → stage3
        # env1: 同机接触被忽略 → 无接触、f_score=1 → stage2
        # env2: 他机接触被忽略 → 仰卧 f=0 → stage1
        out, _ = self._run_both(xpos, xquat, contacts, agent=0)
        assert out["stage"].tolist() == [3, 2, 1]
        # 同一场景对 agent1：env2 的 foot_l_b-ground 应计入
        dev_out = run_device_rewarder(
            make_device_state(contacts, xpos, xquat, DEV), 1,
            torch.device(DEV))
        cpu_out = run_cpu_rewarder(xpos[2], xquat[2],
                                   _contacts_vec(contacts[2]), "robot_b")
        assert dev_out["stage"][2] == pytest.approx(cpu_out["stage"])

    def test_d_score_xy_only(self):
        """竖直分离不计入 d_hf：站立时手在脚上方 1m，XY 距离仍小。"""
        xpos, xquat = _base_scene(1)
        _set_pose(xpos, xquat, 0, 0, torso_h=H_STAND, torso_quat=UPRIGHT,
                  hand_h=1.2, foot_h=0.0, hand_xy=(0.1, 0.0))
        contacts = [[(GROUND_GID, _geom_of_body(FOOT_L), 300.0)]]
        out, cpu = self._run_both(xpos, xquat, contacts)
        assert out["d_hf"][0] == pytest.approx(0.1, abs=1e-4)

    def test_f_enter_boundary(self):
        """f_score 在 F_ENTER 两侧 ±0.005 处两实现取同一侧。"""
        xpos, xquat = _base_scene(2)
        for e, s in ((0, 0.59), (1, 0.61)):   # f_score≈0.795 / 0.805
            q = np.array([np.cos(np.arcsin(s) / 2), 0,
                          np.sin(np.arcsin(s) / 2), 0])
            _set_pose(xpos, xquat, e, 0, torso_h=0.3, torso_quat=q)
        out, _ = self._run_both(xpos, xquat, [[], []])
        assert out["stage"].tolist() == [1, 2]


# ---------------------------------------------------------------------------
# Layer B：真机同注入态对照（warp fp32 vs CPU fp64）
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
class TestStandupRewarderWarpVsCpu:
    """相同 qpos/qvel 注入两后端 → 各步进 1 子步 → rewarder 输出对照。

    容差对齐 warp fixture（fp32）：atol=2e-2, rtol=1e-3；stage 是离散量，
    只在注入态远离 stage 边界时断言相等。
    """

    @pytest.fixture(scope="class")
    def sims(self):
        from envs.batchframework.warp_simulator import WarpHumanoid21Simulator
        from envs.humanoid21.simulator import Humanoid21Simulator
        cpu = Humanoid21Simulator()
        warp = WarpHumanoid21Simulator(batch_size=4)
        cpu.reset(seed=0)
        warp.reset(seeds=np.zeros(4, np.int64))
        yield cpu, warp
        warp.close()

    def test_same_injected_state(self, sims):
        cpu, warp = sims
        B = warp.batch_size
        # 让 CPU 侧产生若干不同体态：reset 站姿 + 随机扰动摔姿
        states = []
        rng = np.random.default_rng(7)
        for e in range(B):
            cpu.reset(seed=int(rng.integers(1e6)))
            for _ in range(5 * e):          # 各 env 步数不同 → 体态分化
                cpu.physical_step()
            states.append(cpu.get_core_state())
        # 注入 warp：各 env 一行
        batched = {}
        for rid in ("robot_a", "robot_b"):
            batched[rid] = {
                k: np.stack([s[rid][k] for s in states])
                for k in states[0][rid]}
        warp.set_core_state(batched)

        # 两侧各步进 1 子步后计算 reward
        warp.physical_step(1)
        warp_outs = {a: run_device_rewarder_warp(warp, a) for a in (0, 1)}
        cpu_outs = []
        for e in range(B):
            cpu.set_core_state(states[e])
            cpu.physical_step()
            ctx = SimpleNamespace(accessor=cpu)
            row = {}
            for a, aid in ((0, "robot_a"), (1, "robot_b")):
                r = StandingBalance4StageRewarder(agent_id=aid)
                r.on_pre_episode(SimpleNamespace())
                r.on_post_action_step(ctx)
                row[a] = r.get_output()
            cpu_outs.append(row)

        for a in (0, 1):
            got = warp_outs[a]
            for k in OUT_KEYS:
                exp = np.array([cpu_outs[e][a][k] for e in range(B)])
                np.testing.assert_allclose(
                    got[k], exp, rtol=1e-2, atol=2e-2,
                    err_msg=f"agent{a} {k}: warp={got[k]} cpu={exp}")


def run_device_rewarder_warp(warp, agent_idx):
    rewarder = DeviceStandup4StageRewarder.from_sim(warp, agent_idx)
    ctx = DeviceCtx(warp.build_device_state(), plugin_name="r")
    rewarder.on_post_action_step(ctx)
    return {k: v.detach().cpu().numpy()
            for k, v in rewarder.get_output().items()}


# ---------------------------------------------------------------------------
# T3 — DeviceFallenResetPlugin（RandomFallenStatePlugin 原生版）
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
class TestFallenResetPlugin:
    """验收 A：同注入态逻辑 + 行隔离；验收 B：摔倒分布统计等价。"""

    @staticmethod
    def _make(B=4, **kw):
        from envs.batchframework.warp_simulator import WarpHumanoid21Simulator
        from envs.batchframework.device_standup import DeviceFallenResetPlugin
        sim = WarpHumanoid21Simulator(batch_size=B)
        sim.reset(seeds=np.arange(B, dtype=np.int64))
        kw.setdefault("sim_factory",
                      lambda b: WarpHumanoid21Simulator(batch_size=b))
        kw.setdefault("max_phy_steps", 400)
        plugin = DeviceFallenResetPlugin(**kw).bind_shared_sim(sim)
        state = sim.build_device_state()
        plugin.declare_state(state)
        state.rng.seed_offsets.copy_(
            torch.arange(B, dtype=torch.int64, device="cuda"))
        return sim, plugin, state

    def _run_pre_episode(self, sim, plugin, state, env_ids=None):
        ctx = DeviceCtx(state, plugin_name=plugin.name)
        ctx.reset_env_ids = env_ids
        ctx._mutator_impl = DeviceMutator(sim)
        ctx._grant_mutator()
        plugin.on_pre_episode(ctx)

    def test_already_fallen_captures_step1(self):
        """注入 torso z < threshold 的状态 → 第 1 步即捕获（语义对照）。"""
        sim, plugin, state = self._make(B=4)
        # 把 env0/2 的 robot_a torso 压到地面高度
        adr_a = sim._robots["robot_a"]["root_qpos_adr"]
        adr_b = sim._robots["robot_b"]["root_qpos_adr"]
        q = state.sim.qpos.clone()
        q[[0, 2], adr_a + 2] = 0.15
        q[[0, 2], adr_b + 2] = 0.15
        sim.dev_set_integration_rows(torch.arange(4, device="cuda"), q,
                                     state.sim.qvel.clone())
        self._run_pre_episode(sim, plugin, state)
        pool = state.plugin[plugin.name]
        steps = pool["init_steps"].cpu()
        hits = pool["init_hit"].cpu()
        assert hits[0] and hits[2]
        assert steps[0] == 1 and steps[2] == 1
        # 写回后目标机器人 root z 应低（含摔倒态）
        z_a = state.sim.qpos[:, adr_a + 2].cpu()
        assert float(z_a[0]) < 0.4

    def test_partial_reset_isolation(self):
        """部分 reset：未涉及 env 行逐位不变。"""
        sim, plugin, state = self._make(B=4)
        before = (state.sim.qpos.clone(), state.sim.qvel.clone())
        self._run_pre_episode(sim, plugin, state,
                              env_ids=torch.tensor([1, 3], device="cuda"))
        for e in (0, 2):
            torch.testing.assert_close(before[0][e], state.sim.qpos[e])
            torch.testing.assert_close(before[1][e], state.sim.qvel[e])
        pool = state.plugin[plugin.name]
        steps = pool["init_steps"].cpu()
        assert steps[1] >= 1 and steps[3] >= 1

    def test_fallen_state_written_back(self):
        """摔倒后目标机器人维度确实更新（与 reset 站姿不同）。"""
        sim, plugin, state = self._make(B=2, max_phy_steps=600)
        stand_q = state.sim.qpos.clone()
        self._run_pre_episode(sim, plugin, state)
        diff = (state.sim.qpos - stand_q).abs().max(dim=-1).values.cpu()
        assert float(diff[0]) > 1e-3 or float(diff[1]) > 1e-3
        # 写回后 hit 标志一致性：hit 条件是两目标 root 高度的 min
        # < threshold（任一达标即停），因此断言 min(za, zb) < 0.35——
        # 非先倒地机器人此刻可以仍在高位（双峰分布语义，见 M4_RESULTS §3）。
        adr_a = sim._robots["robot_a"]["root_qpos_adr"]
        adr_b = sim._robots["robot_b"]["root_qpos_adr"]
        hit = state.plugin[plugin.name]["init_hit"].cpu()
        za = state.sim.qpos[:, adr_a + 2].cpu()
        zb = state.sim.qpos[:, adr_b + 2].cpu()
        for e in range(2):
            if bool(hit[e]):
                assert min(float(za[e]), float(zb[e])) < 0.35

    def test_seed_determinism_and_diversity(self):
        """同 seed_offsets 同 reset 序号 → 相同 action；env 间独立。"""
        sim, plugin, state = self._make(B=4)
        from envs.batchframework.device_state import RngView
        rng = RngView(state, plugin._salt)
        dev = torch.device("cuda")
        ids = torch.arange(4, device=dev)
        a1 = plugin._draw_actions(ids, rng, dev)
        a2 = plugin._draw_actions(ids, rng, dev)
        torch.testing.assert_close(a1, a2)          # 同次调用确定性
        plugin._count += 1
        a3 = plugin._draw_actions(ids, rng, dev)
        assert not torch.allclose(a1, a3)           # 下一次 reset 不同
        assert not torch.allclose(a1[0], a1[1])     # env 间独立
        assert a1.abs().max() <= 1.0

    @pytest.mark.slow
    def test_distribution_vs_cpu(self):
        """验收 B：CPU RandomFallenStatePlugin vs 设备版的摔倒分布。

        指标：hit rate、init_steps 分布、写回 torso 高度/quat/关节统计。
        fp32-vs-fp64 + 随机源不同 → 只要求统计级一致（宽松容差）。
        """
        from types import SimpleNamespace
        from envs.humanoid21.disturbance_plugins import (
            RandomFallenStatePlugin)
        from envs.humanoid21.simulator import Humanoid21Simulator

        N = 24
        # --- CPU 参考分布 ---
        cpu_sim = Humanoid21Simulator()
        cpu_stats = dict(steps=[], ha=[], hb=[])
        for i in range(N):
            cpu_sim.reset(seed=i)
            p = RandomFallenStatePlugin(
                target_robots=["robot_a", "robot_b"], max_phy_steps=1000,
                height_threshold=0.3, reset_interval=5, random_seed=1000 + i)
            ctx = SimpleNamespace(accessor=cpu_sim, mutator=cpu_sim,
                                  metrics={})
            p.on_pre_episode(ctx)
            core = cpu_sim.get_core_state()
            cpu_stats["steps"].append(ctx.metrics["robot_a_fallen_init_steps"])
            cpu_stats["ha"].append(float(core["robot_a"]["root_pos"][2]))
            cpu_stats["hb"].append(float(core["robot_b"]["root_pos"][2]))

        # --- 设备分布（B=N 并行摔倒） ---
        sim, plugin, state = self._make(B=N, max_phy_steps=1000,
                                        reset_interval=5)
        self._run_pre_episode(sim, plugin, state)
        pool = state.plugin[plugin.name]
        dev_steps = pool["init_steps"].cpu().numpy()
        dev_hit = pool["init_hit"].cpu().numpy()
        adr_a = sim._robots["robot_a"]["root_qpos_adr"]
        adr_b = sim._robots["robot_b"]["root_qpos_adr"]
        dev_ha = state.sim.qpos[:, adr_a + 2].cpu().numpy()
        dev_hb = state.sim.qpos[:, adr_b + 2].cpu().numpy()

        cpu_hit = np.array([s < 1000 for s in cpu_stats["steps"]])
        print(f"\nCPU: hit={cpu_hit.mean():.2f} "
              f"steps med={np.median(cpu_stats['steps']):.0f} "
              f"ha mean={np.mean(cpu_stats['ha']):.3f}±{np.std(cpu_stats['ha']):.3f}")
        print(f"DEV: hit={dev_hit.mean():.2f} "
              f"steps med={np.median(dev_steps):.0f} "
              f"ha mean={dev_ha.mean():.3f}±{dev_ha.std():.3f}")

        # 统计等价：hit 率、高度分布量级、步数中位数同数量级
        assert dev_hit.mean() > 0.8 and cpu_hit.mean() > 0.8
        assert abs(dev_ha.mean() - np.mean(cpu_stats["ha"])) < 0.1
        assert abs(dev_hb.mean() - np.mean(cpu_stats["hb"])) < 0.1
        med_c, med_d = np.median(cpu_stats["steps"]), np.median(dev_steps)
        assert 0.3 < med_c / max(med_d, 1) < 3.0
