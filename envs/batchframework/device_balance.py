"""basic_balance 任务的设备端原生单元（E5 W1）。

转换自（每个单元的 CPU 语义单一来源 = 直接引用 CPU 模块，不复制
字面量常量）：

- ``baseline.humanoid21.plugins.imbalance_termination
  :DualImbalanceTerminationPlugin`` → ``DeviceDualImbalancePlugin``
  **逐 agent 终止**（``agents=[a]``）——本框架首个真实不对称终止
  用例：A 终 B 续跑，env 在双 agent 皆终止后才 ENDED。
- ``baseline.humanoid21.rewards.cross_support
  :CrossSupportBalanceRewarder`` → ``DeviceCrossSupportObserver``
  （状态机经 (B,) 张量化；``get_output`` 返回标量 reward (B,)）。
- ``baseline.humanoid21.rewards.posture_reward:PostureRewarder``
  → ``DevicePostureObserver``。
- ``baseline.humanoid21.plugins.height_phi_observer:HeightPhiObserver``
  → ``DeviceHeightPhiObserver``。

精度承诺与 standup 相同：fp32 设备端计算对 CPU fp64/串行结果
**数值近似**（接触力/姿态的物理量本身已是近似），语义逐字段对齐
由 tests/test_device_balance.py 的同输入回放锁定。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from baseline.humanoid21.rewards.posture_reward import PostureRewarder
from baseline.humanoid21.rewards.cross_support import (
    CROSS_SUPPORT_DOUBLE_SUPPORT_MAX_STEPS,
    CROSS_SUPPORT_DOUBLE_SUPPORT_PENALTY_COEF,
    CROSS_SUPPORT_FOOT_LIFT_MIN_HEIGHT,
    CROSS_SUPPORT_FOOT_LIFT_MIN_STEPS,
    CROSS_SUPPORT_FOOT_LIFT_PENALTY_COEF,
    CROSS_SUPPORT_INITIAL_GRACE_STEPS,
    CROSS_SUPPORT_INITIAL_PENALTY_COEF,
    CROSS_SUPPORT_MIN_HEIGHT,
    CROSS_SUPPORT_SINGLE_SUPPORT_BONUS,
    CROSS_SUPPORT_STANDING_FOOT_HEIGHT,
    CROSS_SUPPORT_SWITCH_INTERVAL_MAX_STEPS,
    CROSS_SUPPORT_SWITCH_INTERVAL_PENALTY_COEF,
)

from .device_obs import contact_forces_flat
from .device_plugin import (
    BaseDeviceObserver, BaseDevicePlugin, DeviceCtx,
)

_AGENT_RID = ("robot_a", "robot_b")
_AGENT_AFF = (1, 2)


def _robot_ground_contacts(state, geom_bodyid, geom_aff,
                           ground_gid: int, robot_aff: int,
                           min_force: float = 0.0):
    """→ (worldid (M,), robot_body (M,))：env(ground)↔指定 robot 的活跃接触。

    ``min_force=0`` → "有接触条目即算"（CPU cross_support 语义）；
    >0 → ``force_mag >= min_force``（CPU imbalance 语义）。
    """
    cf = contact_forces_flat(state, geom_bodyid, geom_aff)
    if cf.sel.numel() == 0:
        return None
    ga = (cf.aff1 == 0) & (cf.geom1 == ground_gid) & (cf.aff2 == robot_aff)
    gb = (cf.aff2 == 0) & (cf.geom2 == ground_gid) & (cf.aff1 == robot_aff)
    hit = ga | gb
    if min_force > 0.0:
        hit = hit & (cf.force_mag >= min_force)
    if not bool(hit.any()):
        return None
    m = torch.nonzero(hit, as_tuple=False).squeeze(-1)
    rb = torch.where(ga[m], cf.body2[m], cf.body1[m])
    return cf.worldid[m], rb


def _foot_ground_contact(state, geom_bodyid, geom_aff,
                         ground_gid: int, robot_aff: int,
                         body_ids: Tuple[int, ...],
                         min_force: float = 0.0) -> torch.Tensor:
    """→ (B,) bool：指定 robot 的任一 ``body_ids`` 内 body 触地。"""
    r = _robot_ground_contacts(state, geom_bodyid, geom_aff,
                               ground_gid, robot_aff, min_force)
    out = torch.zeros(state.batch_size, dtype=torch.bool,
                      device=state.sim.qpos.device)
    if r is None:
        return out
    w, rb = r
    hit = torch.zeros_like(w, dtype=torch.bool)
    for b in body_ids:
        hit |= rb == int(b)
    out[w[hit]] = True
    return out


def _nonfoot_ground_contact(state, geom_bodyid, geom_aff,
                            ground_gid: int, robot_aff: int,
                            foot_ids: Tuple[int, int],
                            min_force: float) -> torch.Tensor:
    """→ (B,) bool：指定 robot 的**非足部** body 触地（力≥阈值）。

    CPU ``_is_non_foot_grounded``：body_robot ∉ {foot_left, foot_right}
    的 robot↔ground 接触存在即真。
    """
    r = _robot_ground_contacts(state, geom_bodyid, geom_aff,
                               ground_gid, robot_aff, min_force)
    out = torch.zeros(state.batch_size, dtype=torch.bool,
                      device=state.sim.qpos.device)
    if r is None:
        return out
    w, rb = r
    hit = (rb != int(foot_ids[0])) & (rb != int(foot_ids[1]))
    out[w[hit]] = True
    return out


def _uprightness(xquat_torso: torch.Tensor) -> torch.Tensor:
    """(B,4) wxyz → R[2,2]（CPU ``world_rot_mat[2,2]`` 同式）。"""
    x, y = xquat_torso[:, 1], xquat_torso[:, 2]
    return 1.0 - 2.0 * (x * x + y * y)


def _torso_uprightness(state, torso_id: int) -> torch.Tensor:
    return _uprightness(state.sim.xquat[:, torso_id])


def _root_height(state, root_qpos_adr: int) -> torch.Tensor:
    """(B,) f32——CPU ``root_pos[2]`` = free joint qpos z。"""
    return state.sim.qpos[:, root_qpos_adr + 2]


# ---------------------------------------------------------------------------
# 1) DualImbalanceTerminationPlugin → DeviceDualImbalancePlugin
# ---------------------------------------------------------------------------
class DeviceDualImbalancePlugin(BaseDevicePlugin):
    """逐 agent 失衡终止（per-agent ``agent_done``）。

    CPU 语义逐项对应
    （``imbalance_termination.py:DualImbalanceTerminationPlugin``）：

    - 每 action step 采样一次接触快照（CPU 同名类的判定粒度是
      action step，不是物理子步）；
    - 非足-地接触 = env↔robot_aff 接触，env 侧 geom==ground、
      robot 侧 body 非 foot_left/foot_right、force≥threshold；
    - counter: 命中 +1 / 未命中 -1（下界 0），按 env×agent 独立；
    - counter≥tolerance 且该 agent 未终止 →
      ``request_termination(ids, "imbalance_robot_{a|b}", agents=[a])``；
    - min_height>0 时 env 内两机高度都低于阈值 → 本 env 跳过判定
      （counter 也不更新——CPU 的 all_below 早退语义）。

    状态经 ``declare_state`` 入 plugin pool（``counter`` (B,2) i32），
    行 reset 自动清零 = CPU ``on_pre_episode`` 归零。
    """

    def __init__(self, force_threshold: float = 1.0, tolerance: int = 1,
                 min_height: float = 0.0, *, tables: Any = None):
        self.force_threshold = float(force_threshold)
        self.tolerance = int(tolerance)
        self.min_height = float(min_height)
        self._t = tables  # from_sim 填的任务设备表摘要

    @classmethod
    def from_sim(cls, sim, force_threshold: float = 1.0,
                 tolerance: int = 1, min_height: float = 0.0,
                 obs_builder=None):
        tables = sim.task_tables()
        if obs_builder is None:
            obs_builder = sim.device_obs_builder()
        geom_bodyid, geom_aff = obs_builder.contact_tables
        t = {
            "ground_gid": tables.ground_geom_id,
            "geom_bodyid": geom_bodyid,
            "geom_aff": geom_aff,
            "foot_ids": [(
                tables.robots[rid]["keypoint_body_ids"]["foot_left"],
                tables.robots[rid]["keypoint_body_ids"]["foot_right"],
            ) for rid in _AGENT_RID],
            "root_adr": [tables.robots[rid]["root_qpos_adr"]
                         for rid in _AGENT_RID],
        }
        return cls(force_threshold=force_threshold, tolerance=tolerance,
                   min_height=min_height, tables=t)

    @property
    def name(self) -> str:
        return "device_dual_imbalance_termination"

    @property
    def declared_reads(self):
        return ("qpos", "contacts_flat")

    def declare_state(self, state) -> None:
        state.declare_state(self.name, "counter", (2,), torch.int32)

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        st, ep, t = ctx.state, ctx.state.episode, self._t
        dev = ep.agent_done.device
        B = st.batch_size

        gate = torch.ones(B, dtype=torch.bool, device=dev)
        if self.min_height > 0.0:
            # CPU: 两台 robot 都 < min_height → 整 env 跳过
            below = torch.ones(B, dtype=torch.bool, device=dev)
            for adr in t["root_adr"]:
                below &= _root_height(st, adr) < self.min_height
            gate = ~below

        fell = torch.stack([
            _nonfoot_ground_contact(
                st, t["geom_bodyid"], t["geom_aff"], t["ground_gid"],
                _AGENT_AFF[a], t["foot_ids"][a],
                min_force=self.force_threshold)
            for a in (0, 1)], dim=-1)                      # (B,2)

        counter = ctx.pstate["counter"]
        new_c = torch.where(fell, counter + 1, counter - 1).clamp(min=0)
        counter.copy_(torch.where(gate[:, None], new_c, counter))

        for a, rid in enumerate(_AGENT_RID):
            req = gate & (counter[:, a] >= self.tolerance) \
                & ~ep.agent_done[:, a]
            if bool(req.any()):
                ids = torch.nonzero(req, as_tuple=False).squeeze(-1)
                ctx.request_termination(
                    ids, f"imbalance_{rid}", agents=[a])


# ---------------------------------------------------------------------------
# 2) CrossSupportBalanceRewarder → DeviceCrossSupportObserver
# ---------------------------------------------------------------------------
class DeviceCrossSupportObserver(BaseDeviceObserver):
    """交替支撑平衡奖励的状态机批量版。

    状态变量全部为 (B,) 张量（实例自持；``on_pre_episode`` 全量建、
    ``on_envs_reset`` 按 reset 行重置——对应 CPU ``on_pre_episode``
    归零）。状态编码：state 0=wait/1=tracking；foot 字段 -1=none/
    0=left/1=right。

    CPU 语义对齐点：

    - foot 接触**无力阈值**（有条目即算），加 foot_lift 高度补充
      （anchor z < 站立阈值+min_height → 仍算接触）；
    - wait 态首入 single-support 的当步 reward=0 且不计 timer；
    - tracking 态：段结束结算过短惩罚、对侧首现结算换脚间隔惩罚；
    - ``min_height`` 门只影响 double_support/bonus，不影响段计时。

    ``get_output`` 返回标量 (B,)——CPU 返回 float 标量。
    """

    def __init__(self, agent_idx: int, *,
                 initial_grace_steps=CROSS_SUPPORT_INITIAL_GRACE_STEPS,
                 initial_penalty_coef=CROSS_SUPPORT_INITIAL_PENALTY_COEF,
                 foot_lift_min_steps=CROSS_SUPPORT_FOOT_LIFT_MIN_STEPS,
                 foot_lift_penalty_coef=CROSS_SUPPORT_FOOT_LIFT_PENALTY_COEF,
                 switch_interval_max_steps=CROSS_SUPPORT_SWITCH_INTERVAL_MAX_STEPS,
                 switch_interval_penalty_coef=CROSS_SUPPORT_SWITCH_INTERVAL_PENALTY_COEF,
                 double_support_max_steps=CROSS_SUPPORT_DOUBLE_SUPPORT_MAX_STEPS,
                 double_support_penalty_coef=CROSS_SUPPORT_DOUBLE_SUPPORT_PENALTY_COEF,
                 single_support_bonus=CROSS_SUPPORT_SINGLE_SUPPORT_BONUS,
                 min_height=CROSS_SUPPORT_MIN_HEIGHT,
                 foot_lift_min_height=CROSS_SUPPORT_FOOT_LIFT_MIN_HEIGHT,
                 tables: Any = None):
        assert agent_idx in (0, 1)
        self.agent_idx = int(agent_idx)
        self.initial_grace_steps = int(initial_grace_steps)
        self.initial_penalty_coef = float(initial_penalty_coef)
        self.foot_lift_min_steps = int(foot_lift_min_steps)
        self.foot_lift_penalty_coef = float(foot_lift_penalty_coef)
        self.switch_interval_max_steps = max(0, int(switch_interval_max_steps))
        self.switch_interval_penalty_coef = float(switch_interval_penalty_coef)
        self.double_support_max_steps = max(0, int(double_support_max_steps))
        self.double_support_penalty_coef = float(double_support_penalty_coef)
        self.single_support_bonus = float(single_support_bonus)
        self.min_height = float(min_height)
        self.foot_lift_min_height = float(foot_lift_min_height)
        self._t = tables
        self._v: Optional[Dict[str, torch.Tensor]] = None
        self._out: Optional[Dict[str, torch.Tensor]] = None

    @classmethod
    def from_sim(cls, sim, agent_idx: int, obs_builder=None, **kw):
        tables = sim.task_tables()
        rid = _AGENT_RID[agent_idx]
        kp = tables.robots[rid]["keypoint_body_ids"]
        if obs_builder is None:
            obs_builder = sim.device_obs_builder()
        geom_bodyid, geom_aff = obs_builder.contact_tables
        # ankle_x_{left,right} 的 joint id → xanchor 行
        ankle = tables.robots[rid].get("keypoint_joint_ids") or {}
        t = {
            "ground_gid": tables.ground_geom_id,
            "geom_bodyid": geom_bodyid, "geom_aff": geom_aff,
            "robot_aff": agent_idx + 1,
            "foot_l": int(kp["foot_left"]), "foot_r": int(kp["foot_right"]),
            "root_adr": tables.robots[rid]["root_qpos_adr"],
            "ankle_l_jid": ankle.get("ankle_x_left"),
            "ankle_r_jid": ankle.get("ankle_x_right"),
        }
        return cls(agent_idx, tables=t, **kw)

    # ------------------------------------------------------------------
    @property
    def output_schema(self):
        # 单叶标量——CPU get_output 返回 float；dict 形态供
        # RecordStore 写入，extract_per_step_scalar 取首叶。
        return {"reward": (torch.float32, ())}

    def get_output(self) -> Any:
        return self._out

    # ------------------------------------------------------------------
    def _alloc(self, B: int, dev) -> None:
        z32 = torch.zeros(B, dtype=torch.int32, device=dev)
        self._v = {
            "state": torch.zeros(B, dtype=torch.int8, device=dev),
            "timer": z32.clone(),
            "cur_foot": torch.full((B,), -1, dtype=torch.int8, device=dev),
            "cur_steps": z32.clone(),
            "anchor": torch.full((B,), -1, dtype=torch.int8, device=dev),
            "switch_steps": z32.clone(),
            "ds_steps": z32.clone(),
        }
        self._out = {"reward": torch.zeros(B, dtype=torch.float32,
                                           device=dev)}

    def _reset_rows(self, env_ids) -> None:
        v = self._v
        v["state"][env_ids] = 0
        v["timer"][env_ids] = 0
        v["cur_foot"][env_ids] = -1
        v["cur_steps"][env_ids] = 0
        v["anchor"][env_ids] = -1
        v["switch_steps"][env_ids] = 0
        v["ds_steps"][env_ids] = 0
        self._out["reward"][env_ids] = 0.0

    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        self._alloc(ctx.batch_size, ctx.state.sim.qpos.device)

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        if self._v is None:
            self._alloc(ctx.batch_size, ctx.state.sim.qpos.device)
            return
        if ctx.reset_env_ids is None:
            self._reset_rows(
                torch.arange(ctx.batch_size,
                             device=self._out["reward"].device))
        elif ctx.reset_env_ids.numel():
            self._reset_rows(
                ctx.reset_env_ids.to(self._out["reward"].device))

    # ------------------------------------------------------------------
    def _foot_contacts(self, ctx) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self._t
        st = ctx.state
        B = st.batch_size
        dev = st.sim.qpos.device
        lc = _foot_ground_contact(st, t["geom_bodyid"], t["geom_aff"],
                                  t["ground_gid"], t["robot_aff"],
                                  body_ids=(t["foot_l"],))
        rc = _foot_ground_contact(st, t["geom_bodyid"], t["geom_aff"],
                                  t["ground_gid"], t["robot_aff"],
                                  body_ids=(t["foot_r"],))
        if self.foot_lift_min_height > 0.0 and t["ankle_l_jid"] is not None:
            thr = CROSS_SUPPORT_STANDING_FOOT_HEIGHT + \
                self.foot_lift_min_height
            lh = st.sim.xanchor[:, t["ankle_l_jid"], 2]
            rh = st.sim.xanchor[:, t["ankle_r_jid"], 2]
            lc = lc | (lh < thr)
            rc = rc | (rh < thr)
        return lc, rc

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        if self._v is None:
            self._alloc(ctx.batch_size, ctx.state.sim.qpos.device)
        v, t = self._v, self._t
        dev = self._out["reward"].device
        B = ctx.batch_size
        f32 = torch.float32

        lc, rc = self._foot_contacts(ctx)
        root_h = (_root_height(ctx.state, t["root_adr"])
                  if self.min_height > 0.0
                  else torch.ones(B, device=dev))
        tall = root_h >= self.min_height if self.min_height > 0.0 \
            else torch.ones(B, dtype=torch.bool, device=dev)

        reward = torch.zeros(B, dtype=f32, device=dev)

        # --- double-support 惩罚（CPU 同序，先于状态机） ---
        if self.double_support_max_steps > 0:
            both = lc & rc & tall
            ds = torch.where(both, v["ds_steps"] + 1,
                             torch.zeros_like(v["ds_steps"]))
            v["ds_steps"].copy_(ds)
            excess = (v["ds_steps"] - self.double_support_max_steps) \
                .clamp(min=0).to(f32)
            denom = max(1, self.double_support_max_steps)
            reward -= torch.where(
                v["ds_steps"] > self.double_support_max_steps,
                self.double_support_penalty_coef
                * (excess / denom).clamp(max=1.0),
                torch.zeros_like(reward))

        ss = torch.where(lc & ~rc, torch.zeros(B, dtype=torch.int8,
                                               device=dev),
                         torch.where(rc & ~lc,
                                     torch.ones(B, dtype=torch.int8,
                                                device=dev),
                                     torch.full((B,), -1,
                                                dtype=torch.int8,
                                                device=dev)))

        # tracking 掩码取迁移**前**的状态——CPU 在 WAIT→TRACKING 迁移
        # 当步从 _handle_wait_first_single_support 早退，tracking 逻辑
        # 不在该步运行。
        tracking0 = v["state"] == 1
        waiting = v["state"] == 0
        enter = waiting & (ss >= 0)
        # WAIT → TRACKING 迁移：reward=0、timer 不加（CPU 早退语义）
        v["state"] = torch.where(enter, torch.ones_like(v["state"]),
                                 v["state"])
        v["cur_foot"] = torch.where(enter, ss, v["cur_foot"])
        v["cur_steps"] = torch.where(enter, torch.ones_like(v["cur_steps"]),
                                     v["cur_steps"])
        v["anchor"] = torch.where(enter, ss, v["anchor"])
        v["switch_steps"] = torch.where(
            enter, torch.zeros_like(v["switch_steps"]), v["switch_steps"])
        # WAIT 且未进入 → timer+1，超 grace 线性惩罚
        wait_no = waiting & ~enter
        v["timer"] = v["timer"] + wait_no.to(torch.int32)
        over = (v["timer"] - self.initial_grace_steps).clamp(min=0).to(f32)
        denom = max(1, self.initial_grace_steps)
        reward -= torch.where(
            wait_no & (v["timer"] > self.initial_grace_steps),
            self.initial_penalty_coef * (over / denom).clamp(max=1.0),
            torch.zeros_like(reward))

        # --- TRACKING ---
        tracking = tracking0
        if self.single_support_bonus > 0.0:
            reward += torch.where(
                tracking & (ss >= 0) & tall,
                torch.full_like(reward, self.single_support_bonus),
                torch.zeros_like(reward))
        v["switch_steps"] = v["switch_steps"] + tracking.to(torch.int32)

        cur, cst = v["cur_foot"], v["cur_steps"]
        none_cur = cur < 0
        same = (~none_cur) & (ss == cur)
        trans = tracking & (~none_cur) & (ss != cur)          # 含 ss=-1
        # 段结束结算过短惩罚
        short_pen = torch.where(
            trans & (cst < self.foot_lift_min_steps),
            self.foot_lift_penalty_coef
            * ((self.foot_lift_min_steps - cst).to(f32)
               / max(1, self.foot_lift_min_steps)),
            torch.zeros_like(reward))
        reward -= short_pen
        # 迁移：cur_foot 更新
        new_cur = torch.where(
            tracking & none_cur & (ss >= 0), ss,
            torch.where(trans, ss, cur))
        new_cst = torch.where(
            tracking & none_cur & (ss >= 0),
            torch.ones_like(cst),
            torch.where(same & tracking, cst + 1,
                        torch.where(trans,
                                    torch.where(ss < 0,
                                                torch.zeros_like(cst),
                                                torch.ones_like(cst)),
                                    cst)))
        v["cur_foot"].copy_(new_cur)
        v["cur_steps"].copy_(new_cst)

        # 换脚间隔：对侧首现结算
        crossed = tracking & (ss >= 0) & (v["anchor"] >= 0) \
            & (ss != v["anchor"])
        excess = (v["switch_steps"] - self.switch_interval_max_steps) \
            .clamp(min=0).to(f32)
        denom = max(1, self.switch_interval_max_steps)
        reward -= torch.where(
            crossed & (v["switch_steps"] > self.switch_interval_max_steps),
            self.switch_interval_penalty_coef
            * (excess / denom).clamp(max=1.0),
            torch.zeros_like(reward))
        v["anchor"] = torch.where(crossed, ss, v["anchor"])
        v["switch_steps"] = torch.where(
            crossed, torch.zeros_like(v["switch_steps"]),
            v["switch_steps"])

        self._out["reward"].copy_(reward)

    def on_post_episode(self, ctx: DeviceCtx) -> None:
        pass


# ---------------------------------------------------------------------------
# 3) PostureRewarder → DevicePostureObserver
# ---------------------------------------------------------------------------
_POSTURE_KEYS = ("joint_deviation", "joint_vel", "torso_tilt", "foot_height")


class DevicePostureObserver(BaseDeviceObserver):
    """姿态诊断 observer 的批量版（4 个 (B,) f32 叶）。

    - joint_deviation: |joint_pos_norm - STANDING_JOINT_POS|.mean
      （``STANDING_JOINT_POS`` 引用 CPU 常量单一来源）；
    - joint_vel: |joint_vel_norm|.mean；
    - torso_tilt: arccos(uprightness)（R[2,2]）；
    - foot_height: max(foot_l.z, foot_r.z)。
    """

    def __init__(self, agent_idx: int, *, tables: Any = None):
        assert agent_idx in (0, 1)
        self.agent_idx = int(agent_idx)
        self._t = tables
        self._out: Optional[Dict[str, torch.Tensor]] = None
        self._ref = torch.as_tensor(
            PostureRewarder.STANDING_JOINT_POS, dtype=torch.float32)

    @classmethod
    def from_sim(cls, sim, agent_idx: int, **kw):
        tables = sim.task_tables()
        r = tables.robots[_AGENT_RID[agent_idx]]
        kp = r["keypoint_body_ids"]
        t = {
            "torso": int(r["root_body_id"]),
            "qpos_idx": r["qpos_indices"], "qvel_idx": r["qvel_indices"],
            "norm_ref": r["norm_ref"], "norm_scale": r["norm_scale"],
            "foot_l": int(kp["foot_left"]), "foot_r": int(kp["foot_right"]),
        }
        return cls(agent_idx, tables=t, **kw)

    @property
    def output_schema(self):
        return {k: (torch.float32, ()) for k in _POSTURE_KEYS}

    def get_output(self):
        return self._out

    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        self._out = None

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        self._out = None

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        s, t = ctx.sim, self._t
        ref = self._ref.to(device=s.qpos.device)
        jpn = (s.qpos[:, t["qpos_idx"]] - t["norm_ref"]) / t["norm_scale"]
        jvn = s.qvel[:, t["qvel_idx"]] / t["norm_scale"]
        upr = _torso_uprightness(ctx.state, t["torso"])
        tilt = torch.arccos(upr.clamp(-1.0, 1.0))
        fh = torch.maximum(s.xpos[:, t["foot_l"], 2],
                           s.xpos[:, t["foot_r"], 2])
        self._out = {
            "joint_deviation": (jpn - ref).abs().mean(dim=-1),
            "joint_vel": jvn.abs().mean(dim=-1),
            "torso_tilt": tilt,
            "foot_height": fh,
        }

    def on_post_episode(self, ctx: DeviceCtx) -> None:
        pass


# ---------------------------------------------------------------------------
# 4) HeightPhiObserver → DeviceHeightPhiObserver
# ---------------------------------------------------------------------------
_PHI_KEYS = ("height", "uprightness", "phi", "initial_phi")


class DeviceHeightPhiObserver(BaseDeviceObserver):
    """高度/直立度/φ observer 的批量版。

    ``phi = uprightness * height / standing_height``；
    ``initial_phi`` 在 episode 起点捕获（partial reset 行经
    ``on_envs_reset`` 用复位后姿态重捕获——CPU ``on_pre_episode``
    同语义）。
    """

    def __init__(self, agent_idx: int, standing_height: float = 1.28,
                 *, tables: Any = None):
        assert agent_idx in (0, 1)
        self.agent_idx = int(agent_idx)
        self.standing_height = float(standing_height)
        self._t = tables
        self._init_phi: Optional[torch.Tensor] = None
        self._out: Optional[Dict[str, torch.Tensor]] = None

    @classmethod
    def from_sim(cls, sim, agent_idx: int,
                 standing_height: float = 1.28, **kw):
        tables = sim.task_tables()
        r = tables.robots[_AGENT_RID[agent_idx]]
        return cls(agent_idx, standing_height=standing_height, tables={
            "torso": int(r["root_body_id"]),
            "root_adr": int(r["root_qpos_adr"]),
        })

    @property
    def output_schema(self):
        return {k: (torch.float32, ()) for k in _PHI_KEYS}

    def get_output(self):
        return self._out

    # ------------------------------------------------------------------
    def _phi_now(self, ctx) -> Tuple[torch.Tensor, torch.Tensor]:
        """→ (height, uprightness) 当前帧。"""
        st = ctx.state
        h = _root_height(st, self._t["root_adr"])
        u = _torso_uprightness(st, self._t["torso"])
        return h, u

    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        h, u = self._phi_now(ctx)
        self._init_phi = u * (h / self.standing_height)
        self._out = None

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        if self._init_phi is None:
            self.on_pre_episode(ctx)
            return
        h, u = self._phi_now(ctx)
        ids = (torch.arange(ctx.batch_size, device=h.device)
               if ctx.reset_env_ids is None else ctx.reset_env_ids)
        self._init_phi[ids] = (u * (h / self.standing_height))[ids]
        self._out = None

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        h, u = self._phi_now(ctx)
        phi = u * (h / self.standing_height)
        self._out = {"height": h, "uprightness": u, "phi": phi,
                     "initial_phi": self._init_phi.clone()}

    def on_post_episode(self, ctx: DeviceCtx) -> None:
        pass
