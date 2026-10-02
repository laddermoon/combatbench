"""standup 任务的设备端原生插件（M4）。

转换自：
- ``baseline.humanoid21.rewards.standing_balance_4stage.StandingBalance4StageRewarder``
  → ``DeviceStandup4StageRewarder``（BaseDeviceObserver，per-agent 参数化）

所有阈值/常量**单一来源**：直接 import CPU 模块的常量，不复制字面量。
语义对照点（与 CPU 逐字段对齐）：
- extra_contact_count 是 distinct body 去重计数（非接触点计数）；
- 接触计入条件：env 侧 geom==ground 且 robot 侧 aff==本 agent，
  force_mag ≥ 1.0N；
- w_foot 门限 F_LOAD_MIN=10N 以下置 0；
- d_score 是 XY 平面距离（站立兼容，见 CPU 侧 Stage3 注释）；
- stage 自上而下判定，无迟滞。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from baseline.humanoid21.rewards.standing_balance_4stage import (
    D_GATE,
    D_MAX,
    D_MIN,
    F_ENTER,
    F_LOAD_MIN,
    H_CROUCH,
    H_FOOT_MAX,
    H_HAND_MAX,
    H_STAND,
    OTHER_PENALTY_K,
)

from .device_obs import contact_forces_flat
from .device_plugin import (
    BaseDeviceObserver, BaseDevicePlugin, DeviceCtx,
)
from .device_state import (
    _lshr64 as _lshr,
    RngView,
    splitmix64 as _splitmix64,
)

# 输出字段顺序（get_output dict 的稳定键序）
OUT_KEYS = ("stage", "potential", "f_score", "contact_score", "d_score",
            "d_hf", "w_foot", "h_score", "h_torso")


class DeviceStandup4StageRewarder(BaseDeviceObserver):
    """StandingBalance4StageRewarder 的设备端批量版。

    Args:
        agent_idx: 0=robot_a / 1=robot_b。
        body_tables: 由 ``from_sim`` 构造时填——torso/hand/foot 的 body id
            与 contact 查找表。
    """

    def __init__(self, agent_idx: int, body_tables: Dict[str, Any]):
        assert agent_idx in (0, 1)
        self.agent_idx = int(agent_idx)
        self._t = body_tables
        self._out: Optional[Dict[str, torch.Tensor]] = None

    @classmethod
    def from_sim(cls, sim, agent_idx: int, obs_builder=None) -> "DeviceStandup4StageRewarder":
        """从任务设备表提取 body id / 查找表（W4：不再读 sim 私有字段）。"""
        tables = sim.task_tables()
        rid = "robot_a" if agent_idx == 0 else "robot_b"
        cache = tables.robots[rid]
        kp = cache["keypoint_body_ids"]
        torso_id = kp.get("torso", cache["root_body_id"])
        if obs_builder is None:
            obs_builder = sim.device_obs_builder()
        geom_bodyid, geom_aff = obs_builder.contact_tables
        return cls(agent_idx, dict(
            torso=int(torso_id),
            hand_l=int(kp["hand_left"]), hand_r=int(kp["hand_right"]),
            foot_l=int(kp["foot_left"]), foot_r=int(kp["foot_right"]),
            ground_gid=tables.ground_geom_id,
            robot_aff=agent_idx + 1,
            geom_bodyid=geom_bodyid, geom_aff=geom_aff,
            nbody=tables.nbody,
        ))

    # ------------------------------------------------------------------
    def _contacts_for_agent(self, state):
        """→ (per-env) hand/foot/extra 分类聚合。"""
        cf = contact_forces_flat(state, self._t["geom_bodyid"],
                                 self._t["geom_aff"])
        B = state.batch_size
        dev = cf.worldid.device
        t = self._t
        M = cf.sel.numel()
        zeros = torch.zeros(B, dtype=torch.float32, device=dev)
        extra_bool = torch.zeros(B, t["nbody"], dtype=torch.bool, device=dev)
        cat_force = torch.zeros(B, 4, dtype=torch.float32, device=dev)
        cat_touch = torch.zeros(B, 4, dtype=torch.bool, device=dev)
        # 类别码：0=foot_l 1=foot_r 2=hand_l 3=hand_r；其它 body 单独去重
        if M == 0:
            return dict(cat_force=cat_force, cat_touch=cat_touch,
                        extra_count=zeros)

        strong = cf.force_mag >= 1.0
        # CPU 语义：env 侧 aff==0 且 geom==ground，robot 侧 aff==本 agent
        ga = (cf.aff1 == 0) & (cf.geom1 == t["ground_gid"]) \
            & (cf.aff2 == t["robot_aff"])
        gb = (cf.aff2 == 0) & (cf.geom2 == t["ground_gid"]) \
            & (cf.aff1 == t["robot_aff"])
        rb = torch.where(ga, cf.body2,
                         torch.where(gb, cf.body1,
                                     torch.full_like(cf.body1, -1)))
        valid = strong & (ga | gb)
        m = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if m.numel() == 0:
            return dict(cat_force=cat_force, cat_touch=cat_touch,
                        extra_count=zeros)
        w, body = cf.worldid[m], rb[m]
        f = cf.force_mag[m]

        cat = torch.full((m.numel(),), -1, dtype=torch.long, device=dev)
        cat = torch.where(body == t["foot_l"], torch.zeros_like(cat), cat)
        cat = torch.where(body == t["foot_r"], torch.ones_like(cat), cat)
        cat = torch.where(body == t["hand_l"],
                          torch.full_like(cat, 2), cat)
        cat = torch.where(body == t["hand_r"],
                          torch.full_like(cat, 3), cat)
        hf = cat >= 0
        idx = w[hf] * 4 + cat[hf]
        cat_force.view(-1).index_add_(0, idx, f[hf])
        cat_touch.view(-1)[idx] = True
        # extra = robot 侧 body 不属于手/脚的接触，按 (env, body) 去重
        ex = ~hf
        extra_bool[w[ex], body[ex].clamp(min=0)] = True
        return dict(cat_force=cat_force, cat_touch=cat_touch,
                    extra_count=extra_bool.sum(dim=-1).float())

    # ------------------------------------------------------------------
    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        self._out = None

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        # CPU on_pre_episode 全清零语义；out 缓存在下一步重建，
        # 本步已终止 env 的输出由采集端按 terminated 掩码丢弃。
        self._out = None

    def on_post_action_step(self, ctx: DeviceCtx) -> None:
        s = ctx.sim
        t = self._t

        # --- Signal 1: f_down（朝向） ---
        q = s.xquat[:, t["torso"]]                      # (B,4) wxyz
        x_world_z = 2.0 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
        f_score = ((-x_world_z + 1.0) / 2.0).clamp(0.0, 1.0)

        # --- Signal 2: 支撑 proximity × 额外接触惩罚 ---
        h_hand = (s.xpos[:, t["hand_l"], 2] + s.xpos[:, t["hand_r"], 2]) / 2.0
        h_foot = (s.xpos[:, t["foot_l"], 2] + s.xpos[:, t["foot_r"], 2]) / 2.0
        hand_prox = (1.0 - h_hand / H_HAND_MAX).clamp(0.0, 1.0)
        foot_prox = (1.0 - h_foot / H_FOOT_MAX).clamp(0.0, 1.0)
        support = (hand_prox + foot_prox) / 2.0

        con = self._contacts_for_agent(ctx.state)
        other_pen = 1.0 / (1.0 + OTHER_PENALTY_K * con["extra_count"])
        contact_score = support * other_pen

        # --- Signal 3: hand-mid↔foot-mid XY 距离 ---
        hand_mid = (s.xpos[:, t["hand_l"], :2] + s.xpos[:, t["hand_r"], :2]) / 2
        foot_mid = (s.xpos[:, t["foot_l"], :2] + s.xpos[:, t["foot_r"], :2]) / 2
        d_hf = torch.linalg.norm(hand_mid - foot_mid, dim=-1)
        d_score = ((D_MAX - d_hf) / (D_MAX - D_MIN)).clamp(0.0, 1.0)

        # --- Signal 4: 承重转移 + torso 高度 ---
        f_foot = con["cat_force"][:, 0] + con["cat_force"][:, 1]
        f_hand = con["cat_force"][:, 2] + con["cat_force"][:, 3]
        total = f_foot + f_hand
        w_foot = torch.where(total >= F_LOAD_MIN, f_foot / total.clamp(min=1e-9),
                             torch.zeros_like(total))
        h_torso = s.xpos[:, t["torso"], 2]
        h_score = ((h_torso - H_CROUCH) / (H_STAND - H_CROUCH)).clamp(0.0, 1.0)
        p4 = w_foot * h_score

        # --- stage 自上而下 ---
        hf_touch = con["cat_touch"].any(dim=-1)
        only_hf = (con["extra_count"] == 0) & hf_touch
        stage = torch.ones(ctx.batch_size, dtype=torch.long, device=h_torso.device)
        stage = torch.where(f_score >= F_ENTER, torch.full_like(stage, 2), stage)
        stage = torch.where(only_hf, torch.full_like(stage, 3), stage)
        stage = torch.where(only_hf & (d_score >= D_GATE),
                            torch.full_like(stage, 4), stage)

        potential = torch.where(
            stage == 1, 0.10 * f_score,
            torch.where(
                stage == 2, 0.10 + 0.10 * contact_score,
                torch.where(
                    stage == 3, 0.20 + 0.10 * d_score,
                    0.30 + 0.70 * p4)))

        self._out = {
            "stage": stage.to(torch.float32),
            "potential": potential,
            "f_score": f_score,
            "contact_score": contact_score,
            "d_score": d_score,
            "d_hf": d_hf,
            "w_foot": w_foot,
            "h_score": h_score,
            "h_torso": h_torso,
        }

    def on_post_episode(self, ctx: DeviceCtx) -> None:
        pass  # 输出缓存由下一步重建；episode 汇总由采集端负责

    def get_output(self) -> Dict[str, torch.Tensor]:
        """dict of (B,) 张量；首次 build 前为 None。"""
        return self._out

# ---------------------------------------------------------------------------
# RandomFallenStatePlugin 的设备端原生版（M4 T3）
# ---------------------------------------------------------------------------


class DeviceFallenResetPlugin(BaseDevicePlugin):
    """RandomFallenStatePlugin 的设备端批量版。

    CPU 语义（``envs/humanoid21/disturbance_plugins.py``）逐项对应：

    - 目标机器人收**常量随机 action**（uniform[-1,1]，每 env 独立种子），
      非目标机器人收其摔倒前 ``joint_pos_norm``（保持姿态）；
    - 每 ``reset_interval`` 物理步把非目标机器人写回摔倒前状态
      （目标 blueprint 是双机 ⇒ 分支不触发，仍保留覆盖通用配置）；
    - 每步检查目标机器人 root 高度最小值 < ``height_threshold`` →
      捕获该 env 当步 qpos/qvel（首个达标态，逐 env 独立早停）；
    - 达到 ``max_phy_steps`` 未达标的 env 捕获末态（同 CPU 写回末态）；
    - 只写回目标机器人维度（非目标维度保持 reset 后的摔倒前快照）；
    - pstate 记 ``init_steps``/``init_height``(B,2)/``init_hit``。

    执行形态：

    - **全量 reset**（reset_env_ids=None 或覆盖全部 env）：共享 sim 原地
      摔倒——所有 env 都参与，等价于 CPU 的内部 sim，不占额外汇存。
      摔倒期间覆写 io.action/act_target，结束后从快照恢复。
    - **部分 reset**（env_ids 子集）：用 ``sim_factory`` 惰性构造的专用
      摔倒 sim 跑 rollout，未 reset env 的物理完全不被触碰。

    差异声明：CPU 内部 sim 是 fp64 串行摔倒；本实现 fp32 并行——
    **摔倒姿态分布只承诺统计等价**（验收 B），不承诺逐 seed 逐位一致。
    随机源 = splitmix64(seed_offset, reset_count, salt)，与 CPU 共享
    RandomState 的抽取序列不同构，但 per-env 独立、逐 episode 可复现。
    """

    def __init__(
        self,
        sim_factory,
        target_robots=("robot_a", "robot_b"),
        max_phy_steps: int = 1000,
        height_threshold: float = 0.3,
        reset_interval: int = 5,
        sync_chunk: int = 25,
        salt: int = 0xF411E,
    ):
        """
        Args:
            sim_factory: ``callable(batch_size) -> simulator``，构造专用
                摔倒 sim（仅发生部分 reset 时才实例化）。该 sim 需有
                dev_set_integration_rows / dev_set_action / physical_step /
                views()（即 warp 后端绑定接口）。
            sync_chunk: 每多少物理步做一次 host 侧 done 检查（reset 路径
                的有限同步，不在 rollout 热路径上）。
            salt: 插件随机盐（区分其他用同一 seed_offsets 的插件）。
        """
        if isinstance(target_robots, str):
            ts = {"robot_a", "robot_b"} if target_robots == "both" \
                else {target_robots}
        else:
            ts = set(target_robots)
        for rid in ts:
            assert rid in ("robot_a", "robot_b"), rid
        self._targets = ts
        self.max_phy_steps = int(max_phy_steps)
        self.height_threshold = float(height_threshold)
        self.reset_interval = int(reset_interval)
        self.sync_chunk = int(sync_chunk)
        self._salt = int(salt)
        self._sim_factory = sim_factory
        self._internal = None       # 专用摔倒 sim（惰性）
        self._sim = None            # 共享 sim（bind_shared_sim 注入）
        self._count = None          # (B,) i64 per-env reset 计数——必须在
        # 插件自有属性而非 plugin pool：pool 行会被 partial reset 清零，
        # 清零会让同 env 每次 reset 抽到同一随机序列。

    @property
    def name(self) -> str:
        return "device_random_fallen_state"

    @property
    def require_mutator(self) -> bool:
        return True

    # ------------------------------------------------------------------
    def declare_state(self, state) -> None:
        nq = state.sim.qpos.shape[1]
        nv = state.sim.qvel.shape[1]
        st = state.declare_state(self.name, "captured_qpos", (nq,),
                                 torch.float32)
        state.declare_state(self.name, "captured_qvel", (nv,), torch.float32)
        state.declare_state(self.name, "init_steps", (), torch.int32)
        state.declare_state(self.name, "init_height", (2,), torch.float32,
                            init=float("nan"))
        state.declare_state(self.name, "init_hit", (), torch.bool)
        self._count = torch.zeros(state.batch_size, dtype=torch.int64,
                                  device=st.device)

    def on_attach(self) -> None:
        if self._sim is None:
            raise RuntimeError(
                "DeviceFallenResetPlugin.bind_shared_sim(sim) must be "
                "called before attach.")

    def bind_shared_sim(self, sim) -> "DeviceFallenResetPlugin":
        """注入共享 sim 引用（插件需要 task_tables()/views() 并在全量
        reset 时原地摔倒）。"""
        self._sim = sim
        return self

    def _internal_sim(self, B: int):
        if self._internal is None:
            self._internal = self._sim_factory(B)
            self._internal.reset()
        return self._internal

    @property
    def rng_salt(self) -> int:
        return self._salt

    # ------------------------------------------------------------------
    def _draw_actions(self, env_ids, rng_view, dev) -> torch.Tensor:
        """(K,2,21) uniform[-1,1)——每 env 每机器人每次 reset 独立抽取
        （CPU：每 episode 每目标机器人各抽一次 uniform[-1,1]^21）。
        机器人间共享同一 action 会导致双机同步倒地、初态分布失真。"""
        seeds = (rng_view.unit_seed(env_ids, self._count[env_ids])
                 + env_ids * (-7046029254386353131))            # (K,)
        j = torch.arange(42, dtype=torch.int64, device=dev)
        bits = _splitmix64(seeds[:, None] + j[None, :])          # (K,42)
        u = _lshr(bits, 11).to(torch.float64) * (2.0 ** -53)     # [0,1)
        return (u * 2.0 - 1.0).to(torch.float32).view(-1, 2, 21)

    # ------------------------------------------------------------------
    def on_pre_episode(self, ctx: DeviceCtx) -> None:
        sim = self._sim
        st = ctx.state
        dev = st.sim.qpos.device
        B = st.batch_size
        env_ids = (torch.arange(B, device=dev) if ctx.reset_env_ids is None
                   else ctx.reset_env_ids.to(dev))
        if env_ids.numel() == 0:
            return

        # 摔倒前快照 = CPU 版 real_state（reset 后的站姿）
        qpos0 = st.sim.qpos[env_ids].clone()      # (K,nq)
        qvel0 = st.sim.qvel[env_ids].clone()      # (K,nv)

        # 摔倒载体：全量 reset → 原地共享 sim；子集 → 专用内部 sim
        in_place = env_ids.numel() == B
        if in_place:
            host = sim
            host_qpos, host_qvel = st.sim.qpos, st.sim.qvel
            # in-place 摔倒会覆写 io.action/act_target——先快照后恢复
            io_a0 = st.io.action_a[env_ids].clone()
            io_b0 = st.io.action_b[env_ids].clone()
        else:
            host = self._internal_sim(B)
            hv = host.views()
            host_qpos, host_qvel = hv["qpos"], hv["qvel"]
            host.dev_set_integration_rows(env_ids, qpos0, qvel0)

        t = self._tables()
        # --- action：目标随机（per-robot 独立）；非目标保持 joint_pos_norm ---
        # ctx.rng 由 runtime 按声明的 rng_salt 分配；脱离 runtime 的直接
        # 调用（测试/工具）现场构建同 salt 视图，序列一致。
        rng_view = ctx.rng or RngView(st, self._salt)
        rand = self._draw_actions(env_ids, rng_view, dev)
        act_full = [torch.zeros(B, 21, device=dev),
                    torch.zeros(B, 21, device=dev)]
        for idx, rid in enumerate(("robot_a", "robot_b")):
            if rid in self._targets:
                act_full[idx][env_ids] = rand[:, idx]
            else:
                ref, scale = t.norm_pair(rid)
                qi = t.robots[rid]["qpos_indices"]
                act_full[idx][env_ids] = (
                    (qpos0[:, qi] - ref) / scale).clamp(-1.0, 1.0)
        host.dev_set_action(act_full[0], act_full[1])

        # --- 摔倒循环：逐 env 首个达标态捕获 ---
        tgt_adr = [t.robots[rid]["root_qpos_adr"] + 2
                   for rid in ("robot_a", "robot_b") if rid in self._targets]
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        first_step = torch.zeros(B, dtype=torch.int64, device=dev)
        h_buf = torch.full((B, 2), float("nan"), device=dev)
        pool = st.plugin[self.name]
        cap_q, cap_v = pool["captured_qpos"], pool["captured_qvel"]

        # 非目标维度的定期恢复表（CPU: set_core_state(non_target_state)）
        nt_qp, nt_qv = [], []
        for rid in ("robot_a", "robot_b"):
            if rid in self._targets:
                continue
            c = t.robots[rid]
            nt_qp.extend(range(c["root_qpos_adr"], c["root_qpos_adr"] + 7))
            nt_qp.extend(c["qpos_indices"])
            nt_qv.extend(range(c["root_qvel_adr"], c["root_qvel_adr"] + 6))
            nt_qv.extend(c["qvel_indices"])
        non_t_dims = None
        if nt_qp:
            non_t_dims = (torch.as_tensor(sorted(set(nt_qp)),
                                          dtype=torch.long, device=dev),
                          torch.as_tensor(sorted(set(nt_qv)),
                                          dtype=torch.long, device=dev))

        step = self.max_phy_steps
        for step in range(1, self.max_phy_steps + 1):
            host.physical_step(1)

            if non_t_dims is not None and step % self.reset_interval == 0:
                mqp = host_qpos[env_ids].clone()
                mqv = host_qvel[env_ids].clone()
                mqp[:, non_t_dims[0]] = qpos0[:, non_t_dims[0]]
                mqv[:, non_t_dims[1]] = qvel0[:, non_t_dims[1]]
                host.dev_set_integration_rows(env_ids, mqp, mqv)

            h = torch.stack([host_qpos[env_ids, a] for a in tgt_adr], -1)
            newly = (~done[env_ids]) & (h.min(-1).values
                                        < self.height_threshold)
            if bool(newly.any()):
                ids = env_ids[newly]
                cap_q[ids] = host_qpos[ids]
                cap_v[ids] = host_qvel[ids]
                first_step[ids] = step
                done[ids] = True
                for t_i, adr in enumerate(tgt_adr):
                    h_buf[ids, t_i] = host_qpos[ids, adr]
            if step % self.sync_chunk == 0 or step == self.max_phy_steps:
                if bool(done[env_ids].all()):
                    break

        # 未达标 env → 捕获末态（CPU 同样写回末态）
        pending = env_ids[~done[env_ids]]
        if pending.numel():
            cap_q[pending] = host_qpos[pending]
            cap_v[pending] = host_qvel[pending]
            first_step[pending] = step
            for t_i, adr in enumerate(tgt_adr):
                h_buf[pending, t_i] = host_qpos[pending, adr]

        # --- 只写回目标机器人维度（基底 = 摔倒前快照） ---
        tgt_qp, tgt_qv = [], []
        for rid in self._targets:
            c = t.robots[rid]
            tgt_qp.extend(range(c["root_qpos_adr"], c["root_qpos_adr"] + 7))
            tgt_qp.extend(c["qpos_indices"])
            tgt_qv.extend(range(c["root_qvel_adr"], c["root_qvel_adr"] + 6))
            tgt_qv.extend(c["qvel_indices"])
        tgt_qp = torch.as_tensor(sorted(set(tgt_qp)), dtype=torch.long,
                                 device=dev)
        tgt_qv = torch.as_tensor(sorted(set(tgt_qv)), dtype=torch.long,
                                 device=dev)
        out_q = qpos0.clone()
        out_v = qvel0.clone()
        out_q[:, tgt_qp] = cap_q[env_ids][:, tgt_qp]
        out_v[:, tgt_qv] = cap_v[env_ids][:, tgt_qv]
        ctx.mutator.set_integration_rows(env_ids, out_q, out_v)

        # in-place 路径恢复 io.action / act_target（CPU 真实 env 的
        # action 状态不受内部 sim 影响）
        if in_place:
            st.io.action_a[env_ids] = io_a0
            st.io.action_b[env_ids] = io_b0
            v = sim.views()
            for idx, (a0, rid) in enumerate(
                    ((io_a0, "robot_a"), (io_b0, "robot_b"))):
                ref, scale = t.norm_pair(rid)
                v["act_target"][env_ids, idx * 21:(idx + 1) * 21] = \
                    a0.clamp(-1.0, 1.0) * scale + ref

        # --- metrics ---
        pool["init_steps"][env_ids] = first_step[env_ids].to(torch.int32)
        pool["init_height"][env_ids] = h_buf[env_ids]
        pool["init_hit"][env_ids] = done[env_ids]
        self._count[env_ids] += 1

    def on_envs_reset(self, ctx: DeviceCtx) -> None:
        # 池行已被 runtime 清零；真正逻辑在随后的 on_pre_episode
        return None

    def export_episode_metrics(self, state) -> Dict[str, torch.Tensor]:
        """W4 指标契约：init_steps/init_height/init_hit → 逐 agent 命名。"""
        pool = state.plugin[self.name]
        out: Dict[str, torch.Tensor] = {}
        for i, rid in enumerate(("robot_a", "robot_b")):
            out[f"{rid}_fallen_init_steps"] = pool["init_steps"]
            out[f"{rid}_fallen_init_height"] = pool["init_height"][:, i]
            out[f"{rid}_fallen_init_height_threshold"] = pool["init_hit"]
        return out

    def _tables(self):
        """任务设备表（惰性缓存自 bind_shared_sim 注入的 sim）。"""
        if getattr(self, "_tables_cache", None) is None:
            self._tables_cache = self._sim.task_tables()
        return self._tables_cache
