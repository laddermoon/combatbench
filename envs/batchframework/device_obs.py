"""设备端观测构建器：``_get_robot_view_batch`` 的 torch 复刻。

输入是 ``DeviceBatchState.sim`` 的活跃张量视图（warp 后端零拷贝），
全程无 host 传输。公式逐字段对齐 ``mjx_simulator._get_robot_view_batch``
（即 CPU ``_get_robot_view`` 的批量语义）；两者之间只允许 fp32 噪声级
差异，由 W4 的对照测试锁定。

feet_forces 走 contact **flat** 视图 + index_add 聚合，不经 padded
中间形态（padded 是插件契约；内部消费者用 flat 更省且语义相同）。
"""
from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import torch

KP_NAMES = ("head", "hand_right", "hand_left", "foot_right", "foot_left")


def _quat_to_rot_t(quat: torch.Tensor) -> torch.Tensor:
    """(B,4) [w,x,y,z] → (B,3,3)。"""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    r0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                      2 * (x * z + y * w)], dim=-1)
    r1 = torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                      2 * (y * z - x * w)], dim=-1)
    r2 = torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                      1 - 2 * (x * x + y * y)], dim=-1)
    return torch.stack([r0, r1, r2], dim=-2)


def _sqrt_signed_t(v: torch.Tensor, div2_inside: bool = False) -> torch.Tensor:
    if div2_inside:
        return torch.sign(v) * torch.sqrt(torch.abs(v / 2.0))
    return torch.sign(v) * torch.sqrt(torch.abs(v)) / 2.0


class FlatContactForces:
    """活跃接触的 per-contact 力分解与归属分类（flat packed 视图）。

    由 ``contact_forces_flat`` 计算一次，观测构建器与 rewarder 等
    设备端消费者共享——接触力只有一份实现，避免两处漂移。

    字段全部为长度 M（活跃接触数）的张量；``robot_body``/``ground``
    只覆盖"ground↔robot"接触，其余项为哨兵值。
    """

    def __init__(self, sel, worldid, geom1, geom2, body1, body2,
                 aff1, aff2, force_mag, force_world):
        self.sel = sel              # (M,) 活跃接触在 flat 数组中的下标
        self.worldid = worldid      # (M,) i64
        self.geom1 = geom1          # (M,) i64
        self.geom2 = geom2
        self.body1 = body1          # (M,) i64 — geom→body
        self.body2 = body2
        self.aff1 = aff1            # (M,) i64 — 归属（0=环境, 1=a, 2=b）
        self.aff2 = aff2
        self.force_mag = force_mag      # (M,) f32
        self.force_world = force_world  # (M,3) f32


def contact_forces_flat(state, geom_bodyid: torch.Tensor,
                        geom_aff: torch.Tensor) -> FlatContactForces:
    """flat contacts → 活跃项的力/归属分类。

    condim=3 ⇒ 每接触 4 行 efc：normal=sum, f1=r0-r1, f2=r2-r3；
    ``force_world = frameᵀ @ [normal,f1,f2]``（与 mjx_simulator
    ``_extract_contacts_batch`` 同公式）。不活跃槽位不参与。
    ``geom_bodyid``/``geom_aff`` 是 geom_id→body/affiliation 的
    设备端查找表（后端提供）。
    """
    c = state.sim.contacts_flat
    dev = c.worldid.device
    C = c.worldid.shape[0]
    idx = torch.arange(C, device=dev)
    n_active = c.n_active.reshape(()).long()
    active = (idx < n_active) & (c.dist <= 0)
    sel = torch.nonzero(active, as_tuple=False).squeeze(-1)

    geom = c.geom[sel].long()
    body = geom_bodyid[geom.clamp(min=0)]
    aff = geom_aff[geom.clamp(min=0)]
    w = c.worldid[sel].long()

    adr = c.efc_address[sel].long()
    if adr.ndim > 1:
        adr = adr[:, 0]
    ef_rows = adr.clamp(min=0)[:, None] + torch.arange(4, device=dev)
    ef = c.efc_force[w[:, None].clamp(min=0), ef_rows.clamp(min=0)]
    fl = torch.stack([ef.sum(dim=-1), ef[:, 0] - ef[:, 1],
                      ef[:, 2] - ef[:, 3]], dim=-1)
    fw = torch.einsum("mij,mj->mi", c.frame[sel].transpose(-1, -2), fl)
    fmag = torch.linalg.norm(fw, dim=-1)

    return FlatContactForces(sel, w, geom[:, 0], geom[:, 1],
                             body[:, 0], body[:, 1], aff[:, 0], aff[:, 1],
                             fmag, fw)


class DenseContactForces:
    """``contact_forces_flat`` 的定形版本——供无 host 同步路径使用。

    与 flat 版逐帧语义等价，但不做 ``torch.nonzero``（变长输出强制
    host 同步/图内非法）：所有字段为全 C 槽位张量，非活跃槽位由
    ``active`` 标出（``n_active`` 以设备值参与 mask，数据变化时
    自动更新）。padding 槽位内容是未定义值——消费方必须用
    ``active``/阈值 mask 归零，不得直接读。
    """

    def __init__(self, active, worldid, geom1, geom2, body1, body2,
                 aff1, aff2, force_mag, force_world):
        self.active = active        # (C,) bool
        self.worldid = worldid      # (C,) i64（已 clamp 到 [0,B-1]）
        self.geom1 = geom1          # (C,) i64
        self.geom2 = geom2
        self.body1 = body1          # (C,) i64 — geom→body
        self.body2 = body2
        self.aff1 = aff1            # (C,) i64 — 归属（0=环境, 1=a, 2=b）
        self.aff2 = aff2
        self.force_mag = force_mag      # (C,) f32
        self.force_world = force_world  # (C,3) f32


def contact_forces_dense(state, geom_bodyid: torch.Tensor,
                         geom_aff: torch.Tensor,
                         batch_size: int) -> DenseContactForces:
    """flat contacts → 全槽位 dense 力分解（无 host 同步）。

    与 ``contact_forces_flat`` 同公式（condim=3 ⇒ 4 行 efc：
    normal=sum, f1=r0-r1, f2=r2-r3；``force_world = frameᵀ @ f``），
    对全部 C 槽计算——无效槽的 gather 索引 clamp 到合法域，值由
    ``active``/力阈值 mask 消去。
    """
    c = state.sim.contacts_flat
    dev = c.worldid.device
    C = c.worldid.shape[0]
    active = (torch.arange(C, device=dev)
              < c.n_active.reshape(())) & (c.dist <= 0)

    geom = c.geom.long()                                    # (C,2)
    body = geom_bodyid[geom.clamp(0, geom_bodyid.shape[0] - 1)]
    aff = geom_aff[geom.clamp(0, geom_aff.shape[0] - 1)]
    w = c.worldid.long().clamp(0, batch_size - 1)

    adr = c.efc_address.long()
    if adr.ndim > 1:
        adr = adr[:, 0]
    ef_rows = (adr.clamp(0, c.efc_force.shape[-1] - 4)[:, None]
               + torch.arange(4, device=dev))
    ef = c.efc_force[w[:, None], ef_rows]
    fl = torch.stack([ef.sum(dim=-1), ef[:, 0] - ef[:, 1],
                      ef[:, 2] - ef[:, 3]], dim=-1)
    fw = torch.einsum("cij,cj->ci", c.frame.transpose(-1, -2), fl)
    fmag = torch.linalg.norm(fw, dim=-1)
    # 非活跃槽的 gather 值是垃圾——力先归零，下游 mask 只需管归属
    fmag = torch.where(active, fmag, torch.zeros_like(fmag))

    return DenseContactForces(active, w, geom[:, 0], geom[:, 1],
                              body[:, 0], body[:, 1], aff[:, 0], aff[:, 1],
                              fmag, fw)


class WarpObsBuilder:
    """从 DeviceBatchState 构建 96 维观测（torch，fp32）。

    构造时从 simulator 的 meta 缓存提取全部索引/归一化常量并转为
    CUDA 张量；``build()`` 为纯设备端计算。
    """

    def __init__(self, tables, batch_size: int):
        """Args:
            tables: ``Humanoid21DeviceTables``——任务设备端常量表
                （W4 收口后的唯一任务元数据来源）。
            batch_size: batch 行数 B。
        """
        self._tables = tables
        self._dev = tables.device
        self._B = int(batch_size)
        # E7-W2：build() 的 torch 序列 launch-bound（~66% collect 时间），
        # 用 torch.cuda.graph 重放。固定形状版本见 _feet_forces_dense；
        # 捕获失败永久回退 eager（_graph_broken）。
        self._use_graph = (torch.cuda.is_available()
                           and str(self._dev).startswith("cuda")
                           and os.environ.get("CB_OBS_GRAPH", "1") != "0")
        self._graph = None
        self._graph_out = None
        self._graph_key = None
        self._graph_broken = not self._use_graph
        self._ground_gid = tables.ground_geom_id
        self._geom_bodyid = tables.geom_bodyid
        self._geom_aff = tables.geom_aff
        self._robots: Dict[str, Dict[str, Any]] = {}
        for rid in ("robot_a", "robot_b"):
            cache = tables.robots[rid]
            self._robots[rid] = dict(
                torso_id=cache["root_body_id"],
                root_qva=cache["root_qvel_adr"],
                qpos_idx=cache["qpos_indices"],
                qvel_idx=cache["qvel_indices"],
                norm_ref=cache["norm_ref"],
                norm_scale=cache["norm_scale"],
                kp_ids=cache["keypoint_body_ids"],
                body_weight=cache["body_weight"],
            )

    # ------------------------------------------------------------------
    @property
    def contact_tables(self):
        """(geom_bodyid, geom_aff) 查找表——contact_forces_flat 的入参。"""
        return self._geom_bodyid, self._geom_aff

    def _feet_forces(self, state, rid: str) -> torch.Tensor:
        """双足地面接触力（按体重归一）——共享 contact_forces_flat。"""
        cf = contact_forces_flat(state, *self.contact_tables)
        out = torch.zeros(self._B, 2, dtype=torch.float32, device=self._dev)
        if cf.sel.numel() == 0:
            return out
        g1_ground = cf.geom1 == self._ground_gid
        ground = g1_ground | (cf.geom2 == self._ground_gid)
        other = torch.where(g1_ground, cf.body2, cf.body1)
        kp = self._robots[rid]["kp_ids"]
        fm = cf.force_mag * ground
        out[:, 0].index_add_(0, cf.worldid, fm * (other == kp["foot_right"]))
        out[:, 1].index_add_(0, cf.worldid, fm * (other == kp["foot_left"]))
        return out / self._robots[rid]["body_weight"]

    def _feet_forces_dense(self, state, rid: str) -> torch.Tensor:
        """``_feet_forces`` 的固定形状版本——供 CUDA Graph 路径使用。

        与 ``contact_forces_flat`` 语义逐帧等价，但不做
        ``torch.nonzero``（动态形状 + 隐含 host sync，图内非法）：
        改为全 C 槽位 dense 计算 + ``active`` mask 归零非活跃项。
        ``n_active`` 留在设备上以值参与 mask，重放时随数据自动更新。
        """
        c = state.sim.contacts_flat
        dev = c.worldid.device
        C = c.worldid.shape[0]
        active = (torch.arange(C, device=dev)
                  < c.n_active.reshape(())) & (c.dist <= 0)

        geom = c.geom.long()                              # (C,2)
        body = self._geom_bodyid[geom.clamp(0, self._geom_bodyid.shape[0] - 1)]
        # padding 槽位索引值是垃圾：上下界都钳住，值反正被 active mask 归零
        w = c.worldid.long().clamp(0, self._B - 1)

        adr = c.efc_address.long()
        if adr.ndim > 1:
            adr = adr[:, 0]
        ef_rows = (adr.clamp(0, c.efc_force.shape[-1] - 4)[:, None]
                   + torch.arange(4, device=dev))
        ef = c.efc_force[w[:, None], ef_rows]
        fl = torch.stack([ef.sum(dim=-1), ef[:, 0] - ef[:, 1],
                          ef[:, 2] - ef[:, 3]], dim=-1)
        fw = torch.einsum("cij,cj->ci", c.frame.transpose(-1, -2), fl)
        fmag = torch.linalg.norm(fw, dim=-1)

        g1_ground = geom[:, 0] == self._ground_gid
        ground = (g1_ground | (geom[:, 1] == self._ground_gid)) & active
        other = torch.where(g1_ground, body[:, 1], body[:, 0])
        kp = self._robots[rid]["kp_ids"]
        fm = fmag * ground
        out = torch.zeros(self._B, 2, dtype=torch.float32, device=dev)
        out[:, 0].index_add_(0, w, fm * (other == kp["foot_right"]))
        out[:, 1].index_add_(0, w, fm * (other == kp["foot_left"]))
        return out / self._robots[rid]["body_weight"]

    # ------------------------------------------------------------------
    def _robot_obs(self, state, rid: str, opp: str, *,
                   dense_contacts: bool = False) -> torch.Tensor:
        s, r, ro = state.sim, self._robots[rid], self._robots[opp]
        torso, opp_torso = r["torso_id"], ro["torso_id"]

        self_pos = s.xpos[:, torso]
        self_quat = s.xquat[:, torso]
        opp_pos = s.xpos[:, opp_torso]
        opp_quat = s.xquat[:, opp_torso]

        R = _quat_to_rot_t(self_quat)                 # body→world
        Ri = R.transpose(-1, -2)                      # world→body

        def loc(v):  # 世界系向量 → 自机体系
            return torch.einsum("bij,bj->bi", Ri, v)

        height = self_pos[:, 2:3]
        projected_gravity = -R[:, 2, :]
        rq = r["root_qva"]
        linear_vel = loc(s.qvel[:, rq:rq + 3])
        angular_vel = s.qvel[:, rq + 3:rq + 6]        # 本就机体系
        feet = (self._feet_forces_dense(state, rid) if dense_contacts
                else self._feet_forces(state, rid))
        arena_center_local = loc(-self_pos)
        rel_pos = loc(opp_pos - self_pos)
        rel_vel = loc(s.cvel[:, opp_torso, 3:6])
        opp_fwd = _quat_to_rot_t(opp_quat)[:, :, 0]
        face = loc(opp_fwd)

        kp_pos, kp_vel = {}, {}
        for n in KP_NAMES:
            b = ro["kp_ids"][n]
            kp_pos[n] = loc(s.xpos[:, b] - self_pos)
            kp_vel[n] = loc(s.cvel[:, b, 3:6])

        jpn = (s.qpos[:, r["qpos_idx"]] - r["norm_ref"]) / r["norm_scale"]
        jvn = s.qvel[:, r["qvel_idx"]] / r["norm_scale"]

        return torch.cat([
            jpn, _sqrt_signed_t(jvn), projected_gravity, height,
            linear_vel, _sqrt_signed_t(angular_vel, div2_inside=True),
            feet, arena_center_local, rel_pos, rel_vel, face,
            kp_pos["head"], kp_pos["hand_right"], kp_pos["hand_left"],
            kp_pos["foot_right"], kp_pos["foot_left"],
            kp_vel["head"],                            # head 速度不变换
            torch.cat([_sqrt_signed_t(kp_vel[n])
                       for n in KP_NAMES[1:]], dim=-1),
        ], dim=-1)

    # ------------------------------------------------------------------
    def obs_dim(self) -> int:
        """观测维度（state 无关，固定 96——见 OBSERVATION_zh.md）。"""
        return 96

    def _build_inner(self, state, *, dense_contacts: bool) -> Dict[str, torch.Tensor]:
        obs = {
            "robot_a": self._robot_obs(state, "robot_a", "robot_b",
                                       dense_contacts=dense_contacts),
            "robot_b": self._robot_obs(state, "robot_b", "robot_a",
                                       dense_contacts=dense_contacts),
        }
        if state.io.obs_a is not None:
            state.io.obs_a.copy_(obs["robot_a"])
            state.io.obs_b.copy_(obs["robot_b"])
        return obs

    def _build_graphed(self, state) -> Dict[str, torch.Tensor]:
        """捕获/重放 torch.cuda.graph；state 身份变化时重新捕获。"""
        key = (id(state), id(state.sim.qpos),
               None if state.io.obs_a is None else id(state.io.obs_a))
        if key != self._graph_key:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):          # 侧流预热（句柄/workspace 就位）
                    self._build_inner(state, dense_contacts=True)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._build_inner(state, dense_contacts=True)
            self._graph, self._graph_out, self._graph_key = g, out, key
        self._graph.replay()
        return self._graph_out

    def build(self, state) -> Dict[str, torch.Tensor]:
        """→ {"robot_a": (B,96), "robot_b": (B,96)}；写入 io.obs_* 缓冲。"""
        if not self._graph_broken:
            try:
                return self._build_graphed(state)
            except Exception:
                self._graph_broken = True    # 永久回退 eager
        return self._build_inner(state, dense_contacts=False)
