"""WarpBackend — mujoco-warp 物理后端（E1-W2，实现 PhysicsBackend）。

职责边界（对应 physics.py 契约）：

- **只拥有物理**：mjw.Model/Data、设备视图、pending wrench 与
  force schedule 缓冲、PD kernel（控制由 ControlProgram 经
  ``advance(control=...)`` 注入）、容量状态。不含 episode/IO/RNG/
  plugin 簿记——那些归 runtime。
- **advance 全行推进**：warp 1.12.1 无 masked-step 原语；ENDED 行
  冻结由 runtime 用 capture/restore 组合（W0 探针选定 write-back）。
- **视图借用语义**：``views()`` 返回 mjw.Data/wp.array 的 torch
  零拷贝视图；advance/initialize/apply_patch/restore 可能使其失效。
- **快照近似恢复**：warp 同输入运行间不逐位确定（W0-P2 实测
  ~1e-7/10 步漂移），restore 不承诺逐位续跑。

本模块可 import warp/mjw，但**不依赖** runtime/plugin/collector/任务层。
Humanoid21 相关的 PD/索引表经构造参数注入（``pd_statics`` 为
``Humanoid21Binding.build_statics(np, np.float32)`` 的产物），
后端本身不 import humanoid21。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from ..physics import (
    BackendDescriptor,
    CapacitySpec,
    ContractError,
    FieldSpec,
    RefreshPolicy,
    SamplePhase,
    SnapshotLevel,
)

# capture(level=INTEGRATION) 捕获的字段集——W0-P2 实测为近似恢复
_INTEGRATION_FIELDS = (
    "qpos", "qvel", "ctrl", "act", "time",
    "qacc_warmstart", "xfrc_applied", "qfrc_applied",
)


class WarpPDControl:
    """PD 控制程序：每子步由 ``advance`` 调用，wp.kernel 就地写 ctrl。

    ``statics`` 是 ``build_statics(np, np.float32)`` 的 per-robot 表；
    ``target`` 视图（act_target (B, 42)）由调用方/插件经
    ``views()["act_target"]`` 或 mutator 写入。
    """

    def __init__(self, backend: "WarpBackend", statics: Dict[str, Any]):
        wp = backend._wp
        dev = backend._device_str
        statics_np = statics

        def cat(key):
            return np.concatenate(
                [np.asarray(statics_np["robot_a"][key]),
                 np.asarray(statics_np["robot_b"][key])])

        self._arrays = dict(
            qpos_idx=wp.array(cat("qpos_indices"), dtype=wp.int32, device=dev),
            qvel_idx=wp.array(cat("qvel_indices"), dtype=wp.int32, device=dev),
            act_ids=wp.array(cat("actuator_ids"), dtype=wp.int32, device=dev),
            gear=wp.array(cat("gear"), dtype=wp.float32, device=dev),
            lo=wp.array(cat("ctrl_lo"), dtype=wp.float32, device=dev),
            hi=wp.array(cat("ctrl_hi"), dtype=wp.float32, device=dev),
            kp=wp.array(np.concatenate(
                [statics_np["robot_a"]["kp"],
                 statics_np["robot_b"]["kp"]]), dtype=wp.float32, device=dev),
            kd=wp.array(np.concatenate(
                [statics_np["robot_a"]["kd"],
                 statics_np["robot_b"]["kd"]]), dtype=wp.float32, device=dev),
        )
        self._backend = backend

        @wp.kernel
        def pd_kernel(qpos: wp.array(dtype=wp.float32, ndim=2),
                      qvel: wp.array(dtype=wp.float32, ndim=2),
                      target: wp.array(dtype=wp.float32, ndim=2),
                      qpos_idx: wp.array(dtype=wp.int32),
                      qvel_idx: wp.array(dtype=wp.int32),
                      act_ids: wp.array(dtype=wp.int32),
                      gear: wp.array(dtype=wp.float32),
                      lo: wp.array(dtype=wp.float32),
                      hi: wp.array(dtype=wp.float32),
                      kp: wp.array(dtype=wp.float32),
                      kd: wp.array(dtype=wp.float32),
                      ctrl: wp.array(dtype=wp.float32, ndim=2)):
            w, i = wp.tid()
            t = (kp[i] * (target[w, i] - qpos[w, qpos_idx[i]])
                 - kd[i] * qvel[w, qvel_idx[i]])
            ctrl[w, act_ids[i]] = wp.clamp(t / gear[i], lo[i], hi[i])

        self._kernel = pd_kernel

    @property
    def arrays(self) -> Dict[str, Any]:
        return self._arrays

    def apply(self, views) -> None:
        """ControlProgram 协议：读 views 中的 act_target，写 ctrl。"""
        b = self._backend
        a = self._arrays
        target_wp = b._act_target_wp
        b._wp.launch(
            self._kernel, dim=(b.batch_size, 42),
            inputs=[b._wdata.qpos, b._wdata.qvel, target_wp,
                    a["qpos_idx"], a["qvel_idx"], a["act_ids"],
                    a["gear"], a["lo"], a["hi"], a["kp"], a["kd"],
                    b._wdata.ctrl],
            device=b._device_str)


class WarpBackend:
    """``PhysicsBackend`` 协议的 mujoco-warp 实现。

    Args:
        model: 编译后的 ``mujoco.MjModel``（opt.timestep 已就位）。
        batch_size: world 数 B。
        device: torch 风格设备串（"cuda:0"）。
        nconmax_per_world / njmax_per_world: 每 world 的接触/约束容量
            （put_data 内部总量 = per_world × B；语义曾混淆导致 B² 分配，
            见 warp_simulator 注释与 M6 结果）。
        pd_statics: 可选——``build_statics(np, np.float32)`` 产物；
            提供后 ``pd_control`` 可用。
    """

    def __init__(self, model, batch_size: int,
                 device: str = "cuda:0",
                 nconmax_per_world: int = 48,
                 njmax_per_world: int = 512,
                 pd_statics: Optional[Dict[str, Any]] = None):
        # XLA 预分配防护：warp 路径不消费 jax，禁掉 XLA 默认 ~75% 显存占用
        # （历史上 jax 初始化曾把 mempool 挤到 OOM）。
        import os
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        import warp as wp
        import mujoco_warp as mjw

        wp.init()
        self._wp = wp
        self._mjw = mjw
        self._model = model
        self._batch_size = int(batch_size)
        self._device_str = device
        self._wmodel = mjw.put_model(model)
        self._nconmax_per_world = int(nconmax_per_world)
        # njmax 是 per-world 约束槽上限：接触最多 4 efc 行/contact（worst
        # case cap48 → ~192），再加关节 limit/friction/equality 约百余条；
        # 默认启发式只有 64，u55 曾触发 "nefc overflow" device assert。
        self._njmax_per_world = int(njmax_per_world)
        self._wdata = None            # mjw.Data
        self._ext_dev = None          # wp.array (B, nbody, 6) 挂起外力
        self._sched_dev = None        # wp.array (B, S, nbody, 6) 子步表
        self._act_target_wp = None    # wp.array (B, 42) PD 目标
        self._pd = None               # WarpPDControl
        self._views = None            # dict[str, torch.Tensor] 零拷贝视图
        self._torch_device = torch.device(device)

        if pd_statics is not None:
            self._pd = WarpPDControl(self, pd_statics)
            # PD kernel 输入必须已分配——否则 advance 早于任何
            # action 写入时读 None。
            self.ensure_act_target(
                int(self._pd.arrays["kp"].numpy().shape[0]))

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def pd_control(self) -> Optional[WarpPDControl]:
        return self._pd

    @property
    def device_str(self) -> str:
        return self._device_str

    # ------------------------------------------------------------------
    # PhysicsBackend 契约
    # ------------------------------------------------------------------
    def describe(self) -> BackendDescriptor:
        import mujoco_warp as mjw_pkg
        fields = {}
        for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "act",
                     "time", "xfrc_applied", "qfrc_applied"):
            fields[name] = FieldSpec(name, SamplePhase.INTEGRATION,
                                     writable=name in ("qpos", "qvel"))
        for name in ("xpos", "xquat", "xipos", "xanchor", "cvel",
                     "geom_xpos", "geom_xmat"):
            fields[name] = FieldSpec(name, SamplePhase.POST_INTEGRATE)
        for name in ("act_target", "xfrc_pending", "xfrc_sched"):
            fields[name] = FieldSpec(name, SamplePhase.INPUT, writable=True)
        for name in ("con_worldid", "con_geom", "con_dist", "con_pos",
                     "con_frame", "con_dim", "con_efc_address",
                     "efc_force", "nacon"):
            fields[name] = FieldSpec(name, SamplePhase.POST_INTEGRATE)
        return BackendDescriptor(
            backend="warp",
            backend_version=getattr(mjw_pkg, "__version__", "unknown"),
            device=self._device_str,
            batch_size=self._batch_size,
            nq=int(self._model.nq), nv=int(self._model.nv),
            nbody=int(self._model.nbody), nu=int(self._model.nu),
            fields=fields,
            capacities={
                "nconmax": CapacitySpec("nconmax",
                                        self._nconmax_per_world, True),
                "njmax": CapacitySpec("njmax",
                                      self._njmax_per_world, True),
            },
            capabilities=frozenset({
                "fused_control", "snapshot:integration",
                "capture:masked", "wrench_schedule",
            }),
        )

    def views(self) -> Dict[str, torch.Tensor]:
        """mjw.Data/wp.array 的 torch 零拷贝视图（借用，见类 docstring）。"""
        if self._views is None:
            if self._wdata is None:
                raise ContractError("WarpBackend.views() before initialize()")
            wp, d = self._wp, self._wdata
            wt = wp.to_torch
            v = dict(
                qpos=wt(d.qpos), qvel=wt(d.qvel), ctrl=wt(d.ctrl),
                xpos=wt(d.xpos), xquat=wt(d.xquat), xipos=wt(d.xipos),
                xanchor=wt(d.xanchor), cvel=wt(d.cvel),
                xfrc_applied=wt(d.xfrc_applied),
                qfrc_applied=wt(d.qfrc_applied),
                qacc_warmstart=wt(d.qacc_warmstart),
                geom_xpos=wt(d.geom_xpos), geom_xmat=wt(d.geom_xmat),
                xfrc_pending=wt(self._ext_dev),
                act_target=wt(self._act_target_wp)
                if self._act_target_wp is not None else None,
                # contact flat-packed（跨 world，worldid 归属，前 nacon 项活跃）
                con_worldid=wt(d.contact.worldid),
                con_geom=wt(d.contact.geom),
                con_dist=wt(d.contact.dist),
                con_pos=wt(d.contact.pos),
                con_frame=wt(d.contact.frame),
                con_dim=wt(d.contact.dim),
                con_efc_address=wt(d.contact.efc_address),
                efc_force=wt(d.efc.force),
                nacon=wt(d.nacon),
                xfrc_sched=None,
            )
            self._views = v
        return self._views

    def initialize(self, qpos: torch.Tensor, *,
                   qvel: Optional[torch.Tensor] = None,
                   mask: Optional[torch.Tensor] = None) -> None:
        """写入全新积分状态 + 清求解器残留 + forward 刷新 derived。

        mask=None：全量初始化（首次建 mjw.Data 或整批重写）。
        mask 非 None：仅 mask 行——partial reset 语义。
        """
        wp, mjw = self._wp, self._mjw
        B = self._batch_size
        dev = self._device_str
        qpos = torch.as_tensor(qpos, dtype=torch.float32,
                               device=self._torch_device)

        if self._wdata is None:
            import mujoco
            mjd0 = mujoco.MjData(self._model)
            mujoco.mj_resetData(self._model, mjd0)
            # nconmax/njmax 均为 per-world——put_data 内部乘 nworld。
            # 历史上误传 B×48 曾导致 B² 级分配（B=1536 → ~20GB）。
            self._wdata = mjw.put_data(
                self._model, mjd0, nworld=B,
                nconmax=self._nconmax_per_world,
                njmax=self._njmax_per_world)
        d = self._wdata
        # pending wrench 缓冲必须先于 views() 存在（视图登记其指针）
        if self._ext_dev is None:
            self._ext_dev = wp.zeros((B, self._model.nbody, 6),
                                     dtype=wp.float32, device=dev)
        else:
            self._ext_dev.zero_()
        self._sched_dev = None
        if self._views is not None:
            self._views["xfrc_sched"] = None

        with wp.ScopedStream(wp.stream_from_torch()):
            if mask is None:
                if qpos.shape != (B, self._model.nq):
                    raise ContractError(
                        f"initialize qpos shape {tuple(qpos.shape)} != "
                        f"({B}, {self._model.nq})")
                d.qpos.assign(wp.from_torch(qpos.contiguous()))
                if qvel is None:
                    d.qvel.assign(wp.zeros((B, self._model.nv),
                                           dtype=wp.float32, device=dev))
                else:
                    d.qvel.assign(wp.from_torch(
                        qvel.to(torch.float32).contiguous()))
                d.ctrl.assign(wp.zeros((B, self._model.nu),
                                       dtype=wp.float32, device=dev))
                d.xfrc_applied.assign(
                    wp.zeros((B, self._model.nbody, 6),
                             dtype=wp.float32, device=dev))
                d.qfrc_applied.assign(
                    wp.zeros((B, self._model.nv),
                             dtype=wp.float32, device=dev))
                d.qacc_warmstart.assign(
                    wp.zeros((B, self._model.nv),
                             dtype=wp.float32, device=dev))
            else:
                vv = self.views()
                rows = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                vv["qpos"][rows] = qpos
                if qvel is not None:
                    vv["qvel"][rows] = qvel.to(torch.float32)
                else:
                    vv["qvel"][rows] = 0.0
                for k in ("ctrl", "qacc_warmstart", "xfrc_applied",
                          "qfrc_applied"):
                    vv[k][rows] = 0.0
            mjw.forward(self._wmodel, d)

    def apply_patch(self, mask: torch.Tensor,
                    fields: Mapping[str, torch.Tensor], *,
                    refresh: RefreshPolicy = RefreshPolicy.FORWARD) -> None:
        if self._wdata is None:
            raise ContractError("apply_patch before initialize()")
        vv = self.views()
        rows = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        for k, val in fields.items():
            t = vv.get(k)
            if t is None:
                t = _data_field_torch(self, k)  # act/time 等未登记字段
            if t is None:
                raise ContractError(f"unknown or absent field {k!r}")
            t[rows] = val
        if refresh is RefreshPolicy.FORWARD:
            with self._wp.ScopedStream(self._wp.stream_from_torch()):
                self._mjw.forward(self._wmodel, self._wdata)

    def advance(self, n_substeps: int, control=None) -> None:
        """全行推进 n 子步；control.apply(views) 每子步前调用。

        消费型输入：pending wrench 仅首子步加入；schedule 逐子步取
        sched[:, i]，步末清零（对齐 CPU physical_step 语义）。
        """
        if self._wdata is None:
            raise ContractError("advance before initialize()")
        wp, mjw, d = self._wp, self._mjw, self._wdata
        v = self.views()
        sched = v["xfrc_sched"]
        if sched is not None and sched.shape[1] != n_substeps:
            raise ContractError(
                f"force schedule has {sched.shape[1]} substeps, "
                f"advance got n_substeps={n_substeps}")
        # ScopedStream：warp kernel（control/mjw.step）绑到 torch 当前流，
        # 与 torch 侧 xfrc 组合（zero_/add_）严格有序。
        with wp.ScopedStream(wp.stream_from_torch()):
            for i in range(n_substeps):
                if control is not None:
                    control.apply(v)
                xf, pend = v["xfrc_applied"], v["xfrc_pending"]
                xf.zero_()
                if i == 0:
                    xf.add_(pend)
                if sched is not None:
                    xf.add_(sched[:, i])
                mjw.step(self._wmodel, d)
            # CPU physical_step 末尾清零施加力/求解偏置
            v["xfrc_applied"].zero_()
            pend.zero_()
            v["qfrc_applied"].zero_()
        self._sched_dev = None
        v["xfrc_sched"] = None

    def capture(self, mask: torch.Tensor,
                level: SnapshotLevel = SnapshotLevel.INTEGRATION
                ) -> Dict[str, torch.Tensor]:
        if level is not SnapshotLevel.INTEGRATION:
            raise ContractError(f"snapshot level {level} not supported")
        vv = self.views()
        rows = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        out = {}
        for k in _INTEGRATION_FIELDS:
            t = vv.get(k)
            if t is None:
                t = _data_field_torch(self, k)
            if t is None:
                continue
            out[k] = t[rows].clone()
        return out

    def restore(self, mask: torch.Tensor,
                snapshot: Mapping[str, torch.Tensor]) -> None:
        """快照写回 + forward（近似恢复，见类 docstring）。"""
        writable = {k: v for k, v in snapshot.items()}
        self.apply_patch(mask, writable, refresh=RefreshPolicy.FORWARD)

    def status(self) -> Dict[str, Any]:
        if self._wdata is None:
            return {"initialized": False}
        vv = self.views()
        return {
            "initialized": True,
            "nacon": int(vv["nacon"].reshape(-1)[0].item())
                     if vv["nacon"].numel() else 0,
            "nconmax_total": self._nconmax_per_world * self._batch_size,
            "njmax_total": self._njmax_per_world * self._batch_size,
        }

    def close(self) -> None:
        self._wdata = None
        self._views = None
        self._ext_dev = None
        self._sched_dev = None
        self._act_target_wp = None

    # ------------------------------------------------------------------
    # warp 特有辅助（不进契约，但 facade/binding 需要使用）
    # ------------------------------------------------------------------
    def ensure_act_target(self, n_ctrl: int) -> torch.Tensor:
        """分配 act_target 缓冲（若未建）并返回 torch 视图。"""
        if self._act_target_wp is None:
            self._act_target_wp = self._wp.zeros(
                (self._batch_size, n_ctrl),
                dtype=self._wp.float32, device=self._device_str)
            if self._views is not None:
                self._views["act_target"] = self._wp.to_torch(
                    self._act_target_wp)
        return self._wp.to_torch(self._act_target_wp)

    def set_wrench_schedule(self, sched: Optional[torch.Tensor]) -> None:
        """上传子步外力表 (B, S, nbody, 6)；下一个 advance 消费。"""
        if sched is not None and (
                sched.shape[0] != self._batch_size
                or sched.shape[2:] != (self._model.nbody, 6)):
            raise ContractError(
                f"force schedule must be (B, n_steps, "
                f"{self._model.nbody}, 6), got {tuple(sched.shape)}")
        self._sched_dev = None if sched is None else sched.contiguous()
        self.views()["xfrc_sched"] = self._sched_dev

    def forward(self) -> None:
        """显式 forward（apply_patch refresh=NONE 后的补偿重建）。"""
        with self._wp.ScopedStream(self._wp.stream_from_torch()):
            self._mjw.forward(self._wmodel, self._wdata)

    # ------------------------------------------------------------------
    # host 快照（验证/回放路径；伪 mjx.Data 形态喂 binding 提取函数）
    # ------------------------------------------------------------------
    def host_snapshot(self):
        """mjw.Data → numpy namespace（含 padded contact + efc_force）。

        **这是 host 快照而非活跃数据**——提取公式消费它如同 mjx.Data。
        属验证/回放路径，吞吐路径不经过。
        """
        from types import SimpleNamespace
        d = self._wdata
        if d is None:
            return None
        contact_ns, efc_force = self._contacts_padded(d)
        impl = SimpleNamespace(contact=contact_ns, efc_force=efc_force)
        return SimpleNamespace(
            qpos=d.qpos.numpy().astype(np.float64),
            qvel=d.qvel.numpy().astype(np.float64),
            xpos=d.xpos.numpy().astype(np.float64),
            xquat=d.xquat.numpy().astype(np.float64),
            xipos=d.xipos.numpy().astype(np.float64),
            xanchor=d.xanchor.numpy().astype(np.float64),
            cvel=d.cvel.numpy().astype(np.float64),
            ctrl=d.ctrl.numpy().astype(np.float64),
            _impl=impl,
        )

    def _contacts_padded(self, d):
        """warp flat packed contacts → binding 提取期望的 per-world padded 布局。

        warp: contact.* 形状 (naconmax,)，跨 world，worldid 归属，
        前 nacon 项活跃。转换为 (B, cap) 填充 dist=+inf 的 padding。
        力：d.efc.force 已按 world 分组 (nworld, nefc)，直接透传。
        """
        B = self._batch_size
        cap = self._nconmax_per_world
        c = d.contact
        n_active = int(np.atleast_1d(d.nacon.numpy())[0])
        worldid = c.worldid.numpy()[:n_active]
        geom = c.geom.numpy()[:n_active]
        dist = c.dist.numpy()[:n_active]
        pos = c.pos.numpy()[:n_active]
        frame = c.frame.numpy()[:n_active]
        dim = c.dim.numpy()[:n_active]
        efc_adr = c.efc_address.numpy()[:n_active, 0]  # 每接触首个 efc 行址

        # (B, cap) padding 容器：dist=+inf ⇒ active=False（binding dist<=0 判定）。
        # dim 按非 batched (C,) 消费——本模型 condim 全场为 3 无歧义，
        # 取第一个填充者的值；efc_address 是逐 world 行址，传 (B,cap)。
        geom_pad = np.zeros((B, cap, 2), dtype=np.int64)
        dist_pad = np.full((B, cap), np.inf)
        pos_pad = np.zeros((B, cap, 3))
        frame_pad = np.broadcast_to(np.eye(3), (B, cap, 3, 3)).copy()
        dim_arr = np.ones(cap, dtype=np.int64)
        efc_arr = np.zeros((B, cap), dtype=np.int64)
        slot_seen = np.zeros(cap, dtype=bool)
        counts = np.zeros(B, dtype=np.int64)
        for i in range(n_active):
            w = int(worldid[i])
            slot = counts[w]
            counts[w] += 1
            if slot >= cap:
                continue  # 超容量丢弃（warp 本就 capped，语义一致）
            geom_pad[w, slot] = geom[i]
            dist_pad[w, slot] = dist[i]
            pos_pad[w, slot] = pos[i]
            frame_pad[w, slot] = frame[i]
            efc_arr[w, slot] = efc_adr[i]
            if not slot_seen[slot]:
                dim_arr[slot] = dim[i]
                slot_seen[slot] = True
        contact_ns = SimpleNamespace(
            geom=geom_pad, dist=dist_pad, pos=pos_pad, frame=frame_pad,
            dim=dim_arr, efc_address=efc_arr,
        )
        efc_force = d.efc.force.numpy().astype(np.float64)  # (nworld, nefc)
        return contact_ns, efc_force


def _data_field_torch(backend: WarpBackend, name: str):
    """d.<name> 的 torch 视图（_views 未登记的字段；act/time 等）。"""
    attr = getattr(backend._wdata, name, None)
    if attr is None:
        return None
    try:
        return backend._wp.to_torch(attr)
    except Exception:
        return None
