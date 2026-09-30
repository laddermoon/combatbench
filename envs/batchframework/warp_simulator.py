"""mujoco-warp 后端模拟器：与 MjxHumanoid21Simulator 同契约的 NWORLDS 实现。

设计原则：**最大复用父类的语义代码，只替换数据承载层**。
- 观测/core-state/PD 公式、reset 姿态计算、set_core_state 映射全部继承
  ``mjx_simulator.py`` —— 同一个函数源，不存在"warp 版另写一遍公式"
  的漂移风险。
- 差异仅在数据布局与步进实现：warp 是 NWORLDS 原生批量（字段自带
  leading nworld 维、contact 跨 world packed 存储、内核手写 CUDA），
  不是 vmap + 稠密张量。

布局要点：
- ``mjw.put_data(mjm, mjd, nworld=B)`` 创建批量数据，``mjw.step(m, d)``
  一次推进所有 world（原地修改 d，无返回值）。
- contact 是跨 world 的 flat packed 数组（容量 naconmax=48/world 默认），
  ``worldid`` 字段标记归属，前 ``nacon`` 项为活跃接触。
- 提取路径统一走 ``_wdata()`` 快照：把 warp 数组 pull 成 numpy 并组织成
  父类提取函数期望的形状（mjx.Data 形似的 namespace），因此
  ``_extract_core_state`` / ``_get_robot_view_batch`` 等父类方法零改动复用。
  这是 host 侧路径（验证/语义用途），不是吞吐路径——吞吐路径见
  ``probe_e2e_warp.py``。

精度边界：mujoco-warp 只有 fp32，无法参与 fp64 级容差对照；
跨后端 fixture 的 warp 容差单独标定（见 validation_warp / M2 文档）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .device_state import (
    ContactFlatNamespace,
    DeviceBatchState,
    EpisodeNamespace,
    IoNamespace,
    RngNamespace,
    SimNamespace,
)
from .mjx_simulator import MjxHumanoid21Simulator


class WarpHumanoid21Simulator(MjxHumanoid21Simulator):
    """NWORLDS 批量模拟器（mujoco-warp 后端）。

    公开接口与 MjxHumanoid21Simulator 完全一致：reset / physical_step /
    set_action / apply_external_force / set_core_state /
    set_integration_state / get_observation / get_core_state /
    get_derived_state / get_static_data。
    """

    def __init__(self, batch_size: int = 1,
                 nconmax_per_world: int = 48,
                 njmax_per_world: int = 512, **_kwargs):
        # warp 路径不用 jax：_init_jax=False 让父类跳过 jax.devices()/
        # mjx.put_model/jp.array——XLA 默认预分配 ~75% 显存（24GiB ~18GB），
        # 曾把 warp mempool 挤到 OOM。env 变量是防御性保险：若未来代码
        # 路径重新触碰 jax 后端，至少不会无声吃掉整块卡。
        import os
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        super().__init__(batch_size=batch_size, precision="fp32",
                         _init_jax=False, **_kwargs)
        import warp as wp
        import mujoco_warp as mjw

        wp.init()
        self._wp = wp
        self._mjw = mjw
        self._wmodel = mjw.put_model(self._model)
        self._nconmax_per_world = nconmax_per_world
        # njmax 是 per-world 约束槽上限：接触最多 4 efc 行/contact（worst
        # case cap48 → ~192），再加关节 limit/friction/equality 约百余条；
        # 默认启发式只有 64，u55 曾触发 "nefc overflow - increase njmax to 65"
        # device assert 崩进程。512 余量充足，显存代价可忽略。
        self._njmax_per_world = njmax_per_world
        self._wdata = None          # mjw.Data
        self._ext_dev = None        # wp.array (B, nbody, 6) 挂起外力（设备端）
        self._sched_dev = None      # wp.array (B, S, nbody, 6) 子步 schedule
        self._action_np = None      # {robot: (B,21)} 当前 PD 目标动作
        self._pd_kernel = None
        self._pd_arrays = None
        self._views = None          # dict[str, torch.Tensor] — mjw.Data 零拷贝视图
        self._dev_state = None      # DeviceBatchState
        self._obs_builder = None    # device_obs.WarpObsBuilder（惰性）

    # ------------------------------------------------------------------
    # warp <-> 父类提取函数的桥：numpy 快照伪装成 mjx.Data 形态
    # ------------------------------------------------------------------
    @property
    def _mjx_data(self):
        """父类提取代码读取 ``self._mjx_data``；这里返回 numpy 快照视图。

        注意：这是 host 快照，不是活跃数据；写路径（physical_step/
        set_*）一律操作 self._wdata。
        """
        return self._wdata_snapshot()

    @_mjx_data.setter
    def _mjx_data(self, _value):
        # 父类 __init__ 会赋值 self._mjx_data=None；吞掉以兼容。
        pass

    def _wdata_snapshot(self) -> SimpleNamespace:
        d = self._wdata
        if d is None:
            return None
        B = self._batch_size
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
        """warp flat packed contacts → 父类期望的 per-world padded 布局。

        warp: contact.* 形状 (naconmax,)，跨 world，worldid 归属，
        前 nacon 项活跃。转换为 (B, cap) 填充 dist=+inf 的 padding。
        力：d.efc.force 已按 world 分组 (nworld, nefc)，直接透传。
        """
        B = self._batch_size
        cap = self._nconmax_per_world
        c = d.contact
        n_active = int(d.nacon.numpy()[0])
        worldid = c.worldid.numpy()[:n_active]
        geom = c.geom.numpy()[:n_active]
        dist = c.dist.numpy()[:n_active]
        pos = c.pos.numpy()[:n_active]
        frame = c.frame.numpy()[:n_active]
        dim = c.dim.numpy()[:n_active]
        efc_adr = c.efc_address.numpy()[:n_active, 0]  # 每接触首个 efc 行址

        # (B, cap) padding 容器：dist=+inf ⇒ active=False（父类 dist<=0 判定）。
        # dim 父类按非 batched (C,) 消费——本模型 condim 全场为 3 无歧义，
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

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seeds=None, options=None) -> None:
        wp = self._wp
        B = self._batch_size
        qpos_all, action_a, action_b = self._compute_reset_state(seeds, options)

        import mujoco
        mjd0 = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, mjd0)
        if self._wdata is None:
            # nconmax/njmax 均为 per-world 语义——put_data 内部
            # naconmax = nconmax × nworld。曾误传 B×48 使每 world 槽位
            # = B×48、总量 = B²×48：显存二次方增长（B=1536 → ~20GB）且
            # 单 world 接触上限被放大到可触碰 njmax 溢出断言。
            self._wdata = self._mjw.put_data(
                self._model, mjd0, nworld=B,
                nconmax=self._nconmax_per_world,
                njmax=self._njmax_per_world)
        d = self._wdata
        d.qpos.assign(wp.array(qpos_all.astype(np.float32), device="cuda:0"))
        d.qvel.assign(wp.zeros((B, self._model.nv), dtype=wp.float32, device="cuda:0"))
        d.ctrl.assign(wp.zeros((B, self._model.nu), dtype=wp.float32, device="cuda:0"))
        d.xfrc_applied.assign(
            wp.zeros((B, self._model.nbody, 6), dtype=wp.float32, device="cuda:0"))
        d.qacc_warmstart.assign(
            wp.zeros((B, self._model.nv), dtype=wp.float32, device="cuda:0"))
        self._mjw.forward(self._wmodel, d)
        wp.synchronize()

        self._action_np = {"robot_a": action_a, "robot_b": action_b}
        if self._ext_dev is None:
            self._ext_dev = wp.zeros((B, self._model.nbody, 6),
                                     dtype=wp.float32, device="cuda:0")
        else:
            self._ext_dev.zero_()
        self._sched_dev = None
        self._refresh_pd_target()
        if self._dev_state is not None:
            import torch
            dev = torch.device("cuda:0")
            self._dev_state.io.action_a.copy_(
                torch.as_tensor(action_a, device=dev))
            self._dev_state.io.action_b.copy_(
                torch.as_tensor(action_b, device=dev))
            self._dev_state.clear_step_flags()
            self._dev_state.episode.episode_steps.zero_()
            self._dev_state.episode.active_mask.fill_(True)
            self._dev_state.episode.time.zero_()
        self._history_buffer = None
        self._history_n_steps = 0

    # ------------------------------------------------------------------
    # PD 控制（wp.kernel） + physical_step
    # ------------------------------------------------------------------
    def _build_pd_kernel(self):
        wp = self._wp
        statics = self._jax_statics
        dev = "cuda:0"

        def cat(key):
            return np.concatenate([np.asarray(statics["robot_a"][key]),
                                   np.asarray(statics["robot_b"][key])])

        self._pd_arrays = dict(
            qpos_idx=wp.array(cat("qpos_indices"), dtype=wp.int32, device=dev),
            qvel_idx=wp.array(cat("qvel_indices"), dtype=wp.int32, device=dev),
            act_ids=wp.array(cat("actuator_ids"), dtype=wp.int32, device=dev),
            gear=wp.array(cat("gear"), dtype=wp.float32, device=dev),
            lo=wp.array(cat("ctrl_lo"), dtype=wp.float32, device=dev),
            hi=wp.array(cat("ctrl_hi"), dtype=wp.float32, device=dev),
            kp=wp.array(np.concatenate([self.KP, self.KP]), dtype=wp.float32, device=dev),
            kd=wp.array(np.concatenate([self.KD, self.KD]), dtype=wp.float32, device=dev),
            target=wp.zeros((self._batch_size, 42), dtype=wp.float32, device=dev),
            zero_xfrc=wp.zeros((self._batch_size, self._model.nbody, 6),
                               dtype=wp.float32, device=dev),
        )
        self._norm_ref_cat = np.concatenate(
            [self._norm_params["robot_a"]["reference"],
             self._norm_params["robot_b"]["reference"]])
        self._norm_scale_cat = np.concatenate(
            [self._norm_params["robot_a"]["scale"],
             self._norm_params["robot_b"]["scale"]])

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

        self._pd_kernel = pd_kernel

    def _refresh_pd_target(self):
        """action → target rad (B,42) 写入 warp 数组。"""
        if self._pd_kernel is None:
            self._build_pd_kernel()
        a = np.concatenate([self._action_np["robot_a"],
                            self._action_np["robot_b"]], axis=-1)
        target = a * self._norm_scale_cat + self._norm_ref_cat
        self._pd_arrays["target"].assign(
            self._wp.array(target.astype(np.float32), device="cuda:0"))

    # ------------------------------------------------------------------
    # 设备视图：mjw.Data 字段 → torch 零拷贝视图
    # ------------------------------------------------------------------
    def _torch_views(self) -> Dict[str, Any]:
        """缓存的 torch 视图字典。_wdata/_ext_dev/_pd_arrays 存活期内有效。

        物理循环与设备插件共享同一组视图；torch 侧就地写直接落到
        warp 底层存储。stream 有序性由 physical_step 的 ScopedStream
        （warp kernel 绑到 torch 当前流）保证。
        """
        if self._views is None:
            if self._wdata is None:
                raise RuntimeError("Call reset() before device access")
            import torch  # noqa: F401 — 确保 CUDA 上下文先于 wp 视图创建
            wp, d, pa = self._wp, self._wdata, self._pd_arrays
            wt = wp.to_torch
            self._views = dict(
                qpos=wt(d.qpos), qvel=wt(d.qvel), ctrl=wt(d.ctrl),
                xpos=wt(d.xpos), xquat=wt(d.xquat), xipos=wt(d.xipos),
                xanchor=wt(d.xanchor), cvel=wt(d.cvel),
                xfrc_applied=wt(d.xfrc_applied),
                qfrc_applied=wt(d.qfrc_applied),
                qacc_warmstart=wt(d.qacc_warmstart),
                xfrc_pending=wt(self._ext_dev),
                act_target=wt(pa["target"]),
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
                xfrc_sched=None,   # 上传 schedule 时填入
            )
        return self._views

    def physical_step(self, n_steps: int = 1, keep_history: bool = False) -> None:
        if self._wdata is None:
            raise RuntimeError("Call reset() before physical_step()")
        if keep_history:
            raise NotImplementedError(
                "warp backend: keep_history 未实现（验证/吞吐路径均不需要）")
        wp, mjw, pa = self._wp, self._mjw, self._pd_arrays
        d = self._wdata
        v = self._torch_views()
        xf, pend = v["xfrc_applied"], v["xfrc_pending"]
        sched = v["xfrc_sched"]
        if sched is not None and sched.shape[1] != n_steps:
            raise ValueError(
                f"force schedule has {sched.shape[1]} substeps, "
                f"physical_step got n_steps={n_steps}")
        # ScopedStream：把 warp kernel（pd/mjw.step）绑到 torch 当前流，
        # 与 torch 侧 xfrc 组合（zero_/add_）严格有序，全程零额外 host sync。
        with wp.ScopedStream(wp.stream_from_torch()):
            for i in range(n_steps):
                wp.launch(self._pd_kernel, dim=(self._batch_size, 42),
                          inputs=[d.qpos, d.qvel, pa["target"], pa["qpos_idx"],
                                  pa["qvel_idx"], pa["act_ids"], pa["gear"],
                                  pa["lo"], pa["hi"], pa["kp"], pa["kd"],
                                  d.ctrl],
                          device="cuda:0")
                # CPU xfrc_applied 每子步清零语义：pending 只进首个子步；
                # schedule 逐子步消费（插件"每物理步修改"的设备端等价物）。
                xf.zero_()
                if i == 0:
                    xf.add_(pend)
                if sched is not None:
                    xf.add_(sched[:, i])
                mjw.step(self._wmodel, d)
            # CPU physical_step 末尾清零 data 上的施加力 / solver 偏置
            xf.zero_()
            pend.zero_()
            v["qfrc_applied"].zero_()
        self._sched_dev = None
        v["xfrc_sched"] = None

    # ------------------------------------------------------------------
    # set_action / apply_external_force
    # ------------------------------------------------------------------
    def set_action(self, action: Dict[str, Any]) -> None:
        for rid in ("robot_a", "robot_b"):
            if rid in action and action[rid] is not None:
                act = np.asarray(action[rid], dtype=np.float32)
                if act.shape != (self._batch_size, self.ACTION_DIM):
                    raise ValueError(
                        f"Action for {rid} must have shape "
                        f"({self._batch_size}, {self.ACTION_DIM}), got {act.shape}")
                self._action_np[rid] = np.clip(act, -1.0, 1.0)
        self._refresh_pd_target()

    def get_action(self) -> Dict[str, Any]:
        return {} if self._action_np is None else dict(self._action_np)

    def apply_external_force(self, body_name, force, torque=None,
                             robot_id="robot_a") -> None:
        suffix = self._robots[robot_id]["suffix"]
        body_id = self._body_name_to_id.get(f"{body_name}{suffix}")
        if body_id is None:
            raise ValueError(f"Body not found: {body_name}{suffix}")
        import torch
        pend = self._torch_views()["xfrc_pending"]
        pend[:, body_id, :3] += torch.as_tensor(
            np.asarray(force), dtype=torch.float32, device=pend.device)
        if torque is not None:
            pend[:, body_id, 3:6] += torch.as_tensor(
                np.asarray(torque), dtype=torch.float32, device=pend.device)

    @property
    def _ext_force_jax(self):
        """与父类同名的挂起外力缓冲（validation adapter 直接读）。

        warp 侧真实缓冲是设备端 ``_ext_dev``；此属性返回 host 快照。
        """
        if self._ext_dev is None:
            return None
        return (self._torch_views()["xfrc_pending"]
                .detach().cpu().numpy().astype(np.float64))

    @_ext_force_jax.setter
    def _ext_force_jax(self, _value):
        # 父类 __init__ 赋值 _ext_force_jax=None；warp 侧真实缓冲是 _ext_dev。
        pass

    # ------------------------------------------------------------------
    # 状态写入（host 路径，验证用途）
    # ------------------------------------------------------------------
    def _write_qpos_qvel(self, qpos_new, qvel_new):
        wp = self._wp
        d = self._wdata
        B, nv = self._batch_size, self._model.nv
        d.qpos.assign(wp.array(qpos_new.astype(np.float32), device="cuda:0"))
        d.qvel.assign(wp.array(qvel_new.astype(np.float32), device="cuda:0"))
        # 状态恢复契约：全新求解，清零 warmstart/ctrl/施加力
        d.qacc_warmstart.assign(wp.zeros((B, nv), dtype=wp.float32, device="cuda:0"))
        d.ctrl.assign(wp.zeros((B, self._model.nu), dtype=wp.float32, device="cuda:0"))
        d.xfrc_applied.assign(
            wp.zeros((B, self._model.nbody, 6), dtype=wp.float32, device="cuda:0"))
        d.qfrc_applied.assign(wp.zeros((B, nv), dtype=wp.float32, device="cuda:0"))
        self._mjw.forward(self._wmodel, d)
        wp.synchronize()

    def set_integration_state(self, qpos, qvel, env_ids=None) -> None:
        if self._wdata is None:
            raise RuntimeError("Call reset() before set_integration_state()")
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        if env_ids is None:
            if qpos.shape != (self._batch_size, self._model.nq) or \
               qvel.shape != (self._batch_size, self._model.nv):
                raise ValueError("integration state shape mismatch")
            qpos_new, qvel_new = qpos, qvel
        else:
            env_ids = list(env_ids)
            qpos_new = self._wdata.qpos.numpy().astype(np.float64)
            qvel_new = self._wdata.qvel.numpy().astype(np.float64)
            if qpos.shape != (len(env_ids), self._model.nq) or \
               qvel.shape != (len(env_ids), self._model.nv):
                raise ValueError("per-env integration state shape mismatch")
            qpos_new[env_ids] = qpos
            qvel_new[env_ids] = qvel
        self._write_qpos_qvel(qpos_new, qvel_new)

    def set_core_state(self, state, env_ids=None) -> None:
        if self._wdata is None:
            raise RuntimeError("Call reset() before set_core_state()")
        if env_ids is None:
            env_ids = list(range(self._batch_size))
        env_ids = list(env_ids)
        qpos_new = self._wdata.qpos.numpy().astype(np.float64)
        qvel_new = self._wdata.qvel.numpy().astype(np.float64)
        self._write_core_state(state, env_ids, qpos_new, qvel_new)
        self._write_qpos_qvel(qpos_new, qvel_new)

    # ------------------------------------------------------------------
    # 提取：全部走父类实现（它们读 self._mjx_data → numpy 快照）
    # ------------------------------------------------------------------
    # get_core_state / get_derived_state / get_observation 继承自父类，
    # 唯一差异是 contact 的 _impl 字段由 _contacts_padded 适配。

    # ------------------------------------------------------------------
    # M3 设备数据平面：DeviceBatchState 绑定 + device mutator
    # ------------------------------------------------------------------
    def build_device_state(self) -> DeviceBatchState:
        """构造（或返回已有的）设备数据平面。

        sim 字段是 mjw.Data 的活视图；episode/io/rng 为本对象持有的
        CUDA 张量（reset 不重建——wdata 对象跨 reset 存活，视图不失效）。
        """
        if self._dev_state is not None:
            return self._dev_state
        import torch
        v = self._torch_views()
        B, dev = self._batch_size, torch.device("cuda:0")
        obs_dim = self._obs_dim()
        sim_ns = SimNamespace(
            qpos=v["qpos"], qvel=v["qvel"], ctrl=v["ctrl"],
            xpos=v["xpos"], xquat=v["xquat"], xipos=v["xipos"],
            xanchor=v["xanchor"], cvel=v["cvel"],
            xfrc_applied=v["xfrc_applied"], act_target=v["act_target"],
            contacts_flat=ContactFlatNamespace(
                worldid=v["con_worldid"], geom=v["con_geom"],
                dist=v["con_dist"], pos=v["con_pos"], frame=v["con_frame"],
                dim=v["con_dim"], efc_address=v["con_efc_address"],
                efc_force=v["efc_force"], n_active=v["nacon"],
                cap_per_world=self._nconmax_per_world,
            ),
        )
        episode_ns = EpisodeNamespace(
            episode_steps=torch.zeros(B, dtype=torch.int64, device=dev),
            active_mask=torch.ones(B, dtype=torch.bool, device=dev),
            terminated_flag=torch.zeros(B, dtype=torch.bool, device=dev),
            term_reason=torch.full((B,), -1, dtype=torch.int8, device=dev),
            agent_terminated=torch.zeros(B, 2, dtype=torch.bool, device=dev),
            agent_term_reason=torch.full(
                (B, 2), -1, dtype=torch.int8, device=dev),
            reset_request=torch.zeros(B, dtype=torch.bool, device=dev),
            time=torch.zeros(B, dtype=torch.float32, device=dev),
        )
        io_ns = IoNamespace(
            action_a=torch.zeros(B, self.ACTION_DIM, device=dev),
            action_b=torch.zeros(B, self.ACTION_DIM, device=dev),
            obs_a=torch.zeros(B, obs_dim, device=dev),
            obs_b=torch.zeros(B, obs_dim, device=dev),
            reward=torch.zeros(B, device=dev),
        )
        rng_ns = RngNamespace(
            seed_offsets=torch.zeros(B, dtype=torch.int64, device=dev),
            step_counter=torch.zeros((), dtype=torch.int64, device=dev),
        )
        self._dev_state = DeviceBatchState(B, sim_ns, episode_ns, io_ns, rng_ns)
        return self._dev_state

    def _obs_dim(self) -> int:
        if getattr(self, "_obs_dim_cached", None) is None:
            self._obs_dim_cached = int(
                self.get_observation()["robot_a"].shape[-1])
        return self._obs_dim_cached

    def _norm_consts_t(self):
        """norm ref/scale 的 torch 常量（dev_set_action 用）。"""
        if getattr(self, "_norm_t", None) is None:
            import torch
            self._norm_t = {
                rid: (torch.as_tensor(p["reference"], dtype=torch.float32,
                                      device="cuda:0"),
                      torch.as_tensor(p["scale"], dtype=torch.float32,
                                      device="cuda:0"))
                for rid, p in self._norm_params.items()
            }
        return self._norm_t

    # --- device mutator（插件经 mutator 调用；输入均为 CUDA torch.Tensor） ---

    def dev_set_action(self, action_a, action_b) -> None:
        """动作 → PD target，全设备端。形状 (B,21)，clip 到 [-1,1]。"""
        st = self.build_device_state()
        v = self._torch_views()
        norm = self._norm_consts_t()
        for rid, act, cols in (("robot_a", action_a, slice(0, 21)),
                               ("robot_b", action_b, slice(21, 42))):
            a = act.clamp(-1.0, 1.0)
            getattr(st.io, f"action_{rid[-1]}").copy_(a)
            ref, scale = norm[rid]
            v["act_target"][:, cols] = a * scale + ref

    def dev_add_ext_force(self, body_id: int, force,
                          torque=None) -> None:
        """累加挂起外力（设备端）；作用于下一 physical_step 的首个子步。"""
        pend = self._torch_views()["xfrc_pending"]
        pend[:, body_id, :3] += force
        if torque is not None:
            pend[:, body_id, 3:6] += torque

    def dev_upload_force_schedule(self, sched) -> None:
        """上传子步外力 schedule：(B, n_steps, nbody, 6) torch CUDA 张量。

        下一个 physical_step(n_steps) 逐子步消费后被清除；
        n_steps 必须与届时调用的 n_steps 一致。
        """
        if sched.shape[0] != self._batch_size or \
                sched.shape[2:] != (self._model.nbody, 6):
            raise ValueError(
                f"force schedule must be (B, n_steps, {self._model.nbody}, 6), "
                f"got {tuple(sched.shape)}")
        self._sched_dev = sched.contiguous()
        self._torch_views()["xfrc_sched"] = self._sched_dev

    def dev_reset_rows(self, env_ids) -> None:
        """部分 reset：指定 env 恢复初始姿态（默认 options），其余不动。

        初始姿态在 host 计算（确定性、每 env 独立），仅写回 env_ids 行；
        该 env 的 qvel/ctrl/warmstart/qfrc/xfrc 一并清零（"全新求解"契约），
        io.action/act_target 行恢复为初始姿态对应的 PD 目标。末调用
        ``mjw.forward`` 重建全 world derived 字段（未重置 world 的 qpos
        未变，重算无害）。
        """
        import torch
        st = self.build_device_state()
        v = self._torch_views()
        env_ids = torch.as_tensor(env_ids, dtype=torch.long,
                                  device=v["qpos"].device)
        qpos_all, act_a, act_b = self._compute_reset_state(None, None)
        dev = v["qpos"].device
        qp = torch.as_tensor(qpos_all, dtype=torch.float32, device=dev)
        v["qpos"][env_ids] = qp[env_ids]
        v["qvel"][env_ids] = 0.0
        for k in ("ctrl", "qacc_warmstart", "xfrc_applied", "qfrc_applied"):
            v[k][env_ids] = 0.0
        st.io.action_a[env_ids] = torch.as_tensor(
            act_a, dtype=torch.float32, device=dev)[env_ids]
        st.io.action_b[env_ids] = torch.as_tensor(
            act_b, dtype=torch.float32, device=dev)[env_ids]
        norm = self._norm_consts_t()
        v["act_target"][env_ids, :21] = (
            st.io.action_a[env_ids] * norm["robot_a"][1]
            + norm["robot_a"][0])
        v["act_target"][env_ids, 21:] = (
            st.io.action_b[env_ids] * norm["robot_b"][1]
            + norm["robot_b"][0])
        with self._wp.ScopedStream(self._wp.stream_from_torch()):
            self._mjw.forward(self._wmodel, self._wdata)
        self._wp.synchronize()

    def dev_set_integration_rows(self, env_ids, qpos_t, qvel_t) -> None:
        """原始 qpos/qvel 行写入（跨后端搬运/回放用），随后 forward 刷新。"""
        import torch
        v = self._torch_views()
        env_ids = torch.as_tensor(env_ids, dtype=torch.long,
                                  device=v["qpos"].device)
        v["qpos"][env_ids] = qpos_t
        v["qvel"][env_ids] = qvel_t
        for k in ("ctrl", "qacc_warmstart", "xfrc_applied", "qfrc_applied"):
            v[k][env_ids] = 0.0
        with self._wp.ScopedStream(self._wp.stream_from_torch()):
            self._mjw.forward(self._wmodel, self._wdata)
        self._wp.synchronize()

    def device_obs_builder(self):
        """惰性构造设备端观测器（公式复刻见 device_obs.py）。"""
        if self._obs_builder is None:
            from .device_obs import WarpObsBuilder
            self._obs_builder = WarpObsBuilder(self)
        return self._obs_builder

    def close(self) -> None:
        self._wdata = None
        self._views = None
        self._dev_state = None
        self._ext_dev = None
        self._sched_dev = None
        self._history_buffer = None
