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

from .mjx_simulator import MjxHumanoid21Simulator


class WarpHumanoid21Simulator(MjxHumanoid21Simulator):
    """NWORLDS 批量模拟器（mujoco-warp 后端）。

    公开接口与 MjxHumanoid21Simulator 完全一致：reset / physical_step /
    set_action / apply_external_force / set_core_state /
    set_integration_state / get_observation / get_core_state /
    get_derived_state / get_static_data。
    """

    def __init__(self, batch_size: int = 1,
                 nconmax_per_world: int = 48, **_kwargs):
        super().__init__(batch_size=batch_size, precision="fp32")
        import warp as wp
        import mujoco_warp as mjw

        wp.init()
        self._wp = wp
        self._mjw = mjw
        self._wmodel = mjw.put_model(self._model)
        self._nconmax_per_world = nconmax_per_world
        self._wdata = None          # mjw.Data
        self._ext_np = None         # (B, nbody, 6) 挂起外力（同 _ext_force_jax 语义）
        self._action_np = None      # {robot: (B,21)} 当前 PD 目标动作
        self._pd_kernel = None
        self._pd_arrays = None

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
            self._wdata = self._mjw.put_data(
                self._model, mjd0, nworld=B,
                nconmax=B * self._nconmax_per_world)
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
        self._ext_np = np.zeros((B, self._model.nbody, 6), dtype=np.float64)
        self._refresh_pd_target()
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

    def physical_step(self, n_steps: int = 1, keep_history: bool = False) -> None:
        if self._wdata is None:
            raise RuntimeError("Call reset() before physical_step()")
        if keep_history:
            raise NotImplementedError(
                "warp backend: keep_history 未实现（验证/吞吐路径均不需要）")
        wp, mjw, pa = self._wp, self._mjw, self._pd_arrays
        d = self._wdata
        for i in range(n_steps):
            wp.launch(self._pd_kernel, dim=(self._batch_size, 42),
                      inputs=[d.qpos, d.qvel, pa["target"], pa["qpos_idx"],
                              pa["qvel_idx"], pa["act_ids"], pa["gear"],
                              pa["lo"], pa["hi"], pa["kp"], pa["kd"], d.ctrl],
                      device="cuda:0")
            # 挂起外力只作用于第一个子步（CPU xfrc_applied 每步清零语义）
            if i == 0 and self._ext_np is not None and np.abs(self._ext_np).max() > 0:
                d.xfrc_applied.assign(
                    wp.array(self._ext_np.astype(np.float32), device="cuda:0"))
            else:
                d.xfrc_applied.assign(pa["zero_xfrc"])
            mjw.step(self._wmodel, d)
        # CPU physical_step 末尾清零 data 上的施加力
        d.xfrc_applied.assign(pa["zero_xfrc"])
        d.qfrc_applied.assign(
            wp.zeros((self._batch_size, self._model.nv), dtype=wp.float32,
                     device="cuda:0"))
        if self._ext_np is not None:
            self._ext_np[:] = 0.0

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
        self._ext_np[:, body_id, :3] += np.asarray(force, dtype=np.float64)
        if torque is not None:
            self._ext_np[:, body_id, 3:6] += np.asarray(torque, dtype=np.float64)

    @property
    def _ext_force_jax(self):
        """与父类同名的挂起外力缓冲（validation adapter 直接读）。"""
        return self._ext_np

    @_ext_force_jax.setter
    def _ext_force_jax(self, _value):
        # 父类 __init__ 赋值 _ext_force_jax=None；warp 侧真实缓冲是 _ext_np。
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

    def close(self) -> None:
        self._wdata = None
        self._history_buffer = None
