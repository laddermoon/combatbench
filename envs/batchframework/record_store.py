"""record_store — 波内设备记录缓冲（E3-W3，D10.1 落地）。

语义边界：
- 缓冲所有权归 collector（经 RecordStore），不归插件——不被
  ``partial reset``/``plugin pool`` 清零影响；
- 形状固定 ``(T, B, ...)`` 预分配——**无逐步动态分配**；
- ``frame_valid``：该 env 行该步是否产生 CPU 兼容记录帧（ENDED 后
  False——显式化旧实现里 ``t_use=term_step`` 截断的隐含语义）；
- 终止记录不在 store 冗余存——波末从 ``ep.term_history``（runtime
  权威源）导出；store 只持有 ``env_term_step``/``final_obs``。

observer schema：``{observer_name: {leaf: (dtype, shape_suffix)}}``——
shape_suffix 是 (T,B) 之后的形状尾（标量叶 = ``()``）。声明过的叶
缺失/形状不符 = 装配错误（报错，不静默跳过）；未在 schema 的叶忽略。
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from .binding_registry import IoSchema
from .device_state import DeviceBatchState


class ObserverLeaf(NamedTuple):
    dtype: torch.dtype
    shape_suffix: Tuple[int, ...] = ()


class RecordStore:
    """(T,B,·) 预分配设备缓冲 + 波界收尾。"""

    def __init__(self, io_schema: IoSchema, T: int, B: int,
                 device: torch.device,
                 observer_schemas: Optional[
                     Dict[str, Dict[str, ObserverLeaf]]] = None):
        self.schema = io_schema
        self.T, self.B = int(T), int(B)
        self.device = device
        dev = device
        f32, bl = torch.float32, torch.bool
        self.obs = {rid: torch.zeros(T, B, io_schema.obs_dim,
                                     device=dev, dtype=f32)
                    for rid in io_schema.agent_ids}
        self.act = {rid: torch.zeros(T, B, io_schema.action_dim,
                                     device=dev, dtype=f32)
                    for rid in io_schema.agent_ids}
        self.log_prob = {rid: torch.zeros(T, B, device=dev, dtype=f32)
                         for rid in io_schema.agent_ids}
        self.frame_valid = torch.zeros(T, B, dtype=bl, device=dev)
        self.env_term_step = torch.full((B,), -1, dtype=torch.int32,
                                        device=dev)
        self.final_obs: Dict[str, torch.Tensor] = {
            rid: torch.zeros(B, io_schema.obs_dim, device=dev, dtype=f32)
            for rid in io_schema.agent_ids}
        self._final_captured = torch.zeros(B, dtype=bl, device=dev)

        # observer 叶子缓冲（schema 声明则预分配；否则惰性 (B,) 叶——
        # 与旧 _WaveRecorder 同语义的兼容路径）
        self._obs_schema = observer_schemas or {}
        self.obs_out: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, leaves in self._obs_schema.items():
            self.obs_out[name] = {
                leaf_name: torch.zeros(
                    (T, B, *spec.shape_suffix), device=dev,
                    dtype=spec.dtype)
                for leaf_name, spec in leaves.items()}

        self.term_records: List[Dict[str, List]] = []

    # ------------------------------------------------------------------
    def n_bytes(self) -> int:
        """设备缓冲总字节数（D10.3 预算用）。"""
        def _sum(d):
            return sum(t.numel() * t.element_size() for t in d.values())
        total = _sum(self.obs) + _sum(self.act) + _sum(self.log_prob) \
            + _sum(self.final_obs)
        total += self.frame_valid.numel() + self.env_term_step.numel() * 4
        for bufs in self.obs_out.values():
            total += _sum(bufs)
        return int(total)

    # ------------------------------------------------------------------
    def begin_wave(self) -> None:
        self.env_term_step.fill_(-1)
        self._final_captured.zero_()
        self.frame_valid.zero_()
        self.term_records = [{rid: [] for rid in self.schema.agent_ids}
                             for _ in range(self.B)]

    def write_inputs(self, t: int, state: DeviceBatchState,
                     running: torch.Tensor) -> None:
        """帧输入：当前 obs（本步 action 的输入观测）。"""
        io = state.io
        for rid, obs in zip(self.schema.agent_ids,
                            (io.obs_a, io.obs_b)):
            self.obs[rid][t].copy_(obs)
        self.frame_valid[t] = running

    def write_actions(self, t: int, a_a: torch.Tensor, a_b: torch.Tensor,
                      lp_a: Optional[torch.Tensor],
                      lp_b: Optional[torch.Tensor]) -> None:
        ra, rb = self.schema.agent_ids
        self.act[ra][t].copy_(a_a)
        self.act[rb][t].copy_(a_b)
        if lp_a is not None:
            self.log_prob[ra][t].copy_(lp_a)
            self.log_prob[rb][t].copy_(lp_b)

    def write_observer_step(self, t: int,
                            outputs: Dict[str, Dict[str, torch.Tensor]],
                            rows: Optional[torch.Tensor] = None) -> None:
        """observer 输出 → 缓冲。

        ``rows=None`` 写全部行；否则只写 ``buf[t, rows]``（末帧覆写用）。
        schema 声明的叶缺失 → 报错；未声明的叶忽略。
        """
        B = self.B
        for name, leaves in self._obs_schema.items():
            out = outputs.get(name)
            if out is None:
                raise ValueError(
                    f"observer {name!r} produced no output at step {t} "
                    f"(schema declares {sorted(leaves)})")
            for leaf_name, spec in leaves.items():
                v = out.get(leaf_name)
                if v is None:
                    raise ValueError(
                        f"observer {name!r} missing declared leaf "
                        f"{leaf_name!r} at step {t}")
                if v.shape[0] != B or tuple(v.shape[1:]) != spec.shape_suffix:
                    raise ValueError(
                        f"observer {name!r}.{leaf_name} shape "
                        f"{tuple(v.shape)} != (B={B}, *{spec.shape_suffix})")
                if rows is None:
                    self.obs_out[name][leaf_name][t].copy_(v)
                else:
                    self.obs_out[name][leaf_name][t, rows] = v[rows]
        # 兼容路径：未声明 schema 的 observer 按旧规则收 (B,) 张量叶
        for name, out in outputs.items():
            if name in self._obs_schema or out is None:
                continue
            bufs = self.obs_out.setdefault(name, {})
            for k, v in out.items():
                if not torch.is_tensor(v) or v.shape[:1] != (B,):
                    continue
                buf = bufs.get(k)
                if buf is None:
                    buf = torch.zeros(self.T, B, dtype=v.dtype,
                                      device=self.device)
                    bufs[k] = buf
                if rows is None:
                    buf[t].copy_(v)
                else:
                    buf[t, rows] = v[rows]

    def seal_rows(self, ids: torch.Tensor, ep, io) -> None:
        """本步新 ENDED 行：记 env_term_step + 捕获终止时刻 final_obs。

        只记本波**首次**终止（显式 reset_rows 的再终止不入本波记录）。
        """
        if ids.numel() == 0:
            return
        fresh = ids[self.env_term_step[ids] < 0]
        if fresh.numel() == 0:
            return
        self.env_term_step[fresh] = ep.episode_steps[fresh].to(torch.int32)
        for rid, obs in zip(self.schema.agent_ids, (io.obs_a, io.obs_b)):
            self.final_obs[rid][fresh] = obs[fresh]
        self._final_captured[fresh] = True

    def finalize(self, ep, io) -> None:
        """波末：未终止行 final_obs 兜底 + term_history → records 导出。"""
        live = ~self._final_captured
        if bool(live.any()):
            for rid, obs in zip(self.schema.agent_ids,
                                (io.obs_a, io.obs_b)):
                self.final_obs[rid][live] = obs[live]
        names = {v: k for k, v in ep.reason_registry.items()}
        hist = ep.term_history.cpu()
        hlen = ep.term_history_len.cpu()
        steps = ep.episode_steps.cpu()
        for e in range(self.B):
            for a, rid in enumerate(self.schema.agent_ids):
                recs = self.term_records[e][rid]
                for k in range(int(hlen[e, a])):
                    code, step = int(hist[e, a, k, 0]), int(hist[e, a, k, 1])
                    recs.append((names.get(code, "custom"), step))
                if not recs:
                    recs.append(("abandoned", int(steps[e])))
