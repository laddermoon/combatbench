"""episode_exporter — 向量化 Episode 组装（E3-W4，D10.2）。

替代逐帧 dict + ``Episode.from_buffer_frames`` 的热路径：缓冲已是
``(T,B,·)`` 预分配，按行切片直接构造 frozen dataclass 字段——
语义等价但无 T×B 次 Python dict 中转。

``from_buffer_frames`` 保留为 golden 参考：等价测试对同一批 np
数据跑两条路径逐字段比对。
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from baseline.framework.rollout.episode import Episode
from baseline.framework.rollout.job import Job


def export_episode(np_bufs: Mapping[str, Any],
                   *,
                   job: Job,
                   ep_index: int,
                   row: int,
                   agent_ids: Sequence[str],
                   ef_pair,
                   stochastic: bool,
                   env_hash: str,
                   term_records: Mapping[str, Sequence],
                   metrics_np: Mapping[str, np.ndarray],
                   T: int) -> Episode:
    """按行切片直接构造 Episode（向量路径）。

    Args:
        np_bufs: 波末批量 D2H 后的 numpy 缓冲字典：
            obs/act/log_prob/final_obs {rid: ndarray}、
            obs_out {observer: {leaf: ndarray}}、env_term (B,)。
        ef_pair: ``((ef_a, df_a), (ef_b, df_b))``——per-agent
            (explore_factor, delta_factor) 标量。
    """
    # env_term_step 已是含端点边界（本步 1-based 帧序号 = 帧索引+1），
    # 终止帧包含在 t_use 内；退化帧（物理增量 0）由 Episode 的
    # agent_frame_boundary 在消费侧排除，这里仍忠实导出全部记录帧。
    term_step = int(np_bufs["env_term"][row])
    t_use = term_step if 0 < term_step <= T else T

    observations = {rid: np_bufs["obs"][rid][:t_use, row]
                    for rid in agent_ids}
    actions = {rid: np_bufs["act"][rid][:t_use, row]
               for rid in agent_ids}

    # action_extras / explore_factors——与 CPU SamplingPolicy 写入的
    # 帧 dict 经 _stack_* 后的形态一致：每键 (T,) f32。
    # stochastic=False 波与 CPU 对齐：extras=None → 整组为空。
    if stochastic:
        action_extras = {}
        explore_factors = {}
        for rid, (ef, df) in zip(agent_ids, ef_pair):
            lp = np_bufs["log_prob"][rid][:t_use, row].astype(np.float32)
            # 优先用记录的逐帧 ef（ef 程序路径的真实值）；无记录
            # 时回退标量广播（静态 ef 两值本就相同）。
            rec = np_bufs.get("ef", {}).get(rid)
            if rec is not None:
                ef_arr = rec[:t_use, row].astype(np.float32)
            else:
                ef_arr = np.full(t_use, np.float32(ef),
                                 dtype=np.float32)
            df_arr = np.full(t_use, np.float32(df), dtype=np.float32)
            action_extras[rid] = {
                "log_prob": lp,
                "explore_factor": ef_arr,
                "sctx__delta_factor": df_arr,
            }
            explore_factors[rid] = ef_arr
    else:
        action_extras, explore_factors = {}, {}

    # observer 输出叶——与 _try_stack 同语义：每帧标量叶 → list[T]，
    # 向量叶 → (T,*) ndarray。
    def _leaf(arr):
        a = arr[:t_use, row]
        return a.tolist() if a.ndim == 1 else a
    observer_outputs = {
        name: {k: _leaf(v) for k, v in fields.items()}
        for name, fields in np_bufs["obs_out"].items()}

    metrics = {"backend": "warp-fp32"}
    for k, arr in metrics_np.items():
        v = arr[row]
        metrics[k] = v.item() if np.ndim(v) == 0 else v.tolist()

    return Episode(
        base_seed=int(job.seed),
        episode_index=int(ep_index),
        blueprint_hash=str(env_hash),
        num_frames=t_use,
        episode_options=dict(job.episode_options),
        agent_termination_proposal_records={
            rid: tuple(term_records[rid]) for rid in agent_ids},
        observations=observations,
        actions=actions,
        action_extras=action_extras,
        explore_factors=explore_factors,
        observer_outputs=observer_outputs,
        final_observation={rid: np_bufs["final_obs"][rid][row]
                           for rid in agent_ids},
        episode_metrics=metrics,
        physics_steps=(
            np_bufs["phys"][:t_use, row].astype(np.int64)
            if np_bufs.get("phys") is not None else None
        ),
    )
