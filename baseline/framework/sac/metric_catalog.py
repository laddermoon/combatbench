"""Metric catalog for ``sac_metrics_v1`` events.

The catalog intentionally uses SAC namespaces only.  PPO-specific names such
as ``advantage.*``, ``ratio.*``, ``clip.*`` or ``gae.*`` are rejected rather
than being silently reinterpreted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


EVENT_TYPES = (
    "round",
    "tick",
    "eval",
    "export",
    "checkpoint",
    "debug",
    "config",
    "terminal",
)

_ALLOWED_PREFIXES = {
    "round": (
        "collection.",
        "episode.",
        "reward.",
        "replay.",
        "timing.",
        "task.",
        "behavior.",
    ),
    "tick": (
        "batch.",
        "critic.",
        "actor.",
        "temperature.",
        "target.",
        "replay.",
        "timing.",
        "debug.",
        "regularizer.",
    ),
    "eval": ("eval.", "collection.", "timing.", "task."),
    "export": ("export.", "timing."),
    "checkpoint": ("checkpoint.", "timing.", "replay."),
    "debug": ("debug.", "timing.", "replay.", "collection.", "batch."),
    # Config events carry their payload in context; metrics are optional but
    # are kept in a distinct namespace if present.
    "config": ("config.",),
    # Terminal events carry status/reason/final_checkpoint in context; the
    # metric payload is a presence flag only.
    "terminal": ("run.",),
}

_FORBIDDEN_PREFIXES = (
    "advantage.",
    "ratio.",
    "clip.",
    "gae.",
    "ppo.",
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    event_type: str
    description: str = ""


class MetricCatalog:
    def allowed_prefixes(self, event_type: str) -> Tuple[str, ...]:
        try:
            return _ALLOWED_PREFIXES[event_type]
        except KeyError as exc:
            allowed = ", ".join(EVENT_TYPES)
            raise ValueError(
                f"unknown SAC metric event type {event_type!r}; allowed={allowed}"
            ) from exc

    def validate_metric_name(self, event_type: str, name: str) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("metric name must be a non-empty string")
        if name.startswith(_FORBIDDEN_PREFIXES):
            raise ValueError(
                f"metric {name!r} uses a PPO-only namespace and is not "
                "part of sac_metrics_v1"
            )
        prefixes = self.allowed_prefixes(event_type)
        if not any(name.startswith(prefix) for prefix in prefixes):
            raise ValueError(
                f"metric {name!r} is not allowed for event {event_type!r}; "
                f"expected prefixes {prefixes}"
            )

    def validate_metrics(
        self,
        event_type: str,
        metrics: Mapping[str, object],
    ) -> None:
        for name in metrics:
            self.validate_metric_name(event_type, str(name))


DEFAULT_METRIC_SPECS = tuple(
    MetricSpec(name=name, event_type=event)
    for event, names in {
        "round": (
            "collection.episodes",
            "collection.env_steps",
            "collection.agent_transitions",
            "collection.wall_time_s",
            "replay.transitions_added",
            "replay.size",
            "timing.collection_s",
            "timing.slice_s",
            "task.online_success",
            "task.final_potential_mean",
            "behavior.explore_factor",
            "behavior.random_start",
        ),
        "tick": (
            "batch.size",
            "critic.loss",
            "critic.q_mean",
            "critic.td_mean",
            "critic.q1_loss.<channel>",
            "critic.q2_loss.<channel>",
            "critic.updated.<channel>",
            "critic.valid_weight.<channel>",
            "critic.grad_norm.<channel>",
            "critic.clip_applied.<channel>",
            "actor.loss",
            "actor.grad_norm",
            "actor.log_prob_mean",
            "actor.pair1_frac",
            "actor.valid_count",
            "actor.weight.<channel>",
            "actor.weight_next.<channel>",
            "actor.reg_term",
            "regularizer.shannon",
            "temperature.alpha",
            "temperature.loss",
            "temperature.grad_norm",
            "target.tau",
            "target.pair1_frac",
            "replay.size",
            "timing.update_s",
        ),
        "eval": (
            "eval.episodes",
            "eval.success",
            "eval.survival_rate",
            "eval.survived",
            "eval.ep_len_mean",
            "eval.max_pot",
            "eval.max_stage",
            "eval.max_h",
            "eval.final_pot",
            "timing.eval_s",
        ),
        "export": ("export.bytes", "timing.export_s"),
        "checkpoint": ("checkpoint.bytes", "timing.checkpoint_s"),
        "debug": (
            "debug.status",
            "debug.capture_s",
            "debug.bytes",
            "debug.request",
        ),
        "config": (),
        "terminal": ("run.terminal",),
    }.items()
    for name in names
)


def metric_specs() -> Iterable[MetricSpec]:
    return DEFAULT_METRIC_SPECS


# =====================================================================
# Sectioned dashboard catalog (sac_catalog_v1)
#
# Same editorial role as the PPO dumpkit catalog: one data source for
# what each metric means and how the run dashboard is laid out.  The
# six sections below use SAC semantics only — no PPO concept reuse.
#
# Flat-key conventions served by ``/api/run/metrics`` (the round-row
# table produced by ``runview.run_events_table``):
#
# - round-event metrics verbatim: ``collection.*``, ``replay.*``,
#   ``behavior.*``, ``timing.*`` (round-level ones), ``task.*``;
# - eval metrics merged by ``collection_round``: ``eval.*`` (sparse);
# - ``tck.<metric>`` = per-round aggregate of the same-named tick
#   metric (mean by default; ``tck.timing.update_s`` is a *sum* — see
#   ``TCK_SUM_KEYS``).
# - channel split: SAC encodes the channel as the key *suffix*
#   (``critic.q1_loss.r_fall``); chart specs use ``{"ch": prefix}``
#   (per-channel chart) and ``{"chm": name, "metrics": [...]}``
#   (merged per-channel chart) — the SAC analog of PPO's pc/pcm.
# =====================================================================

SUPPRESS_KEYS = frozenset({
    "param_patches",     # row metadata — shown in Round Detail
})
SUPPRESS_PREFIXES = ()

# Tick metrics aggregated as round sums rather than means.
TCK_SUM_KEYS = frozenset({"timing.update_s"})

SECTIONS: List[Dict[str, Any]] = [

    # 1 ────────────────────────────────────────────────────────────
    {"id": "task",
     "title": "任务表现",
     "question": "训练有没有让任务做得更好？",
     "curated": [
        {"zone_all": "eval",
         "title": "评估表现",
         "subtitle": "held 段 eval 种子的任务能力曲线——实验 on_eval 输出的全部指标（稀疏）。",
         "guide": "值得注意：持续停滞、突然退化、波动变大；success/survival 与 max_pot 背离。\n"
                  "下一步：看对应 eval 视频；检查该 round 附近的 param patch；对照更新执行与探索分区。\n"
                  "不能说明：eval 种子与训练种子同分布（held-out 另评）；单点不等于趋势。"},
     ],
     "expanded": [
        {"title": "在线指标（实验 post_round_metrics 输出）", "zone_rest": "task"},
     ]},

    # 2 ────────────────────────────────────────────────────────────
    {"id": "update",
     "title": "更新执行",
     "question": "每轮梯度更新是否按预算执行，优化器是否健康？",
     "curated": [
        {"title": "Critic & Actor Loss",
         "keys": ["tck.critic.loss", "tck.actor.loss"],
         "right": ["tck.temperature.alpha"],
         "subtitle": "每轮内全部 critic tick 的损失均值（右轴 α）。",
         "guide": "critic.loss：多通道 MSE(td) 合计；actor.loss：−(Σw·Q + entropy + reg)。\n"
                  "值得注意：critic loss 持续上升/发散；actor loss 单调漂移但任务停滞；α 塌到边界。\n"
                  "下一步：展开 per-channel 分解；进 dump 的 ②③ 看 target/TD 构成。\n"
                  "不能说明：loss 下降≠任务提升；SAC 是 off-policy，近期 batch 分布不代表全 replay。"},
        {"title": "更新预算执行",
         "keys": ["collection.updates_executed", "collection.updates_dropped"],
         "subtitle": "每轮实际执行的 critic tick 数 vs 被 max_grad_steps_per_round 丢掉的请求量。",
         "guide": "utd_ratio × added transitions = 请求梯度步数；超过 max_grad_steps_per_round 的部分记 dropped。\n"
                  "值得注意：dropped 长期>0 说明 UTD 预算吃满——提 episodes_per_update 只会浪费采集。\n"
                  "不能说明：dropped=0 也不代表数据被充分复用——看 replay 人口学与 batch 归因。"},
     ],
     "expanded": [
        {"title": "Critic per-channel", "charts": [
            {"chm": "critic loss", "metrics": ["critic.q1_loss", "critic.q2_loss"],
             "guide": "每通道 twin-Q 各自 MSE——颜色=通道、虚线=q2。两通道应同量级，长期分离提示某 critic 失配。"},
            {"ch": "critic.valid_weight",
             "guide": "该通道在本 batch 上的有效权重均值（课程 mask/terminated 后）。0=该通道此轮没参与。"},
            {"ch": "critic.clip_applied",
             "guide": "该通道 critic 梯度触发 clip 的 tick 占比。持续=1 说明阈值绑死。"},
        ]},
        {"title": "优化动力学", "charts": [
            {"keys": ["tck.actor.grad_norm"], "title": "Actor Grad Norm",
             "guide": "actor 目标总梯度 L2 范数（clip 前）。持续≈0 警惕梯度死亡；尖刺多为 batch 异常——进 grad-attr 归因。"},
            {"ch": "critic.grad_norm",
             "guide": "每通道 critic 梯度范数（clip 前）。配合 clip_applied 看是否常年顶着阈值。"},
            {"keys": ["tck.actor.valid_count", "tck.actor.pair1_frac",
                      "tck.target.pair1_frac"], "title": "Expectation 采样",
             "guide": "valid_count=batch 有效行；pair1_frac=actor/target 期望对被采到 pair-1 的比例（expectation_samples>1 时有意义）。"},
        ]},
     ]},

    # 3 ────────────────────────────────────────────────────────────
    {"id": "signal",
     "title": "训练信号",
     "question": "critic 的 Q/TD 是否健康，通道加权是否在漂移？",
     "curated": [
        {"title": "Q 均值与 TD 均值",
         "keys": ["tck.critic.q_mean", "tck.critic.td_mean"],
         "subtitle": "每轮 batch 上 critic 输出均值与时序差分误差均值。",
         "guide": "q_mean 长期单调上升可能是值膨胀（Q divergence 前兆）；td_mean 偏离 0 表示系统性高估/低估。\n"
                  "下一步：进 dump ②看 target 分解（reward·scale vs softQ vs reg）；③看 td 分布尾部。\n"
                  "不能说明：q_mean 高≠策略好——reward_scale 放大所有值；要结合 success 看。"},
        {"chm": "actor_weight", "metrics": ["actor.weight", "actor.weight_next"],
         "title": "通道 Actor Weight",
         "subtitle": "本轮 actor 目标里各通道的权重（weight）与下一轮生效值（weight_next）。",
         "guide": "颜色=通道；实线=本批生效、虚线=下批生效。课程调度（param patch）会造成阶跃。\n"
                  "值得注意：某通道权重长期≈0；weight/weight_next 持续错位（调度抖动）。"},
     ],
     "expanded": [
        {"title": "Batch 构成", "charts": [
            {"keys": ["tck.batch.size"], "title": "Batch Size",
             "guide": "每次更新的 batch 行数——恒定配置值，变化说明 param patch 改了 batch_size。"},
        ]},
     ]},

    # 4 ────────────────────────────────────────────────────────────
    {"id": "explore",
     "title": "探索状态",
     "question": "熵/不确定性正则是否在起作用，探索是否坍缩？",
     "curated": [
        {"title": "温度与正则项",
         "keys": ["tck.temperature.alpha", "tck.actor.log_prob_mean"],
         "right": ["tck.actor.reg_term"],
         "subtitle": "shannon：α 与 logp 均值（右轴 reg 项）；U 模式：α 恒为 0、reg_term 为 λ·U 惩罚。",
         "guide": "shannon：α 由 temperature loss 自适应，logp↑=分布变窄（探索收缩）。\n"
                  "u_floor：auto_alpha 关闭，regularizer.shannon=0，actor.reg_term=λ·relu(f−U)²。\n"
                  "值得注意：α 塌到 log_alpha_min；logp 持续上升且任务停滞；U 模式下 reg_term 长期=0（floor 未激活）或恒最大（floor 失效）。\n"
                  "不能说明：α 高≠探索好——要结合行为指标；reg_term 是均值，掩盖分布。"},
     ],
     "expanded": [
        {"title": "温度动力学", "charts": [
            {"keys": ["tck.temperature.loss", "tck.temperature.grad_norm",
                      "tck.regularizer.shannon"], "title": "Temperature & Regularizer Bits",
             "guide": "temperature.loss=α·(logp−target)；regularizer.shannon 是 shannon 项是否计入的开关位（U 模式恒 0）。"},
            {"keys": ["behavior.explore_factor", "behavior.random_start"],
             "title": "Behavior 生效值",
             "guide": "采集侧行为参数：explore_factor=策略蓝图探索系数；random_start=随机起点开关——两者由 param patch 调度。"},
        ]},
     ]},

    # 5 ────────────────────────────────────────────────────────────
    {"id": "sampling",
     "title": "采集与缓存",
     "question": "本轮采了什么数据，replay 是什么形态？",
     "curated": [
        {"title": "采集量",
         "keys": ["collection.episodes", "collection.env_steps"],
         "right": ["collection.agent_transitions"],
         "subtitle": "每轮 episode 数、env step 与注入的 agent transition（右轴）。",
         "guide": "self-play 下 agent_transitions ≈ 2×env_steps（双 agent 都产数据）。\n"
                  "值得注意：episodes 不变但 env_steps 掉=提前终止增多；transitions/env_step 比变化=agent 参与率变了。"},
        {"title": "Replay 规模与注入",
         "keys": ["replay.size", "replay.transitions_added"],
         "subtitle": "replay 占用 vs 每轮新增——size 见顶即开始 FIFO 淘汰。",
         "guide": "size 顶到 capacity 后曲线变平；transitions_added 恒定属正常。\n"
                  "下一步：看 replay 人口学（run 页 replay 卡 / debugkit replay）确认复用次数与来源分布。"},
     ],
     "expanded": [
        {"title": "采集耗时", "charts": [
            {"keys": ["collection.wall_time_s"], "title": "Collection Wall Time",
             "guide": "本轮并行采集墙钟（含 worker 冷启动）。尖刺多为资源竞争或池重建——对照 timing 分区。"},
        ]},
     ]},

    # 6 ────────────────────────────────────────────────────────────
    {"id": "cost",
     "title": "运行开销",
     "question": "时间花在哪里，是否出现运行瓶颈？",
     "curated": [
        {"title": "每轮耗时分解",
         "keys": ["timing.collection_s", "timing.slice_s",
                  "tck.timing.update_s"],
         "right": ["timing.eval_s"],
         "subtitle": "采集 / 切片 / 训练（该轮全部 tick 之和）/ 评估（右轴，稀疏）。",
         "guide": "tck.timing.update_s 是轮内所有 tick 的耗时*合计*（不是均值）——train 段成本。\n"
                  "值得注意：collection 尖刺（资源竞争）、train 阶跃（UTD cap 或 batch 变化）、eval 变贵。\n"
                  "不能说明：更快≠更高效——可能只是预算/数据量变了。"},
     ],
     "expanded": [
        {"title": "工件与诊断耗时", "charts": [
            {"keys": ["export.bytes", "checkpoint.bytes"], "title": "Artifact Bytes",
             "guide": "每轮 policy 导出与 checkpoint 大小。"},
            {"keys": ["debug.capture_s", "debug.bytes"], "title": "Debug Capture",
             "guide": "dump 抓取耗时与体积——只在触发的 tick 上有值（稀疏）。"},
        ]},
     ]},
]

# Open-ended zones — prefixes whose keys are defined at runtime.
ZONES: List[Dict[str, Any]] = [
    {"prefix": "eval.",
     "zone": "eval",
     "title": "Eval metrics",
     "color": "#5bb8d4",
     "sparse": True,
     "hint": "评估指标——只在 eval 事件上有值（held 段种子）；曲线为稀疏点直连。"},
    {"prefix": "task.",
     "zone": "experiment",
     "title": "Experiment metrics",
     "color": "#7fb069",
     "sparse": False,
     "hint": "实验自定义指标——experiment.post_round_metrics() 的返回值，每 round 一条。"},
]

# Per-flat-key hints for the fallback bucket and `metric_doc`.
TCK_HINTS: Dict[str, str] = {
    "critic.loss": "该 round 全部 critic tick 的多通道 MSE(td) 合计均值。",
    "critic.q_mean": "该 round batch 上 critic 输出均值——值膨胀/塌缩的第一信号。",
    "critic.td_mean": "该 round batch 的时序差分误差均值——系统性偏离 0 = 估计偏差。",
    "actor.loss": "该 round actor 目标均值：−(Σw·Q + entropy + reg)。",
    "actor.grad_norm": "actor 目标总梯度范数（clip 前）。",
    "actor.log_prob_mean": "采样动作的 logp 均值——上升 = 策略分布变窄。",
    "actor.pair1_frac": "actor 期望对被采到 pair-1 的比例。",
    "actor.reg_term": "actor 目标中的正则项均值（shannon=熵项；U 模式=λ·U 惩罚）。",
    "actor.valid_count": "本 batch 参与 actor 目标的有效行数。",
    "temperature.alpha": "熵温度 α（shannon 自适应；U 模式恒为常数/0）。",
    "temperature.loss": "温度损失 α·(logp−target_entropy)。",
    "temperature.grad_norm": "α 的梯度范数。",
    "target.tau": "目标网络软更新系数（EMA）。",
    "target.pair1_frac": "target 计算里 expectation pair-1 的占比。",
    "batch.size": "每次更新的 batch 行数。",
    "replay.size": "tick 时刻的 replay 占用。",
    "regularizer.shannon": "shannon 熵项是否计入（U 模式恒 0）。",
    "timing.update_s": "该轮全部 critic tick 耗时合计（sum 聚合）。",
}
CH_METRIC_HINTS: Dict[str, str] = {
    "critic.q1_loss": "该通道 q1 网络的 MSE(td)。",
    "critic.q2_loss": "该通道 q2 网络的 MSE(td)。",
    "critic.grad_norm": "该通道 critic 梯度范数（clip 前）。",
    "critic.clip_applied": "该通道 critic 梯度 clip 触发占比。",
    "critic.updated": "该通道 critic 本轮被执行更新的 tick 占比。",
    "critic.valid_weight": "该通道在 batch 上的有效权重均值。",
    "actor.weight": "该通道进入 actor 目标的权重（本批生效）。",
    "actor.weight_next": "该通道下批生效的 actor 权重（调度量）。",
}
ROUND_HINTS: Dict[str, str] = {
    "collection.episodes": "本轮采集的 episode 数（配置 episodes_per_update）。",
    "collection.env_steps": "本轮 env step（episode 帧数合计）。",
    "collection.agent_transitions": "注入 replay 的 agent transition 数（self-play≈2×env_step）。",
    "collection.wall_time_s": "本轮并行采集墙钟。",
    "collection.updates_executed": "本轮实际执行的 critic tick 数。",
    "collection.updates_dropped": "被 max_grad_steps_per_round 丢掉的请求梯度步数。",
    "replay.transitions_added": "本轮注入 replay 的 transition 数。",
    "replay.size": "round 末 replay 占用。",
    "timing.collection_s": "采集段耗时。",
    "timing.slice_s": "episode→transition slice 构建耗时。",
    "timing.eval_s": "本轮 eval 耗时（稀疏）。",
    "behavior.explore_factor": "本轮生效的采集探索系数。",
    "behavior.random_start": "随机起点是否启用。",
    "export.bytes": "policy 导出字节数。",
    "checkpoint.bytes": "checkpoint 字节数。",
    "debug.capture_s": "debug dump 抓取耗时（仅触发的 tick）。",
    "debug.bytes": "debug dump 体积。",
}

KEY_LABELS: Dict[str, str] = {
    "critic.loss": "critic_loss (Σ MSE)",
    "actor.loss": "actor_loss (−J)",
    "critic.q_mean": "q_mean",
    "critic.td_mean": "td_mean",
    "temperature.alpha": "alpha (α)",
    "actor.log_prob_mean": "logp_mean",
    "actor.reg_term": "reg_term",
    "regularizer.shannon": "shannon_bit",
    "timing.update_s": "train_s (Σ ticks)",
    "collection.updates_executed": "updates",
    "collection.updates_dropped": "dropped",
}


def _zone_for(key: str) -> Dict[str, Any]:
    for z in ZONES:
        if key.startswith(z["prefix"]):
            return z
    return {"zone": "framework", "prefix": "", "title": "Framework metrics",
            "sparse": False, "color": "#4fa3ff", "hint": ""}


def _channel_key(key: str) -> Tuple[str, Optional[str]]:
    """Split ``<prefix>.<metric>.<channel>`` → (prefix.metric, channel)."""
    parts = key.split(".")
    if len(parts) >= 3 and parts[-2] not in ("loss",):
        return ".".join(parts[:-1]), parts[-1]
    return key, None


def metric_doc(key: str) -> Dict[str, Any]:
    """Resolve one flat dashboard key to its semantic doc.

    Returns ``{"key", "zone", "hint", "label"}`` — SAC analog of PPO's
    ``metric_doc``: framework keys resolve via ROUND_HINTS/TCK_HINTS/
    CH_METRIC_HINTS, zone keys via their namespace hint.
    """
    z = _zone_for(key)
    hint = ""
    if z["zone"] == "framework":
        if key.startswith("tck."):
            bare = key[4:]
            hint = TCK_HINTS.get(bare) or CH_METRIC_HINTS.get(
                ".".join(bare.split(".")[:-1]), ""
            )
        else:
            hint = ROUND_HINTS.get(key, "")
            if not hint:
                stem, _ch = _channel_key(key)
                hint = CH_METRIC_HINTS.get(
                    stem.split(".", 1)[-1] if stem.startswith("tck.")
                    else stem, "")
    else:
        hint = z.get("hint", "")
    bare = key[4:] if key.startswith("tck.") else key.split(".", 1)[-1]
    label = KEY_LABELS.get(bare, bare)
    return {"key": key, "zone": z["zone"],
            "hint": hint or z.get("hint", ""), "label": label}


def catalog() -> Dict[str, Any]:
    """JSON-safe catalog served at ``/api/catalog`` and consumed by the
    viewer — same shape as the PPO dumpkit catalog."""
    layout: List[Dict[str, Any]] = []
    for sec in SECTIONS:
        layout.extend(sec.get("curated") or [])
        for grp in sec.get("expanded") or []:
            if grp.get("zone_rest"):
                layout.append({"zone_rest": grp["zone_rest"]})
            layout.extend(grp.get("charts") or [])
    return {
        "sections": SECTIONS,
        "layout": layout,
        "zones": ZONES,
        "tck_hints": TCK_HINTS,
        "ch_hints": CH_METRIC_HINTS,
        "round_hints": ROUND_HINTS,
        "labels": KEY_LABELS,
        "suppress": sorted(SUPPRESS_KEYS),
        "suppress_prefixes": list(SUPPRESS_PREFIXES),
    }


__all__ = [
    "DEFAULT_METRIC_SPECS",
    "EVENT_TYPES",
    "MetricCatalog",
    "MetricSpec",
    "metric_specs",
    "catalog",
    "metric_doc",
    "TCK_SUM_KEYS",
    "SECTIONS",
    "ZONES",
]
