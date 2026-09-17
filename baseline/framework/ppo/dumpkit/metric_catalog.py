"""Run-metrics catalog — single source of truth for metric semantics.

Everything the Debug viewer knows about *what a metric means* lives
here as data: the framework chart layout (grouping, ordering, hints),
per-channel / leftover-stat hint tables, and the open-ended zone
definitions (experiment / eval / policy namespaces).

Consumers:
- ``viewer/server.py`` serves ``catalog()`` at ``/api/catalog``.
- ``viewer/index.html`` fetches it and renders — the JS holds no hint
  text of its own.
- Agents (and future ``debug.py`` CLI commands) can import this module
  directly — it has no heavy dependencies — to resolve metric keys to
  meanings with ``metric_doc()``.

Key namespaces emitted by ``RunData._flatten_update``:

- ``stats.*``   framework-owned PPO scalars (UpdateStats.to_log_dict)
- ``ep.*``      rollout episode_stats
- ``pc.<m>.<ch>`` per-channel buffer stats
- ``time.*``    per-update phase timings
- ``exp.*``     experiment.on_update() return value (ZONES)
- ``eval.*``    experiment.on_eval() info dict — sparse (ZONES)
- ``policy.*``  policy_stats contributed by the policy (ZONES)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------
# Framework chart layout — fixed editorial order.  Spec shapes:
#   {"title": str, "keys": [flat_key, ...], "hint": str}
#   {"pc": "<metric>"}                        per-channel chart
#   {"pcm": "<name>", "metrics": [...], "hint": str}  merged per-channel
#   {"timing": true, "hint": str}             time.* catch-all chart
# ---------------------------------------------------------------------

FRAMEWORK_LAYOUT: List[Dict[str, Any]] = [
    {"title": "Episodes, Trajectories & Active Trajectories",
     "keys": ["ep.n_episodes", "stats.n_trajectories"],
     "pc": "n_active_trajs",
     "hint": "ep.n_episodes：本 update 真实跑的环境 episode 数。\n"
             "stats.n_trajectories：进入 buffer 的轨迹段总数——一个 episode 可拆出多条（如每 agent 一条），PPO 的训练单位是轨迹。\n"
             "n_active_trajs.*：各 reward channel 的活跃轨迹数（该 channel 有非零 reward 的 trajectory 数量），与总数对比可看出哪些 channel 在本 update 被激活。"},
    {"title": "Episode Length & Trajectory Length",
     "keys": ["ep.ep_len_mean", "ep.ep_len_min", "ep.ep_len_max",
              "stats.traj_len_mean", "stats.traj_len_min", "stats.traj_len_max"],
     "hint": "ep.ep_len_*：环境 episode 的帧数统计（episode_stats，真实对局长度）。\n"
             "stats.traj_len_*：buffer 轨迹段的帧数统计（Σ=total_steps）。\n"
             "一条轨迹恰好覆盖一条 episode 时两者相等。"},
    {"keys": ["stats.n_batches"],
     "hint": "每个 update 的 minibatch 数 = tensor_split(总帧数, n_batches)。"},
    {"keys": ["stats.total_steps"],
     "hint": "本 update buffer 的总帧数 = Σ traj_len。"},
    {"pcm": "reward",
     "metrics": ["reward_min", "reward_max", "reward_std", "reward_mean"],
     "hint": "该 channel 逐帧原始 reward 在活跃轨迹上的统计。\n"
             "min=点线、max=长虚线（上下界），std=短虚线（离散度），mean=加粗实线（主体）。\n"
             "颜色=channel、线型=指标；两行图例可分别 toggle channel 或指标。"},
    {"pcm": "actor_weight",
     "metrics": ["actor_weight_min", "actor_weight_max", "actor_weight_mean"],
     "hint": "实验侧 build_trajectories 给每条轨迹该 channel 的 actor_weight——课程权重。\n"
             "它决定该 channel 的 advantage 进入 combined_adv 的相对权重：combined = Σ aw·confidence·norm_adv，aw 逐帧 L1 归一化。"},
    {"pcm": "ret", "metrics": ["ret_mean", "ret_std"],
     "hint": "该 channel 折扣回报 return 的 mean/std——critic 的拟合目标（GAE 的 γ 口径）。"},
    {"pcm": "adv", "metrics": ["adv_mean", "adv_std"],
     "hint": "该 channel 原始 GAE advantage 的 mean/std（仅活跃帧、未归一化）。\n"
             "真正送进 actor 的是另一份量：norm_adv = 本图经 z-score 后 × actor_weight × confidence。"},
    {"pcm": "ev & confidence", "metrics": ["ev", "confidence"],
     "hint": "ev = 1 − Var(ret−V)/Var(ret)：critic 对该 channel return 的解释度——1=完美拟合，<0=不如直接猜均值。\n"
             "confidence = √clip(ev,0,1)：乘进 combined_adv，自动压低不可信 critic 的通道权重。"},
    {"title": "Uncertainty & Exploration Spec",
     "keys": ["stats.uncertainty_mean", "stats.uncertainty_floor", "stats.uncertainty_coef"],
     "hint": "合并图：buffer 全帧在 θ_old 下的 ActorEval.uncertainty 均值 U，以及本 update 实际生效的 exploration spec（floor + coef）。\n"
             "U ∈ [0,1]、与 action 无关；floor 线可视作 U 的警戒下限。floor_loss_mean = coef·mean(relu(floor−U)²) —— U 低于 floor 时 hinge 开始激活。\n"
             "spec 由 experiment.exploration(update) 逐 update 下发；恒定直线 = 未启用调度。"},
    {"title": "Learning Rate", "keys": ["stats.actor_lr", "stats.critic_lr"],
     "hint": "本 update 实际生效的学习率（从 optimizer param_groups 读取）。\n"
             "由 experiment.lr_schedule(update) 逐 update 下发 LRSpec 绝对值；None = 保持现状。\n"
             "恒定直线 = 未启用调度；逐步下降 = 学习率衰减生效中。"},
    {"pc": "vloss_mean"},
    {"title": "Policy & Floor Loss",
     "keys": ["stats.policy_loss_mean", "stats.floor_loss_mean"],
     "hint": "policy_loss_mean：PPO-clip 替代目标 −mean(min(r·A, clip(r)·A))，minibatch 平均。\n"
             "为何初始≈0：epoch0 首 minibatch 所有 r=1 → loss = −mean(w·A)；仅当该批加权 A 均值=0 时严格为 0（单通道 z-score 归一化下近似成立）。\n"
             "为何会变正/上升：有利方向 r 越过 ε 边界后该项目标值进入平台、该项梯度归零（收益封顶 ε·|A|）；不利方向不封顶（A<0 且 r>1+ε 时贡献 r·A 无界）。少数恶化样本可抵消大量改善 → 均值偏正是结构性现象，正负不代表学没学。\n"
             "floor_loss_mean：uncertainty 单边二次 hinge coef·mean(relu(floor−U)²·fw)，未激活时为 0。\n"
             "两者相加（非本图）才是实际反传的总损失。"},
    {"title": "Loss → Actor ∇",
     "keys": ["stats.action_grad_pol_mean", "stats.action_grad_floor_mean"],
     "hint": "两种损失各自对 actor 全部参数的梯度 L2 范数（autograd.grad，未触达的参数计 0）。\n"
             "每 epoch 首个 minibatch 采样、按 update 平均。\n"
             "· action_grad_pol_mean：clipped surrogate 对动作网络的拉力\n"
             "· action_grad_floor_mean：floor hinge 的拉力——按设计应只落在探索参数上、幅值小；异常升高提示 floor 泄漏进了动作参数"},
    {"title": "Actor Gradient Norm", "keys": ["stats.grad_norm_actor_mean"],
     "hint": "每个 update 一个点：该 update 内所有 actor minibatch 梯度 L2 范数的均值，clip 前原始值。\n"
             "clip 阈值 1.0：范数超过时所有梯度等比缩放至范数 1（方向不变）。\n"
             "读法：衡量\"策略想迈多大步\"vs clip 允许的实际步长。\n"
             "· 持续远高于 1 → clip 在主导步长，有效步长被压缩\n"
             "· 孤立尖刺 → 多为某个 batch 的 advantage 异常，可对照 KL/Timeline\n"
             "· 持续趋近 0 → 警惕梯度死亡（advantage ~0 或 tanh 饱和）"},
    {"pc": "grad_norm_mean"},
    {"title": "Ratio",
     "keys": ["stats.ratio_mean", "stats.ratio_min", "stats.ratio_max"],
     "hint": "ratio = exp(new_lp − old_lp)，新旧策略对同一动作的分歧度（1=不变）。\n"
             "mean 是 update 内所有 actor minibatch ratio 均值的平均；max 是上尾（加压方向）；min 是下尾（压制方向——趋 0 = 某些已采样动作被新策略近乎清零，探索坍缩前兆）。\n"
             "within-update ratio 天然从 1 起步、随 minibatch 推进扩散，所以均值会被 early minibatch 拉低；末态 ratio 的精确分布请看\"Clip Fraction & Post-Update Ratio Bins\"图。"},
    {"title": "Clip Fraction & Post-Update Ratio Bins",
     "keys": ["stats.clip_frac_mean", "stats.clip_frac_hi_mean", "stats.clip_frac_lo_mean",
              "stats.rbin_pos_ltlo", "stats.rbin_pos_lo", "stats.rbin_pos_hi",
              "stats.rbin_pos_gthi", "stats.rbin_neg_ltlo", "stats.rbin_neg_lo",
              "stats.rbin_neg_hi", "stats.rbin_neg_gthi", "stats.rbin_zero"],
     "hint": "合并图：过程 clip 参与度 + update 结束后最终 ratio 分布。\n"
             "clip_frac_mean/hi/lo：update 内 actor minibatch 上 ratio 越界 [1−ε,1+ε] 的样本占比；hi = r>1+ε 加压尾、lo = r<1−ε 压制尾，两尾不相交故 clip_frac_mean = hi + lo。它是\"clip 平均参与度\"，不是\"末态 clip 强度\"；想看 within-update 形态用 dump timeline。\n"
             "rbin_*：update 结束后最终 actor 的 ratio 按 advantage 符号分 9 段（ltlo: r<1−ε｜lo: [1−ε,1)｜hi: [1,1+ε]，含 r=1｜gthi: r>1+ε），zero = A=0 样本占比，合计=1。\n"
             "理想形态：pos 样本集中在 hi/gthi（概率被抬高）、neg 样本集中在 lo/ltlo（被压低）；pos_ltlo 或 neg_gthi 占比高 = 大量样本被推向反方向。"},
    {"title": "Post-Update ΔClipLoss", "keys": ["stats.post_clip_dloss_mean"],
     "hint": "update 结束后用最终 actor 对全部样本重算 ratio，再算\"双向 clip\"surrogate 相对 r=1 基线的变化：−mean[w·A·(clip(r,1−ε,1+ε)−1)]。\n"
             "与 policy_loss_mean 的区别：不取 min、两尾都截断——每样本贡献限在 ±ε·w·|A|，不被极端 ratio 主导。\n"
             "读法：负 = 本 update 对该批固定 advantage 净顺应（越负越好）；≈0 或正 = 无一致方向（信号弱/相互冲突/已过时的 adv）。衡量的是对当前 adv 估计的顺应度，不直接等于真实回报提升。"},
    {"title": "KL", "keys": ["stats.post_kl_mean", "stats.post_kl_max",
                             "stats.post_kl_pos_mean", "stats.post_kl_neg_mean",
                             "stats.kl_mean", "stats.kl_max",
                             "stats.early_stop_kl_mean"],
     "hint": "post_kl_*：update 结束后用最终 actor 在全 buffer 上重算的 k3 KL（(r−1)−log r）——本次更新的真实位移，跨 update 可比，是\"推了多远\"的权威读数。\n"
             "mean：全 buffer 平均位移（信任域距离）；max：单样本最大位移（位移是否集中于少数样本）；pos_mean/neg_mean：A>0 / A<0 样本上的平均位移——理想是 pos 侧位移占优。\n"
             "kl_mean：过程量——update 内所有 actor minibatch 的 k3 均值，每个 minibatch 在当时迭代点上测量。读作\"update 过程中 actor 平均工作的位移水平\"，不是端点位移（端点看 post_kl_mean）。它恒受 ramp 结构影响（首 minibatch ≈0 后单调爬升）且分母随早停截断变化，绝对值系统性低于 post_kl_mean 属正常。\n"
             "kl_max：过程量中单个 minibatch 的 k3 峰值，噪声大、受个别异常样本主导，主要作为异常尖刺报警（突然冲高 = 某个 minibatch 有 outlier 优势样本被大幅加压）。\n"
             "early_stop_kl_mean：触发 KL 早停当刻的\"本 epoch running mean KL\"（0=未触发）。与 kl_mean 同数量级，放在 KL 图便于直接比较\"日常过程位移\"和\"触发早停的位移阈值\"——若 kl_mean 持续逼近 early_stop_kl_mean 说明更新正贴着 target_kl 走。\n"
             "潜在用途：与 post_kl_mean 对比揭示位移时序——post_kl_mean ≫ 2×kl_mean = 位移集中在末段爆发；post_kl_mean < kl_mean = 中途位移被后续 minibatch 回退（churn）。\n"
             "逐 minibatch/逐 epoch 的 k3 序列见 dump Timeline 与 epoch_kl_stats。"},
    {"title": "Epochs & Early Stop",
     "keys": ["stats.epochs_done", "stats.actor_epochs_done"],
     "right": ["stats.early_stop_kl_mean"],
     "hint": "epochs_done：完成的 epoch 数（恒 = update_epochs——actor 被 KL 早停后 critic 仍继续跑完全部 epoch）。\n"
             "actor_epochs_done：actor 至少跑过一个 minibatch 的 epoch 数——actor 早停则它 < epochs_done。\n"
             "early_stop_kl_mean（右轴）：触发早停当刻的\"本 epoch running mean KL\"（0=未触发）。它是滑动均值不是单点峰值，所以与 kl_max 不相等是正常的；现在使用独立的右轴，避免被 epoch 计数量级压扁。"},
    {"timing": True,
     "hint": "每 update 各阶段耗时（s）：jobs/rollout/buffer/ppo/eval/export + total。"},
]

# Per-channel metric hints — used by pc.* fallback charts; pcm-merged
# charts carry their own hint on the layout spec.
PC_HINTS: Dict[str, str] = {
    "n_active_trajs": "该 channel 活跃轨迹数。活跃 = 该 trajectory 的 channels 中含此 key；不活跃轨迹 reward=0、不参与该 channel 的 advantage。",
    "active_ratio": "n_active_trajs / 总轨迹数——该 channel 的活跃占比。",
    "vloss_mean": "该 channel critic 的 (V−ret)² 帧加权 MSE，minibatch 平均。critic 跑满全部 epoch（actor 早停不影响它）。",
    "grad_norm_mean": "该 channel critic 网络的梯度 L2 范数（clip 前、minibatch 平均）。\nactor 被 KL 早停后 critic 仍跑满全部 epoch，采样窗口比 actor 梯度大。",
    "reward_min": "该 channel 逐帧原始 reward 的最小值（活跃轨迹全部帧）。",
    "reward_max": "该 channel 逐帧原始 reward 的最大值（活跃轨迹全部帧）。",
    "reward_mean": "该 channel 逐帧原始 reward 的均值（活跃轨迹全部帧）。",
    "reward_std": "该 channel 逐帧原始 reward 的标准差（活跃轨迹全部帧）。",
    "actor_weight_min": "该 channel 轨迹 actor_weight 的最小值。",
    "actor_weight_max": "该 channel 轨迹 actor_weight 的最大值。",
    "actor_weight_mean": "该 channel 轨迹 actor_weight 的均值。",
    "ret_mean": "该 channel 折扣回报 return 的均值（活跃轨迹）。",
    "ret_std": "该 channel 折扣回报 return 的标准差（活跃轨迹）。",
    "adv_mean": "该 channel 原始 GAE advantage 的均值（仅活跃帧、未归一化）。",
    "adv_std": "该 channel 原始 GAE advantage 的标准差（仅活跃帧、未归一化）。",
    "ev": "explained variance = 1 − Var(ret−V)/Var(ret)：critic 拟合质量。",
    "confidence": "√clip(ev,0,1)——乘进 combined_adv 的通道置信权重。",
}

# Fallback hints for leftover scalar stats.* keys not covered by the
# layout. Empty for now — all framework scalars are either in the layout
# or in the open-ended zones.
STATS_HINTS: Dict[str, str] = {
}

# Conventional-name aliases for chart legends — several renamed metrics
# correspond to names the PPO literature uses, so legends display
# e.g. "kl_mean (approx_kl)".
KEY_LABELS: Dict[str, str] = {
    "kl_mean": "kl_mean (approx_kl)",
    "kl_max": "kl_max (max_kl)",
    "early_stop_kl_mean": "early_stop_kl_mean (early_stop_kl)",
    "clip_frac_mean": "clip_frac_mean (clip_frac)",
    "clip_frac_hi_mean": "clip_frac_hi_mean (clip_frac_hi)",
    "clip_frac_lo_mean": "clip_frac_lo_mean (clip_frac_lo)",
    "policy_loss_mean": "policy_loss_mean (policy_loss)",
    "floor_loss_mean": "floor_loss_mean (floor_loss)",
    "action_grad_pol_mean": "action_grad_pol_mean (action_grad_pol)",
    "action_grad_floor_mean": "action_grad_floor_mean (action_grad_floor)",
    "grad_norm_actor_mean": "grad_norm_actor_mean (grad_norm_actor)",
    "uncertainty_mean": "uncertainty_mean (uncertainty)",
    "post_clip_dloss_mean": "post_clip_dloss_mean (post_clip_dloss)",
    "post_kl_pos_mean": "post_kl_pos_mean (post_kl_pos)",
    "post_kl_neg_mean": "post_kl_neg_mean (post_kl_neg)",
}

# ---------------------------------------------------------------------
# Open-ended zones — metrics whose keys are defined by the experiment
# or policy at runtime, so the catalog can only document the namespace,
# not enumerate keys.  Rendered after framework charts, in this order.
# ---------------------------------------------------------------------

ZONES: List[Dict[str, Any]] = [
    {"prefix": "exp.",
     "zone": "experiment",
     "title": "Experiment metrics",
     "color": "#7fb069",
     "sparse": False,
     "hint": "实验自定义指标——experiment.on_update() 的返回值，每个 update 记录一次。\n"
             "语义由实验定义（如在线成功率的连续曲线），框架仅透传展示。"},
    {"prefix": "eval.",
     "zone": "eval",
     "title": "Eval metrics",
     "color": "#5bb8d4",
     "sparse": True,
     "hint": "评估指标——experiment.on_eval() 返回的 info，仅在 eval_interval 的 update 上有值。\n"
             "曲线为稀疏点直连（无评估的 update 跳过不画）。"},
    {"prefix": "policy.",
     "zone": "policy",
     "title": "Policy-defined metrics",
     "color": "#c98b3f",
     "sparse": False,
     "hint": "策略自定义统计量——policy 的 evaluate_actions(want_stats) 发出的 policy_stats。\n"
             "语义与计算完全由策略决定，框架仅透传展示。"},
]


def catalog() -> Dict[str, Any]:
    """JSON-safe catalog dict served at /api/catalog and consumed by
    index.html — and by agents / future debug.py CLI commands."""
    return {
        "layout": FRAMEWORK_LAYOUT,
        "pc_hints": PC_HINTS,
        "stats_hints": STATS_HINTS,
        "zones": ZONES,
        "labels": KEY_LABELS,
    }


def _zone_for(key: str) -> Dict[str, Any]:
    for z in ZONES:
        if key.startswith(z["prefix"]):
            return z
    return {"zone": "framework", "prefix": "", "title": "Framework metrics",
            "sparse": False, "color": "#4fa3ff", "hint": ""}


def metric_doc(key: str) -> Dict[str, Any]:
    """Resolve one flattened metric key to its semantic doc.

    Returns ``{"key", "zone", "hint", "label"}`` — ``hint`` may be empty
    when no specific doc exists (open-ended zones / undocumented stats
    keys); the zone-level hint still describes the namespace.  ``label``
    carries the conventional-name alias for renamed framework keys
    (e.g. ``kl_mean (approx_kl)``).
    """
    z = _zone_for(key)
    hint = ""
    if z["zone"] == "framework":
        if key.startswith("pc."):
            # pc.<metric>.<channel>
            parts = key.split(".", 2)
            if len(parts) == 3:
                hint = PC_HINTS.get(parts[1], "")
        elif key.startswith("stats."):
            hint = STATS_HINTS.get(key[6:], "")
            # Layout keys carry a group-level hint — surface it so a
            # documented layout key resolves to its chart's hint.
            if not hint:
                for spec in FRAMEWORK_LAYOUT:
                    if key in (spec.get("keys") or []):
                        hint = spec.get("hint", "")
                        break
    else:
        hint = z.get("hint", "")
    label = (
        KEY_LABELS.get(key[6:], key[6:])
        if key.startswith("stats.") else key.split(".", 1)[-1]
    )
    return {"key": key, "zone": z["zone"],
            "hint": hint or z.get("hint", ""), "label": label}
