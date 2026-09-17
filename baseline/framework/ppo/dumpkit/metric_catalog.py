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
    {"title": "Episodes & Trajectories",
     "keys": ["ep.n_episodes", "stats.n_trajectories"],
     "hint": "ep.n_episodes：本 update 真实跑的环境 episode 数。\n"
             "stats.n_trajectories：进入 buffer 的轨迹段数——一个 episode 可拆出多条（如每 agent 一条），PPO 的训练单位是轨迹。"},
    {"title": "Episode Length & Trajectory Length",
     "keys": ["ep.ep_len_mean", "ep.ep_len_min", "ep.ep_len_max",
              "stats.traj_len_mean", "stats.traj_len_min", "stats.traj_len_max"],
     "hint": "ep.ep_len_*：环境 episode 的帧数统计（episode_stats，真实对局长度）。\n"
             "stats.traj_len_*：buffer 轨迹段的帧数统计（Σ=total_steps）。\n"
             "一条轨迹恰好覆盖一条 episode 时两者相等。"},
    {"pc": "n_active_trajs"},
    {"pc": "active_ratio"},
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
    {"title": "Exploration Spec",
     "keys": ["stats.uncertainty_floor", "stats.uncertainty_coef"],
     "hint": "本 update 实际生效的 ExplorationSpec：uncertainty_floor（U 下限）与 uncertainty_coef（hinge 系数）。\n"
             "由 experiment.exploration(update) 逐 update 下发；恒定直线 = 未启用调度。\n"
             "floor_loss = coef·mean(relu(floor−U)²)，floor 线低于 Uncertainty 图的 U 曲线时 hinge 开始激活。"},
    {"title": "Learning Rate", "keys": ["stats.actor_lr", "stats.critic_lr"],
     "hint": "本 update 实际生效的学习率（从 optimizer param_groups 读取）。\n"
             "由 experiment.lr_schedule(update) 逐 update 下发 LRSpec 绝对值；None = 保持现状。\n"
             "恒定直线 = 未启用调度；逐步下降 = 学习率衰减生效中。"},
    {"title": "Uncertainty", "keys": ["stats.uncertainty"],
     "hint": "框架指标：buffer 全帧在 θ_old（训练前一次前向）下的 ActorEval.uncertainty 均值。\n"
             "U ∈ [0,1]、与 action 无关；被 floor 损失 relu(floor−U) 消费。σ 收缩或截断区间变化都会压低 U。"},
    {"pc": "vloss"},
    {"title": "Policy & Floor Loss", "keys": ["stats.policy_loss", "stats.floor_loss"],
     "hint": "policy_loss：PPO-clip 替代目标 −mean(min(r·A, clip(r)·A))，minibatch 平均。\n"
             "为何初始≈0：epoch0 首 minibatch 所有 r=1 → loss = −mean(w·A)；仅当该批加权 A 均值=0 时严格为 0（单通道 z-score 归一化下近似成立）。\n"
             "为何会变正/上升：有利方向 r 越过 ε 边界后该项目标值进入平台、该项梯度归零（收益封顶 ε·|A|）；不利方向不封顶（A<0 且 r>1+ε 时贡献 r·A 无界）。少数恶化样本可抵消大量改善 → 均值偏正是结构性现象，正负不代表学没学。\n"
             "floor_loss：uncertainty 单边二次 hinge coef·mean(relu(floor−U)²·fw)，未激活时为 0。\n"
             "两者相加（非本图）才是实际反传的总损失。"},
    {"title": "Loss → Actor ∇", "keys": ["stats.action_grad_pol", "stats.action_grad_floor"],
     "hint": "两种损失各自对 actor 全部参数的梯度 L2 范数（autograd.grad，未触达的参数计 0）。\n"
             "每 epoch 首个 minibatch 采样、按 update 平均。\n"
             "· action_grad_pol：clipped surrogate 对动作网络的拉力\n"
             "· action_grad_floor：floor hinge 的拉力——按设计应只落在探索参数上、幅值小；异常升高提示 floor 泄漏进了动作参数"},
    {"title": "Actor Gradient Norm", "keys": ["stats.grad_norm_actor"],
     "hint": "每个 update 一个点：该 update 内所有 actor minibatch 梯度 L2 范数的均值，clip 前原始值。\n"
             "clip 阈值 1.0：范数超过时所有梯度等比缩放至范数 1（方向不变）。\n"
             "读法：衡量\"策略想迈多大步\"vs clip 允许的实际步长。\n"
             "· 持续远高于 1 → clip 在主导步长，有效步长被压缩\n"
             "· 孤立尖刺 → 多为某个 batch 的 advantage 异常，可对照 KL/Timeline\n"
             "· 持续趋近 0 → 警惕梯度死亡（advantage ~0 或 tanh 饱和）"},
    {"pc": "grad_norm"},
    {"title": "Ratio & Clip Fraction",
     "keys": ["stats.ratio_mean", "stats.ratio_min", "stats.ratio_max",
              "stats.clip_frac", "stats.clip_frac_hi", "stats.clip_frac_lo"],
     "hint": "ratio = exp(new_lp − old_lp)，新旧策略对同一动作的分歧度（1=不变）。mean 是均值，max 是上尾（加压方向），min 是下尾（压制方向——趋 0 = 某些已采样动作被新策略近乎清零，探索坍缩前兆）。\n"
             "clip_frac = |ratio−1| > clip_eps 的帧占比；hi = r>1+ε 上尾、lo = r<1−ε 下尾，两尾不相交故 clip_frac = hi + lo。注意越界只在一半情况下真杀梯度：hi 配合 A>0、lo 配合 A<0 才被掐——所以 hi/lo 是\"越界占比\"而非\"被掐占比\"。"},
    {"title": "Post-Update ΔClipLoss", "keys": ["stats.post_clip_dloss"],
     "hint": "update 结束后用最终 actor 对全部样本重算 ratio，再算\"双向 clip\"surrogate 相对 r=1 基线的变化：−mean[w·A·(clip(r,1−ε,1+ε)−1)]。\n"
             "与 policy_loss 的区别：不取 min、两尾都截断——每样本贡献限在 ±ε·w·|A|，不被极端 ratio 主导。\n"
             "读法：负 = 本 update 对该批固定 advantage 净顺应（越负越好）；≈0 或正 = 无一致方向（信号弱/相互冲突/已过时的 adv）。衡量的是对当前 adv 估计的顺应度，不直接等于真实回报提升。"},
    {"title": "Post-Update Ratio Bins",
     "keys": ["stats.rbin_pos_ltlo", "stats.rbin_pos_lo", "stats.rbin_pos_hi",
              "stats.rbin_pos_gthi", "stats.rbin_neg_ltlo", "stats.rbin_neg_lo",
              "stats.rbin_neg_hi", "stats.rbin_neg_gthi", "stats.rbin_zero"],
     "hint": "update 结束后最终策略的 ratio 分布，按 advantage 符号分组：pos_*/neg_* 各 4 段（ltlo: r<1−ε｜lo: [1−ε,1)｜hi: [1,1+ε]，含 r=1｜gthi: r>1+ε），zero = A=0 样本占比。\n"
             "9 条线都是全部样本的占比，合计=1。\n"
             "理想形态：pos 样本集中在 hi/gthi（概率被抬高）、neg 样本集中在 lo/ltlo（被压低）。pos_ltlo 或 neg_gthi 占比高 = 大量样本被推向反方向。"},
    {"title": "KL", "keys": ["stats.post_kl_mean", "stats.post_kl_max",
                             "stats.post_kl_pos", "stats.post_kl_neg"],
     "hint": "update 结束后用最终 actor 在全 buffer 上重算的 k3 KL（(r−1)−log r）——本次更新的真实位移，跨 update 可比。\n"
             "mean：全 buffer 平均位移（信任域距离）；max：单样本最大位移（位移是否集中于少数样本）；pos/neg：A>0 / A<0 样本上的平均位移——理想是 pos 侧位移占优。\n"
             "读法：mean ≈ target_kl 且 actor 早停 = 健康撞墙；mean 明显低于 early_stop_kl（Epochs 图）= 早停后位移部分回退。\n"
             "注：过程指标 approx_kl/max_kl（minibatch 时刻值）仍在 Timeline 页可见。"},
    {"title": "Epochs & Early Stop",
     "keys": ["stats.epochs_done", "stats.actor_epochs_done", "stats.early_stop_kl"],
     "hint": "epochs_done：完成的 epoch 数（恒 = update_epochs——actor 被 KL 早停后 critic 仍继续跑完全部 epoch）。\n"
             "actor_epochs_done：actor 至少跑过一个 minibatch 的 epoch 数——actor 早停则它 < epochs_done。\n"
             "early_stop_kl：触发早停当刻的\"本 epoch running mean KL\"（0=未触发）。它是滑动均值不是单点峰值，所以与 max_kl 不相等是正常的；量级远小于 epoch 计数所以贴底。"},
    {"timing": True,
     "hint": "每 update 各阶段耗时（s）：jobs/rollout/buffer/ppo/eval/export + total。"},
]

# Per-channel metric hints — used by pc.* fallback charts; pcm-merged
# charts carry their own hint on the layout spec.
PC_HINTS: Dict[str, str] = {
    "n_active_trajs": "该 channel 活跃轨迹数。活跃 = 该 trajectory 的 channels 中含此 key；不活跃轨迹 reward=0、不参与该 channel 的 advantage。",
    "active_ratio": "n_active_trajs / 总轨迹数——该 channel 的活跃占比。",
    "vloss": "该 channel critic 的 (V−ret)² 帧加权 MSE，minibatch 平均。critic 跑满全部 epoch（actor 早停不影响它）。",
    "grad_norm": "该 channel critic 网络的梯度 L2 范数（clip 前、minibatch 平均）。\nactor 被 KL 早停后 critic 仍跑满全部 epoch，采样窗口比 actor 梯度大。",
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
# layout (e.g. value_loss).
STATS_HINTS: Dict[str, str] = {
    "value_loss": "所有 channel critic value loss 的均值标量（分 channel 明细见 vloss 图）。",
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
    }


def _zone_for(key: str) -> Dict[str, Any]:
    for z in ZONES:
        if key.startswith(z["prefix"]):
            return z
    return {"zone": "framework", "prefix": "", "title": "Framework metrics",
            "sparse": False, "color": "#4fa3ff", "hint": ""}


def metric_doc(key: str) -> Dict[str, Any]:
    """Resolve one flattened metric key to its semantic doc.

    Returns ``{"key", "zone", "hint"}`` — ``hint`` may be empty when no
    specific doc exists (open-ended zones / undocumented stats keys);
    the zone-level hint still describes the namespace.
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
    return {"key": key, "zone": z["zone"], "hint": hint or z.get("hint", "")}
