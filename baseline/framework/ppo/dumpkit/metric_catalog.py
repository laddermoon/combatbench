"""Run-metrics catalog — single source of truth for metric semantics.

Everything the Debug viewer knows about *what a metric means* lives
here as data: the sectioned chart layout (six themed sections, each
with curated charts plus expandable detail groups), per-channel /
leftover-stat hint tables, and the open-ended zone definitions
(experiment / eval / policy namespaces).

Dashboard boundary (see ``../DESIGN_run_dashboard_zones.md``): curated
charts exist to surface problems and point at evidence; expanded groups
aid triage; root-causing belongs to Resume/Dump/Delta tooling.  Every
curated chart carries a ``subtitle`` (what question it answers) and a
``guide`` (what changes matter, what to do next, what it cannot show).

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

Section spec shapes::

    {"id": str, "title": str, "question": str,
     "curated": [chart_spec, ...],
     "expanded": [{"title": str, "charts": [chart_spec, ...]}, ...]}

Chart spec shapes (unchanged ones kept for expanded groups):
  {"title": str, "keys": [flat_key, ...], "right": [...],
   "subtitle": str, "guide": str}
  {"spike_keys": [...]}            render only non-zero values (markers)
  {"pc": "<metric>"}               per-channel chart
  {"pcm": "<name>", "metrics": [...]}  merged per-channel chart
  {"zone_all": "eval"|"exp"|"policy"}
      render ALL keys of the zone in a single chart — for namespaces
      where every emitted key is headline by definition (eval.*)
  {"zone_pick": "eval"|"exp"|"policy", "prefer": [key, ...], "max": int}
      pick up to ``max`` present keys of the zone, prefer-listed first
  {"zone_rest": "eval"|"exp"|"policy"}
      render all keys of the zone not already consumed by zone_pick
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------
# Keys that must never render as charts — they are metadata, not
# metrics, or belong to investigation tooling rather than the dashboard.
# ---------------------------------------------------------------------

SUPPRESS_KEYS = frozenset({
    "param_overrides",  # per-update dict — shown in Update Detail
})

# stats.* prefixes suppressed entirely: the gradsig (per-frame gradient
# vs aggregate) diagnostic moved out of the dashboard into on-demand
# dump tooling — its scalars still exist in old logs but must not
# auto-chart.
SUPPRESS_PREFIXES = ("stats.grad_sig_",)


# ---------------------------------------------------------------------
# Sectioned layout — fixed editorial order, six themed sections.
# ---------------------------------------------------------------------

SECTIONS: List[Dict[str, Any]] = [

    # 1 ────────────────────────────────────────────────────────────
    {"id": "task",
     "title": "任务表现",
     "question": "训练有没有让任务做得更好？",
     "curated": [
        {"zone_all": "eval",
         "title": "评估表现",
         "subtitle": "固定评估条件下策略的任务能力曲线——实验 on_eval 输出的全部指标。",
         "guide": "值得注意：持续停滞、突然退化、波动变大；两个指标一升一降。\n"
                  "下一步：看对应评估视频；检查配置/课程变化；对照策略更新与探索状态分区。多比较几个评估点，别盯单点。\n"
                  "不能说明：单次评估不能证明稳定提升；受评估样本、场景、随机性影响，也不说明变化的原因。"},
     ],
     "expanded": [
        {"title": "在线指标（实验 on_update 输出）", "zone_rest": "exp"},
     ]},

    # 2 ────────────────────────────────────────────────────────────
    {"id": "policy_update",
     "title": "策略更新",
     "question": "PPO 是否在正常推进——更新被截断了吗，漂移正常吗？",
     "curated": [
        {"title": "策略漂移与早停",
         "keys": ["stats.post_kl_mean", "stats.kl_mean"],
         "spike_keys": ["stats.early_stop_kl_mean"],
         "subtitle": "本轮策略相对采样策略改变了多少，以及是否触发 KL 早停。",
         "guide": "post_kl_mean：update 结束后用最终 actor 在全 buffer 上重算的平均 k3 KL——端点位移。\n"
                  "kl_mean：update 内所有 actor minibatch 的过程均值，系统性低于端点值属正常（首 minibatch≈0）。\n"
                  "early_stop 标记：只在触发早停的 update 上出现（值为触发当刻的窗口均值 KL）。\n"
                  "值得注意：KL 突然升高；早停开始频繁；KL 长期≈0；过程/末态关系改变。\n"
                  "下一步：先看 Actor 更新执行量；展开 ratio/梯度辅助图；局部突变则定位首次异常 update，进 Dump timeline。\n"
                  "不能说明：KL 更低≠更好、更高≠更坏；是采样估计不是参数位移，更不是任务收益。"},
        {"title": "Actor 更新执行量",
         "keys": ["stats.actor_epochs_done", "stats.epochs_done"],
         "right": ["stats.actor_steps"],
         "subtitle": "Actor 实际执行了多少更新；Critic 继续时它是否提前停。",
         "guide": "actor_epochs_done vs epochs_done：actor 早停则前者 < 后者（critic 不受影响跑满）。\n"
                  "actor_steps（右轴）：actor 实际跑的 minibatch 总数——早停 epoch 只算到触发处。\n"
                  "值得注意：步数突然下降；actor/总 epoch 开始分离；更新量长期偏低。\n"
                  "下一步：对照早停标记；检查 epochs/minibatch/buffer 配置变化，区分主动改配置与提前截断。\n"
                  "不能说明：跑满 epoch 不等于学习效率高；步数变化也可能只是数据量或 minibatch 配置变了。"},
     ],
     "expanded": [
        {"title": "概率变化", "charts": [
            {"title": "Ratio",
             "keys": ["stats.ratio_mean", "stats.ratio_min", "stats.ratio_max"],
             "guide": "ratio = exp(new_lp − old_lp)，新旧策略对同一动作的分歧度（1=不变）。\n"
                      "mean 是 update 内所有 actor minibatch ratio 均值的平均；max 是上尾（加压方向）；min 是下尾（压制方向——趋 0 = 某些已采样动作被新策略近乎清零，探索坍缩前兆）。\n"
                      "within-update ratio 天然从 1 起步、随 minibatch 推进扩散，所以均值会被 early minibatch 拉低；末态分布见 Clip Fraction & Ratio Bins。"},
            {"title": "Clip Fraction & Post-Update Ratio Bins",
             "keys": ["stats.clip_frac_mean", "stats.clip_frac_hi_mean",
                      "stats.clip_frac_lo_mean",
                      "stats.rbin_pos_ltlo", "stats.rbin_pos_lo",
                      "stats.rbin_pos_hi", "stats.rbin_pos_gthi",
                      "stats.rbin_neg_ltlo", "stats.rbin_neg_lo",
                      "stats.rbin_neg_hi", "stats.rbin_neg_gthi",
                      "stats.rbin_zero"],
             "guide": "clip_frac_mean/hi/lo：update 内 actor minibatch 上 ratio 越界 [1−ε,1+ε] 的样本占比；hi = r>1+ε 加压尾、lo = r<1−ε 压制尾，clip_frac_mean = hi + lo。过程参与度，不是末态强度。\n"
                      "rbin_*：update 结束后最终 actor 的 ratio 按 advantage 符号分 9 段（ltlo/lo/hi/gthi + zero），合计=1。\n"
                      "理想形态：pos 集中在 hi/gthi、neg 集中在 lo/ltlo；pos_ltlo 或 neg_gthi 高 = 大量样本被推向反方向。"},
        ]},
        {"title": "优化信号", "charts": [
            {"title": "Policy Loss",
             "keys": ["stats.policy_loss_mean"],
             "guide": "policy_loss_mean：PPO-clip 替代目标 −mean(min(r·A, clip(r)·A))，minibatch 平均。\n"
                      "会变正/上升是结构性现象：有利方向收益封顶 ε·|A|，不利方向不封顶（A<0 且 r>1+ε 时贡献 r·A 无界），少数恶化样本可抵消大量改善——正负不代表学没学。"},
            {"title": "Actor Gradient Norm",
             "keys": ["stats.grad_norm_actor_mean", "stats.grad_clip_frac"],
             "guide": "grad_norm_actor_mean：该 update 内所有 actor minibatch 梯度 L2 范数均值，clip 前原始值（阈值见 grad_clip_norm）。\n"
                      "持续远高于阈值 → clip 主导步长；孤立尖刺 → 多为某 batch 的 advantage 异常；持续趋 0 → 警惕梯度死亡。\n"
                      "grad_clip_frac：pre-clip 范数超过 grad_clip_norm 的 minibatch 占比——1=每步都被裁（cap 绑死，mb 间被压成同权）；0=clip 从未触发。"},
        ]},
        {"title": "更新末态", "charts": [
            {"title": "Post-Update KL 细分",
             "keys": ["stats.post_kl_max", "stats.post_kl_pos_mean",
                      "stats.post_kl_neg_mean", "stats.kl_max"],
             "guide": "post_kl_max：单样本最大位移——位移是否集中于少数样本。\n"
                      "post_kl_pos/neg_mean：A>0 / A<0 样本上的平均位移。\n"
                      "kl_max：过程量中单个 minibatch 的峰值，噪声大，用作尖刺报警。\n"
                      "post_kl_mean ≫ 2×kl_mean = 位移集中在末段；post_kl_mean < kl_mean = 中途位移被后续 minibatch 回退（churn）。"},
            {"title": "Post-Update ΔClipLoss",
             "keys": ["stats.post_clip_dloss_mean",
                      "stats.post_clip_dloss_gain",
                      "stats.post_clip_dloss_harm"],
             "guide": "update 结束后用最终 actor 对全部样本重算 ratio 的双向 clip surrogate 变化：−mean[w·A·(clip(r)−1)]。每样本贡献限 ±ε·w·|A|，不被极端 ratio 主导。\n"
                      "dloss_mean = gain + harm 严格成立：gain ≤ 0 = 顺 adv 方向的有利贡献；harm ≥ 0 = 逆向的不利贡献。\n"
                      "变差时看拆项：gain→0 没学够；harm 升 = 学歪了；同升 = 位移大但方向混杂。衡量对当前 adv 估计的顺应度，不直接等于真实回报提升。"},
        ]},
        {"title": "配置与调度", "charts": [
            {"title": "Learning Rate",
             "keys": ["stats.actor_lr", "stats.critic_lr"],
             "guide": "本 update 实际生效的学习率（从 optimizer param_groups 读取，含 lr_schedule 与 param_overrides 的结果）。\n"
                      "恒定直线 = 未启用调度；阶跃 = param override 或 schedule 切换。"},
            {"title": "Batches per Epoch",
             "keys": ["stats.n_batches"],
             "guide": "每个 epoch 的 minibatch 数 = total_frames / minibatch_size。变化说明 buffer 规模或 minibatch 配置变了。"},
        ]},
     ]},

    # 3 ────────────────────────────────────────────────────────────
    {"id": "signal",
     "title": "训练信号",
     "question": "Critic 产生的学习依据是否可靠，哪些通道实际在参与训练？",
     "curated": [
        {"pcm": "ev & confidence", "metrics": ["ev", "confidence"],
         "title": "Critic 解释度与通道置信度",
         "subtitle": "各 reward channel 的 critic 拟合质量与置信权重。",
         "guide": "ev = 1 − Var(ret−V)/Var(ret)：critic 对该 channel return 的解释度——1=完美，<0=不如猜均值。\n"
                  "confidence = √clip(ev,0,1)：乘进 combined_adv，自动压低不可信 critic 的通道权重。\n"
                  "值得注意：EV 突降或长期为负；confidence 长期≈0；某通道明显落后。\n"
                  "下一步：展开该通道的 reward/ret/adv/vloss/actor_weight，分辨目标分布变化、拟合问题还是权重机制失效。\n"
                  "不能说明：高 EV 不保证 adv 无异常尾部或 value 无偏差；confidence 高≠该通道重要或贡献大。"},
     ],
     "expanded": [
        {"title": "通道信号量", "charts": [
            {"pcm": "reward",
             "metrics": ["reward_min", "reward_max", "reward_std", "reward_mean"],
             "guide": "该 channel 逐帧原始 reward 在活跃轨迹上的统计。min=点线、max=长虚线、std=短虚线、mean=加粗实线；颜色=channel。"},
            {"pcm": "ret",
             "metrics": ["ret_min", "ret_max", "ret_std", "ret_mean"],
             "guide": "该 channel 折扣回报 return（critic 拟合目标，GAE γ 口径）。min≈0（末帧 return≈末帧 reward）；max 反映最佳轨迹质量。"},
            {"pcm": "adv",
             "metrics": ["adv_min", "adv_max", "adv_std", "adv_mean"],
             "guide": "该 channel 原始 GAE advantage 统计（活跃帧、未归一化）。min/max 反映分布上下尾——极端值提示异常帧幅值。\n"
                      "注意：真正进 actor 的是 norm_adv = 本图经 adv_norm 变换后 × actor_weight × confidence。"},
            {"pcm": "actor_weight",
             "metrics": ["actor_weight_min", "actor_weight_max",
                         "actor_weight_mean"],
             "guide": "build_trajectories 给的课程权重，决定该 channel advantage 进入 combined_adv 的相对权重（逐帧 L1 归一化）。"},
        ]},
        {"title": "Critic 健康", "charts": [
            {"pc": "vloss_mean"},
            {"pc": "grad_norm_mean"},
            {"pc": "grad_clip_frac"},
        ]},
     ]},

    # 4 ────────────────────────────────────────────────────────────
    {"id": "explore",
     "title": "探索状态",
     "question": "策略是否过早收缩，探索保护是否持续介入？",
     "curated": [
        {"title": "不确定性与保护下限",
         "keys": ["stats.uncertainty_mean", "stats.uncertainty_floor"],
         "subtitle": "策略不确定性均值 vs 配置的保护下限。",
         "guide": "uncertainty_mean：buffer 全帧在 θ_old 下的 ActorEval.uncertainty 均值 U ∈ [0,1]。\n"
                  "uncertainty_floor：本 update 生效的保护下限（exploration spec）。\n"
                  "值得注意：不确定性快速下降；长期低于下限；下限调整后不响应；任务停滞同时持续收缩。\n"
                  "下一步：展开 floor loss 与梯度分解；看视频判断行为是否僵化。\n"
                  "不能说明：均值不代表所有状态；高于 floor 不代表局部保护项未激活；高不确定性≠有用探索。"},
     ],
     "expanded": [
        {"title": "保护项与梯度", "charts": [
            {"title": "Floor Loss & Coef",
             "keys": ["stats.floor_loss_mean", "stats.uncertainty_coef"],
             "guide": "floor_loss_mean = coef·mean(relu(floor−U)²·fw)：单边二次 hinge，未激活为 0。\n"
                      "uncertainty_coef：本 update 生效的保护系数（exploration spec）。"},
            {"title": "Loss → Actor ∇",
             "keys": ["stats.action_grad_pol_mean",
                      "stats.action_grad_floor_mean"],
             "guide": "两种损失各自对 actor 参数的梯度 L2 范数（每 epoch 首 minibatch 采样、按 update 平均）。\n"
                      "action_grad_floor 按设计只应落在探索参数上、幅值小；异常升高提示 floor 泄漏进了动作参数。"},
        ]},
        {"title": "策略自定义指标", "zone_rest": "policy"},
     ]},

    # 5 ────────────────────────────────────────────────────────────
    {"id": "sampling",
     "title": "采样与轨迹",
     "question": "本轮用什么规模、什么形态的数据在训练？",
     "curated": [
        {"title": "Episode 与 Trajectory 数量",
         "keys": ["ep.n_episodes", "stats.n_trajectories"],
         "subtitle": "实际采集的 episode 数 vs 进入 buffer 的轨迹段数。",
         "guide": "n_episodes：本 update 真实跑的环境 episode 数；n_trajectories：buffer 轨迹段总数（一个 episode 可拆多条，如每 agent 一条）。\n"
                  "值得注意：数量意外下降；两者比例突变；与 episodes_per_update 配置不符。\n"
                  "下一步：检查 trajectory 切分、agent 参与和通道激活；必要时看 Dump 映射。\n"
                  "不能说明：数量多不等于信息量大——重复、强相关数据不因数量大变成高质量样本。"},
        {"title": "Episode 与 Trajectory 长度",
         "keys": ["ep.ep_len_mean", "stats.traj_len_mean"],
         "subtitle": "环境交互长度与训练轨迹长度的均值趋势。",
         "guide": "ep_len_mean：真实对局长度均值；traj_len_mean：buffer 轨迹段长度均值——一一对应时两者相等。\n"
                  "值得注意：长度突降；两者开始明显分离；长期固定 timeout。\n"
                  "下一步：检查终止原因、时间上限和切分规则；结合任务表现及 episode 证据判断。\n"
                  "不能说明：长不一定好、短不一定坏——可能是更早失败或更快成功，须由任务语义解释。"},
     ],
     "expanded": [
        {"title": "规模与覆盖", "charts": [
            {"keys": ["stats.total_frames"],
             "title": "Total Frames",
             "guide": "本 update buffer 总帧数 = Σ traj_len。"},
            {"pc": "n_active_trajs"},
            {"pc": "active_ratio"},
        ]},
        {"title": "长度范围", "charts": [
            {"title": "Episode & Trajectory Length (range)",
             "keys": ["ep.ep_len_min", "ep.ep_len_max",
                      "stats.traj_len_min", "stats.traj_len_max"],
             "guide": "episode / trajectory 长度的 min/max 范围——发散说明数据形态不均一。"},
            {"pc": "traj_len_mean"},
        ]},
     ]},

    # 6 ────────────────────────────────────────────────────────────
    {"id": "cost",
     "title": "运行开销",
     "question": "时间花在哪里，是否出现运行瓶颈？",
     "curated": [
        {"title": "每轮耗时与阶段组成",
         "keys": ["time.total", "time.rollout", "time.ppo", "time.eval"],
         "subtitle": "每 update 总耗时及主要阶段分解。",
         "guide": "total：一轮总耗时；rollout/ppo/eval：三大阶段（buffer/jobs/export 等小项见展开）。\n"
                  "值得注意：总耗时阶跃增长；某阶段持续变慢；周期性尖峰。\n"
                  "下一步：对照数据规模、actor 步数、评估周期和诊断开关，再查资源竞争或具体阶段。\n"
                  "不能说明：每轮更快≠效率更高——可能只是数据变少或提前停止；不据此声称样本效率提升。"},
     ],
     "expanded": [
        {"title": "全部阶段耗时", "charts": [
            {"title": "Minor Phases",
             "keys": ["time.buffer", "time.jobs", "time.export"],
             "guide": "buffer 构建、jobs 构建、policy 导出等小阶段耗时。"},
        ]},
     ]},
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
    "ret_mean": "该 channel 折扣回报 return 的均值（活跃帧）。",
    "ret_std": "该 channel 折扣回报 return 的标准差（活跃帧）。",
    "ret_min": "该 channel 折扣回报 return 的最小值（活跃帧；逐帧极值，非逐轨迹）。",
    "ret_max": "该 channel 折扣回报 return 的最大值（活跃帧；逐帧极值，非逐轨迹）。",
    "adv_mean": "该 channel 原始 GAE advantage 的均值（仅活跃帧、未归一化）。",
    "adv_std": "该 channel 原始 GAE advantage 的标准差（仅活跃帧、未归一化）。",
    "adv_min": "该 channel 原始 GAE advantage 的最小值（仅活跃帧、未归一化；逐帧极值）。",
    "adv_max": "该 channel 原始 GAE advantage 的最大值（仅活跃帧、未归一化；逐帧极值）。",
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
    "post_clip_dloss_gain": "post_clip_dloss_gain (dloss_gain, <=0)",
    "post_clip_dloss_harm": "post_clip_dloss_harm (dloss_harm, >=0)",
    "post_kl_pos_mean": "post_kl_pos_mean (post_kl_pos)",
    "post_kl_neg_mean": "post_kl_neg_mean (post_kl_neg)",
    "grad_sig_g_norm": "grad_sig_g_norm (‖G‖)",
    "grad_sig_coherence": "grad_sig_coherence (‖G‖/mean‖g‖)",
    "grad_sig_dir_cos": "grad_sig_dir_cos (dir persist)",
    "grad_sig_proj_mean": "grad_sig_proj_mean (mean p ≈ ‖G‖)",
    "grad_sig_proj_std": "grad_sig_proj_std (std p)",
    "grad_sig_frac_neg_mean": "grad_sig_frac_neg_mean (P(p<0))",
    "grad_sig_norm_mean": "grad_sig_norm_mean (mean ‖g_i‖)",
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
    # "layout" is the flattened view of every chart spec across all
    # sections — kept for agents/tools that enumerate charts without
    # caring about the section structure.
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
        "pc_hints": PC_HINTS,
        "stats_hints": STATS_HINTS,
        "zones": ZONES,
        "labels": KEY_LABELS,
        "suppress": sorted(SUPPRESS_KEYS),
        "suppress_prefixes": list(SUPPRESS_PREFIXES),
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
            # Keys placed in a section chart carry the chart's guide —
            # surface it so a documented key resolves to its chart's doc.
            if not hint:
                for sec in SECTIONS:
                    specs = list(sec.get("curated") or [])
                    for grp in sec.get("expanded") or []:
                        specs.extend(grp.get("charts") or [])
                    for spec in specs:
                        if key in (spec.get("keys") or []):
                            hint = spec.get("guide") or spec.get("hint", "")
                            break
                    if hint:
                        break
    else:
        hint = z.get("hint", "")
    label = (
        KEY_LABELS.get(key[6:], key[6:])
        if key.startswith("stats.") else key.split(".", 1)[-1]
    )
    return {"key": key, "zone": z["zone"],
            "hint": hint or z.get("hint", ""), "label": label}
