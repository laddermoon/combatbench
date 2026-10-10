# SAC 实验框架使用指南

> 类型：指南

面向想用 SAC 框架训练自己实验的用户。读完本指南，你就能写出一个完整的 SAC 实验、跑起来、知道去哪看诊断。

相关文档：`PLAN.md`（路线图）、`DECISIONS.md`（设计决策记录）、`DEBUG_PLAYBOOK.md`（故障排查手册）、`ARTIFACTS.md`（工件版本登记）。

---

## 1. 这个框架是什么

一个**多 critic 的 off-policy SAC 训练框架**，与 PPO 框架完全独立（不共享算法内部，只共享环境基础设施）。核心能力：

- **多 reward channel**：每个 channel 独立 twin-Q critic、独立 gamma、独立 actor 门控/权重
- **TN actor 家族**：截断正态策略，8 种架构格（单分布 `s*` / 混合 `m*`，shared/state σ，bounded/unbounded）
- **FIFO 可溯源 replay**：每条样本带 source_key 全链路溯源（run→round→job→episode→frame）
- **env_step 时钟**：eval/checkpoint 按环境步数调度，与 UTD 无关
- **完整 checkpoint/resume**：模型+优化器+replay+全部 RNG+实验状态，分段恢复与连续训练逐位一致
- **按需 debug dump**：训练中写哨兵文件即可在下一个 critic tick 捕获完整截面

---

## 2. 数据流

```
每个 collection round:

1. experiment.build_jobs(policy_bp, seed, n_episodes, collection_round=r)
     实验构建 rollout 任务（环境、对手、种子、deterministic=False）
2. rollouter.collect(jobs) → List[SACCollectedEpisode]
     并行环境交互（当前仅支持 data source kind="self"）
3. experiment.build_slices(episodes) → List[SACTransitionSlice]
     实验把 episode 转成 sac_transition_v2 切片——reward 语义、
     per-channel gate/weight、终止/截断标志、source_key 的唯一出处
4. replay.add_slices(slices)
     校验 + 去重（source_key 冲突 fail-loud）+ 写入 FIFO buffer
5. UTD 训练段：n_steps = utd_ratio × n_new_transitions（封顶
   max_grad_steps_per_round，尾差计入 utd_credit 滚动）
   每步：replay.sample(batch_size) → sac_update_v2
     critic: per-channel twin-Q TD（target min over in_target_min 个 head）
     actor: 加权 Q 最大 + regularizer（shannon α·logp 或 u_bonus/u_floor）
     alpha: auto_alpha 时朝 target_entropy 调节（夹于 log_alpha_min/max）
6. do_eval（env_step 跨过 eval_interval）:
     导出确定性 blueprint → eval jobs → experiment.on_eval()
     → is_new_best 时刷新 runs/<run>/policy/，可触发视频录制
7. checkpoint（每个 eval round + round 1）:
     checkpoint_s<env_step:08d> 目录 bundle（manifest+sha256 校验）
     checkpoint_keep_last（默认 3）剪枝旧 bundle，.pinned 豁免
```

关键语义：SAC 是 off-policy——replay 样本可被复用多个 update（replay 人口学见 debug 工具）；`bootstrap` 标志区分真终止（0）与截断（1），由 build_slices 保证正确性。

---

## 3. 写一个实验

继承 `ExperimentSAC`（`framework/sac/experiment.py`），放在 `experiments_sac/exp_sac_<name>.py`，定义 `EXPERIMENT_CLASS` 即自动注册。

必须实现：

| 方法 | 职责 |
|---|---|
| `common_params()` | `CommonParamsSAC`：lr、clip、预算 max_env_steps、eval 节奏、workers、seed |
| `sac_params()` | `SACParams`：replay/batch/UTD/τ/α 控制/正则模式（构造期强校验） |
| `reward_channels()` | `SACRewardChannel` 元组：name/gamma/n_critics/in_target_min |
| `build_actor(device)` | 构造 actor（通常 `TNActor(..., arch=...)`） |
| `build_q_critic(name, device)` | 单 channel Q critic（框架包成 MultiHeadQCritic） |
| `data_sources()` | 数据源声明（当前仅 `kind="self"` 被支持） |
| `build_jobs(...)` | rollout/eval 任务构建 |
| `build_slices(episodes)` | episode → `sac_transition_v2` 切片（reward/gate/终止语义唯一出处） |
| `on_eval(episodes, env_step)` | 返回 `{is_new_best, info, stop_training?}` |

可选覆写：`replay_plan()`（采样/保留策略）、`pre_action_fact_specs()`（action 前捕获的 task facts，如 `phi_pre`）、`post_round_metrics(episodes)`（`task.*` 指标）、`state()/load_state()`（实验自状态——**resume 等价性要求把恢复所需的游标/调度全部放进来**，例如随机化进度、方向余弦调度）。

参考实现：`experiments_sac/exp_sac_standup.py`（单通道）、`exp_sac_balance.py`（双通道双 agent）。

---

## 4. 启动训练

```bash
# 必须在 things/combatbench/ 下且 PYTHONPATH 指向它，否则 --background 静默失败
cd /data1/mono/things/combatbench

# 列出实验
PYTHONPATH=. python3 baseline/framework/train.py --list-experiments

# 冒烟（短跑 sanity check）
PYTHONPATH=. python3 -B baseline/framework/train.py --experiment sac_balance --algo sac --smoke

# 正式后台训练（--background 自带 fork+setsid，不要再套 nohup）
PYTHONPATH=/data1/mono/things/combatbench CUDA_VISIBLE_DEVICES=0 \
  python3 -B baseline/framework/train.py \
  --experiment sac_balance --algo sac --background

# 自定义 run 名 / 恢复
--run-name my_run_v2
--resume-from baseline/runs/<run>/checkpoints/checkpoint_s00100000
--config-lock   # 严格模式：除白名单外任何配置差异都拒绝 resume
```

每次 run 写 `config.json`（含全部参数与实验 knobs）、`code_snapshot.json`（git 快照分支）、`REPRODUCE.md`。**训练跑的是代码快照**，主线后续改动不影响在训 run。

---

## 5. Resume 语义

| 方式 | 效果 |
|---|---|
| `--resume-from <ckpt_dir>` | **full resume**：模型+优化器+log_alpha+replay（含采样 rng）+clocks+utd_credit+n_evals_done+experiment state+全部 RNG 流。已证明与连续训练逐位一致（test_segmented_resume_matches_continuous_training） |
| `--resume-from <ckpt_dir> --reset-update` | **warm start**：只载模型权重，其余清零重来 |
| `--resume-from <file.pt>` | model-only 热启动 |

config fingerprint 默认校验：除 `RESUME_ALLOWED_OVERRIDES`（lr/critic_lr/utd_ratio/max_grad_steps_per_round/alpha_lr/saved_at/config_schema）外任何配置差异都会 fail-loud 拒绝 resume；`--config-lock` 连白名单也不放行。

---

## 6. 关键配置旋钮

`SACParams`（全部构造期校验，非法值立即报错）：

| 旋钮 | 默认 | 说明 |
|---|---|---|
| `replay_buffer_size` | 500K | FIFO 容量 |
| `batch_size` / `warmup_steps` | 256 / 10K | 采样批量 / 首次更新前最小数据量 |
| `utd_ratio` | 1.0 | 每条新样本的梯度步数（尾差 utd_credit 滚动） |
| `max_grad_steps_per_round` | 10K | 单 round 梯度步封顶（超出的计入 dropped） |
| `tau` | 0.005 | target net 软更新（0 = 冻结 target） |
| `init_alpha` / `auto_alpha` / `target_entropy` | 0.2 / True / -act_dim | 熵温度控制；target_entropy 必须落在 init_alpha 可表达的均衡区（P6 教训：过低会熵崩） |
| `log_alpha_min/max` | -10 / 2 | α 夹界；log(init_alpha) 必须在界内 |
| `q_hidden_dim` / `q_layer_norm` | 256 / False | critic 容量与 LayerNorm（高维任务防 Q 发散的关键开关） |
| `reward_scale` | 1.0 | 全局 reward 缩放（小 per-step reward 任务需要放大，但过大会加大 TD） |
| `expectation_samples` | 1 | actor/target 期望的枚举样本数 |
| `regularizer_mode` | "shannon" | `u_bonus`/`u_floor` 为不确定性正则路线（需 actor 提供 uncertainty；不与 auto-alpha 叠加） |
| `use_grad_norm` | False | per-channel actor 梯度归一化（多 channel 权重均衡） |

实验层旋钮（写进 `config.json` 的 `knobs`）：`actor_arch`（s00/s01/s10/s11/m00/m01/m10/m11）、`actor_hidden_dim`、`actor_n_components`、`behavior_explore`（探索放大）、`random_start_transitions`（开局随机动作填充）。

---

## 7. Debug 工具链

```bash
DK="python3 -m baseline.framework.sac.debugkit"

$DK runs baseline/runs                     # run 总览
$DK run  baseline/runs/<run>               # 单 run 状态+配置+最近 eval
$DK series <run_dir> --metric critic.q_mean        # 指标时间序列
$DK dump  <run_dir> [--hypothesis "..."] [--at-tick N]
      # 写 dump_request.json 哨兵 → 运行中的 loop 下一个 critic tick 捕获截面
$DK inspect <dump_dir>                     # dump 总览：consistency/update_stats/通道/batch 构成
$DK samples <dump_dir> --sort td_abs       # 样本表（td/q/logp 排序）
$DK trace <dump_dir> --sample-id N         # 单样本全链溯源+逐通道明细
$DK replay <run_dir|ckpt|replay.pt>        # replay 人口学：年龄/复用/来源/policy fingerprint
$DK catalog [--prefix critic]              # 可用指标键
$DK recompute <dump_dir>                   # 离线重算 critic 目标对拍
$DK serve baseline/runs --port 8766 --host 0.0.0.0
      # 只读 JSON API + HTML viewer（http://<host>:8766/）
```

浏览器 viewer：run 列表、指标曲线、dump inspect/样本/trace、eval 视频直接播放。API 端点见 `debugserver.py` 头注释。

---

## 8. 工件与磁盘

- 工件版本登记见 `ARTIFACTS.md`；checkpoint bundle 原子写（tmp dir + rename + sha256 manifest）。
- **磁盘**：checkpoint ~1GB/个（含 replay），默认 `checkpoint_keep_last=3` 自动剪枝；`.pinned` 文件可钉住重要 checkpoint（如留给 held-out eval 的）；debug dump 默认保留最近 8 个。
- 长期跑之前预估：`eval 次数 × ~1GB`（剪枝前），盯 `df -h`。

## 9. 出问题时

按 `DEBUG_PLAYBOOK.md` 的症状→命令→证据表走（假设→证据→单变量→复验）。已实战验证过的两类：

- **熵崩**：α 撞 `log_alpha_min` 地板 + log_prob 冲高 → target_entropy 均衡过低，调高之；
- **Q 发散/幻想固定点**：q_mean 远超真实回报尺度、corr(frame,Q)≈0 → 开 `q_layer_norm`。

`regularizer_mode="u_bonus"/"u_floor"` 与 `auto_alpha` 的组合会被 trainer 层 fail-loud 拦截（防 α 双叠加）。

## 10. 常见坑

- `PYTHONPATH` 必须指向 `things/combatbench/`，`--background` 下 import 失败静默；
- `--background` 自带 setsid，不要 `nohup ... &` 包一层；
- resume 的 `source_key` 去重意味着实验必须把切片游标放进 `state()`，否则恢复后 source_key 冲突；
- `max_env_steps` 不在 resume 白名单——分段训练要在同一预算下中断/续跑，不能靠 resume 时改预算；
- divergence guard 阈值（Q>1e4/loss>1e3）只管爆炸，不管慢漂移——慢漂移要靠 dump+series 诊断。
