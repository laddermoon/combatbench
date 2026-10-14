# SAC 诊断 Playbook（阶段六 P6-PROTO-1）

> 类型：指南

用途：pathfinder/正式 run 出现异常时，按本表定位到具体链环。每条规则：症状 → 命令 → 看什么 → 结论写法。所有命令从 `things/combatbench/` 运行，`PYTHONPATH=.`。

```bash
DK() { PYTHONPATH=. python3 -B -m baseline.framework.sac.debugkit "$@"; }
```

## 0. 通用入口

| 目的 | 命令 |
|---|---|
| run 总览 | `DK runs baseline/runs_sac` / `DK run <run_dir>` |
| 指标序列 | `DK series <run_dir> --metric <key> [--event tick]` |
| 按需截面 | `DK dump <run_dir> --hypothesis "..." [--at-tick N]`（训练在线时生效） |
| dump 总览 | `DK inspect <run_dir>/debug_dumps/critic_tick_NNNNNNNN` |
| 样本表 | `DK samples <dump> --sort td_abs|q_mean|logp --limit N` |
| 单样本全链 | `DK trace <dump> --sample-id N` |
| 来源×贡献归因 | `DK attribute <dump> [--group-by collection_round|policy_fingerprint|explore_factor|random_start]` |
| 目标/actor 分解 | `DK decomp <dump> [--sample-id N]` |
| 分组梯度归因（重） | `DK grad-attr <dump> [--group-by ...]`（重建 pre-state + autograd，离线按需） |
| 前后策略漂移 | `DK drift <dump>`（batch obs 上 det-action/logπ/Q 位移） |
| slice 上下文 | `DK slice-ctx <run|ckpt|replay.pt> [--source-key K|--slice-id S|--sample-id N]` |
| replay 人口学 | `DK replay <run_dir>` |
| 精确复算 | `DK recompute <dump>`（全正则模式，应 `max_abs_diff≈0`） |
| JSON API | `DK serve baseline/runs_sac --port 8766` |

## 1. 不收敛 / 回报平台期

1. `DK series <run> --metric eval.survival_rate`（或 `eval.success_rate`）——先确认"真的不涨"而非噪声。
2. `DK series <run> --metric temperature.alpha`：α 撞 `alpha_min`/`alpha_max` 边界 → 熵调节失效，见 §4。
3. `DK series <run> --metric critic.td_mean` + `critic.q_mean`：TD 长期不降 → critic 跟不上（UTD 太低/目标非稳态）。
4. `DK replay <run>`：检查 `collection_round_counts` 是否单一 round 独占（旧数据堆积，见 §5）、`policy_fingerprint_counts` 是否多版本混杂异常。

## 2. Q 异常 / 发散

- `DK series <run> --metric critic.q_mean`：`|Q|` 单调膨胀 → Q 爆炸。divergence guard 会自动 checkpoint + 终止；拿终止前 checkpoint 所在 run 的最近 dump：`DK inspect <dump>` 看 `per_channel.*_absmax` 与 `td_abs_max`，`DK samples <dump> --sort td_abs` 找爆点样本，`DK trace --sample-id` 看该样本 `terminated/bootstrap` 与 reward 是否异常。

## 3. 终止语义错误嫌疑

- `DK inspect <dump>` → `consistency.violations`：`terminated_with_bootstrap`、`truncated_without_bootstrap` 非零即有错。正常 run 应为空。
- 交叉验证：`DK trace` 某 terminated 样本 → `metadata.termination_reason`、`truncated`、`bootstrap` 三字段应与任务语义一致（timeout 必须 bootstrap=1）。

## 4. 温度失控（alpha collapse / runaway）

- `DK series --metric temperature.alpha`：跌至 `log_alpha_min` 且 `actor.log_prob_mean` 持续上升 → 熵崩（探索死亡）。升至 `log_alpha_max` → 温度失控。
- `DK inspect <dump>` → `actor_side.integration_weights`/`log_probs` 分布退化（logp 极端负）佐证。

## 5. 旧数据过多 / 行为策略漂移

- `DK replay <run>`：`sample_age` quantiles、`policy_fingerprint_counts`、`explore_factor_counts`、`random_start_rows`。若最老 bucket 占比过高且 `policy_fingerprint` 陈旧 → off-policy 陈旧主导。
- `DK inspect <dump>` → `batch.collection_round_counts` 与 `analysis.replay_stats` 对照：batch 分布应接近 buffer 分布（uniform 采样）；偏差大说明 buffer 构成问题。
- **`DK attribute <dump> --group-by collection_round`**：来源×贡献联合分布——某老 round 的行数占比小但 `critic_loss_share` 占比大 → 陈旧样本主导了这次更新（高 TD 残留），坐实 off-policy 问题。
- **`DK grad-attr <dump> --group-by collection_round`**（重）：每组对 critic 梯度的范数与 `cos_to_total`——老 round 梯度与总梯度方向对冲（cos<0 或近 0）说明陈旧数据在反向拉扯。

## 6. 通道压制

- `DK inspect <dump>` → `update_stats.critic_updated_<ch>` 应为 1；`critic_valid_weight_<ch>` 过低 → 该通道 valid 稀疏。`actor_valid_count` 远小于 batch → 全有效行稀缺。
- `DK trace` 逐样本看 `actor_weight`/`channel_valid` 定位压制发生在 gate 还是 valid。

## 7. actor 无学习信号

- `DK inspect` → `param_delta.actor_state_dict.l2`：长期≈0 且 `actor.loss` 平 → 梯度死亡。
- mixture：`samples/trace` 看 `integration_weights` 是否退化到单一分量（`argmax` 概率≈1）且 `actor_pair_index` 单一——分量坍缩。
- `DK inspect` → `pair_index.actor_unique` 只剩一个 index 表示双塔长期同选一塔（不必然是错，但值得记录）。
- **`DK decomp <dump>`**：actor 目标三分解——`per_channel_q` 近 0 而 `entropy_term` 主导 → Q 信号死、纯熵驱动；`reg_term`（U 模式）占比过大 → 正则项淹没任务目标。
- **`DK grad-attr <dump>`** → `actor.terms`：Q 项梯度范数≈0 或 `cos_to_total`≈0 → Q 对参数无有效拉力；entropy 项反向（cos<0）说明两项在打架。
- **`DK drift <dump>`**：`det_action_drift.l2_mean` 长期≈0 但 `param_delta.actor.l2`>0 → 参数在动但行为不动（死区/饱和）；l2 大但 eval 不动 → 改了但改错方向。
- 梯度动力学：`critic.grad_norm.<ch>`、`actor.grad_norm`、`temperature.grad_norm`（tick 指标）；某通道 `critic.clip_applied.<ch>` 长期=1 → 该通道梯度持续越界被裁剪。

## 8. eval 与训练脱节

- `DK series --metric eval.*` 与 round `reward.*`/`task.*` 对照：训练 reward 涨但 eval 不涨 → 检查 eval 用 deterministic 导出（`policy_bp.config.stochastic`）与 eval seed 分布。

## 9. 证据记录格式

每次诊断结论写入当次 run 的 `debug_notes.md`（run_dir 下新建即可）：

```
## [date] 症状: <一句话>
假设: <哪一环>
命令/证据: <命令 + 关键数字>
结论: <定位到 X>
动作: <单变量改动 or 继续观察>
```

禁止：无假设的"调参重跑"。
