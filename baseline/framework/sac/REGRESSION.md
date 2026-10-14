# SAC 回归训练规程（R2 后基线复核）

> 目的：R2 深化改动（生命周期/契约/诊断/交付）后，用两实验 × 两熵配置
> 共 4 条全量训练做回归验收。Shannon 支路复核"改动未破坏原能力"，
> U-floor 支路是 U 正则路线首次真实训练实证。
>
> 状态：R1 已完成，held-out success=1.000；R2–R4 待执行。
> 规程版本：reg-sac-v1（2026-10-13）

## 1. 回归矩阵

| # | run 名 | experiment | 熵配置 | 参照 |
|---|---|---|---|---|
| R1 | `reg_standup_shannon_s42` | sac_standup | shannon + auto_alpha | 对照第一轮 `sac_standup_s01_s42_te0_ln`（held-out 0.992） |
| R2 | `reg_standup_ufloor_s42` | sac_standup | u_floor（λ=0.5, f=0.5） | 首次实证，无门槛 |
| R3 | `reg_balance_shannon_s42` | sac_balance | shannon + auto_alpha | 对照第一轮 `sac_balance_s01_s42`（held-out 0.953） |
| R4 | `reg_balance_ufloor_s42` | sac_balance | u_floor（λ=0.5, f=0.5） | 首次实证，无门槛 |

**统一**：actor_arch=s01、seed=42、第一轮同预算（不截短）。

U-floor 参数取值说明：λ=0.5、f=0.5 无标定先例（设计文档只定语义
未定数值），取测试中位量值。smoke 门（§3-Step0）必须检查
`temperature.reg`/`actor` reg 项量级与 Q 量级的相对大小后再放行
全量；若 reg 项主导目标或恒为 0，停下来调参再跑。

## 2. 启动命令

全部命令在 `things/combatbench/` 下执行，`PYTHONPATH` 必须指向该目录。
shannon 支路的 `--set` 还原第一轮验收配置（standup 类默认
target_entropy=-21 / q_layer_norm=False 与 R1 run 实际值不同，
必须显式给 `--set target_entropy=0 --set q_layer_norm=true`）。

```bash
cd /data1/mono/things/combatbench
export PYTHONPATH=/data1/mono/things/combatbench

# R1 standup + shannon（复现 R1 配置：te=0, q_ln=true）
CUDA_VISIBLE_DEVICES=0 python3 -B baseline/framework/train.py \
  --experiment sac_standup --algo sac --run-name reg_standup_shannon_s42 \
  --set target_entropy=0 --set q_layer_norm=true --background

# R2 standup + u_floor（λ=0.5, f=0.5；auto_alpha 必须关，否则 trainer 拒绝）
CUDA_VISIBLE_DEVICES=1 python3 -B baseline/framework/train.py \
  --experiment sac_standup --algo sac --run-name reg_standup_ufloor_s42 \
  --set target_entropy=0 --set q_layer_norm=true \
  --set regularizer_mode=u_floor --set reg_lambda=0.5 --set u_floor=0.5 \
  --set auto_alpha=false --background

# R3 balance + shannon（类默认即 R1 配置：te=-10, q_ln=true）
CUDA_VISIBLE_DEVICES=2 python3 -B baseline/framework/train.py \
  --experiment sac_balance --algo sac --run-name reg_balance_shannon_s42 \
  --background

# R4 balance + u_floor
CUDA_VISIBLE_DEVICES=3 python3 -B baseline/framework/train.py \
  --experiment sac_balance --algo sac --run-name reg_balance_ufloor_s42 \
  --set regularizer_mode=u_floor --set reg_lambda=0.5 --set u_floor=0.5 \
  --set auto_alpha=false --background
```

**并发约束**：每 run `rollout_workers=96`（CPU rollout），机器 192
核 → **同时最多 2 条**。建议顺序：R1+R3 先行，R2+R4 随后；或
`--set rollout_workers=48` 四条并行（吞吐略降，结果不受影响）。

预算：standup max_env_steps=2M / balance 10M；eval 各 16/32 episodes。

## 3. 分步验证规程

### Step 0 — smoke 门（4 条，必须先过）

每条正式跑之前先 `--smoke`（不占用预算，验证管线端到端）：

```bash
# 例：R2 的 smoke（去掉 --background、--run-name 改 *_smoke）
python3 -B baseline/framework/train.py --experiment sac_standup --algo sac \
  --run-name reg_standup_ufloor_s42_smoke --smoke \
  --set target_entropy=0 --set q_layer_norm=true \
  --set regularizer_mode=u_floor --set reg_lambda=0.5 --set u_floor=0.5 \
  --set auto_alpha=false
```

u_floor 两条额外检查（smoke 输出/事件流）：

- tick 指标 `actor.reg_term` 非恒 0、非数量级爆炸（相对于
  `critic.q_mean` 的量级）；`regularizer.shannon`=0 确认模式生效；
- `run_terminal.json` status=`completed`，无 early error。

smoke 不过 → 调参（λ/f）并重冒烟，不进全量。

### Step 1 — 启动全量

按 §2 命令起后台训练（并发 ≤2）。记录每条 run_dir。

### Step 2 — 运行中监控（每小时级巡检）

```bash
# 进度与最近 eval
PYTHONPATH=. python3 -m baseline.framework.sac.debugkit run baseline/runs/<run>
# 单指标曲线（--metric 每次一个；常用 eval.success / critic.q_mean /
# temperature.alpha；u_floor run 再看 actor.reg_term）
PYTHONPATH=. python3 -m baseline.framework.sac.debugkit series \
  baseline/runs/<run> --metric eval.success
```

停训/异常处理：

- 正常完成/达标：`run_terminal.json` status=`completed`/`experiment_stop`；
- 外部停训：SIGTERM 一次 → 等安全边界落盘（`signal_stop` + 终态
  checkpoint + `run_terminal.json`）；
- `status=diverged` → 按 DEBUG_PLAYBOOK 定位（Q 发散/熵崩路径已登记）；
- u_floor run 额外看 `actor.reg_term` 明细指标是否偏离预期。

### Step 3 — held-out 复核（每条完成后必做）

```bash
PYTHONPATH=. python3 -m baseline.framework.sac.held_out_eval \
    --run baseline/runs/<run> --policy policy \
    --episodes 64 --workers 32
```

输出 `held_out_eval.json`：聚合 + `episode_details` 逐 agent 明细 +
`policy` 身份块（fingerprint/sha256/code_version）。

### Step 4 — 验收口径与记录

| 支路 | 通过口径 |
|---|---|
| shannon ×2 | held-out 达第一轮判据（standup `eval.success`≥0.9、balance `survival_rate`≥0.9），且与第一轮 s42 结果（0.992/0.953）无显著退化 |
| u_floor ×2 | **首次实证，不设阈值**：训练稳定完成（terminal=completed/experiment_stop，无 diverged）、held-out 报告完整产出；结果如实记录并与 shannon 对照 |

结果表写入 DECISIONS.md 新增节（4 条 run 的 held-out verdict +
逐 agent 明细摘要 + 与第一轮对照），作为本轮回归的证据记录。

## 4. 如实声明的边界

- u_floor 的 λ/f 是未标定取值；本回归产出的是"该配置下的首个实证
  点"，不是"U 路线有效性"的结论——后者需要对照实验矩阵另行设计。
- 回归只验证 s42；不覆盖多 seed 结论。
- 训练跑的是 git 快照分支（code_snapshot），回归期间主线改动不影响
  在训 run，但结果对应的是快照代码而非最新主线。
- `--resume-from` 恢复语义已在 R2-1 逐位验证，回归中若中断可用
  断点续训，不算失败。

## 5. 工件清单（每 run 预期产出）

`config.json` / `code_snapshot.json` / `metrics/events.jsonl` /
`checkpoints/`（末 3 个）/ `policy/`（含 R2-5 起新增的
`meta.json`）/ `videos/` / `run_terminal.json` / `held_out_eval.json`。
