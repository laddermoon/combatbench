# DETERMINISM — 训练层确定性担保

> 类型：契约

本文件定义**训练路径上"设了 seed 能复现到什么程度"**的权威边界。
框架层（回合内 seed 派生、reset 链）见 `envs/framework/SEED.md`——
两者互补：那篇管单回合，这篇管训练全过程。

## 1. seed 流向

```
train.py --seed S  ──► experiment.seed = S（不设则用实验内置）
                          │
                          ▼
                set_seed(S)          # loop 开头：py-random + np + torch CPU/CUDA
                          │
        ┌─────────────────┼──────────────────────────────┐
        ▼                 ▼                              ▼
rollout 种子            eval 种子                     gradsig 采样种子
S + u·episodes_       S + 100000 + u·97              S·1000003 + u
per_update            （每 update 独立）               （dump 诊断专用）
```

- **job seed 全由公式预计算**——`collect()` 保序返回，`num_workers`
  改变只影响并行度，**不影响结果**（种子与 worker 无关）。
- checkpoint 保存/恢复 RNG 状态（numpy state），resume 链不断。
- 设备端（batchframework）的 seed 语义以 `SEMANTICS.md` 为准——
  该域在飞，不在本文件担保范围。

## 2. 担保等级

| 层级 | 担保 | 依据 |
|---|---|---|
| **resume 等价性（CPU/同硬件同依赖）** | **位级相等**：续训从断点跑出的 update 与不中断连续训**全字段一致**（actor/critic 参数、Adam 状态、RNG 状态、RAW_STATS 除 timing、gradsig 数组） | `test_resume_equivalence.py` 旗舰测试——跨进程边界逐位比对整个 checkpoint payload |
| **同 seed 同配置重训（同机同依赖）** | 与 resume 同级——种子公式确定，RNG 链封闭 | 同上测试的推论 |
| **GPU 训练** | **不承诺位级相等**——全仓未设 `torch.backends.cudnn.deterministic`/`CUBLAS_WORKSPACE_CONFIG`；统计等价预期成立 | 源码核查（无确定性开关） |
| **跨硬件/跨依赖版本** | 不承诺位级；统计等价受 MuJoCo/torch 版本影响 | — |
| **device/batchframework 采样** | fp32 近似，不承诺与 CPU bit-identical | batchframework README 明示 |

## 3. REPRODUCE.md / code_snapshot 的真实担保

每次 run 生成的 `REPRODUCE.md` + `code_snapshot.json` 担保的是
**代码状态可复现**（git worktree 取同一份代码）+ **命令可重放**；
**不担保**跨环境位级一致——依赖版本、硬件、GPU 配置仍是变量。

## 4. 实践建议

- 要位级对照实验：固定同一台机器、同一依赖环境、CPU 训练
  （`--collector cpu`），设 `--seed`。
- 怀疑非确定性时先排除：`num_workers` 变化**不是**非确定源；
  GPU 训练才是。
- dumpkit 的 `delta`/`rollout` 回放都是 deterministic act() 链路，
  跨进程可复现。
