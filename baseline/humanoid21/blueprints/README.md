# blueprints/

环境蓝图与初始策略蓝图目录。文件清单与分类见
[`../README.md`](../README.md#blueprints)。

## 活跃 vs Legacy（2026-10 审计划定）

**活跃**（被 `experiments_ppo` 顶层实验直接引用）：

- 环境蓝图：`basic_balance_v2_phi_dual_env.yaml`、`standup_4stage_dense_v2_env.yaml`
- 初始策略蓝图（10 个，对应现役 TruncNorm 策略族）：
  `init_policy_truncated_normal.yaml`、`init_policy_state_truncated_normal.yaml`、
  `init_policy_bounded_std_truncated_normal.yaml`、`init_policy_state_bounded_std_truncated_normal.yaml`、
  `init_policy_mixture_truncated_normal.yaml`、`init_policy_state_mixture_bounded_std_truncated_normal.yaml`、
  `init_policy_shared_mixture_truncated_normal.yaml`、`init_policy_shared_mixture_bounded_std_truncated_normal.yaml`、
  `init_policy_pre_tanh_normal.yaml`、`init_policy_state_pre_tanh_normal.yaml`

**Legacy**（其余 ~48 个）：V1/课程时代实验资产，保留供参考与复用（多数被
`experiments_ppo/archive|todo` 或 `curriculum/` 引用），不在当前主线训练路径上。
其中 4 个参数化蓝图（`fight_mixed.yaml`、`fight_mixed_v2.yaml`、`mixed.yaml`、
`standup_fallback.yaml`）的默认 `*_policy_bp` 指向已不存在的历史 `runs/`
导出——使用时必须经 `pb.build(...)` 显式覆盖对应参数。

## 调试环境（录制 + 回放）

用任一训练环境蓝图跑一回合并录制帧数据：

```bash
PYTHONPATH=. python3 -m envs.framework.round_runner \
    --env-blueprint baseline/humanoid21/blueprints/basic_balance_env.yaml \
    --policy-a-blueprint policy/blueprints/random.yaml \
    --policy-b-blueprint policy/blueprints/random.yaml \
    --recorder "envs.framework.recorder:BaseFrameRecorder?output_dir=baseline/humanoid21/blueprints/out"
```

启动回放查看器（默认端口 8765）：

```bash
PYTHONPATH=. python3 -m envs.framework.recorder_viewer --no-browser baseline/humanoid21/blueprints/out
```

在浏览器打开 `http://localhost:8765/viewer.html` 查看逐帧数据。
