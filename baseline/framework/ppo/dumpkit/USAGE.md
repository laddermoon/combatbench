# dumpkit 使用指南（USAGE）

> 类型：指南

dumpkit 的**操作向**入口。概念地图（数据阶梯/指标命名空间/catalog）
见 `CONTEXT.md`；内部数据流设计见 `DATA_FLOW.md`。

统一 CLI：`baseline/framework/ppo/debug.py`（PYTHONPATH=仓根）。
除 `viewer` 外全部离线、输出 JSON（`--pretty` 美化）。

## 1. 触发一次 dump（三种途径）

```bash
# A. 训练启动时预约：到 update 282 完整抓取一次
python3 baseline/framework/train.py --experiment X --algo ppo \
    --dump-at 282 --dump-hypothesis "KL 为什么在 250 飙高" [--dump-full-grad]

# B. 运行中追加请求：写哨兵文件，下一个 update 被抓
python3 baseline/framework/ppo/debug.py dump baseline/runs/<run> \
    --hypothesis "调查梯度消失" [--full-grad]

# C. resume 精确复现：从 u280 checkpoint 续训 + dump u282 =
#    完全重现那次 update 的输入与动力学
python3 baseline/framework/train.py --experiment X --algo ppo \
    --resume-from runs/<run>/checkpoints/checkpoint_u00280.pt --dump-at 282
```

## 2. dump 产物结构

```
<run>/dumps/uNNNNN/
├── episodes.npz        # 本 update 采到的回合逐帧数据
├── buffer.npz          # 训练轨迹（ret/adv/actor_weight/confidence）
├── update.npz          # update 内部动力学（epoch/minibatch 级 KL、ratio、loss）
├── gradsig.npz         # ADV 梯度信号分布（cos(g_i,G)×‖g_i‖ 分箱 + 逐帧数组）
├── manifest.json       # 抓取清单 + hypothesis
├── episode_options.json
├── policy 导出          # 本 update 后的策略蓝图
└── 派生：record/（render PNG）、delta/（跨代漂移）、rollout/（realized ε）
```

## 3. 分析子命令（`debug.py <cmd>`）

| 子命令 | 回答的问题 |
|---|---|
| `runs` | 有哪些 run？（列 `baseline/runs`，JSON） |
| `summary <run>` | run 整体怎么样？每指标摘要+语义——**先看这个** |
| `metrics <run>` | 指标时序（扁平化，与 viewer API 同数据） |
| `catalog [--key K]` | 指标语义目录（metric_catalog 单一信息源） |
| `inspect <dump>` | 一次 dump 的截面概览（能力/ADV/gradsig/update 摘要） |
| `query <dump> <api>` | dump 任意 API 离线查询 |
| `samples <dump>` | 采样梯度帧排名（含 episode/trajectory 溯源） |
| `trace <dump>` | 单个 buffer 帧跨阶段 join 的全部记录 |
| `delta <dump> [--episode N]` | 策略跨代漂移（确定性回放对老代） |
| `rollout <dump>` | realized exploration ε（线性空间残差 a−a_det） |
| `render <run>` | dump 回合逐帧 PNG + 一致性自校验 |

## 4. viewer（交互 web UI）

```bash
python3 baseline/framework/ppo/debug.py viewer <path> [--port 8766] [--no-browser]
```

`<path>` 三种粒度：`baseline/runs`（多 run 索引）/ 单个 run 目录 /
单个 dump 目录（直达该 update 截面）。浏览器开 `http://localhost:8766/`。
四场景设计见 `DESIGN_viewer_scene*.md`。

## 5. 常见排查路径

| 症状 | 路径 |
|---|---|
| KL/ratio 异常 | `summary` → `inspect uNNNNN` → `metrics` 看 `stats.*` 时序 |
| 梯度信号弱 | `inspect` 的 gradsig 摘要 → `samples` 看排名帧 |
| 探索量不对 | `rollout` 看 realized ε → `delta` 看跨代漂移 |
| 环境行为怪 | `render` 逐帧看 → `trace` 追单帧全记录 |
