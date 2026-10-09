# DOCS_INDEX — 文档路由器

> 类型：指南

按**任务**找文档，不按目录翻。每行 = 一条最短阅读链。
写文档的规范在 `DOCS.md`；目录树在 `CLAUDE.md`。

## 开发任务

| 我要… | 按序读 |
|---|---|
| 写一个新 PPO 实验 | `baseline/README.md` → `baseline/framework/ppo/GUIDE.md` → `envs/framework/BLUEPRINTS.md` |
| 加一个 reward/observer | `envs/humanoid21/DATASPEC.md` → `OBSERVATION_zh.md` → `ppo/GUIDE.md` |
| 写一个扰动/规则插件 | `envs/framework/DESIGN.md` → `envs/framework/EPISODE_OPTIONS.md` → `envs/humanoid21/disturbance_plugins.py` 源码 |
| 新建/改环境蓝图 | `envs/framework/BLUEPRINTS.md` → `baseline/humanoid21/blueprints/README.md` |
| 导出自包含策略 | `policy/README.md` → `BLUEPRINTS.md` §2（file: 形态） |

## 运行任务

| 我要… | 按序读 |
|---|---|
| 启动/复现训练 | `baseline/README.md` ② → `CLAUDE.md` §Training CLI → run 的 `REPRODUCE.md` |
| 跑单回合/整场评估 | `envs/framework/RUNNERS.md` |
| 查看训练指标/抓 dump | `ppo/dumpkit/USAGE.md` → `CONTEXT.md` |
| GPU/多卡批量采样 | `envs/batchframework/README.md` → `PUBLIC_INTERFACE.md` → `SEMANTICS.md` |
| 提交策略到平台 | `docs/SUBMISSION.md`（zh 同名） |

## 理解边界

| 我要搞清… | 读 |
|---|---|
| 框架职责边界（什么插件能干什么） | `envs/framework/DESIGN.md` |
| reset/seed 语义 | `envs/framework/RESET.md`、`SEED.md` |
| 训练确定性担保到哪儿 | `baseline/framework/DETERMINISM.md` |
| episode_options 有哪些键 | `envs/framework/EPISODE_OPTIONS.md` |
| rollout 数据模型（Job/Episode） | `baseline/framework/rollout/README.md` |
| 回合数据/观测规范 | `envs/humanoid21/DATASPEC.md`、`CONTROLSPEC.md` |

## 状态判断

| 这份东西还能用吗 | 看 |
|---|---|
| 某能力什么状态（STABLE/LEGACY/…） | `CAPABILITY_LEDGER.md` |
| 某文档是不是历史墓碑 | 文档头 `类型：` 行（历史类标注取代者） |
| 已知问题/审计进度 | `AUDIT.md`、`AUDIT_DOCS.md`、`ISSUES.md`（只读） |
| SAC 到哪一步了 | `baseline/framework/sac/PLAN.md` |
| batchframework 路线图 | `envs/batchframework/ROADMAP.md` |

## 遗留/参考（能读不能跑）

`baseline/humanoid21/curriculum/`（v1 框架已删）、
`baseline/framework/obsolete/`、`balance_recover/` 的 MANIFEST/PLAN
系文档——头部 `类型：历史` 或各 README 的 ⚠️ 标记为准。
