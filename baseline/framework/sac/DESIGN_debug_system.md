# SAC Debug 系统对齐 PPO — 第三轮总体规划

> 类型：规划
>
> 状态：待批准执行
> 依据：用户三条要求——
> ① 页面框架和 Layout 与 PPO 相同（代码不复用）；
> ② 页面逻辑与 PPO 保持一致；
> ③ 功能不照抄 PPO，按 SAC 自身计算链需求设计。

## 0. 定位

本轮只攻 **viewer/debug 系统的用户界面层**。SAC 的分析层与 CLI
（`debugkit` 18 个子命令、`analysis.py` ~1400 行、`sac_dump_v3` 工件）
已达 R2-3 验收深度，不是本轮主体。缺口集中在：

| 层 | PPO | SAC 现状 | 差距 |
|---|---|---|---|
| CLI | `debug.py` 19 子命令 | `debugkit` 18 子命令 | 已基本对齐，补齐 `viewer`/`metrics --docs`/`summary` 语义挂接 |
| 分析层 | `dump_analysis.py` 单计算源 | `analysis.py` 单计算源 | 对齐 |
| HTTP API | 60+ 端点、模式化 `/run/<n>/api/...`、POST 单 job 槽 | 14 个 GET，仅 query-string 定位，无 POST | **大** |
| Viewer | 6019 行 SPA：runs/run/dump 三模式、六分区 dashboard、compare、dump 主页八模块卡 + 工具页 + 面包屑 + 上下文栏 + 共享帧游标 | 426 行：runs 侧栏 + 单图 metrics + dumps tab | **本轮主体** |
| metric catalog | 六分区 sectioned（curated chart spec + hint/subtitle/zone_pick） | 纯白名单 spec，无 zone/chart/hint | **大** |

## 1. Layout 对齐契约（要求①+② 的可验收清单）

以下条目**逐条复刻 PPO viewer 的框架与页面逻辑**，但全部 SAC 自有实现：

### 1.1 运行模式（与 PPO 相同的三模式）

```
runs 模式  顶层 = 多 run 索引页（/api/runs 分页/排序/搜索）
run 模式   顶层 = 单 run dashboard（六分区指标仪表板 + run info +
           dumps + videos + log tail 分页）
dump 模式  顶层 = 单 dump 主页（管线模块卡 + open tool → 链接）
```

### 1.2 页面骨架

- **SPA 单文件 HTML + vanilla JS/CSS**，无构建系统（与 PPO 同技术方案）；
- **Header**：标题 + 模式/对象标识 + 面包屑；
- **底部上下文栏**始终可见（PPO 显示 episode/trajectory/frame/epoch/step；
  SAC 显示 `run / dump(critic_tick) / sample_id / slice_id / collection_round`）；
- **面包屑导航**可逐级返回，状态沿 drill-down 链传递。

### 1.3 共享选择状态（SAC 版传递规则，结构与 PPO 同构）

| 状态 | runs→run | run→dump | dump 主页→工具页 | 工具页↔工具页 |
|---|---|---|---|---|
| run | ✅ | ✅ | ✅ | ✅ |
| dump(critic_tick) | — | ✅ | ✅ | ✅ |
| sample_id | — | — | 进入页自带 | ✅ 跨页保持 |
| slice_id / round | — | — | 随样本联动 | 随样本联动 |

SAC 的游标单位是 **sample_id（batch 行）** 而非 PPO 的 trajectory/frame。
选中一个样本后，trace/decomp/slice-ctx/grad-attr 各页共享该游标——与
PPO "选帧后跨页保持" 是同一交互逻辑。

### 1.4 run 级页面（与 PPO 同逻辑）

- runs 索引：状态判定复刻 PPO 规则
  （running=pid 存活或日志 90s 内活跃 / finished=到预算 /
  failed=进程死且尾有 Traceback / stopped / unknown）；
- dashboard：**六分区 catalog 驱动**（curated 图 + expanded 折叠组 +
  hint/subtitle），zones 见 §3；
- compare 页：`/compare/<runA>,<runB>,...`，跨图同色、按 run 拆分超
  12 线面板、URL 可分享——同 PPO 逻辑；
- log 卡：`/api/run/log?lines&before=` 字节偏移分页 + `/train.log`
  整文件下载；
- update detail：SAC 等价物 = 该 round 的 `--param` 生效值 +
  config patch 记录（▲ 标记被调度覆盖的字段）。

### 1.5 dump 级页面（管线主页 + 工具页，结构同 PPO）

- dump 主页 = **功能汇总卡 + `open tool →`**，每卡一模块；
- 工具页结构同构：**选择器 + 骨架大图 + 页专属图表**，跨页共享
  sample 游标；
- 重量计算走 **单后台 job 槽**（PPO 的 render/delta/rollout 共享单槽 +
  409 语义 + 轮询）：SAC 对应物 = `grad-attr` / `drift` /
  `slice-ctx(大 replay)` / `realized-ε` 重算。

### 1.6 POST 与离线等价（PPO 页面逻辑的两侧）

- `POST /run/<n>/api/run/dump-request {hypothesis}`：写 dump_request
  哨兵（loop 已支持该协议）；
- `POST /api/dump/<d>/job {kind}` → 单 job 槽 + `GET /api/job-status`；
- 所有 dump 读端点必须有 `debugkit query` 离线等价物（现有机制保留，
  新端点注册进同一 dispatch）。

## 2. SAC 管线模块映射（要求③：功能按 SAC 计算链设计）

PPO 的八卡链（⓪ε→①ep→traj→②ADV→③norm→④merge→⑤gradsig→⑥update
→⑦post→⑧delta）**不照搬**。SAC 的一跳 critic_tick dump
（`batch.pt` + `forward.pt` + `trainer_pre/post.pt` + `spec.json`）
对应以下八模块：

| 卡 | 模块 | 数据 | 回答的问题 |
|---|---|---|---|
| ⓪ | Round & Collection | manifest/clocks + replay provenance | 这个 tick 属于哪个 collection round、用了哪版策略采的数据 |
| ① | Replay → Batch | `/api/dump/batch` + `replay_report` | batch 的 256 行来自哪些 slice/round/fingerprint、各被复用几次 |
| ② | Reward → Target | `decomp`（reward·scale + γ·b·(softQ+reg)） | 每通道 target 怎么构成；γ 滑杆服务端重算（PPO 的 GAE 重算对应物） |
| ③ | TD → Critic | forward.pt per-channel | td 分布、q1/q2/loss/q_mean、各通道 grad_norm 与 clip |
| ④ | Actor Objective | `decomp.actor` + pair selection | Σw·Q + entropy + reg 三分解、pair1_frac、expectation 采样对 |
| ⑤ | Temperature & Regularizer | forward + spec.sac_params | shannon：α/loss/grad；U 模式：U(s) 分布 vs u_floor、λ·reg 项 |
| ⑥ | Update Process | events.jsonl tick 序列按 round 过滤 | 本 round 10K tick 的 loss/q/td/α 轨迹 + 本 dump 所在位置（PPO timeline 的 SAC 对应物——SAC 无 epoch/minibatch，单位是 critic tick） |
| ⑦ | Post Update | `param_delta`（trainer_pre/post 递归 diff） | 一跳更新实际把哪层参数推了多远 |
| ⑧ | Drift & Context | `drift` + `slice-ctx` | 固定输入 pre→post actor/Q 漂移 + 源 episode 上下文重建 |

工具/钻取页（与 PPO 工具页同构：选择器+骨架+页内图表）：

- `/samples` — 排序样本表（td_abs/q/logp/draws），行动作 →
  trace/decomp/slice-ctx；
- `/trace/<sid>` — 单样本 batch→forward→target parts→actor parts 全记录；
- `/decomp` — target/actor 双分解 + 行级下钻；
- `/attribute` — 来源 × 更新贡献联合分布；
- `/grad-attr` — 分组梯度归因（重计算，job 槽触发）；
- `/drift` — 固定输入漂移分布图；
- `/tickline` — ⑥的钻取页；
- `/slice-ctx` — replay 上下文重建（run 级，从 run 页进入）。

## 3. SAC metric catalog 升级（zone 化）

把白名单 catalog 升级为 sectioned 目录（同 PPO 的
zone_all/zone_pick/zone_rest 机制），SAC 语义六分区：

| zone | 收录前缀 | 类比 PPO |
|---|---|---|
| task | `eval.*`, `task.*` | task |
| update | `critic.*`, `actor.*`, `temperature.*` | policy_update |
| signal | `critic.td_*`, `critic.grad_norm.*`, `actor.grad_norm`, `temperature.grad_norm` | signal |
| explore | `behavior.*`, `collection.*`, `regularizer.*`, `actor.reg_term` | explore |
| sampling | `replay.*`, `batch.*`, `actor.pair1_frac`, `target.pair1_frac` | sampling |
| cost | `timing.*`, `export.*`, `checkpoint.*`, `debug.*` | cost |

catalog spec 携带 subtitle/guide/hint 元数据（单一信息源，CLI
`catalog`/`metrics --docs` 与 `/api/catalog` 同源）。PPO 命名空间
继续禁用。

## 4. API surface（SAC 化 PPO 形状）

```
GET  /api/mode | /api/catalog | /api/job-status
GET  /api/runs?q=&page=&size=&sort=&order=
GET  /run/<n>/api/run/{info,dumps,metrics,videos,log?lines&before}
GET  /run/<n>/api/run/replay | /run/<n>/api/run/slice-ctx
GET  /run/<n>/train.log
GET  /api/dump/<d>/{manifest,batch,inspect,pipeline,samples,trace,
       decomp,attribute,grad-attr,drift,tickline,recompute}
GET  /compare/<runA>,... (SPA 页)
POST /run/<n>/api/run/dump-request {hypothesis}
POST /api/dump/<d>/job {kind: grad-attr|drift|slice-ctx|eps-realize}
```

只读语义不变；仅有的写 = 哨兵文件 + job 状态内存态（不落盘修改
run 工件）。

## 5. 工作包与出口标准

| 包 | 内容 | 出口 |
|---|---|---|
| **R3-W0** | 本文档契约冻结 + 逐条 parity checklist 落成表格 | checklist 写入本文件 §6，提交 |
| **R3-W1** | sectioned catalog + runs 索引/状态 + 六分区 dashboard + compare + log 分页 + run detail | run 级页面逻辑全过 checklist；CLI `metrics --docs`/`summary` 与 `/api/catalog` 同源 |
| **R3-W2** | dump 骨架：主页八模块卡 + `/api/dump/pipeline` + 面包屑/上下文栏框架 | dump 主页可进任意模块页，面包屑可逐级返回 |
| **R3-W3** | 工具页批 1：⓪①②③④⑤ + samples/trace/decomp/attribute + 共享 sample 游标 | 每页结构同构、游标跨页保持、数据与 CLI 同源一致 |
| **R3-W4** | 工具页批 2：⑥tickline ⑦postupdate ⑧drift/slice-ctx + 单 job 槽（grad-attr/drift/slice-ctx/eps-realize）+ POST dump-request | job 槽 409 语义、轮询、哨兵触发在真实 run 上验证 |
| **R3-W5** | dogfood + 收口：在 R2_b/R4 在训 run 上抓 u_floor dump 走全链路；USAGE/CONTEXT 文档；测试；PPO viewer 零影响确认 | 全量回归 + parity checklist 全绿 + 登记 DECISIONS |

## 6. 验收 checklist（执行时逐条打勾）

- [ ] 三模式入口、面包屑、上下文栏、共享游标：结构与 PPO 一致
- [ ] runs 状态判定规则与 PPO 同逻辑
- [ ] dashboard 六分区由 catalog 单一信息源驱动
- [ ] compare 页交互与 PPO 一致
- [ ] dump 主页 = 八模块卡 + open tool；模块语义为 SAC 版 §2 映射
- [ ] 每工具页：选择器+骨架+页内图，sample 游标跨页保持
- [ ] POST job 单槽 + 409 + 轮询 + dump-request 哨兵
- [ ] 全部 dump 读端点有 `debugkit query` 离线等价
- [ ] 零 PPO import：只拷交互/布局概念，不共享代码
- [ ] 在训回归 run 不受影响（viewer 只读；哨兵协议既有）

## 7. 不做的事

- 不做多 dump 对比（同 PPO：每 viewer 实例一个 dump）；
- 不做实时训练控制/参数修改（只读 + 哨兵）；
- 不做前端构建系统；
- 不重写 `analysis.py` 计算层（本轮只补它缺的端点，不动语义）；
- 不承诺 realized-ε 回放一定进本轮（⑧的 ε 计算视 dogfood 价值决定，
  先登记为可选项）。
