# CombatBench 正则化总纲（REGULARIZATION）

日期：2026-10-01
状态：**v0 草案 —— 待用户确认后冻结为 v1**
维护规则：本文件是"正则化"这个长期任务的唯一权威总纲。任何会话开始正则化工作前，
必须先读本文件与能力总账（`CAPABILITY_LEDGER.md`，Phase 0 建立）。本文件冻结后，
修改需经用户确认；执行层事实（盘点结果、各能力状态）不写进本文，写进总账。

---

## 0. 本文定位

本文回答四个问题：

1. 正则化要做什么、不做什么（§1–§2、§8）
2. 对项目的核心理解——任务执行者的心智底座（§3）
3. 三层文档体系规范——给谁看、写什么、放哪里（§4）
4. 成熟度模型与工作方式——怎么判定"做完了"、怎么持续推进（§5–§7）

不写具体实现细节。实现细节属于各工作包的产物。

## 1. 一句话任务定义

把 CombatBench 从"持续演进的研究代码库"收敛为**"用户决策、AI 执行"的策略开发平台**：

- 盘点全部能力，按成熟度分级；
- 可用的能力**暴露出来**（有契约、有选型说明、有测试证据）；
- 差一点就好的能力**补齐**（含补测试），不留"差不多能用"的灰色地带；
- 差得远的能力**显式标记不可用**，不留"看起来能用"的陷阱；
- 建立三层文档体系，使 AI **不读源码**即可正确使用每个杠杆。

## 2. 终态判定标准（Definition of Done）

正则化完成的标志是以下场景成立：

> 用户说"我要做 XX 实验" → AI 读本项目的用户层文档 → 正确选择杠杆 →
> 正确写出实验代码 → 通过项目自带验证 → 达到目的。

拆开为四个可检查项：

1. **每个被暴露的能力都有**：AI 契约文档（接口/参数/扩展规范/坑）、人类选型说明
   （适合什么场景）、测试证据、总账登记条目。
2. **不可用的能力显式标记**：总账和入口文档都能看到状态；调用方得到显式拒绝
   而非静默走错路。
3. **入口可达**：新人（人或 AI）从 README 出发能到达正确的文档层，不需要
   先理解仓库结构。
4. **CPU→GPU 加速路径有转换规范**：AI 可照规范做代码转换；不支持的能力被
   显式拒绝，而不是静默转换出一个不同的任务。

## 3. 对项目的核心理解

### 3.1 项目是什么

CombatBench 是 MuJoCo 人形机器人（21-DOF × 2）对战仿真平台 + RL 训练框架 +
线上对战平台（combatbench.tech，Elo 打榜）。核心资产不是某一个 benchmark 场景，
而是**一套可扩展的环境运行时框架**和**一套自研训练框架**：

```
实验配置层   baseline/experiments_ppo|sac/   exp_*.py 自动发现
训练框架层   baseline/framework/             多 critic PPO/SAC + dumpkit debug
训练适配层   baseline/humanoid21/            rewards / plugins / blueprints
环境框架层   envs/framework/                 Accessor/Mutator + SimContext + 插件生命周期
物理仿真层   envs/humanoid21/                MuJoCo simulator + combat plugins
加速路径     envs/batchframework/            MJX/Warp device-resident（M0–M8，进行中）
```

### 3.2 产品逻辑：用户是决策者，AI 是执行者，文档是接口

AI 时代用户不手写代码。分工是：

- **用户**：对项目有心智层面的判断——知道有哪些杠杆、各适合什么场景——
  然后对 AI 下达方向性指令（"用这个杠杆做 XX 实验"）；
- **AI（用户侧）**：按项目文档把用户意图实现为正确代码，不需要通读源码
  （读片面的源码反而会得到偏的系统理解）；
- **AI（开发者侧）**：维护本项目本身，需要理解内部实现与设计动机。

推论：**文档的完备性和正确性就是这个产品的可用性**。文档缺失或说谎，
等于功能不存在或功能是坏的。

### 3.3 杠杆（Lever）

**杠杆 = 用户可以选择的、已建成的机制**。用户心智模型是"杠杆目录 + 选型直觉"，
AI 心智模型是"每个杠杆的契约"。当前已识别的杠杆域：

| 域 | 杠杆举例 |
|---|---|
| 策略分布 | TruncNorm 家族 8 格（truncated / bounded / state-σ / pre-tanh / mixture 及组合） |
| 探索控制 | `explore_factor`（rollout 侧 [-1,1]）、`uncertainty_floor`+`coef`（训练侧）、SamplingContext delta/reference_action 机制 |
| 奖励/评估 | 多 critic per-channel actor_weight、φ² 动态权重、PBRS 势能族 reward、8+ reward channels |
| 算法与执行 | PPO / SAC；CPU ParallelRollouter / DeviceRollouter（WIP） |
| 环境规则 | world plugins（scoring / non-fall / wind / instant push / fallen reset / timeout）、observer plugins、blueprints |
| 调试分析 | dumpkit 全家（runs/summary/metrics/dump/render/delta/rollout/inspect/trace/viewer + HTTP API） |
| 运行器 | RoundRunner / MatchRunner / recorder / replay |
| 课程 | 四阶段 curriculum（legacy 资产，可用性待核实） |

### 3.4 两条执行路径与一组共享资产

- **CPU 主路径**（EnvRuntime + MuJoCo）：任务语义的权威来源、原型开发主路径，
  灵活、可调试。
- **GPU 加速路径**（batchframework，MJX/Warp）：大规模训练路径，从主路径**单向
  派生**。**明确不做**自动适配层——不提供"主路径代码直接跑上加速路径"的复杂
  适配代码；提供的是**转换规范 + 模板 + 验证工具**，由 AI 按规范做转换。
  见 `envs/batchframework/discuss.md`（总纲）与 `ROADMAP.md`（M0–M8）。
- **共享资产**：Episode/Trajectory/PPO/checkpoint/dump 契约和 dumpkit 分析
  工具跨路径复用；加速路径不得反向修改任务语义。

正则化与 batchframework M8（AI 迁移体系验收）的关系：**正则化是上游**——
M8 要的"AI 入口说明、转换规范、支持矩阵"就是本文 A 层文档在加速路径上的特例。

### 3.5 已有的两类范本（推广对象）

正则化不是从零发明格式，项目里已有两份经过实战的范本：

| 范本 | 文件 | 对应层 |
|---|---|---|
| 人类选型文档 | `baseline/framework/ppo/policies/POLICY_SELECTION.md` —— 8 格策略的定位/入选理由/淘汰判据/前提条件 | U 层 |
| AI 契约文档 | `baseline/framework/ppo/dumpkit/CONTEXT.md` —— 能力地图、命令速查、数据阶梯、命名空间、gotchas、红线 | A/D 层混合（见 §4.4） |
| AI 转换规范 | `envs/batchframework/discuss.md` + `ROADMAP.md` —— 转换规则、验证层级 V0–V5、Agent 禁止事项、升级条件 | A 层（加速路径特例） |

## 4. 三层文档体系

### 4.1 三层受众

| 层 | 代号 | 读者 | 读者的任务 | 核心问题 |
|---|---|---|---|---|
| 用户层 | **U** | 人类用户 | 决策：定方向、选杠杆 | "有哪些杠杆？各自适合什么场景？做了会得到什么？" |
| 接口层 | **A** | 用户侧 AI | 使用：按契约实现用户的实验意图 | "怎么正确调用/扩展这个杠杆？签名、参数、不变量、坑？" |
| 开发层 | **D** | 开发者 AI | 修改：改动框架本身 | "内部怎么实现的？为什么这样设计？改动的约束和坑？" |

判别规则：**读者想"用"它 → A 层；想"改"它 → D 层；想"选"它 → U 层。**

### 4.2 各层内容规范

**U 层（给人）**——写心智模型，不写签名：

- 杠杆是什么、解决什么问题、适合/不适合什么场景；
- 选型决策（对比、前提条件、已知失败模式）；
- 指向 A 层文档的链接（"选定后让 AI 读这个"）；
- 可以有人话、有判断、有推荐；不要求完整枚举参数。

**A 层（给用户 AI）**——写契约，不写实现：

- 接口签名、参数语义、返回结构、单位与坐标系；
- 不变量与使用前置条件（"rollout 与 PPO 重算必须用同一 explore_factor"）；
- 扩展规范：新增一个 plugin/reward/policy/experiment 必须满足什么；
- 坑与禁止事项（fail-loud：不支持就显式报错而非静默回退）；
- 验证方式：怎么确认"我用对了"（测试命令、检查项、对照工具）；
- **不复制实现细节**；实现归 D 层。A 层文档撒谎的代价 = 用户实验做错。

**D 层（给开发者 AI）**——写内部与设计动机：

- 架构分层、数据流、模块关系；
- 设计动机与已否定的方案（避免重复犯错）；
- 改动约束、内部 gotchas、性能注意事项；
- 现有 `CONTEXT.md`（per-directory AI memo）、`DESIGN_*.md`、`REVIEW_*.md`、
  `CLAUDE.md` 均属此层。

### 4.3 放置与命名约定（提案，待用户确认后冻结）

| 层 | 位置 | 命名 | 例子 |
|---|---|---|---|
| U | `docs/` 与根目录 | 人读 markdown，双语随现有惯例 | `README.md`、`docs/RULE.md`、杠杆目录文档（名待定，如 `docs/LEVERS.md`） |
| A | 与杠杆同目录 | `USAGE.md`（新增约定）或沿用 `*SPEC.md` | `baseline/framework/ppo/policies/USAGE.md`、`envs/humanoid21/DATASPEC.md` |
| D | 与代码同目录 | `CONTEXT.md` / `DESIGN_*.md` / `REVIEW_*.md` | `baseline/framework/ppo/dumpkit/CONTEXT.md` |
| 跨层入口 | 根目录 | `REGULARIZATION.md`（本文）、`CAPABILITY_LEDGER.md`、README 文档体系节 | — |

总账 `CAPABILITY_LEDGER.md` 放项目根目录，格式见 §6。

### 4.4 README 入口义务

正则化落地后，`README.md` 必须有一节"文档体系 / 阅读指南"，明确：

- 三类文档给谁看、怎么看（U/A/D 的定义与判别规则）；
- 各层入口文件清单；
- "我是用户，我想做实验，从哪开始" / "我是 AI，我被要求实现实验，读什么" /
  "我要改框架，读什么" 三条导航路径。

（README 改动在本文冻结、目录约定确认后执行。）

## 5. 能力成熟度模型

每个能力在总账中标记且仅标记以下一种状态：

| 级别 | 含义 | 对用户是否暴露 | 硬性要求 |
|---|---|---|---|
| **STABLE** | 契约清晰、测试覆盖关键语义、A 层文档齐备 | 暴露 | 测试通过 + A 层文档 + U 层选型说明 |
| **USABLE** | 可用但有已知限制/边界 | 暴露并标注限制 | 限制必须写进 A 层契约，不留隐性坑 |
| **WIP** | 开发中，暂不可用 | 不暴露（状态可见） | 总账登记缺口清单与所属里程碑 |
| **LEGACY** | 能跑但已被取代 | 不暴露 | 指明替代者；保留原因写明 |
| **UNSUPPORTED** | 不可用 / 不会支持 | 显式拒绝 | 调用时明确失败或文档显式劝退 |

规则：

- **"差一点就好"的一律做到好**。包括补测试。成熟度证据是"验证过"，
  不是"看起来能跑"。
- **差得远的显式标记**，不留半成品模糊态。WIP/UNSUPPORTED 也是有效状态，
  不是耻辱——模糊才是问题。
- **状态必须可验证**：标记 STABLE 的能力，要么有测试在跑，要么有明确的
  验证命令写进契约文档。
- 与项目代码守则一致：**fail loud**——不支持的能力让调用方显式失败，
  不静默回退。

## 6. 能力总账（CAPABILITY_LEDGER.md）

正则化的持久状态载体，Phase 0 建立，此后每次正则化工作必须更新它。
一个能力一行，字段：

```
| 能力 | 域 | 状态 | 证据（测试/验证命令） | A 层文档 | U 层说明 | 缺口/备注 |
```

辅助视图（总账内按域分组，不自建多套账本）：

- 按成熟度分组的概览表（STABLE/USABLE/WIP/LEGACY/UNSUPPORTED 各多少）；
- "差一点就好"清单（优先级最高的补齐对象）；
- 明确不可用清单（用户需要知道边界的）。

## 7. 工作方式

### 7.1 阶段划分

- **Phase 0 — 立规矩**（本轮）：本文冻结 v1；README 文档体系说明；建立
  `CAPABILITY_LEDGER.md` 骨架（初始盘点快照先登记"初判待核实"）。
- **Phase 1 — 全量盘点**：按域逐个核实真实状态（读代码、跑测试、查文档），
  把总账从"初判"变成"核实"。产出：可信的总账 + 补齐优先级清单。
- **Phase 2+ — 逐域正则化**：每次会话认领一个工作包（一个域或一组杠杆），
  做"收敛代码 → 补测试 → 写 A 层契约 → 写 U 层选型 → 更新总账"全流程。
- **持续**：此后新能力落地时按同一规范带齐三层文档与总账登记，
  正则化变成项目的落地标准而非一次性运动。

### 7.2 每个工作包的完成标准

沿用 `envs/batchframework/ROADMAP.md` §14 的精神，每次工作交代：

1. 所属域、目标、明确不做的事；
2. 改动范围（动了哪些代码/文档）；
3. 验证证据（跑了什么测试/命令，结果是什么）；
4. 文档产物（U/A/D 各层动了哪些文件）；
5. 总账更新（哪些能力状态变了，依据是什么）；
6. commit（仓库规则：每个任务完成即提交）。

### 7.3 红线

- **文档不撒谎**：标记 WIP 的能力，文档必须明说"当前不可用"，不能含糊；
  能力边界内的承诺必须是真的。
- **fail loud 原则贯通到文档层**：不支持的能力/用法，文档和代码都应
  显式拒绝，不静默回退。
- **归档优先于删除**：legacy 资产先确认无活跃引用再归档；runs/ 历史
  训练数据不是正则化对象。
- **不改任务语义**：正则化不借机"优化"奖励、观测、物理参数——那是
  实验工作，不属于本任务。
- **会话纪律**：沿用仓库规则，一个会话只做一个项目（combatbench）内
  一个明确的工作包；总账保证跨会话连续性。

## 8. 不做的事（Non-goals）

- 不重写架构、不换训练框架、不动物理参数与任务语义；
- 不做 CPU→GPU 自动适配层（只产出转换规范，转换由 AI 按规范执行——这是
  既定架构决策，见 `envs/batchframework/discuss.md` §7）；
- 不清理 `baseline/runs/`（gitignored 训练产物，规模自理）；
- 不承诺加速路径效果达标（那是 M6/M7 验收的事，正则化只登记其真实状态）；
- 不为"文档好看"而隐藏复杂度：杠杆的限制与前提必须如实写。

## 9. 初始盘点快照（v0，全部"初判待核实"——Phase 1 的输入）

| 域 | 内容 | 初判状态 | 备注 |
|---|---|---|---|
| `envs/framework/` | runtime/plugin/blueprint/runner/recorder/replay | STABLE? | 契约层固化；部分文档引用已删文件（REVIEW 记录） |
| `envs/humanoid21/` | simulator + combat/observer/disturbance plugins + XML | STABLE? | DATASPEC/OBSERVATION/CONTROLSPEC 已齐 |
| `envs/batchframework/` | MJX/Warp 设备运行时 + DeviceRollouter | WIP | M5 闭环已过，M6 学习中；按 ROADMAP 登记 |
| `baseline/framework/` PPO | trainer/loop/multi-critic/confidence/dump | STABLE? | 主训练路径 |
| `baseline/framework/` SAC | sac/* | USABLE? | 成熟度待核实 |
| `baseline/framework/ppo/policies/` | TruncNorm 8 格 + SamplingContext | USABLE→STABLE | 有测试与选型文档；部分格子已淘汰需标 |
| `baseline/framework/ppo/dumpkit/` | debug 全家 | STABLE | A/D 层范本已存在 |
| `baseline/experiments_ppo|sac/` | 实验注册表 + base.py | USABLE | README 已有；契约文档待补 |
| `baseline/humanoid21/rewards/` | PBRS reward 族 | 待核实 | 8+ channel，变体碎片化（REVIEW 记录） |
| `baseline/humanoid21/blueprints/` | 48 yaml | USABLE? | 数量膨胀，需盘点合并方向 |
| `baseline/humanoid21/curriculum/` | 四阶段 + gating + mixed policy + V1 实验 70+ | LEGACY | 被 experiments_ppo 取代；归档边界待划 |
| `baseline/framework/obsolete/` | 旧框架 | LEGACY | 已标 obsolete，确认无引用即可 |
| `baseline/humanoid21/{fight,follow,balance_recover,end2end,mocap}` | 各任务适配 | 待核实 | 可用性未知 |
| `policy/` | 预置策略 + blueprints | USABLE? | 哪些是有效资产待盘 |
| `examples/` `tests/` `scripts/` | 示例/测试/脚本 | 待核实 | — |
| combatbench.tech 平台 | 网站/后端/Elo | 范围外? | 属 combatbench-research，边界待确认 |

## 10. 待决问题（冻结前需确认）

1. **U 层杠杆目录的文件名与位置**（提案：`docs/LEVERS.md` 或根目录一份——
   需用户定夺）。
2. **A 层契约文档命名**：新约定 `USAGE.md` vs 沿用现有 `*SPEC.md` 惯例 vs
   其他——需统一后冻结。
3. **语言约定**：U 层双语（现有 README/docs 惯例）？A/D 层中文为主（现有
   CONTEXT/DESIGN 惯例）？
4. **combatbench.tech 平台侧是否在正则化范围内**（代码在 sibling 目录
   `combatbench-research/`）。
5. **Phase 1 盘点的优先顺序**：建议从"用户最先接触的杠杆"开始
   （experiment 注册表 → 策略族 → 探索杠杆 → reward/plugin → 环境 → 加速路径），
   待确认。
