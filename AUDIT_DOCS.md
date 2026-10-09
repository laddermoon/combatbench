# AUDIT_DOCS — 文档正则化审计台账

> 类型：记录

日期：2026-10-09
状态：进行中（逐项核实 → 建议 → 用户裁决 → 执行 → 追加结案记录）
姊妹文档：`AUDIT.md`（代码/行为审计，已收官）。本文件只管**文档系统本身**。

## 0. 目标与判据

目标（用户原意）：文档**全面**（有东西的地方有文档，AI 找得到）且**有规范**
（同类文档同构，生命周期可读，约定成文而非口口相传）。

审计判据（rubric）：

| 编号 | 维度 | 规范状态 |
|---|---|---|
| R1 | 生命周期标注 | 每份入库文档能一眼分辨：规范契约 / 活跃指南 / 设计记录 / 时点产物 / 历史留档 |
| R2 | 命名与类型约定 | 文档类型前缀（README/DESIGN_*/RESULTS_*/PLAN/CONTEXT…）有成文定义 |
| R3 | 私有/共享边界 | 哪些文档不入库、为何、惯例是什么——成文 |
| R4 | 覆盖度 | 每个活模块有入口文档；跨模块有地图 |
| R5 | 交叉引用 | 文档间引用不腐化（禁 file:line 或接受腐化并声明） |
| R6 | 双语同步 | en/zh 对有同步协议；单语文档显式声明语言 |
| R7 | 仓库机制一致性 | .gitignore 与文档惯例不自相矛盾、无静默陷阱 |

## 1. 文档清单盘点（基线事实）

**入库文档（git tracked）：128 份**。分布：

| 域 | 数量 | 构成 |
|---|---|---|
| 根目录 | 8 | README×2、CLAUDE、AUDIT、CAPABILITY_LEDGER、REGULARIZATION、ISSUES、V2_TRAINING_TIME_LOG |
| docs/ | 6 | RULE/ENVIRONMENT/SUBMISSION × en+zh |
| envs/framework | 6 | CONTEXT、DESIGN、README、RESET、SEED、tests/README |
| envs/humanoid21 | 7 | README、DATASPEC、CONTROLSPEC、OBSERVATION_zh、CONTACT_DESIGN、ACCEPTANCE_CRITERIA、tests/README |
| envs/batchframework | 31 | README、CONTEXT、DESIGN、SEMANTICS、PUBLIC_INTERFACE、MIGRATION_GUIDE、ROADMAP、CPU_REQUIREMENTS、BLACKBOARD_DESIGN、LIFECYCLE_TRACE、BATCHFRAMEWORK_AUDIT、discuss、R3_TASK_BRIEF、TERMINAL_FRAME_PLAN、E1-E8_PLAN、E7_BASELINE/E7_RESULTS/E8_SUPPORT_MATRIX、M0-M6 系列 |
| baseline/framework/ppo | 9 | GUIDE、README、4×DESIGN_*、FIXPLAN、RESULTS_、TODO_ |
| ppo/dumpkit | 8 | CONTEXT、DATA_FLOW、6×DESIGN_* |
| ppo/policies | 8 | 6×DESIGN_*、POLICY_SELECTION、2×RESULTS_* |
| ppo/policies/todo | 8 | CONTEXT_、DECISIONS、DESIGN_OVERVIEW、4×DESIGN_migration、TODO_ |
| baseline/framework/sac | 4 | PLAN、DECISIONS、IMPLEMENTATION_SUMMARY、DEBUG_PLAYBOOK |
| baseline/humanoid21 | ~15 | README、balance_recover×6、blueprints/README、curriculum×4、end2end×4、follow/README |
| experiments_ppo | 1 | README |
| policy / examples / scripts / tests | 4 | 各一个 README |

**不入库文档（.gitignore 隐式约定）：~18 份**——`bootstrip.md`×7、
`REVIEW_*.md`×6、`VALUE_PLAN.md`、`STANDUP_ORIG_*`×2、其余散件。
**这是已有的"私有工作文档"惯例**——但只存在于 .gitignore 注释，无成文政策。

**已存在的良性规范**（审计前就有，应沿用而非另起炉灶）：

- `CONTEXT.md` 单目录 AI 上下文惯例（monorepo 级，context-curator 维护）
- batchframework 的文档分层：PUBLIC_INTERFACE（契约）/ SEMANTICS（规格）/
  ROADMAP（索引）/ E*_PLAN（计划，含"状态："行）——**全仓最规范的文档域**
- RESULTS_*/E*_PLAN/REGULARIZATION 已有"日期+状态"行（9/128 份有显式状态）
- CAPABILITY_LEDGER 的能力状态词表（STABLE/USABLE/WIP/LEGACY/UNSUPPORTED/OUT-OF-SCOPE）

## 2. 发现项

### A 类：规范缺失（系统性）

**D-DOC-1｜无文档生命周期标注规范**
现状：128 份中仅 ~9 份有"状态："行，格式各异（"状态：**W0-W5 已实施**"/
"日期：…"/"v0 草案"）。AI/读者无法不看内容就区分"规范"与"时点产物"。
建议：定一个最小头部约定——`类型：`（Contract/Guide/Record/Artifact/Historical）
+ `日期：`。批量加头属机械工作，逐域过。

**D-DOC-2｜文档类型命名体系未文档化**
现状：前缀有机生长：`DESIGN_*`/`RESULTS_*`/`*_PLAN`/`MEMO_*`/`TODO_*`/
`FIXPLAN`/`CONTEXT`/`bootstrip`/`discuss`。不一致实例：`STEP_MEMO.md` vs
`MEMO_training_speed.md`；`bootstrip.md`（小写，私有笔记性质）。
建议：在文档政策文件里定义类型表（新文档按表命名），存量不强制改名。

**D-DOC-3｜私有/共享文档边界无成文**
现状：.gitignore 忽略 `bootstrip.md`/`REVIEW_*.md`/`VALUE_PLAN.md`/`_*/`/
`debug_*`——这是一个**事实存在的文档政策**（"工作笔记不入库"），但只写在
ignore 注释里，CLAUDE.md 和任何文档都没说明。
建议：写进文档政策（CLAUDE.md 或新 DOCS 政策文件），让 AI 不把它们当缺文档。

**D-DOC-4｜.gitignore 文档资产陷阱**
现状：(a) `obsolete` 裸模式——`baseline/framework/obsolete/` 内任何新文件
被静默忽略（实测 `check-ignore` 命中 L46）；(b) `*.png`/`*.json` 全挡，
265 个 PNG 靠 force-add  grandfathered——新文档配图/配数据会静默不入库。
建议：`obsolete` 改 `/obsolete/` 锚定或注释说明；文档资产在 .gitignore
注明"文档资产需 git add -f"或收窄模式。

**D-DOC-5｜无文档索引/路由器**
现状：CLAUDE.md 是事实上的 AI 入口但混合了"仓库规则+架构+CLI"三重职责；
CAPABILITY_LEDGER 记能力账不记文档账；"某问题该看哪份文档"无答案表。
建议：建 `docs/INDEX.md`（或并入 CLAUDE.md 一节）——按"我要做X→读Y"组织。

**D-DOC-6｜交叉引用腐化无防**
现状：64/128 份文档含 .md 引用；CONTEXT.md/DESIGN.md 大量使用
`file.py:NNN` 行号引用——行号会腐化（审计期间已有多处漂移被修）。
建议：约定"引用符号名不引用行号"（如 `simulator.py::get_core_state`）；
存量行号引用不强制修，新增禁用。

**D-DOC-7｜双语文档无同步协议**
现状：4 对 en/zh（README、RULE、ENVIRONMENT、SUBMISSION）+ OBSERVATION_zh
仅中文版 + DATASPEC 内联双语。RULE 对行数 81/87、结构微分叉；SUBMISSION
自 6 月未动（外部平台流程，本仓无法验证）。
建议：约定——docs/ 下用户-facing 文档必须双语同步更新；内部文档单语即可
（zh 为主）；每对顶部互相链接。

### B 类：覆盖缺口

**D-DOC-8｜`baseline/framework/` 根无 README**
ppo/、sac/、rollout/ 各自为政，无汇总层说明"统一训练框架是什么、algo 怎么分"。

**D-DOC-9｜`baseline/framework/rollout/` 零文档**
`Episode`/`Job`/`ParallelRollouter`/`EpisodeRecorder` 是 PPO+bench+评估三方
共用的 API 面，无任何文档——bench_rollout 三处过期正是这种"无契约文档"
的直接后果。优先级高。

**D-DOC-10｜`policy/` 子目录无 README**
`policy/random/`、`policy/humanoid21/`、`policy/blueprints/` 各零文档；
`policy/README.md` 存在但覆盖契约层。（P-POL-2 已修主体。）

**D-DOC-11｜`baseline/experiments_sac/` 无 README**
experiments_ppo 有；SAC 在飞，可随 SAC 成熟补，先标记。

**D-DOC-12｜`baseline/framework/ppo/tests/` 无 README**
framework/tests、humanoid21/tests 已有测试目录 README 惯例；ppo/tests 没有。

**D-DOC-13｜继承自 AUDIT.md Phase 4 D 档的缺失文档**
（当时裁决"暂不做"，本轮文档正则化应重新认领）：
- round_runner / match_runner CLI 参考
- blueprint YAML schema 文档（env/policy 两套）
- 训练路径地图（experiments_ppo→train.py→PPO GUIDE 的完整链路已散落在
  CLAUDE.md/GUIDE/README，无单页导图）
- `episode_options` 键目录（`video_output_path`/`impulse_params`/"episode"
  replay 键等——审计中一个个挖出来的隐性键，应有登记表）
- determinism/seed 担保边界文档（SEED.md 有框架层，训练层无）
- dumpkit 用户指南（CONTEXT+DATA_FLOW 是设计向，无"怎么用"）

### C 类：重复/一致性

**D-DOC-14｜`GATING_REDESIGN.md` 双份分叉**
`balance_recover/GATING_REDESIGN.md`（93 行）与
`balance_recover/gating/GATING_REDESIGN.md`（120 行）——后者含更新的
"策略驱动沉降"节，前者是过期快照。
建议：父目录版改为指针或直接删（内容已被子版覆盖）。

**D-DOC-15｜batchframework 31 份文档状态不全**
ROADMAP 是索引且声明"M0-M8 为历史协议"；但 E*/M* 各 artifact 的"状态："行
覆盖不全（E1_PLAN 有，其余待扫）。该域在大改中——建议**只加索引级标注**，
内容留给重构者。

**D-DOC-16｜SAC 与 PPO 文档体系不对称**
SAC：PLAN/DECISIONS/IMPLEMENTATION_SUMMARY/DEBUG_PLAYBOOK（无 GUIDE）；
PPO：GUIDE + DESIGN_*。SAC 在飞故可接受，标记即可。

**D-DOC-17｜根目录四份台账职责重叠**
`REGULARIZATION.md`（总纲）/`CAPABILITY_LEDGER.md`（能力账）/
`AUDIT.md`（代码审计）/`AUDIT_DOCS.md`（本文档）+ `ISSUES.md`（历史 issue 池）
——需在各自头部互相指路，声明"谁管什么"。ISSUES.md 已有 AUDIT 指路。

**D-DOC-18｜examples/out/ 生成产物**
`examples/out/06_evaluate_policy/match_report.md` 是运行产物（untracked）。
建议：`examples/out/` 加进 .gitignore，examples/README 声明 out/ 为输出目录。

## 3. 处置记录

（按工作流逐项追加：核实结论 → 用户裁决 → 执行 → 验证）

## [2026-10-09] D-DOC-1 结案：文档规范成文 + 全量头部标注

**处置**：新建 `DOCS.md` 仓级文档政策（类型词表 契约/指南/记录/产物/历史
五类 + 头部格式 + 命名约定 + 私有/共享边界 + 引用约定 + 双语协议 +
台账分工）；130 份入库文档全部插入 `> 类型：X` 头（英文面用 Type:，
含 docs/ 3 对 + README/CAPABILITY_LEDGER）；17 份历史类补"取代者"字段。
已有 `状态：`/`日期：` 行的文档（batchframework E*/M*、RESULTS_* 等）
保留原行，类型行加在其上方——类型（静态类别）与状态（动态进度）正交。

**连带解决**：D-DOC-2（命名表入 §4）、D-DOC-3（私有边界入 §3）、
D-DOC-6（符号引用约定入 §5）、D-DOC-7（双语协议入 §6）、D-DOC-17
（台账分工入 §8）的**成文一半**——政策已立，存量违规的逐处整改仍挂账。

**遗留**：`.gitignore` 陷阱（D-DOC-4）、doc index（D-DOC-5）、覆盖缺口
（D-DOC-8~13）、GATING 双份（D-DOC-14）等按序处置。
