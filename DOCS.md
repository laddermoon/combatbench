# DOCS.md — 文档规范（Documentation Policy）

> 类型：契约 ｜ 生效：2026-10-09

本文件是 CombatBench **文档系统的唯一规范**。新文档按本规范创建；
存量文档按 `AUDIT_DOCS.md` 的处置进度逐步对齐（不追求一次改完）。

## 1. 文档类型词表

每份入库文档属于且仅属于一种类型，类型写在标题行之下（见 §2）：

| 类型 | 含义 | 典型命名 |
|---|---|---|
| **契约**（Contract） | 规范性约束——实现以它为准，违反即 bug | `DATASPEC`/`CONTROLSPEC`/`SEED`/`RESET`/`SEMANTICS`/`PUBLIC_INTERFACE`/`ACCEPTANCE_CRITERIA`，docs/ 下的 RULE/ENVIRONMENT/SUBMISSION |
| **指南**（Guide） | 使用/上手指引、目录说明、AI 上下文 | `README`、`GUIDE`、`CONTEXT`、`MIGRATION_GUIDE`、`DEBUG_PLAYBOOK`、`POLICY_SELECTION` |
| **记录**（Record） | 设计推演、决策依据、计划书、评审记录 | `DESIGN_*`、`*_PLAN`、`DECISIONS`、`discuss`、`*_PROPOSAL` |
| **产物**（Artifact） | 时点性结果/日志/备忘——写定后不改，只追加 | `RESULTS_*`、`*_LOG`、`*_MEMO`、`*_HISTORY`、`*_BASELINE`、`*_RESULTS` |
| **历史**（Historical） | 已被取代但保留参考（墓碑） | 旧版 `TRAINING_V1` 之类；历史文档头部必须注明"取代者是谁" |

## 2. 头部格式

H1 标题行之后插入一行（英文面文档用英文标签）：

```markdown
# 标题
> 类型：契约            # 中文文档
> Type: Contract       # 英文文档
```

- **已有 `状态：`/`日期：` 行的文档保留原行**，`类型` 行加在其上方——
  `类型` 是静态类别，`状态` 是动态进度，两者正交共存。
- 历史（Historical）文档的头部**必须**写明取代者，如
  `> 类型：历史 ｜ 取代者：docs/X.md`。
- 产物（Artifact）文档若原无日期，头部补 `日期：`（取 git 首提交日期）。

## 3. 私有/共享文档边界

以下属**本地工作文档，不入库**（.gitignore 已挡，勿 force-add）：

- `bootstrip.md`（个人工作清单）
- `REVIEW_*.md`（架构复盘草稿）
- `VALUE_PLAN.md`、`*_DETAIL.md` 一类过程稿
- `_*/`、`debug_*` 目录内一切（调试产物）

惯例：**没入库的文档不存在**——不要把它们当文档缺口，也不要在
入库文档中引用它们的路径。要共享就先改名入库（去掉私有前缀）。
反之，入库文档默认面向"人 + AI"双读者，写时注意措辞可被索引。

## 4. 命名约定

- 目录入口文档统一 `README.md`；目录 AI 上下文统一 `CONTEXT.md`（monorepo 惯例）。
- 专题文档按类型词表用对应前缀；避免生造新前缀（`MEMO_*` vs `*_MEMO`
  以 `*_MEMO` 为准，存量不强制改）。
- 同一目录下不得有同名不同内容的文档（`GATING_REDESIGN` 双份事故）。
- 英文面文档（docs/、README.md）文件名用英文；内部文档不限语言。

## 5. 交叉引用约定

- 引用代码：**引用符号名，不引用行号**——`simulator.py::get_core_state`
  而非 `simulator.py:257`（行号随重构腐化，已有前车之鉴）。
  - **豁免**：记录/产物/历史类文档（AUDIT、DECISIONS、*_PLAN、RESULTS_*）
    的行号引用是**时点证据**（"当时代码长这样"），允许腐化、不回填。
    本规则约束的是契约/指南类活文档与新增内容。
  - 任何情况下**禁止绝对路径**（`/data1/...`）——一律仓根相对路径。
- 引用文档：相对路径 + 文件名即可；引用外部锚点写明文档名+小节名。
- 历史/挂起的引用对必须在两端之一标注状态（"见 X.md（历史）"）。

## 6. 双语协议

- `docs/` 下用户-facing 文档（RULE/ENVIRONMENT/SUBMISSION）必须 en+zh
  **同步更新**——一对文件同 commit 修改，结构对等（小节序号对应）。
- 仓内工程文档单语即可（以中文为主）；`README.md`/`README_zh.md` 同步维护。
- 每对双语文件在头部互相指认：`中文版本：README_zh.md` / `English: README.md`。

## 7. 仓库机制

- `.gitignore` 的 `*.png`/`*.json`/`obsolete` 模式会静默挡掉文档资产与
  obsolete 目录内新文件——需要入库时用 `git add -f` 并在提交信息中注明。
- 生成产物目录（`runs/`、`examples/out/`、`_*/`）不入库；工具文档里
  应声明输出目录归属。

## 8. 台账分工（谁管什么）

| 文件 | 职责 |
|---|---|
| `DOCS.md` | 本文档规范（本文件） |
| `DOCS_INDEX.md` | 文档路由器：按任务找文档 |
| `REGULARIZATION.md` | 正则化总纲：范围、阶段、验收标准 |
| `CAPABILITY_LEDGER.md` | 能力账：每个能力的状态+证据+缺口 |
| `AUDIT.md` | 代码/行为审计台账（逐条问题→处置记录） |
| `AUDIT_DOCS.md` | 文档审计台账（本文档规范的落实进度） |
| `ISSUES.md` | 历史 issue 池（已被 AUDIT.md 接管，只读） |
| `CLAUDE.md` | AI 入口：仓库规则 + 架构速览 + CLI；详表见各专文 |
