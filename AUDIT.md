# CombatBench Audit Report

Append-only audit report for the regularization effort (see `REGULARIZATION.md`
§7.3). One entry per directory pass or per notable finding. Negative results
("checked X, it's fine") count — the report must show what was inspected,
not only what was changed.

**Audit discipline (Phase 1):** record-only. No edits or deletions of existing
code/docs; findings are *suggestions* for the user to decide. New
`test_audit_*.py` files may be written to prove/expose problems.

Entry format:

```
## [YYYY-MM-DD HH:MM] Action

**Domain/object:** <capability/dir/file>
**Category:** discovery | code-fix | doc-change | status-change | spec-change
**What:** <the action itself>
**Why:** <motivation; for bugs: symptom + root cause; for renames: rationale>
**Result/evidence:** <test output, command result, or commit ref>
**Next:** <follow-ups / ledger rows touched>
```

---

## [2026-10-01] REGULARIZATION.md v0.2 — charter drafted

**Domain/object:** `REGULARIZATION.md`
**Category:** doc-change
**What:** Created the regularization master charter; converged scope to a
per-directory README pass (intuitive → macro → micro, English) that doubles
as the capability audit.
**Why:** Long-running effort needs a frozen mandate; user directed to collapse
open questions and start with READMEs.
**Result/evidence:** commits `550927ee`, `3e4518cb` (+ follow-up edits).
**Next:** Create ledger + this log; start Phase 1 with `envs/framework`.

## [2026-10-01] Ledger + audit log created

**Domain/object:** `CAPABILITY_LEDGER.md`, `AUDIT.md`
**Category:** doc-change
**What:** Created both root files. Ledger seeded with SAC = OUT-OF-SCOPE only.
**Why:** Phase 0 deliverables; all future discoveries land here.
**Result/evidence:** files committed.
**Next:** First directory pass — `envs/framework`.

## [2026-10-01] Audit rule added: record-only, no fixes

**Domain/object:** `REGULARIZATION.md` §7.1
**Category:** spec-change
**What:** Added Phase 1 hard constraint — no edits/deletions of existing code;
findings go to this log as *suggestions* for the user to decide. New test
files may be written; existing tests are not "fixed" during audit.
**Why:** User directive — audit must not mutate the codebase it audits.
**Result/evidence:** charter updated.
**Next:** apply to all subsequent directory passes.

## [2026-10-01] envs/framework — audit findings

**Domain/object:** `envs/framework/`
**Category:** discovery
**What:** Ran `pytest envs/framework/tests/ -q`: **157 passed, 3 failed,
5 collection errors**. Root cause traced: the runner layer was deliberately
refactored (`73fe8da3 "refactor round runner"` removed `parallel_runner.py`;
`07072439` removed `runtime_plugin.py`; `EpisodeRunner` is now a thin loop
that returns `None` — data capture moved to `PostActionRecorder`). Stale
artifacts of the OLD API remain:

- `CONTEXT.md` — describes `parallel_runner.py`, `runtime_plugin.py`,
  `ObserverBinding`/`RolloutConfig`/`AgentTrajectory`/`EpisodeResult`/
  `run_n_episodes`, and "RoundRunner = thin subclass of EpisodeRunner"
  (RoundRunner is now a standalone blueprint-based class). **Largely stale.**
- `tests/test_episode_runner.py` — imports 6 removed names; tests old API.
- `tests/test_parallel_runner.py` — tests deleted `parallel_runner.py`.
- `tests/test_policy.py` — imports removed `call_policy`/`coerce_action`;
  helper policies use old `act(obs)->array` signature (now
  `act(obs, want_extra)->(action, extra)`).
- `tests/test_reset_chain.py` — imports `ObserverBinding`/`RolloutConfig`;
  one test uses removed `run_n_episodes`; most runtime invariants (I1–I6,
  G4, G5) still valid but file can't collect.
- `tests/test_seed.py` — imports `EpisodeSeeds`(now `_EpisodeSeeds`)/
  `_derive_batch_seeds`/`_parallel_derive_seeds`; `runner.policies` dict gone.
- `tests/test_video_recorder.py::TestRoundRunnerVideoSavePath` — 3 tests fail:
  `RoundRunner._merge_video_path_into_options`/`videosave_path` kwarg removed.
- `envs/framework/episode_runner.py` docstring (line ~46) still says
  "cross-process orchestration is handled by ``parallel_runner``" — file gone.
- `envs/framework/README.md` — stale (documents `runtime_plugin.py` section
  and old runner semantics). A rewritten README was drafted then **reverted**
  per user decision — doc-writing paused; audit only.

**Why:** Documents which capabilities/tests are real vs. ghost; feeds Phase 2
task pool.
**Result/evidence:** `pytest` output; git log `73fe8da3`, `07072439`;
`episode_runner.py` docstring "Non-responsibilities" confirms removal was
deliberate.
**Next — SUGGESTIONS (user decides):**
- S1: delete `test_episode_runner.py`, `test_parallel_runner.py` (dead APIs;
  parallelism now lives in `baseline/framework/rollout/`).
- S2: fix `test_policy.py` — drop `TestCoerceAction`/`TestCallPolicy`, update
  `act` signatures to tuple form.
- S3: fix `test_reset_chain.py` — new `EpisodeRunner(runtime, policy_a,
  policy_b)` ctor; drop `run_n_episodes` test; rewrite `TestI3` (recorder-based).
- S4: fix `test_seed.py` — import `_EpisodeSeeds`; drop `TestDeriveBatchSeeds`
  + parallel parts; `runner.policies` → `policy_a`/`policy_b`.
- S5: delete `TestRoundRunnerVideoSavePath` (removed API).
- S6: rewrite or retire `CONTEXT.md` — it's the stalest doc; its accurate
  gotchas should live in README micro section or a fresh CONTEXT.md.
- S7: fix `episode_runner.py` docstring `parallel_runner` mention (line ~46).
- S8: new thin-runner behaviors (`post_termination_action="hold"`,
  `want_extras` forwarding, duck-type policy check) have NO tests —
  candidates for new test file.

## [2026-10-01] Pivot: docs paused, audit-only mode; log renamed AUDIT.md

**Domain/object:** `REGULARIZATION.md` §7.1, `REGULARIZATION_DETAIL.md`,
`envs/framework/README.md`
**Category:** spec-change
**What:** User rejected the rewritten `envs/framework/README.md` ("says
everything, says nothing") and paused all doc-writing. Phase 1 redefined as
**pure audit**: directory-by-directory inspection, findings appended to this
report. `REGULARIZATION_DETAIL.md` → `AUDIT.md` (this file is now the audit
report itself). Stale-test fixes in `test_policy.py` and the README rewrite
were fully reverted to the pre-audit state.
**Why:** Audit findings must precede doc design; writing docs before
understanding the codebase produced low-value output.
**Result/evidence:** `envs/framework/README.md` restored from `dc9a2d06^`;
`test_policy.py` restored to HEAD; working tree clean except the three
bookkeeping files.
**Next:** formal Phase 1 starts at `envs/framework` (this entry's findings
stand as its first record).

---

> 自此以下审计日志一律用中文书写。

## [2026-10-01] envs/framework —— 逐文件深审（第一轮完整盘点）

**Domain/object:** `envs/framework/`（全部 14 个 .py + 5 个 .md + tests/）
**Category:** discovery
**What:** 逐文件精读全部源码，对照文档逐项验证契约，并新增 1 个审计探针
测试文件 `tests/test_audit_terminal_frame.py`（2 个用例，均通过——其中
一个用例**证实了一个真实的语义问题**）。当前测试基线：
**159 passed / 3 failed / 5 collection errors**（新增 2 个测试算入
passed；基线仍是 157+3+5，口径不变）。

### 通过项（查过且确认没问题）

- `backend.py` —— `IDataAccessor`/`IDataMutator`/`BaseSimulator` 契约干净；
  `get_observation`/`get_action` 已在 accessor 契约内。
- `context.py` —— `_AccessorView`/`_MutatorView` 白名单代理真实有效：
  `__slots__` + `__setattr__` 封锁 + `__getattr__` 白名单转发 +
  `__sim` 名字改写，sandbox 不是摆设。
- `plugin.py` —— 生命周期钩子文档与实际派发一致；`require_mutator`/
  `priority` 语义清晰；`on_attach`/`on_detach` 契约完整。
- `observer_plugin.py` —— dispatcher 优先级常量 `OBSERVER_DISPATCHER_PRIORITY`
  集中暴露；`BaseObserverPlugin` 现为 `BaseRuntimeUnit` 纯别名（向后兼容，
  文档有说明）；去重 token 机制是显式契约（Note C2）。
- `episode_runner.py` —— 薄 runner 语义清晰：seed 派生链
  （SeedSequence.spawn：runtime + 2 policy + seedable plugins）、
  `post_termination_action`（policy/hold）、`run_episode` 返回 None、
  duck-type policy 校验。`_reset_all` 顺序正确（plugin 先 reseed，
  再 runtime.reset，最后 policy.reset）。
- `blueprint.py` / `parameterized_blueprint.py` —— ClassSpec 往返协议
  干净；`BLUEPRINT_EXCLUDE` 过滤逻辑正确；参数化替换的全引用/内联两种
  规则实现与文档一致。
- `round_runner.py` / `match_runner.py` —— 薄组装层；options 合并逻辑
  正确（caller options 优先）；CLI 完整。
- `replay.py` —— ReplaySimulator 实现完整：read-only 违例抛
  `ReplayReadOnlyError`（`set_action` 刻意静默是文档化契约）、
  manifest_version>=2 门控、单集/目录两种模式、dtype 再水化。
- `recorder.py` —— `EpisodeBufferRecorder`/`BaseFrameRecorder` 职责清晰，
  on-disk schema（manifest_version=2）与 replay.py 对齐。
- `recorder_viewer.py` + `_recorder_viewer.html` —— bundled asset 存在，
  HTTP viewer 正常。
- `__init__.py` —— 所有导出符号均存在且可导入（逐一核对）。
- `common_plugins.py::TimeoutPlugin` —— 在 on_post_action_step 中判终止，
  与 episode_step 递增时序一致（已被审计测试的对照用例证实）。

### 发现的问题

**P-FW-1（文档大规模漂移，`get_termination_flags` 不存在）**
- 现象：`runtime.get_termination_flags()` 在 **6 处文档**被当作公共 API
  示例：根 `README.md:59`、`README_zh.md:59`、`CLAUDE.md:195`、
  `envs/framework/README.md:173`、`envs/framework/CONTEXT.md:159`、
  `envs/framework/DESIGN.md:178-184`。
- 实情：`EnvRuntime` 上没有这个方法。真实的终止查询接口是
  `is_episode_over()` / `is_agent_active(aid)` / `get_agent_termination()` /
  `is_episode_active`。
- 证据：`grep -rn get_termination_flags` 命中全部是文档，无一处实现。
- 建议：全局替换为真实接口；这是"用户/AI 读文档照抄必炸"级问题。

**P-FW-2（语义问题，已被测试证实）：物理步中途终止 → 终止帧 observer 输出陈旧 + 落盘帧文件名碰撞**
- 现象：`_RuntimeCore.step` 在 `on_pre_phy_step`/`on_post_phy_step` 中
  检测到全员终止时**提前 return**，跳过了 `on_post_action_step` 钩子，
  因此 observer dispatcher 不会为这最后一步刷新 observer 输出。但
  `EnvRuntime.step` 仍然触发 recorder 的 `on_post_action_step`——
  recorder 收到的 `observer_outputs` 是**上一步的陈旧值**。
- 次生问题：终止帧里 `ctx.episode_step` 未递增（与上一帧相同），
  `BaseFrameRecorder` 用它做文件名 → `step_XXXXX.json/png` **被覆盖**，
  落盘录制少一帧。
- 影响：KO/倒地这类在物理循环中产生的终止，其"致命一击"那一步的
  reward observer 输出不会被计入；`EpisodeBufferRecorder`（训练侧数据来源）
  最后一帧 observer_outputs 是旧值。
- 证据：`tests/test_audit_terminal_frame.py`（2 passed）：
  kill_at=15, phy_steps=10 → recorder 收到 2 帧，observer 只刷新 1 次，
  第二帧 observer_outputs == 第一步的旧值；对照组（TimeoutPlugin 在
  on_post_action_step 终止）行为正常。
- 文档矛盾：`recorder.py` 模块 docstring 声称"recorders always run
  after the observer dispatcher has refreshed"——物理中终止时不成立。
- 建议：需决策——是设计意图（终止帧不做全量刷新）还是 bug；
  至少要修 docstring 或让 `_RuntimeCore` 在早退前补一轮 observer 刷新。
  文件名碰撞则需要 recorder 用单调序号而非 episode_step 命名。

**P-FW-3（Policy 契约双轨并存，测试测的是已删除 API）**
- 现状契约（policy.py + EpisodeRunner + 所有已部署 policy）：
  `act(obs, want_extra=False) -> (action, extra)` 二元组。
- `tests/test_policy.py`（HEAD 与工作区一致）测试的是**旧契约**：
  `act(obs) -> action`、`act_with_extras()`、`call_policy()`、
  `coerce_action()`。后两个函数在 `27fd9add` 引入、`029b18af`/
  `7fd36063` 重构期删除，测试从未同步。
- `policy/README.md:54,127` 文档写的也是旧契约 `act(obs) -> ndarray`。
- 建议：确认新契约是终态后重写该测试文件；CONTEXT.md:63 的
  `coerce_action`/`call_policy` 描述同步删除。

**P-FW-4（沙箱可被绕过）：`ctx._simulator` 直通裸模拟器**
- `SimContext.__init__` 里 `self._simulator = simulator` 是普通属性，
  插件在只读钩子中可通过 `ctx._simulator.set_core_state(...)` 绕过整个
  accessor/mutator 授权机制。`_AccessorView` 用名字改写防 `._simulator`，
  但 ctx 自己就把裸对象挂在外面。
- 可能是刻意留的逃生口，但未文档化；与"沙箱真实"的设计宣称矛盾。
- 建议：确认是有意逃生口（写进文档）还是漏洞（改成名字改写）。

**P-FW-5（静默失败，违 fail-loud）**
- `IDataMutator.apply_external_force` 默认实现是 `pass`——后端没实现时
  调用静默成功（backend.py:71-91，注释说是"可选实现"）。
- `VideoRecorderPlugin.on_post_episode` catch 所有异常只 print——
  strict=True 下视频保存失败也不抛出。
- `BaseFrameRecorder._safe_accessor_call` 把 accessor 异常吞成
  `{"__error__": ...}`——有文档说明，属刻意防御，但值得知晓。
- `_safe_call` 的 strict=False 日志格式串 `"%s '%s' failed at %s"`
  传参是 `(label, hook_name, label)`——log 文案打出来是
  `Plugin 'x' 'hook' failed at Plugin 'x'`，语义错误（应为
  `(label, hook_name, hook_name)` 之类）。小问题。

**P-FW-6（测试现状，与前次记录一致并复核）**
- 5 个 collection error 文件确认根因是 runner 重构遗留：
  `test_episode_runner.py` / `test_parallel_runner.py` /
  `test_policy.py` / `test_reset_chain.py` / `test_seed.py`。
- 3 个 fail：`test_video_recorder.py::TestRoundRunnerVideoSavePath`
  （旧 `videosave_path`/`_merge_video_path_into_options` API 已删，
  现机制为 `episode_options["video_output_path"]`）。
- 新薄 runner 行为无测试覆盖：`post_termination_action="hold"`、
  `want_extras` 转发、duck-type 校验。
- `tests/README.md` 测试清单表格未列 `test_seed.py` 等已存在文件，
  也滞后。

**P-FW-7（文档过期清单，本轮新增确认）**
- `DESIGN.md:150` `ctx.termination_proposals: List[str]`——现已改为
  per-agent `agent_termination_proposals` + `agent_terminated`。
- `SEED.md:56,130-131` 引用 `run_n_episodes`/`parallel_runner._derive_seeds`。
- `RESET.md:199,286` 引用 `run_n_episodes(options_fn=...)`。
- `episode_runner.py:46` docstring 引用已删的 `parallel_runner`。
- `policy.py:5` 模块 docstring 引用 `ParallelRunner`。
- `REVIEW_SUMMARY.md` 是历史评审文档（自己也记录了部分过期问题），
  属考古资料，建议保留但标记。

**P-FW-8（小问题）**
- `EpisodeBufferRecorder.get_episode_data` docstring 的 frame schema
  漏列了实际会写入的 `"observation"` 键。
- `EnvBlueprint.from_runtime` 遇多个 TimeoutPlugin 时后者静默覆盖
  `max_steps_from_plugin`（边界情况，通常不触发）。
- `EnvBlueprint.load` 不做 `${DIR}` 替换，而 `PolicyBlueprint.load` 做
  ——不对称；env blueprint 的 config 无法引用同目录资源文件，
  可能是有意也可能是遗漏。
- `env_runtime.py` 顶部 TODO(framework/B2) 记录了 batched runtime
  的规划债（与 batchframework 对应，属已知）。

**Result/evidence:** 上述逐条对应源码行号与测试输出；
`pytest -q --ignore=<5个stale文件>` = 159 passed / 3 failed。
**Next:** envs/framework 审计暂告一段落（建议项交用户裁决）；
下一目录 `envs/humanoid21`。

## [2026-10-01] envs/humanoid21 —— 逐文件深审（模拟器 + 插件 + 扰动族）

**Domain/object:** `envs/humanoid21/`（simulator 1381 行、plugins 612、
observer_plugins 602、disturbance_plugins 1551、meta 558、3 个 XML、
7 个测试文件 + 资产目录）
**Category:** discovery
**What:** 逐文件精读；跑通现有测试 **43 passed / 0 failed / 19.8s**；
新增审计探针 `tests/test_audit_stale_contacts.py`（2 用例均通过——
证实了一个缓存失效缺口）；手动调用了 5 个验收测试的内部逻辑拿到
pytest 看不到的真实结果。

### 通过项

- `simulator.py` —— 96 维观测拼装与 OBSERVATION_zh.md 布局一致；
  PD 控制向量化路径与标量路径按位等价（注释说明）；`set_core_state`/
  `get_core_state` 的机体系↔世界系速度转换对称；blueprint 往返正常；
  `to_blueprint` 捕获全部构造参数。
- `meta.py` —— Humanoid21Meta 静态参数单一数据源 + 模型加载时校验
  （`validate(model)` 在 `__init__` 中执行，fail-loud）。
- `plugins.py::CombatScoringPlugin` —— 每物理子步算伤害
  （quadratic threshold + part weight + dt），KO 判定在
  `on_post_action_step`；`priority = OBSERVER_DISPATCHER_PRIORITY + 1`
  保证 observer 读到本步伤害——与 observer_plugin.py 文档约定一致。
  score_log_file 每回合可通过 episode_options 覆盖，句柄复用正确。
- `plugins.py::FrozenRobotPlugin` —— 状态冻结实现正确。
- `disturbance_plugins.py` —— 12 个扰动/状态池类均有
  `set_episode_seed`（接 EpisodeRunner 派生链）+ `require_mutator` 声明；
  被 `baseline/experiments_ppo/exp_standup*`/`exp_step` 等活实验引用——
  真实可用能力，非死代码。
- `observer_plugins.py::Humanoid21BalanceAnalysisObserver` —— 完整的
  去脚质心/双踝支撑几何分析 + plan-view 渲染，公开接口只走
  accessor（合规），规划视图为自绘像素图。
- `blueprint.yaml` —— 参数化蓝图（`initial_distance`/`max_steps` 旋钮），
  引用的类全部存在。
- `ReplaySimulator`/contacts SoA/`get_derived_state(fields)` per-field
  缓存——设计一致。

### 发现的问题

**P-H21-1（真实 bug，测试证实）：`_cached_contacts_vec` 在 physical_step 后不失效**
- `simulator.py` 有两套缓存：`_data_cache`（physical_step/reset/
  set_action 时清空）和 `_cached_contacts_vec`（只在 reset 清空，
  `get_derived_state(['contacts'])` 时写入，**physical_step 不清**）。
- `_get_feet_forces` 优先读 `_cached_contacts_vec` → 若上一动作步有
  人取过 contacts（恰好 CombatScoringPlugin 每子步都取），本步观测的
  feet_forces（96 维中的 2 维）用的是**上一步的接触数据**。
- 缓解：挂 CombatScoringPlugin 时它每 post_phy_step 都调
  `get_derived_state(['contacts'])` 刷新 cv，所以标准战斗蓝图下被掩盖；
  但任何不持续取 contacts 的配置都会中招。
- 证据：`test_audit_stale_contacts.py` 2 passed——① step 后缓存对象
  还是同一个（未失效）；② 4000N 上推一物理步后，stale 路径读数与
  fresh `_extract_contacts` 重算显著不一致。
- 建议：`_cached_contacts_vec` 并入 `_data_cache` 统一失效（一行修复，
  但留给用户裁决）。

**P-H21-2（真实 bug，静态证据确凿）：`CombatScoringObserver` 读的是
`metrics['events']`，而击中事件写在 `ctx.events`**
- `plugins.py:508-514` 往 `ctx.events` append hit 事件；
  `observer_plugins.py:582` 读 `metrics.get("events", [])`——
  全仓库没有任何地方写 `metrics['events']`。
- 后果：observer 输出的 `events`/`step_hit_events`/`step_damage_taken`
  **恒为空/0**；`health`/`cumulative_damage_taken`/`is_ko` 正常（走 metrics）。
- 另注意：`ctx.events` 整局累计、不清步——即便改读 ctx.events 也仍不是
  "本步"语义，需要另行按步截断/清空。
- 建议：改读 `ctx.events` 并定义清步语义（谁清、何时清需要决策）。

**P-H21-3（死插件，静默失效）：`NonFallConstraintPlugin` 对当前
simulator 是 no-op**
- 它读 `static_data['robot_info'][robot_id]` 和
  `static_data[robot_id]['norm_params']`——当前 `get_static_data()`
  的 schema 里**两个键都不存在**（norm_params 是 simulator 私有
  `_norm_params`）。`robot_info` 为空 dict → 循环 continue → 插件
  什么都不做，也不报错。
- 全仓库（含 blueprints）无任何引用。疑似按旧版 static_data schema
  写的遗留。
- 建议：删除或重写；至少文档标记不可用。属"fail-silent 违例"典型。

**P-H21-4（测试形同虚设）：7 个测试无法失败，且验收标准当前就未达标**
- `test_acceptance.py` 5 个用例（tracking_error / jump /
  response_latency / zero_oscillation / absolute_stability）全部
  `return {'pass': ...}` 而**不 assert**——pytest 永远判过。
- 手动调用拿到真实结果：**tracking_error=False, response_latency=False,
  zero_oscillation=False**（站姿 PD 控制的跟踪误差/延迟/振荡验收当前
  不达标）；jump=True、absolute_stability=True。
- `test_observation_symmetry.py` `return True/False` 不 assert
  （当前实际 True）。`test_data_interfaces.py::test_static_data`
  return simulator 同样虚过。
- `test_videos/` 12 个 mp4 是这些测试 `record_video=True` 默认值的
  历史产物，已提交进仓库。
- 建议：acceptance 用例改成真 assert（这样会把"43 passed"变成
  3 failed——**这正是审计要的诚实信号**）；或标记为 manual/benchmark。

**P-H21-5（README 多处过期）**
- 引用 `rule_blueprint.yaml`，实际文件叫 `blueprint.yaml`。
- CLI 示例用 `--blueprint/--policy-a/--policy-b`，实际参数是
  `--env-blueprint/--policy-a-blueprint/--policy-b-blueprint`。
- 结果示例 `termination_reasons: ['timeout']`——实际是 per-agent dict。
- 末尾引用 `SPEC.md`——不存在（实际是 DATASPEC/CONTROLSPEC/
  OBSERVATION_zh）。
- 目录结构图没列 `disturbance_plugins.py`/`meta.py`/3 个 XML。

**P-H21-6（资产盘点）**
- 当前生效 arena：`battle_circular_v2.xml`（`ARENA_XML` 硬编码；
  condim=3、impratio=10、24 段圆墙）。
- `battle_v1.xml`/`battle_v2.xml` 不再是默认路径，仅被
  `scripts/migrate_feet_forces_norm.py`、`baseline/humanoid21/mocap/`、
  `balance_recover/gating/debug_mujoco_reset.py` 等旧代码引用；
  `REVIEW_SUMMARY.md` 还在说"battle_v1 是当前模型、battle_v1_new 待
  切换"——`battle_v1_new.xml` 已删，整篇是考古文档。
- `CLAUDE.md` 目录结构也只列 v1/v2，没提 circular_v2。
- `obs_analysis/`、`pose_images/`：分析产物/姿态配图，留作档案合理，
  但属非代码资产。

**P-H21-7（小问题集合）**
- `simulator.py:3` `os.environ['MUJOCO_GL'] = 'egl'` 硬覆盖（不是
  setdefault），而下一行 PYOPENGL_PLATFORM 用 setdefault——不一致，
  且 import 时改全局环境对嵌入方不友好。
- `reset(seed=...)` 的 seed **从未被使用**（mj_resetData 确定论）——
  框架种子契约承诺 simulator 消费 seed，这里静默丢弃；当前无 RNG
  所以无害，但任何未来随机初始化都不会有种子效果。
- `get_sensor_data()` 恒返回 `{}`（DATASPEC 没定义传感器——契约内 stub，
  记录为事实）。
- `get_broadcastview_image` catch 所有异常 → `warnings.warn` + 返回
  全黑 720×1280——渲染失败在录制里静默变黑帧。
- 相机代码硬编码 `arena_radius = 3.44`（与 circular XML 耦合，改 XML
  半径就错）。
- `DAMAGE_TARGET_PARTS` 含 `waist_upper/waist_lower`，但
  `_get_part_category` 永远只产出 `torso`——两个死枚举值。
- 伤害力有 `min(force, 1200)` 上限——docstring 未写，只在代码里。
- `plugins.py:266` 用 `while len(ctx.events)>0: pop()` 清列表
  （等价 clear()，风格怪但无害）。

**Result/evidence:** pytest 43 passed（其中 7 个不可失败）；
`test_audit_stale_contacts.py` 2 passed；手动调 acceptance 函数输出
`{'pass': False}`×3。
**Next:** `envs/batchframework`。

## [2026-10-01] envs/batchframework —— 设备批量路径（Warp/MJX，M0–M6 进行中）

**Domain/object:** `envs/batchframework/`（~7.8k 行 py：两套契约
`batch_plugin/batch_context`（numpy 原型）+ `device_*`（torch 设备路径）、
`mjx_simulator` 1063 行、`warp_simulator` 642、`validation*` 治具体系、
`capability_registry`、`device_rollouter`、12 份 M 系列里程碑文档、
3 个 probe 脚本、34 个 validation_fixtures）
**Category:** discovery
**What:** 逐文件结构核查 + 在 4090 上**实跑**了验证链路与测试套件。
这是目前审计过工程质量最高的目录：digest 治具门禁、能力注册表
fail-closed、每阶段 plan/results 文档链完整。

### 架构实况（与 ROADMAP/discuss.md 一致）

- 两代契约并存且是**有意设计**：`batch_plugin.py`/`batch_context.py`
  （numpy BatchSimContext）是 legacy 兼容层的契约载体——`host_compat.py`
  的 `_CachingSimProxy`/`HostBatchCompatAdapter`/`LegacyPluginAdapter`
  把旧插件桥接进设备运行时，ctx 逐字复用 batch_context——**不是死代码**。
- 现行主干是 `device_*`（torch）：`DeviceBatchState`/`BatchRuntime`/
  `BaseDevicePlugin`/`DeviceCtx`；`device_rollouter.py::DeviceRollouter`
  被 `baseline/framework/ppo/loop.py` 引用——设备 rollout 已接入
  训练主路径作为 collector 选项。
- `capability_registry.py`：插件/observer 显式登记 NATIVE/COMPAT/
  HOST_SLOW/UNSUPPORTED，未注册即启动失败——discuss.md 的
  "不猜测映射、不静默回退"红线在代码里真实落实。
- 注册表现状：NATIVE 仅 3 项（DeviceTimeoutPlugin、
  RandomFallenStatePlugin→DeviceFallenResetPlugin、
  StandingBalance4StageRewarder→DeviceStandup4StageRewarder）+
  1 项 UNSUPPORTED（StandupTerminationPlugin）。**即 CPU 侧几十个
  插件/observer 中目前只有 3 个有设备原生实现**——转换面还很大，
  这是"加速路径覆盖面"的真实水位。
- `validation.py` 治具体系：capture（CPU 参考捕获，含 50 个源文件
  sha256 + 环境指纹）→ digest → replay（任一候选适配器逐字段比较，
  支持 per-field 容差）。stale 检测是真正的 fail-closed 门禁。

### 实测结果

- `pytest tests/`（7 个 batchframework 文件，GPU1）：
  **64 passed / 1 failed / 34min**（耗时大头是 warp 内核 JIT 首编译）。
- **实跑 MJX 对照**：现场 capture `dyn-standing-s25`（当前代码）→
  MJX replay = **PASS**（1e-5 容差内 0 failure）。
- **实跑 Warp 对照**：同 fixture = 6 个 contact 力字段超 1e-5
  （max ~2e-3 abs / ~1.5e-5 rel @139N）。**这不是回归**——
  M2_RESULTS 已记录 warp 仅 fp32，其专属容差为 {atol 2e-2, rtol 1e-3}
  （实测 2e-3 << 2e-2，在声明范围内）。
- **唯一失败 `test_mjx_validation.py::test_m2_cross_backend_fixtures`
  （该文件最后一个测试）**：committed fixtures 的源指纹与当前代码不符
  （4 个文件已变：`experiments_ppo/base.py`、`ppo/experiment.py`、
  `ppo/sampling_context.py`、`rollout/job.py`）→ replay 全部判 stale。
  门禁按设计工作，但意味着**整条跨后端 fixture 验证链当前对任何
  修改过这些文件的提交都是红的**——需要 `make-fixtures` 重新捕获 +
  重新批准 digest 才能恢复绿灯。
- MJX 适配器按设计拒绝两类 case：reward/trajectory（"host-side oracle"）
  和 action_sequence 非 dynamics 输入（"无 mjSTATE blob"）——
  显式 unsupported 而非假装支持，正确。

### 问题与观察

**P-BF-1（流程风险，非代码 bug）：fixture 批准滞后于代码演进**
34 个 committed fixtures 的整体 stale 说明"代码变了 → 治具重捕获 +
人工批准"这一步没有跟上主路径变更节奏。ROADMAP 风险表自己也写了
"主路径变更后加速实现静默过期→stale 检测"——检测有效，但**响应靠
人工**。建议：文档化"哪些源文件变更必须触发 fixture 重批准"的清单
（现在 50 个指纹文件谁来背锅不明确），或在 CI 定期跑 replay 告警。

**P-BF-2（组织）：测试与源码分离且命名无归属**
batchframework 的 7 个测试文件住在仓库根 `tests/`（该目录还混着
`test_stage_seg_rewards.py`（属 baseline curriculum）和
`debug_fall_images/` 产物目录）。模块内无 tests/，新人/AI 不易发现
归属关系。建议：README/CONTEXT 里明确指向根 tests/ 的归属清单，
或迁回模块内。

**P-BF-3（小）：`probe_*`/`validation_*`/`m6_compare` 是里程碑手脚架**
probe_e2e_jax/warp（M2）、probe_standup_xeval（M4）、m6_compare（M6
对照工具）都是一次性验证脚本，留档合理（M 文档引用它们），但无
README 说明各自何时该跑。`m4_t4_results.json` 是 probe 的数据产物
（默认输出路径就指它）。

**P-BF-4（状态核对）**：M6（learning-equivalence pilot）有 M6_PLAN +
m6_compare.py 但**无 M6_RESULTS.md**——按 ROADMAP 语义 M6 未结题；
计划中的 ~28h 训练臂是否跑过需向用户确认（runs/ 下应有
m6_pilot_* 目录可查）。

**能力入账**：BatchRuntime+device 插件体系（USABLE，fake_backend
可 CPU 侧测生命周期）、DeviceRollouter（USABLE，已接 PPO collector）、
WarpHumanoid21Simulator（USABLE@fp32 容差）、mjx_simulator
（USABLE@1e-5）、validation 治具体系（STABLE，fail-closed 实测有效）、
capability_registry（STABLE 设计，覆盖面待扩）、host_compat 桥
（USABLE，COMP/HOST_SLOW 路径）。

**Result/evidence:** pytest 64 passed/1 failed（stale gate）；
dyn-standing-s25 现场对照 MJX pass / Warp 6 字段超 fp64 容差
（fp32 容差内）；注册表静态核查。
**Next:** `baseline/framework`（最大的一块）。

## [2026-10-01] baseline/framework —— 训练框架（PPO 主干 + dumpkit + rollout）

**Domain/object:** `baseline/framework/`（~50k 行 py 不含 sac：train.py、
code_snapshot、critic_mlp、`ppo/`（experiment/loop/trainer/trajectory/
algos/policies×8/dumpkit/tests）、`rollout/`（ParallelRollouter/
inference_server/remote_policy/episode/job）、`obsolete/`）
**Category:** discovery
**What:** 测试全量实跑 + 关键路径静态核查。**597 passed / 6 failed /
2+3 collection errors / ~6min**——6 个失败全部证实为**测试过时**
（实现是有意改动），4 个 collection error 是候选设计占位和私有名漂移。

### 通过项（测试覆盖很好）

- `ppo/trainer.py`（2.2k 行实现 + 3.8k 行测试）、`loop.py`、
  `experiment.py`/`CommonParams`/`PPOParams`、GAE algos、trajectory——
  多 critic 主干测试密集且全绿。
- `ppo/policies/` 8 个生产策略族全部有独立测试文件且通过
  （truncated_normal / bounded_std / state / pre_tanh / mixture /
  state_mixture / shared_mixture / shared_mixture_bounded_std），
  配 `_export_template_*.py` 导出模板 + POLICY_SELECTION.md 选型文档——
  策略族是**文档与测试双达标**的区域。
- `rollout/`：ParallelRollouter、Job/Episode、episode_recorder、
  test_exploratory_policy/test_remote_inference 全过。
- `inference_server.py`+`remote_policy.py`：远程 GPU 推理服务模式
  （UDS、固定 batch 容量保证 bit-identical、per-request 噪声注入），
  被 `parallel_rollouter.py` 的 `_remote_addr` spec 路径真实接线——
  活特性不是摆设。
- `dumpkit/`：capture→dataset→analysis→viewer 管线 + CONTEXT.md
  （既有 D 层文档范本），dump_analysis/frame_access/dump tests 通过。
- `code_snapshot.py`：git 分支快照机制（REPRODUCE.md 生成）。

### 发现的问题

**P-TF-1（4 个过时测试）：dump_delta 语义已改，测试夹具没跟上**
- `test_dump_delta.py` 4 个失败全部指向同一处：`compute_delta` 的
  gen 语义在 `85738c03`（"delta ref fix"）被**有意**改为
  row0=post-update u_N（原 row0=rollout 用的 u_{N-1}）——commit
  message 写明了动机和正确性论证。
- 测试夹具还按旧语义建 `export_updates=(0,1,2,3)` for update=4，
  断言旧 `gen_updates=[3,2,1,0]`——**实现正确、测试过期**。
- 建议：夹具改为 `export_updates=(1,2,3,4)` + 断言 `[4,3,2,1]`。

**P-TF-2（1 个过时测试）：`test_checkpoint_rng_state_roundtrip`**
- `9af767fc` 新增守护：checkpoint 里 prev_gvec 长度 ≠ actor 参数量时
  丢弃（obs/param 扩展场景防呆）。测试传 5 维 gvec 给 198 参数的假
  actor——新守护正确丢弃，`resume_ctx["prev_gvec"]=None` → 断言失败。
- **实现正确、测试没适配**。建议：gvec 改成与 actor 参数等长。

**P-TF-3（1 个过时测试）：`test_humanoid21_back_compat_alias`**
- 断言 `baseline.humanoid21.base.Critic` 别名存在——该模块已在
  common→framework/critic_mlp 的迁移中删除。**别名被有意移除**，
  测试是删模块时漏网的。

**P-TF-4（2 个 collection error）**
- `ppo/tests/test_viewer.py`：import `_dump_gradsig`——该私有函数已
  移入 `dump_analysis.dump_gradsig`（server.py 内现为 `_da.dump_gradsig`）。
  viewer 测试文件整体过时（1661 行，覆盖面大，值得修而非删）。
- `ppo/policies/todo/test_*.py` ×3：import `tanh_gaussian_mlp`、
  `FixedSigmaGaussianMLPPolicy` 等——`todo/` 目录是**策略候选设计
  停车场**（DESIGN_*.md×6 + 未实现代码桩），测试本就跑不起来。
  属"设计稿的测试"，不算回归，但建议在 todo/README 里写明"这些
  测试是目标规格的草稿，依赖未实现的类"。

**P-TF-5（文档小滞后）**：`ppo/README.md` 目录结构说 policies/
  = "truncated_normal_mlp + todo/ 在建策略族"——实际 8 个族都已毕业
  为生产策略，todo/ 剩的是未实现候选。

**P-TF-6（资产标记）**：`obsolete/`（旧 ppo_loop/ppo_trainer/sac_*
  + eval/probe 脚本 + golden_data）按命名契约是留档——确认无活代码
  引用即可，不细审。

**能力入账**：ExperimentPPO 契约/CommonParams（STABLE）、多 critic
ppo_update+PPOBuffer（STABLE）、loop 编排含 checkpoint/resume/dump
调度（STABLE）、ParallelRollouter（STABLE）、RemoteSamplingPolicy+
inference_server（USABLE，GPU 推理服务）、dumpkit 全家桶（USABLE，
viewer 测试过时）、8 策略族+导出模板（STABLE）、code_snapshot
（STABLE）、policies/todo 候选族（UNSUPPORTED，仅设计文档）、
obsolete/（LEGACY）。

**Result/evidence:** `pytest ppo/ rollout/ test_critic_mlp.py`:
597 passed / 6 failed（全过时测试）/ 4 collection errors；
git log 85738c03 + 9af767fc 证实两处语义改为有意。
**Next:** `baseline/experiments_ppo`。

## [2026-10-01] baseline/experiments_ppo —— PPO 实验注册表

**Domain/object:** `baseline/experiments_ppo/`（28 个活实验 + base.py
583 行 + 注册表 + `archive/` 28 个 V2 时代实验 + `todo/` ~20 个
V1 时代实验/设计稿 + 3 个测试文件）
**Category:** discovery
**What:** 注册表实测 + 测试实跑 + 结构抽查。**51 passed / 0 failed**。

### 实况

- **注册表健康**：`exp_*.py` 自动发现 + `EXPERIMENT_CLASS` 约定实测
  可用——`list_ppo_experiments()` 返回 31 个实验（含 `__init__` 内
  注册的 minimal）。KeyError 时列出全部可选名，fail-loud。
- **`CombatExperimentPPOBase`（base.py）**：declared-attribute kwargs
  白名单（`test_set_kwargs.py` 覆盖 `--set KEY=VALUE` 注入），
  exploration/ppo_params/build_jobs/sampling_spec/env_bp 全链默认，
  子类只改类属性+少量方法——**实验定义成本极低，这正是"AI 按用户
  意图写实验"的正确形态**。
- 实验文件本身是剂量扫描阵：`standup_floor04` 系列 ~20 个变体
  （ef 剂量 0.3/0.5/0.8、policy 族×8、lam/lr/ufloor/floorsched/
  tklearly），docstring 写明假设与参照实验——实验即文档，质量好。
- `archive/`（28 个 `exp_basic_balance_v2*`）+ `todo/`（~20 个 V1 时代
  exp + EXPERIMENT_LOG + REVIEW_SUMMARY）：**有意分层的历史 strata**，
  不在 glob 发现范围内，不会污染注册表。
- 测试：test_set_kwargs / test_standup_step_v3_floor /
  test_step_detect 共 51 用例全过——测的是 base 的 kwargs 注入和
  step 检测工具函数。

### 观察（非问题）

- **W-EXP-1**：实验命名即文档，但**没有"当前推荐跑哪个"的索引**——
  README 只讲机制（怎么列/怎么冒烟/怎么加实验），不讲选型。对一个
  想让 AI 代做实验的用户来说，"31 个实验里哪个是当前主线"要靠读
  文件名猜（standup_floor04* 是最新剂量阵）。建议：README 加一节
  "当前活跃实验线"（几行即可，或指向某个 LEVER 目录）。
- **W-EXP-2**：`todo/` 和 `archive/` 的命名语义不同（todo=未做/
  候选，archive=做过已归档）但外人看都是"旧东西"——目录名语义
  建议各加一行 README 说明。

**Result/evidence:** `pytest baseline/experiments_ppo` 51 passed；
`list_ppo_experiments()` 实测 31 项；`exp_*_ef03/ef08` 抽查确认
薄子类模式。
**Next:** `baseline/humanoid21`（rewards/plugins/blueprints/curriculum）。

## [2026-10-01] baseline/humanoid21 —— 训练资产层（rewards/plugins/blueprints + 历史层积）

**Domain/object:** `baseline/humanoid21/`（~35k 行：rewards/24、
plugins/11、blueprints/**62 个 yaml**、curriculum/87 py 旧注册表、
end2end/26、mocap/10、fight/follow/balance_recover、tests/、
顶层 replay_* 脚本）
**Category:** discovery
**What:** 测试实跑 + 引用图核查。**tests/: 0 run / 1 collection error /
2 failed**——本目录测试面很薄且全有问题。

### 测试实况（覆盖薄弱）

- `tests/` 仅 2 个文件：
  - `test_curriculum_gate.py` —— **collection error**：import 已删除的
    `baseline.humanoid21.common`（CurriculumStageGate 随旧注册表迁移消失）。
  - `test_fight_mixed_policy.py` —— 2 个用例都失败：需要
    `baseline/humanoid21/runs/curriculum_follow_20260615_131515/policy_exports/u10294/`
    的历史训练产物（runs/ 被 gitignore，环境里不存在）——**环境依赖型
    测试，不是代码回归**，但也等于从来没法在干净环境跑。
- 另外根 `tests/test_stage_seg_rewards.py`（23 用例）测本目录
  `curriculum/experiments/exp_basic_balance_v2_stage_seg`——属于本域
  但被放在根 tests/，且测的是 archive 级实验。
- **结论：35k 行训练资产只有 ~2 个有效测试文件的覆盖**——rewards/
  plugins/ 的核心 reward channel 实现几乎无单测（它们的验证依赖
  "训练曲线正确"这种间接证据 + batchframework 的 fixture 对照）。

### 分层实况

- **活资产**：`rewards/standing_balance_4stage.py`（standup 主线
  rewarder，batchframework 已做 NATIVE 设备版）、
  `plugins/standup_termination.py` 等被 live exp 引用；
  62 个 blueprints 中**活实验只引用 2 个**
  （`standup_4stage_dense_v2_env.yaml` 给 standup 系 +
  `basic_balance_v2_phi_dual_env.yaml` 给 basic_balance；
  `exp_standup_step_v3` 直覆 `_env_pb`）——**~60 个 yaml 是历史层积**
  （balance_recover_×6、standup_orig_×6、rollover_×N、V1/V2 各阶段）。
- **旧注册表**：`curriculum/experiments/`（87 py，自己的 base.py）
  是 superseded 的 V1/V2 实验注册表，只被 `curriculum/run_*_chain.py`
  链式脚本和 obsolete/ 引用——CLAUDE.md 已声明 superseded，属实。
- **mocap/**：AMC/ASF 动捕解析 + retarget，引用 `battle_v1.xml`
  （P-H21-6 已记）——独立的动捕工具链，与主线训练无耦合。
- **end2end/、fight/、follow/、balance_recover/**：V1 时代实验资产
  （状态机、混合策略、env yaml），名义上还可实例化但与当前
  experiments_ppo 主线脱节。
- 顶层 `replay_4stage_recorder.py`/`replay_hybrid.py`/
  `replay_standup_switch.py`：旧策略回放脚本。

### 问题

**P-BH-1（资产膨胀核心）**：62 blueprints / 87 curriculum exps /
~20 rewards 中大量是为单次消融/阶段服务的"一次性"文件——
无"活/死"标记，靠考古才能分辨。这是项目**杠杆目录膨胀最大的一处**。
建议：blueprints/ 下按时代分子目录（`v1/`、`v2/`、`standup4stage/`）
或至少在 README 里列"当前引用中的蓝图清单"（就 2 个）。

**P-BH-2**：`test_curriculum_gate.py` 整文件过时（import 已删模块）。

**P-BH-3**：`test_fight_mixed_policy.py` 依赖历史 run 产物——
要么改成生成 fixture，要么标 skipif 缺失。

**能力入账**：reward 族（standing_balance_4stage 等 ~15 个 channel
实现，USABLE——设备版已验证等价但 CPU 侧无单测）、训练 plugins
（USABLE）、blueprints（USABLE 但膨胀）、curriculum/ 全套（LEGACY）、
mocap 工具链（USABLE，独立）、end2end/fight/follow/balance_recover
资产（LEGACY）。

**Result/evidence:** pytest tests/ 1 collection error + 2 failed；
62 yaml 中 grep 出仅 2 个被活实验引用；87 个 curriculum 实验文件
无外部引用。
**Next:** `policy/`（参考策略目录）。
