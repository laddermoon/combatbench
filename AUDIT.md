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
