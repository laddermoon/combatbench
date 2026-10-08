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
- `recorder.py` —— `PostActionRecorder`/`BaseFrameRecorder` 职责清晰，
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

**P-FW-2（已解决——契约修订，`082187be` 起）：物理步中途终止 → 终止帧语义修订为合法终止 transition**
- 原现象：`_RuntimeCore.step` 在 `on_pre_phy_step`/`on_post_phy_step` 中
  检测到全员终止时提前 return，跳过 `on_post_action_step` 与
  `episode_step += 1`——终止帧 observer 输出陈旧、`term_step` 排他截断
  丢弃该帧、`BaseFrameRecorder` 文件名碰撞。
- 裁决：**是契约缺陷**。终止帧四要素齐全（obs_t/action_t/终止态/done），
  丢弃是实现事故不是设计——"action 导致终止"在 post_action 提出保留、
  子步内提出被丢，同逻辑事件结果不同。
- 修订后语义（详见 `envs/batchframework/TERMINAL_FRAME_PLAN.md`）：
  - `episode_step` = step() 调用计数，无条件 +1 → records 值天然为
    含端点边界，文件名碰撞随之消失；
  - `on_post_action_step` 每进入步恰好触发一次、作用于步末态
    （终止帧 = 终止态刷新值，observer 无需 post_episode 特判）；
  - **退化帧**（pre_action/首子步 pre_phy 全员终止，physics delta=0，
    action 未物理生效）仍记录在案但按 `agent_frame_boundary` 排除出轨迹；
  - 终止 transition 的 reward 从此计入轨迹——reward 记账变化，
    历史 run 与改后不可直接比（预期内）。
- 验证：`envs/framework/tests/test_audit_terminal_frame.py` 已按新契约
  重写断言。

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

## [2026-10-01] policy/ —— 参考策略库（一个全量损坏点 + 一处过时契约文档）

**Domain/object:** `policy/`（random/、humanoid21/standing/、
baseline/ 5 族 ~138 个训练策略快照 57MB、blueprints/、README.md）
**Category:** discovery + test
**What:** 逐目录核查 + 实际加载验证。

### 实况

- `policy/random/policy.py` —— RandomCombatPolicy 实现**当前** Policy
  ABC（`act(obs, want_extra)`），`policy/blueprints/random.yaml`
  加载实测 OK。
- `policy/humanoid21/standing/policy.py` —— StandingCombatPolicy 实测
  import + act 正常（21 维动作）。
- `policy/blueprints/` 只有 random + standing 两个蓝图——**baseline
  族的快照没有对应顶层 blueprint**（各自目录内带 policy_blueprint.yaml）。

### 发现的问题

**P-POL-1（全量损坏，但一处之遥）：`policy/baseline/` 下 81 个策略
快照全部无法加载**
- 每个快照的 `policy.py` 都 `import baseline.framework.ppo.policies
  .tanh_gaussian_mlp`——该模块在 `f232b8c5` 被移入 `policies/todo/`
  （commit 明说 "import chains intentionally left broken"）。
- **实测**：`policy/baseline/fight/u11868/policy.py` import 失败；
  但把模块路径改到 `...policies.todo.tanh_gaussian_mlp` 后，
  `model.pt` 权重正常加载、`act()` 输出合法 21 维动作——
  **权重完好，只差一行 import 路径**。
- 规模：81 个 policy.py ×（fight/fight_v2/fight_v2_oppopool/follow/
  follow_v2 五族），57MB 权重全部不可用。
- 影响：这是"参考对手池/基线对战"能力——打榜前的自对弈对手全灭。
- 建议（用户裁决）：批量把 import 改到 todo 路径（机械修复，
  81 文件同改一行）；或把 tanh_gaussian_mlp 移回正式位置（它已
  被验证能承载这些权重）+ 更新导出模板。

**P-POL-2（契约文档整体过时，对 AI 有误导性）：`policy/README.md`**
- 描述的核心类 `BaseCombatPolicy`（ABC + gym spaces + kwargs 透传）
  **在代码中不存在**——README 里整段类定义是旧世代的伪代码。
- 描述的旧 `act(observation) -> ndarray` 签名 + `act_with_extras`
  钩子——当前契约是 `act(observation, want_extra) -> (action, extra)`。
- 提到的 `load_policy(...)` 函数、`ParallelRunner` 都不存在。
- **按这个 README 写策略的 AI 会产出接口错误的代码**——这是"文档
  比没有更糟"的典型，优先级高。
- 建议：重写为指向 `envs/framework/policy.py` 的当前契约 +
  `random/policy.py` 作为最小样板。

**P-POL-3（小）**：README 目录规范仍说"必须继承 BaseCombatPolicy"——
同上，契约已变为 `envs.framework.policy.Policy`。

**能力入账**：RandomCombatPolicy（STABLE）、StandingCombatPolicy
（USABLE）、baseline 快照库 81 个（**UNSUPPORTED——import 路径损坏，
一处之遥可修**）、policy/blueprints（USABLE 但只覆盖 2 个策略）。

**Result/evidence:** import 实测 random/standing OK、fight/u11868
ModuleNotFoundError、todo 路径补丁后加载+act 成功；
`git show f232b8c5` 证实迁移为有意。
**Next:** 剩余目录——docs/、scripts/、examples/、根 tests/、
assets/、debug_approach/_debug 等边角。

## [2026-10-01] 剩余目录收尾（docs/examples/scripts/assets/根 tests/本地资产/根文档）

**Domain/object:** 仓库其余部分
**Category:** discovery + test

### docs/（平台向文档，6 文件双语）

- **`ENVIRONMENT.md`（+_zh）严重过期**：描述"四面墙方形房间、9 个
  固定相机、`assets/humanoid.xml`、`assets/textures/wall.png`"——
  实际是 `battle_circular_v2.xml`（24 段圆形墙）、纹理在
  `envs/humanoid21/textures/`（wall_0..23.png + floor_circular.png），
  `assets/` 下只有 `images/hero.png`。**对外规则文档描述的场地几何
  与实际物理不符**——如果平台方/用户按它理解比赛空间会被误导。
- `RULE.md`：规则语义（30s×6 回合、手/脚→头/躯干 HP 判定）与
  CombatScoringPlugin 一致（DAMAGE_TARGET_PARTS 含 waist_upper/lower
  对应"上下腰"——虽然代码里这两个枚举是死的见 P-H21-7）。
- `SUBMISSION.md`：combatbench.tech CLI 提交流程——平台侧无法在本
  仓库验证，未见明显矛盾。

### examples/（实测可跑，文档标签过时）

- 9 个生命周期示例（01 认识环境 → 09 recorder/round runner）+
  `_common.py`——**01 实测跑通**（96 维 obs 摘要正常写出）。
- README 自称"提案文档 v2，等待确认后再落地"——但示例已全部实现。
  标签过时（小问题）。

### scripts/

- 仅 `migrate_feet_forces_norm.py`：一次性权重迁移脚本（给旧
  checkpoint 的 feet_forces 维度补归一化），引用 `battle_v1.xml`。
  留档合理，建议 README 注明"一次性脚本，已执行过"。

### 根 tests/

- 归属混杂：7 个文件属 batchframework（已审）、
  `test_stage_seg_rewards.py` 属 baseline/humanoid21 curriculum、
  `debug_fall_images/` 是图像 fixture 目录。建议按归属迁移或加索引。

### 本地资产（不入库，确认无泄漏）

- `1.1 T800 模型资源/`（113MB）、`1.2 题目拳击数据/`（9.5MB）、
  `_debug/`（506MB 录制 dump）、`debug_approach/`（54MB PNG）——
  `git ls-files` 均为 0，`8dac0e30` 已把它们移出跟踪。**磁盘占用
  ~680MB 但不污染仓库**，状态正确。

### 根文档状态

- `README.md`/`README_zh.md`/`CLAUDE.md` 仍引用
  `get_termination_flags()`（P-FW-2 已记）、battle_v1/v2
  （P-H21-6 已记）。
- `ISSUES.md`、`REVIEW_OVERVIEW.md`、`REVIEW_TIMELINE.md`、
  `VALUE_PLAN.md`、`V2_TRAINING_TIME_LOG.md`、`bootstrip.md`：
  历史规划/审查文档，作为时间线档案保留合理。

### 本轮审计总结（Phase 1 完成度）

已深审：`envs/framework`、`envs/humanoid21`、`envs/batchframework`、
`baseline/framework`（不含 sac）、`baseline/experiments_ppo`、
`baseline/humanoid21`、`policy/`、其余全部目录。

**测试面总账**（本轮实跑）：

| 域 | 结果 |
|---|---|
| envs/framework | 159 pass / 3 fail（缺方法）/ 5 collection err |
| envs/humanoid21 | 43 pass（7 个不可失败）+ 审计探针 2 pass |
| batchframework（根 tests/） | 64 pass / 1 fail（stale 门禁） |
| baseline/framework | 597 pass / 6 fail（全过时）/ 4 collection err |
| baseline/experiments_ppo | 51 pass / 0 fail |
| baseline/humanoid21 | 1 collection err / 2 fail（环境依赖） |

**高优先级裁决项汇总**（详见上文各条目）：

1. P-POL-1：81 个 baseline 策略快照一行修复可救（建议优先）
2. P-H21-4：humanoid21 acceptance 测试形同虚设且**标准当前未达标**
   （tracking/latency/oscillation 实测 False——PD 控制质量存疑，
   这可能是个被掩盖的真实性能问题）
3. P-H21-2：CombatScoringObserver events 字段恒空（read 错容器）
4. P-H21-1：contacts 缓存跨步失效（masked by scoring plugin）
5. P-BF-1：fixture 批准链滞后，跨后端验证当前全红
6. P-FW 系列：framework 测试债 + CONTEXT/DESIGN 文档过期
7. P-POL-2：policy/README 契约性错误（会误导 AI 写出错误接口）
8. docs/ENVIRONMENT.md：场地几何与实现不符

**Next:** 审计第一轮完成。等用户对建议清单（S1-S8 + P-* 各项）
裁决后进入修复/文档阶段；或按指示继续深挖特定方向。

## [2026-10-01] Phase 2 深挖 #1 —— P-H21-4 验收不达标的根因测量

**Category:** test + discovery（在 `envs/humanoid21` 内深挖）
**What:** 不动代码，直接跑验收方法的物理测量，回答"是差一点不达标
还是差得远"。

### 实测数据（全部复现 ACCEPTANCE_CRITERIA.md 的方法）

**验收 1（1Hz 满量程正弦跟踪，关重力，要求 heavy<0.05 / light<0.02 rad
均值误差）**：
- 当前 heavy 均值误差 **0.59 rad**（超标 12×）、light **0.47 rad**
  （超标 24×）；逐关节最大误差达 1.37 rad。
- **误差与指令幅度严格线性**（amp 1.0→0.59，0.5→0.30，0.2→0.12，
  0.1→0.07）——典型相位滞后主导的一阶滞后特性，等效滞后时间常数
  ~0.12s；不是饱和失控（饱和率：仅 2 个关节 >40%，多数 <5%）。
- 要达到指标（滞后 <0.05rad @ full-range 1Hz）需 τ≈10ms——当前
  结构下需要扭矩量级提升或前馈补偿，**靠微调 KP 做不到**。

**验收 2（阶跃 90% 到达 <100 步=0.2s）**：
- 实测 t90 = **115–186 步**（0.23–0.37s），21 个关节全部超标；
  2 个关节 400 步内**到不了 90%**（饱和受限）。
- 失败幅度 ~1.5–2×，不是临界。

**验收 3（静态站立零指令：torque 一阶导极小 + 承重关节 <30% 额定力矩）**：
- 静态站立时腹部/腰部关节（0-2）持续占用 **66–90%** 额度力矩，
  手部关节 ~43%——**远超 30% 上限**；等效于机器人站姿本身就接近
  执行器极限（初始位形静态不自平衡 + PD 无积分项→持续姿态误差
  换持续力矩）。
- 力矩变化率 mean|Δctrl| 在关节 0 达 **1.8/step**，"稳态" qvel 峰值
  0.89 rad/s——静止站立有明显高频力矩振荡，"零震荡"不达标。

**历史核查**：`ACCEPTANCE_CRITERIA.md` 与 `test_acceptance.py` 写于
`a7a36357`/`542ee5fb`（4 月初，声称"编写并通过"），此后 simulator
有十几次变更（观测维度、contact 管线、向量化 PD、圆形场地）但
**KP/KD/执行器 gear 一行未动**（git log -S 确认）。疑点：
要么当初就没真跑过全量验收就声明了"通过"，要么 battle_v1→circular_v2
的物理差异（condim 1→3、impratio 1→10、接触模型质变）改变了结论。
**无论哪种，当前现实是：PD 底层不满足它自己的验收文档，而
return-dict 测试结构把这件事藏住了。**

**衍生风险**：`policy/humanoid21/standing` 和所有 standup 系列实验的
"站立"都建立在这套欠阻尼/欠刚度的 PD 上——训练能收敛是因为 RL 会
适应环境特性，但任何声称"PD 达标"的文档都在误导。

## [2026-10-01] Phase 2 深挖 #2 —— 三个框架级问题测试证实/排除

**Category:** test
**What:** 新增两个 audit 测试 + 两个实测验证。

### P-H21-2 已证实（test_audit_combat_observer_events.py，2 passed）

- `CombatScoringObserver` 读 `ctx.metrics['events']`（无写入方），
  真实事件在 `ctx.events`——**health/cumulative_damage 正常，
  events/step_hit_events/step_damage_taken 恒为空**。
- 测试锁死当前行为（observer miss event；若 metrics 里有则能读到——
  证明纯粹是容器错位）。
- **影响面已确认**：`envs/humanoid21/blueprint.yaml`（标准战斗蓝图）
  就挂了这个 observer——所有标准战斗 rollout 的 combat_scoring 输出
  里事件字段全是空的；任何想基于击中事件造 reward 的用户会拿到静默
  全零。

### P-FW-9（新发现，测试证实）：mutator 沙箱是**宣示性的，不是强制的**

- `test_audit_mutator_leak.py`（1 passed）证实：`_MutatorView` 无
  生命周期校验，`_revoke_mutator` 只是 `ctx.mutator = None`——
  插件在可写钩子里 `self._m = ctx.mutator` 缓存引用后，在只读钩子
  `on_post_action_step`（此时 ctx.mutator 已为 None）里用缓存引用
  `set_action` **成功写入**。
- 含义：capability 分离（accessor/mutator）对守规矩的插件是约束，
  对不守规矩的插件**没有任何强制力**。对"AI 写的插件"这个场景
  是真实风险——AI 生成的插件完全可能意外缓存 mutator。
- 建议：`_MutatorView` 加 validity flag（grant 时置真、revoke 时置假，
  方法入口检查），成本 ~10 行。

### 确定性实测（P-DET，排除一个历史疑点）

- **同 seed 同动作序列：进程内 + 跨进程 bit-identical**（500 物理步
  qpos/qvel/contact.dist 全部 SHA256 相等）——当前 XML 未开
  MuJoCo 线程/island，单机完全确定性。
- `REVIEW_OVERVIEW.md:118` 引用的
  `MUJOCO_CROSS_PROCESS_NONDETERMINISM.md` **文件已删除**——历史
  问题档案丢失，当前实测不复现（以前的问题可能与多线程渲染或
  旧 XML 配置有关）。
- **不同 seed → 完全相同的轨迹**：`simulator.reset(seed)` 的 seed
  参数静默丢弃（P-H21-7 实测确认后果）——**初始状态多样性完全
  依赖插件**（RandomFallenStatePlugin 等），simulator 层贡献为零。
  当前设计下是自洽的（固定初始位形是刻意选择），但契约语义上
  "reset(seed)"承诺了它没兑现的东西。

### EpisodeRunner seed 链（静态核查，设计正确）

- `SeedSequence.spawn(n)` 端到端派生：runtime/policy_a/policy_b/
  各 seedable 插件各占一个孩子序列；插件按 attach 顺序按位置分配
  （换插件顺序会改派生结果——固有语义，文档化即可）。
- `set_episode_seed` 在 `runtime.reset` **之前**调用——保证
  `on_pre_episode` 里插件就能用已播种的 RNG 采样初始扰动。顺序正确。
- `seed=None` 在入口解析为具体 uint32，不再向下传 None。链路完整。

### MatchRunner ↔ RoundRunner 契约核对

- `RoundRunner.run(seed, initial_health_a, initial_health_b,
  score_log_file)` 返回 `health_a/health_b`——MatchRunner 的 HP
  结转、KO 即时终止、双边归零判平局逻辑均一致，未见 bug。

**能力/状态修订**：`policy/baseline` 快照库与 `docs/ENVIRONMENT.md`
维持前判；新增：**框架 mutator 沙箱 = USABLE-but-advisory**（记入
总账备注）。

**Result/evidence:** 新增 2 个 audit 测试（共 3 passed）；确定性
SHA256 两次跨进程一致；阶跃/正弦/静置三组物理实测数据如上。
**Next:** 继续深挖（按价值：blueprint 物化边界、recorder/replay
保真度、device 路径 obs 逐维对照、experiments_ppo 里 param 注入的
边界行为）或等用户裁决。

## [2026-10-01] Phase 2 深挖 #3 —— rollout 一致性 / blueprint 边界 / replay 精度

**Category:** test
**What:** 三件实证验证 + 一处文档缺口。

### ParallelRollouter 跨 worker 数一致性（实测通过）

- 4 个相同 Job（random policy，600 帧/ep）分别跑 `num_workers=1`
  （进程内串行）与 `num_workers=2`（spawn 子进程）：
  obs+actions 全量 SHA256 **完全一致**。
- 坑位记录：`mp_context="spawn"` 默认意味着调用方必须是真实
  .py 文件 + `__main__` 守卫——stdin/REPL 里直接跑 ParallelRollouter
  会 BrokenProcessPool（worker 重 import `__main__` 递归 spawn）。
  这是 spawn 的固有约束，但**没有任何文档提示**——AI/用户在 notebook
  或脚本片段里调它会撞上。建议在 ParallelRollouter docstring 加一行。

### episode_options ≠ blueprint 参数（语义边界，文档已有但易误用）

- 实测：`episode_options={"max_steps": 40}` **不生效**——options 只进
  `simulator.reset(options)`，blueprint 参数（TimeoutPlugin 的
  max_steps）在 materialize 时固化。episode 仍跑满 600 帧（=30s×20Hz，
  恰好印证 RULE.md 的回合时长）。
- `Job.episode_options` docstring 已写明 "environment-only，
  forwarded to simulator.reset"——契约正确，但命名上 `episode_options`
  读起来像能改 episode 长度。属易误用命名，非 bug。

### ParameterizedEnvBlueprint（实测验过）

- `materialize(unknown_key=1)` → `ValueError`（列出合法参数名）；
  缺必填参数 → `ValueError`；`${param}` 占位符引用未知参数 →
  `KeyError`。全 fail-loud，无误吞。

### ReplaySimulator 精度说明（静态核查）

- `_rehydrate` 把 JSON 数值一律降为 `float32`——而 CPU 物理是
  float64。**回放状态有 fp32 量化**，对逐帧数值复算（如拿回放状态
  重算 reward）会引入 ~1e-7 级误差。对录像/可视化场景无影响，
  但"回放即重演"的语义边界值得在 replay.py docstring 里写明
  （目前只提了 stride 匹配和 manifest 版本）。
- 32 个 replay/recorder 测试全过。

### 指标目录一致性抽查（阴性结果）

- dumpkit `metric_catalog.py`（97 条目）抽查 KL/grad_sig/adv 等关键
  键与 loop/trainer 发射端命名对得上——未发现目录漂移。

**Result/evidence:** workers 1 vs 2 指纹相等；materialize 三类边界
异常实测；replay rehydrate 降精度属设计内（docstring 未提）。
**Next:** Phase 2 暂告一段落。剩余可挖方向：dumpkit viewer API
契约测试重建（test_viewer 重写）、standup rewarder 四阶段门限的
对照测试、设备端 obs 与 host 逐维 diff 细查。等用户裁决或指示。

---

## 2026-07-12 Phase 3 — envs/framework 逐文件精读（注释/死代码专场）

**Scope:** envs/framework 全部 13 个源文件逐行过一遍，专题是"注释与实现不符、
残留死代码、半成品痕迹"。新增 2 个审计探针。
**发现 15 项（P3-1 ~ P3-15），其中实锤 bug 2 个、探针锁死 2 个、
死代码/陈旧注释 8 处、契约不对称 3 处。**

### P3-1（实锤 bug，探针锁死）：abandoned/未终结回合的 recorder `on_post_episode` 丢失

- **发现方法：** 精读 `_RuntimeCore.reset`（env_runtime.py:134-138）。注释声称
  abandon 时 "so recorder manifests and observer state are flushed"，但
  `_handle_termination()` 只调 `plugin_manager.invoke`；recorder 存在
  `EnvRuntime._recorders`，由 `_invoke_recorders` 分发——core 层根本够不到。
- **后果一**：`reset()` 放弃进行中的回合时，插件侧收到
  `on_post_episode(reason="abandoned")`，**recorder 侧什么都不收到**——
  正在写的 episode 目录没有 manifest flush，index.json 里这个 episode
  永远处于未收尾状态。
- **后果二**（同根因）：`EnvRuntime.close()` 只调 `recorder.on_detach`，
  对仍活跃的回合同样不发 `on_post_episode`。
- **探针**：`tests/test_audit_reset_recorder_gap.py`——两次 reset 后
  recorder 计数为 `pre_episodes=2, post_episodes=0`，锁死两个缺口。
- **建议**：`_RuntimeCore.reset` 的 abandon 分支与 `EnvRuntime.close` 都应
  把终止事件透传给 recorder（例如 core 回调或 EnvRuntime 包一层）。

### P3-2（实锤 bug，探针锁死）：VideoRecorderPlugin 的回合级 output_path 覆盖是永久污染

- **发现方法：** 精读 `common_plugins.py:68-77`。docstring 声称
  `episode_options["video_output_path"]` "覆盖**本次** episode 的保存位置"，
  实现却是 `self.output_path = Path(override)` 直接改写实例状态，无 restore。
- **后果**：某回合传了 override 后，之后所有不传 override 的回合继续写到
  上次的路径（互相覆盖同一个 mp4）。MatchRunner 每回合 new 一个插件所以
  不踩坑；但 EnvRuntime 直挂共享插件的用法会中招。
- **探针**：`tests/test_audit_video_path_leak.py`——三回合后
  `output_path` 仍停留在 episode-2 的 override，锁死该行为。
- **建议**：ctor 默认值存 `_default_output_path`，每回合 pre_episode 里
  `self.output_path = Path(override or self._default_output_path)`。

### P3-3（小 bug）：`_safe_call` 日志格式参数错位

- `env_runtime.py:51`：`"%s '%s' failed at %s", label, hook_name, label`——
  第三个位置又传了一遍 label，日志输出形如
  `Plugin 'x' 'on_post_episode' failed at Plugin 'x'`。应传 hook 上下文
  或干脆删掉第三个占位。仅影响 strict=False 路径的错误日志可读性。

### P3-4 ~ P3-8：死代码 / 陈旧注释 / 失效引用（5 处）

| 编号 | 位置 | 内容 |
|---|---|---|
| P3-4 | round_runner.py:19 | `import numpy as np` 全文未使用——死 import |
| P3-5 | episode_runner.py:46 | docstring 引 `parallel_runner`（已删模块） |
| P3-6 | policy.py:5 | docstring 引 `ParallelRunner`（已删类） |
| P3-7 | policy.py:69 | "The :func:`load_policy` loader"——本模块无此函数（真实入口是 PolicyBlueprint.build）；同名函数只存在于 batchframework 的探针脚本里 |
| P3-8 | env_runtime.py:14 | `TODO(framework/B2)` 提议 VectorizedSimulator/EnvRuntimeBatched 并提到 `RolloutCollector`（旧名）——该 TODO 已被 envs/batchframework 整体实现，注释未更新指向 |
| — | round_runner.py:8-10 | 模块 docstring 的 `run()` 签名过时：缺 want_extras/initial_health_*/score_log_file 参数和 health_a/health_b 返回键 |
| — | recorder.py:55-66 | per-step JSON schema 注释缺 `observation` 键——实现 L381 实际写它，ReplaySimulator.get_observation 也读它 |
| — | round_runner.py:148/194/263 | 三处函数内 `import json`（其中 263 又 `as _json`）——重复残留 |

### P3-9 ~ P3-11：契约不对称（3 处，均确认存在、严重度低）

- **P3-9 `${DIR}` 不对称**：`PolicyBlueprint.load` 做 `${DIR}` 替换
  （policy.py:397），`EnvBlueprint.load` / `ParameterizedEnvBlueprint.load`
  不做。当前所有 env blueprint 的 simulator config 只有标量
  （ARENA_XML 是类常量不入蓝图），暂无实际影响；一旦有人给 env 蓝图配
  外部资产路径就会踩到。
- **P3-10 `is_agent_terminated` 签名不对称**：`SimContext` 上是方法
  `is_agent_terminated(agent_id) -> bool`（context.py:237），
  `ReadOnlySimContext` 上是字段 `is_agent_terminated: Dict[str,bool]`
  （context.py:285）。同名成员两种调用约定——把插件代码搬到 observer
  （read-only ctx）会 TypeError。属设计不一致，非 bug。
- **P3-11 浅快照**：`ReadOnlySimContext.from_sim_context` 的
  metrics/events/proposals 都是 `MappingProxyType(dict(...))` 浅拷贝——
  嵌套可变对象仍与活 ctx 共享。observer 若改了 metrics 里的嵌套 dict 会
  污染真黑板。与 mutator-leak 同类（宣示性隔离），严重度更低。

### P3-12 ~ P3-14：实现细节不符/隐性坑（确认存在，文档未写）

- **P3-12**：`attach_observer_plugin` 在活跃回合中 attach 会立刻
  `refresh(force=True)`，但新 observer 的 `on_pre_episode` 从未被调——
  依赖回合初始化的 observer 会带脏状态上场。无文档说明。
- **P3-13**：`detach_observer_plugin` 后 `observer_plugins` dict 保留
  `name: None` 键，`get_observer_outputs()` 会返回 `{name: None}`。
  to_blueprint 跳过 None 没问题，但输出字典里残留幽灵键。
- **P3-14**：`ReplaySimulator.reset(options={"episode": N})` 的 "episode"
  键经 `EnvRuntime.reset` 一并写进 `ctx.episode_options`——插件黑板会
  看到一个 replay 专用的 stray key。无污染后果，但命名空间未隔离。
  另外 `ReplaySimulator.get_derived_state(fields=...)` 接受但忽略
  `fields` 参数（返回全量）——超集返回语义安全，签名兼容性 OK。

### P3-15（遗留调试插桩）：`_TURB_DEBUG` 打印体系

- `envs/humanoid21/simulator.py:14-15` + `disturbance_plugins.py:19-20`：
  `COMBATBENCH_TURB_DEBUG` / `COMBATBENCH_TURB_DEBUG_MAX_PHYS_STEPS`
  环境变量门控的 `print(..., flush=True)` 调试输出（turb_apply /
  turb_phys_pre / turb_phys_post / turb_debug），源自 441a73fe
  "fix random push"（2026-04-12）扰动力调试会话。
- **核查**：全仓库 `.md` 零引用；非 logging 走 print；模块级 env 读取
  （import 时定型，进程内不可切换）。默认关闭所以无害。
- **判定**：留存的诊断插桩，非死代码但**完全无文档**——AI/用户无法发现
  这个杠杆。建议：要么写进 disturbance_plugins 的文档并改用 logging，
  要么随扰动功能稳定后删除。

### 阴性结果（本专场查了没问题的）

- `blueprint.py` 主体、`_resolve_class` 的 dotted-form 兼容路径、
  `from_runtime` 的 TimeoutPlugin 特例——实现与注释一致。
- `observer_plugin.py` 的 BaseRuntimeUnit/CompositeObserver/dispatcher
  注释全部属实（含 _process_ctx token 跳过 metrics 的 Note 是诚实的）；
  `BaseObserverPlugin = BaseRuntimeUnit` 别名有明确注释说明来历。
- `parameterized_blueprint.py` 与 policy.py 的 Parameter/替换逻辑重复是
  有意为之（policy.py:451 注释说明为避免跨模块 import）。
- `match_runner.py`：HP 结转/round seed 派生/契约与 RoundRunner 对得上；
  `load_env_blueprint` 用 ParameterizedEnvBlueprint.load 统吃两种文档
  （无参数节则空参数物化）——实现正确。
- `recorder_viewer.py`：86 行小工具，无死代码。

**Result/evidence:** 2 个新探针（reset_recorder_gap、video_path_leak）全过
——均锁死当前缺陷行为，注释注明修复后应翻转断言。
**Next:** 继续 Phase 3 下一个域：envs/humanoid21 的工具性文件
（disturbance_plugins/plugins/observer_plugins 的注释与死代码复查，
第一轮审的是行为，这轮专看注释与残留）。

---

## 2026-07-12 Phase 3 续 — envs/humanoid21 逐文件精读（注释/死代码专场）

**Scope:** plugins.py / disturbance_plugins.py / observer_plugins.py /
simulator.py / meta.py + 资产目录。本轮看注释准确性与残留代码
（第一轮审的是行为）。
**发现 13 项（P3-16 ~ P3-28），含同类 bug 第二实例、CPU/device 语义分歧 1 处、
4 套无文档调试通道的集中登记。**

### P3-16（实锤 bug，与 P3-2 同类）：ConstantForcePlugin 的 episode_options 覆盖永久污染实例

- **位置：** `disturbance_plugins.py:1216-1224`。`on_pre_episode` 从
  `ctx.episode_options["impulse_params"][agent_id]` 读覆盖值，直接写进
  `self.force / self.direction / self.duration_action_steps / self.body_name`。
- **后果**：传过一次 impulse_params 的回合之后，不带参数的回合继续用上次
  的覆盖值；且 `to_blueprint()` 会把被污染的值序列化——蓝图快照失真。
- **对照**：CombatScoringPlugin 的 `score_log_file` 做的是**正确**范式
  （plugins.py:274 —— 每回合读 `opts.get(..., self.score_log_file)` 比较
  路径、不改写 ctor 字段）。三类"per-episode override"已有两种实现，
  一正两误（P3-2 + 本条），值得统一。
- **建议**：ctor 默认值存私有字段，per-episode override 走本地变量；
  顺手把 VideoRecorderPlugin 一起改。

### P3-17（语义分歧）：CPU 与 device 的 `ctx.events` 生命周期不同

- **CPU**（context.py:256）：`events` 只在 `clear_episode_state`（reset）清空，
  **整个回合持续累积**。
- **Device**（batch_context.py `clear_step_state` + host_compat.py:169/406）：
  每个 action step 开始前清空，并 drain 到 `last_events`。
- **后果**：插件/observer 写 "事件" 语义时两条路径行为不同——CPU 上是
  回合累积列表，device 上是单步瞬态。与 P-H21-2 叠加构成三层错位：
  CombatScoringObserver (a) 读了错误容器 `metrics["events"]`；
  (b) 即便改读 `ctx.events`，CPU 语义也是"回合至今全部事件"而非
  docstring 声称的 "current step"；(c) device 上则是"本步事件"。
- **建议**：定一个权威语义（推荐 per-step 瞬态，与 device 对齐），
  CPU `_RuntimeCore.step` 开头清 events；CombatScoringObserver 改读
  ctx.events。

### P3-18 ~ P3-22：死代码 / 死分支 / 死 import（5 处）

| 编号 | 位置 | 内容 |
|---|---|---|
| P3-18 | plugins.py:147 | `DAMAGE_TARGET_PARTS` 含 `waist_upper`/`waist_lower`，但 `_get_part_category`（L310-324）只会产出 head/torso/hand/larm/uarm/thigh/shin/foot——两个条目永远匹配不到（waist 已折叠进 'torso'）。死集合条目，误导读者以为 waist 是独立伤害目标 |
| P3-19 | plugins.py:362 | `hit_cat in ('torso','waist_upper','waist_lower')` 同理——后两个是永远进不来的死分支 |
| P3-20 | plugins.py:56,110 | `NonFallConstraintPlugin.on_post_phy_step` 的 `changed` 变量赋值后从未读取——死变量 |
| P3-21 | observer_plugins.py:3 | `import mujoco` 全文未使用——死 import |
| P3-22 | simulator.py:249 | `get_sensor_data` 永远返回 `{}`——自述"暂时返回空字典，未来可扩展"的占位实现。注意后果面：BaseFrameRecorder 的 step JSON 里 `sensor_data` 字段恒空，ReplaySimulator 回放也恒空——能力处于半成品状态但接口已就位 |

### P3-23（隐性坑）：KO 判定只在动作步边界，HP=0 后伤害继续累计

- `CombatScoringPlugin`：伤害在 `on_post_phy_step` 逐物理步施加
  （正确——注释里写明了动机是不错过瞬时接触），但
  `request_termination(KO)` 只在 `on_post_action_step` 判定
  （plugins.py:520-524）。HP 到 0 之后，本动作步剩余物理步仍继续
  累计 `damage_taken` 并写 score log。
- health 有 `max(0.0, ...)` 钳制所以不会变负；但 damage_taken 会超杀
  累计、terminal 帧的 hit events 里含"死人又挨了几步打"。语义上
  可辩护（伤害是物理事实），但与"KO 立即终止"直觉不符，且无注释
  说明这是有意选择。

### P3-24（性能/资源）：FrozenRobotPlugin 每物理步全量重写两个机器人状态

- `plugins.py:585-612`：每个 post_phy_step 都构造含**对方机器人**的完整
  core_state 并 `set_core_state`（触发 mj_forward）。对方机器人的值是
  刚读出来的原值——25 次/动作步的无谓全量写回。功能正确但浪费；
  且 `initial_state` 只在 `frozen_robot_id in core_state` 时设置，
  配置错误的 robot_id 会沿用上个回合的初始状态（静默失效模式）。

### P3-25（资源泄漏级）：内部仿真实例从不关闭

- `RandomFallenStatePlugin._internal_sim` / `ImpulsePerturbationPlugin._internal_sim`
  懒构造 `Humanoid21Simulator()`，插件无 `on_detach`——MuJoCo model/data
  （及潜在 renderer）挂到进程结束。训练 worker 每 worker 一个插件实例，
  量不大，但插件规范没要求/没示范清理内部资源。

### P3-26（无文档调试通道集中登记）

本目录共有 **4 套独立调试通道**，全部 `print`/文件直写，零文档：
| 通道 | 位置 | 开关 |
|---|---|---|
| TURB_DEBUG | simulator.py:14, disturbance_plugins.py:19 | `COMBATBENCH_TURB_DEBUG`, `..._MAX_PHYS_STEPS` |
| FALL_DEBUG | disturbance_plugins.py:818 | `COMBATBENCH_FALL_DEBUG`, `..._DIR` |
| SCORE_DEBUG | plugins.py:193 | `COMBAT_SCORE_DEBUG_FILE` |
| debug_torque | simulator.py:55,1090 | ctor 参数（蓝图可见，唯一半正式的） |

建议：统一收敛到一个 debug 规范（至少一份 CONTEXT/README 登记四者），
或删除已完成使命的（TURB/FALL 是 4 月扰动调试遗留）。

### P3-27（资产残留）

- `test_videos/` 下 2 个 mp4 是 acceptance 测试输出残留被提交
  （test_acceptance.py:23 的 VIDEO_DIR 指向这里——属输出目录被入库）。
- `battle_v1.xml` / `battle_v2.xml` 仍在目录内；当前 ARENA_XML 指向
  circular_v2，但旧 XML 仍被 baseline 工具链引用（retarget.py、
  debug_mujoco_reset.py）——非纯死资产，但新旧混用无文档说明。
- `obs_analysis/` `pose_images/` 为有意提交的研究产物（commit 86b4f141
  force-add）——保留合理，但目录用途缺说明。
- `observer_plugins.py:32` `ARENA_HALF_EXTENT=3.05` 是方形场遗留常量
  （现圆形墙），仅影响可视化贴图裁剪，cosmetic。
- `Humanoid21BalanceAnalysisObserver._last_accessor` 把活 accessor 缓存
  出场（L210），`get_visualization_image` 事后用它取**当前**帧配
  **历史**分析输出——错配风险，属于 accessor 常驻能力的另一面
  （与 mutator-leak 同族，只读方向危害低）。

**Result/evidence:** 全部经逐行阅读 + grep 交叉验证；P3-16/P3-17 与已有
探针同族（video_path_leak / combat_observer_events）。
**Next:** Phase 3 继续 envs/batchframework 工具性文件精读
（device_runtime/device_plugin/host_compat/capability_registry/validation 的
注释与残留），然后是 baseline/framework/rollout + ppo/dumpkit。

---

## 2026-07-12 Phase 3 续 — envs/batchframework 逐文件精读（注释/死代码专场）

**Scope:** capability_registry / device_runtime / device_plugin / host_compat /
batch_context / batch_plugin / backend / device_state / device_obs /
fake_backend / device_standup / device_rollouter / probes。总体印象：
**该域代码质量显著高于其他域**（M 编号溯源、注释诚实、
fail-loud 边界显式）——但仍有 7 项发现。

### P3-29（死枚举 + docstring 不符）：Capability.PENDING 不可达

- `capability_registry.py:10` docstring 声称"未注册的类 = pending"，
  但 `lookup()`（L99-103）对未注册类返回的是 `UNSUPPORTED` 条目。
  枚举值 `PENDING` 在全注册表中无任何条目使用、lookup 也不产生——
  **死枚举值**。语义上 "pending 按 UNSUPPORTED 拒绝" 与实现结果相同，
  但 docstring 的"= pending"表述错了。

### P3-30（半成品管道 ×2）：`last_events` 与 `SyncStats.summary()` 均无消费者

- `HostBatchCompatAdapter._push_ctx`（host_compat.py:167-169）与
  `LegacyPluginAdapter._drain_env_ctx`（L404-406）把 batch ctx 的 events
  drain 进 `self.last_events`——**全仓库无任何读取方**。
  后果：(a) 兼容插件发出的事件在 device 路径上实际进入无人看的列表
  （等价丢弃）；(b) 该列表无限增长，长跑 rollout 内存缓涨。
- `SyncStats` 统计 host↔device 传输，`resolve_plugin(stats=)` 可注入，
  但 `summary()` 无任何调用方（grep 全仓仅定义处）——计量管道建好了
  但没人看表。与 dumpkit 的 SyncStats 审计意图脱节。
- **建议**：last_events 接到 device rollouter 的 Episode 事件字段或删除；
  SyncStats.summary 接到 rollout 结束报告或删除。

### P3-31（实锤缺陷）：LegacyPluginAdapter 的子步钩子拒绝检查是浅的

- `host_compat.py:320-327`：`if h in type(plugin).__dict__`——只检查插件
  **自己类的** `__dict__`，不看 MRO。`class MyPush(RandomPushPlugin)`
  这种继承覆写会穿透检查被接受；随后 `on_pre_batch_step`（L433-436）
  真的会把其继承的 `on_pre_phy_step` 在块边界调一次——**正是报错信息里
  声称不允许的 "silent block-boundary fallback"**。
- 对不覆写子步钩子的插件，这两个 dispatch（L433-439）永远命中基类
  no-op——**双死调用**（检查挡住的人进不来，进来的人没有该钩子）。
- **建议**：检查改为 `type(plugin).on_pre_phy_step is not
  BasePlugin.on_pre_phy_step`（MRO 感知）；或删掉块级 dispatch 死代码。

### P3-32（契约弱化）：device observer 的"只读"只是约定

- CPU 侧 observer 拿 `ReadOnlySimContext`（构造性只读：accessor 白名单
  + 无 mutator 字段）。device 侧 `DeviceObserverDispatcher.on_*` 直接把
  **活 `DeviceCtx`** 传给 observer（device_plugin.py:286-299）——
  `ctx.state.episode`/`ctx.state.sim` 的张量全部可原地写，
  `ctx.mutator` 只是恰好为 None。observer 可直接改
  `ep.terminated_flag` 或物理状态而框架无感知。
- 同类弱化：`DeviceCtx._grant/_revoke` 同样是把共享的
  `self._mutator`（BatchRuntime 里唯一实例）赋给 ctx.mutator——
  缓存引用的 stash 问题与 CPU `P-FW-9` 完全相同，且单例共享更脆。
- `ReadOnlyBatchSimContext` docstring（batch_context.py:308-311）声称
  "快照后修改不影响只读视图"，但 `self.accessor = ctx.accessor` 共享
  活 accessor——ctx 字段快照了，accessor 读的是实时状态。
- **建议**：至少 docstring 写明 device observer 的只读是约定；
  DeviceCtx 的 mutator 加有效性位（同 CPU 修复方案）。

### P3-33（零租户设施）：整个 numpy 兼容契约当前无一个具体插件

- `batch_plugin.py`（427 行）+ `batch_context.py`（342 行）+
  `HostBatchCompatAdapter` 构成完整的 BaseBatchPlugin 兼容契约，
  但 `REGISTRY` 中 **COMPAT / HOST_SLOW 条目数为零**——grep 全仓无
  任何 BaseBatchPlugin 具体实现。LegacyPluginAdapter（单 env BasePlugin
  兼容）同样无注册租户。
- **判定**：设施先于需求建好（M3 W3 按 ROADMAP 交付），不是死代码
  但属"建好未用"——当前唯一生产路径是纯 NATIVE（standup 实验）。
  若长期无租户，batch_plugin.py 是候选收缩对象。

### P3-34 ~ P3-35：小项

- `batch_plugin.py` 与 `device_plugin.py` 的 `OBSERVER_DISPATCHER_PRIORITY`
  重复定义（=1_000_000 各一份）——双源常量，改一处漏另一处的风险。
- `device_runtime.py` 注释完备、行级 reset 语义与实现一致；
  `fake_backend.py` 定位清晰（契约后端 + 新后端模板）；
  `warp_simulator.py` 头部对 fp32-only/布局差异/host 快照路径的声明
  全部属实；`probe_*`/`m6_compare`/`device_examples` 头部注释准确。

**Result/evidence:** grep 交叉验证（last_events/summary 无调用方、
COMPAT 条目为零、PENDING 无产生路径）；MRO 检查缺陷为静态确认。
**Next:** Phase 3 继续 baseline/framework/rollout +
baseline/framework/ppo/dumpkit 的工具性代码精读。

---

## baseline/framework/rollout 第二轮：逐文件精读（2026-02 补）

> 范围：rollout 工具层全部文件逐文件过——job/episode/episode_recorder/
> episode_collection/exploratory_policy/remote_policy/inference_server/
> parallel_rollouter/bench_rollout/observer_utils/__init__/_bench_export。
> 定位：训练管线的数据采集层——EpisodeRunner→Episode 序列化→收集/落盘，
> 以及可选的 GPU 远程推理路径。

### 文件状态总览

- `job.py` / `episode.py` / `episode_collection.py` / `observer_utils.py` /
  `__init__.py`：**干净**——docstring 与实现一致，契约文档质量高
  （blueprint_hash、EfSpec、split_by_termination 的 `""`-reason 约定
  与 `Episode.agent_termination_reason` 的空串回退自洽）。
- `exploratory_policy.py`：**干净且注释诚实**——reference-ensemble 快路径
  首帧对账机制、sctx__ 前缀的 pickle 安全动机、SUPPORTS_REFERENCE_DELTA
  意图握手都有真实说明。
- `remote_policy.py` + `inference_server.py`（1050+ 行）：活跃子系统，
  线协议/确定性边界（GPU↔GPU bit-identical、CPU↔GPU ~1e-7）文档完备；
  `test_remote_inference.py` 18 项实测通过（40s，真实协议往返）。
- `parallel_rollouter.py`：主路径——spawn-context 陷阱此前已记
  （stdin/REPL 无 `__main__` 守卫即崩）。

### 本轮新发现问题（2 项确认 + 2 项记录）

- **P-RO-1（确认，工具失效）**：`bench_rollout.py` 双重过期——
  (a) import `baseline.framework.ppo.policies.tanh_gaussian_mlp`，
  该模块已移入 `policies/todo/`（与 P-POL-1 同一 stale-import 类）；
  (b) docstring 用法示例引用 `baseline/humanoid21/blueprints/
  stage1_env.yaml`——文件已不存在。性能基准工具当前完全不可运行；
  `_bench_export/model.pt` 是其专属 fixture 但随工具一起失用。
  建议：修复 import+blueprint 引用，或将工具与 fixture 一并归档 obsolete。
- **P-RO-2（确认，与 P3-1 同根）**：`episode_recorder.py` 整条
  Episode 产出依赖 `on_post_episode`——回合被 reset 放弃（P3-1 的
  死路径）时该钩子不触发，`get_last_episode()` 静默返回**上一个**
  回合的数据。正常 ParallelRollouter 流程不会放弃回合（跑到终止），
  严重度低，但同一根因在数据管线上表现为"重复旧 Episode"而非显错。
  另：`on_post_episode` 内 `except Exception` 裸捕获包住了观测读取
  ——后端任何异常都变成空 obs 静默入库，建议至少记 warning。
- **P-RO-3（记录，残留风险）**：`EfSpec` 类型别名在
  `exploratory_policy.py:43` 与 `job.py` 重复定义（当前完全一致，
  但漂移后两边注释也不会互相提醒）。建议收敛到 job.py 单源。
- **P-RO-4（记录，布局非标准）**：两个测试文件直接放在包目录
  （`rollout/test_*.py`）而非 `tests/` 子目录——pytest 照常收集，
  但与仓库其他域的 tests/ 约定不一致。不影响功能。

### 深度复核确认无问题的点（阴性结果）

- `SamplingSpec`/`ReferenceSpec`/`Job` 验证链：delta_mode 取值、
  weights 归一、frozen 依赖 reference——均有 fail-loud 校验。
- `Episode.save/load`：stem 路径约定 + npz/json 双写一致；
  `_to_jsonable` 递归处理 numpy 类型。
- `EpisodeCollection.append` 校验 blueprint_hash 一致性（异构混排
  拒绝）；`split_by_termination` 对 timeout/"" 语义与上游字段吻合。
- `SamplingPolicy.reset()` 正确重置 `_step` 计数器与 inner.reset。
- 远程推理握手：spec 注册幂等（内容哈希缓存）、确定性 padding、
  cuDNN benchmark 显式关闭（L597）保证 kernel 选择稳定。

---

## baseline/framework 顶层 + ppo 核心 第二轮：逐文件精读（2026-02 补）

> 范围：`code_snapshot.py`、`train.py`、`critic_mlp.py`、`__init__.py`；
> `ppo/experiment.py`（1613 行）、`ppo/loop.py`（1370）、`ppo/trainer.py`
> （2165）、`ppo/sampling_context.py`、`ppo/stochastic_policy.py`、
> `ppo/trajectory.py`、`ppo/debug.py`；`ppo/policies/` 目录结构。
> 定位：训练框架主干——实验契约、训练循环、PPO 更新、调试工具入口。

### 总体评价

这是目前全仓库注释质量最高的区域：几乎每个数值决策都有 provenance
注释（P0-1/P0-3/B1/B8/A4 等编号可追溯历史修复），契约文档与实现高度
一致（UpdateStats 命名规范、ctx 字段即契约、checkpoint v2 格式、
KL 滑窗跨 epoch 的设计理由全部对得上）。本区问题集中在**重构后的
文档残留**——docstring 引用了已被新机制取代的旧 API。

### 本轮新发现问题

- **P-BFW-1（确认，复现命令不完整）**：`code_snapshot.py`
  `format_repro_command` 只回放部分 CLI 参数。被静默丢弃的有：
  `--seed`、`--param`（整个 per-update patch 机制）、`--collector`、
  `--collector-batch-size`、`--adv-winsorize-sigma`、
  `--adv-winsorize-from-update`、`--adv-norm`、`--adv-norm-from-update`、
  `--dual-clip-c`、`--dual-clip-from-update`、`--reset-update`。
  例如原命令带 `--collector device --param dual_clip_c=3@282` 的运行，
  按 REPRODUCE.md 复现会变成 CPU collector 且无 dual-clip——
  **可复现性工具的复现命令本身不忠实**，且无任何提示。
- **P-BFW-2（确认，docstring 引用已删 API）**：
  - `trainer.py:829` `ppo_update` docstring 说 explore_factor "was
    already applied to the policy before rollout via
    ``set_exploration``"——`set_exploration` 已不存在（仅剩
    `policies/todo/` 归档文件里的引用），现机制是 SamplingContext
    逐帧回放；
  - `critic_mlp.py:6` docstring 称对应 actor backbone 是
    `TanhGaussianMLPPolicy`——已移入 `policies/todo/`。
- **P-BFW-3（确认，示例代码误导）**：`ExperimentPPO` 类 docstring 的
  "Typical subclass structure" 仍列出 `compute_episode_metrics` /
  `compare_eval` / `scheduler_info` 三个方法——它们已被 `on_eval`
  取代、框架从不调用。照此示例写实验的 AI 会产出三个死方法。
  另：`on_eval` docstring 标 update "0-based"，实际 `start_update=1`
  全循环 1-based（`build_jobs`/`post_update` 文档均标 1-based，
  仅此一处不一致）。
- **P-BFW-4（确认，模块图过时）**：`experiment.py` 头部数据流图
  （L79）把 `explore_factor` 画为 Trajectory 字段——现字段为
  `sampling_ctx` dict。
- **P-BFW-5（确认，归档区自测失效）**：`policies/todo/` 内 3 个测试
  （test_blueprint_episode/test_ou_exploration/test_policy_families）
  仍 import 移动前路径 `policies.tanh_gaussian_mlp`——移动时没带走
  自身测试的 import，归档区连自测都跑不起来（即 P-POL-1 的第 4 个
  collection error 来源之一）。
- **P-BFW-6（记录，日志措辞错）**：`loop.py:1075` early-stop 行硬编码
  "no improvement for {_no_improvement_limit} evals"——读私有属性
  且假定停止原因；`stop_training` 可因任何原因返回，措辞会误导。
- **P-BFW-7（记录，注释措辞错）**：`loop.py` ~L699 注释称 exploration
  spec "kept so ppo_update can read its trust-region fields"——
  ExplorationSpec 只携带 uncertainty_floor/coef，trust-region 字段
  在 PPOParams。无害但措辞误导。

### 复核确认无问题（阴性结果）

- `code_snapshot.py` 核心机制可靠：`.git/snapshot.lock` flock 串行化、
  `write-tree`/`commit-tree` 不动 HEAD、`read-tree` 恢复原暂存区、
  非 git 环境静默跳过。问题只在 repro 命令生成（P-BFW-1）。
- checkpoint 链完整：原子写（tmp+os.replace）、v2 格式存 RNG
  （python/numpy/torch_cpu/torch_cuda）+ loop_state、resume 时
  CUDA device 数变化有显式警告、obs 扩展 zero-pad 权重+optimizer
  state、prev_gvec 长度校验防错、`saved_update+1` 语义注释明确。
- PPOBuffer：`sampling_ctx` schema 不一致（部分 trajectory 缺字段）
  fail-loud 报错而非静默补零；zero-length trajectory 的 offset
  簿记一致；empty buffer 双路径（构造时+ppo_update 入口）均安全。
- `_grad_signal_diag`：identity 校验（mean(p_i)=‖G‖）用 z-score
  而非固定阈值、非有限梯度显式排除并计数、dedicated RNG 不碰
  训练流、norm bins 跨 update 冻结且 meta.json 有 norm_axis 语义
  守卫——设计严密。
- `trainer.py` 早停语义与文档一致：k3 KL、滑窗跨 epoch、B1 actor
  停而 critic 续、timeline/epoch_frames 只在 dump_callback 激活时
  产生开销。
- `SamplingContext`/`StochasticPolicy`/`Trajectory` 三个契约文件
  注释全部与实现一致（record_fields 的 None=缺失语义、from_batch
  的未知字段容忍、has_delta 的逐元素保守判据）。
- `debug.py` CLI：14 个子命令均有实现，epilog 与 dumpkit/CONTEXT.md
  呼应；`dump_request.py` sentinel 协议与 loop.py 的轮询消费吻合。

### 建议清单（留待裁决，不在本轮动手）

- S9：修 `format_repro_command` 补齐参数回放（或至少在命令头部
  注明"以下参数不会回放"清单）。
- S10：同步 `ExperimentPPO`/`ppo_update`/模块头图/`critic_mlp`
  四处 stale 文档（P-BFW-2/3/4）——全部是纯文档修复。
- S11：决定 `policies/todo/` 归属——修测试 import 让归档自洽，
  或整体移入 `obsolete/` 并改 import 前缀。
- S12：`loop.py:1075` 的 early-stop 日志改为通用措辞。

---

## baseline/framework/ppo policies + algos + tests 第二轮：精读续（2026-02 补）

> 范围：`ppo/policies/truncated_normal_mlp.py` 全读（706 行，家族基类），
> 其余 8 个变体抽查 docstring；`ppo/algos/advantages.py` 全读（227 行）；
> `ppo/tests/` + `policies/test_*.py` 全量实测。
> 定位：8 个策略族的共享实现与优势估计算子。

### 测试实测（本单元新跑）

`pytest policies/ tests/ algos/`（剔除 todo 与 test_viewer）：
**577 passed / 5 failed**，两簇：

- **test_dump_delta.py ×4（测试过时，非实现回归）**：
  `compute_delta` 现行语义为 row 0 = dump 所属 update 的 post-update
  策略（`policy_exports/u{u}`），docstring（L5-16）与实现一致并给出
  了语义理由（"row 1 = rollout policy 本身"）。但 4 个测试仍按旧
  语义断言（row 0 = rollout 策略 u−1，gen_updates 从 u−1 起计），
  且 fixture 不建 u4 的 export 导致 missing_updates 多出 4。
  判定：**有意的语义变更后测试未跟进**。修测试即可。
- **test_trainer.py::test_checkpoint_rng_state_roundtrip（测试过时）**：
  新增正确的 `prev_gvec` 长度守卫（loop.py:471-481，obs 扩展后
  丢弃尺寸不符的旧梯度向量）把测试里长度 5 的假向量当畸形数据丢
  弃——守卫行为正确，测试应改用与 actor 参数量一致的 gvec。
- 另确认 2 个 collection error：`tests/test_viewer.py` import
  `_dump_gradsig`——该函数已移至 `dumpkit/dump_analysis.py`
  （`server.py:1296` 现以 `_da.dump_gradsig` 调用），重命名后测试
  未跟进；`policies/todo/` 3 个测试同 P-POL-1 类。

### 本轮新发现（文档/残留级）

- **P-BFW-8（确认）**：`algos/advantages.py:3` docstring 引用
  `baseline/DESIGN.md` §3.6——该文件不存在。
- **P-BFW-9（确认）**：`truncated_normal_mlp.py:627` docstring 声称
  "A parity test in ``test_truncated_normal.py`` verifies …
  bit-identical"——文件存在且通过，引用为真（阴性，记入以免再查）。
- `policies/__init__.py` 导出 10 个策略类（含 pre_tanh 族并注明
  "unfinished/on hold"——注释诚实）；`todo/` 归档含 8 个旧实现 +
  设计文档 + 自身已失效测试（见 P-BFW-5）。

### 阴性结果（复核确认）

- `TruncatedNormalPolicy`：截断正态采样（inverse-CDF 重参数化 +
  归一化项入 log_prob）、U=σ√(2π)Z/2 的精度越界说明（大 σ 时
  ~4e-4 超 1 并 clamp）、export 自包含（P0-6：模板文件非字符串
  拼接、无 repo import）、SUPPORTS_REFERENCE_DELTA 握手——实现与
  注释逐条吻合。
- `delta_max_sigma`：frozen/dynamic 两路语义、σ-floor
  （`max(c·|Δ|, σ)` 逐元素）、Δ detach 的理由、payload 缺失时
  fail-loud——均与注释一致；导出版本注意事项（跨进程旧类实例
  只能读属性不能调方法）有显式注释。
- `compute_gae`/`normalize_advantages`：terminated vs truncated
  契约（last_value 显式传入）、gauss_rank 的 tied-rank 处理、
  std<1e-8 短路——实现与注释一致；入参校验 fail-loud。

---

## ppo/dumpkit 第二轮：逐文件精读（2026-02 补）

> 范围：`dump_request.py`、`dump_capture.py`（954 行全读）、
> `dump_rollout.py`（全读）、`dump_render.py`（前 120 行 + 引用核对）、
> `frame_access.py`（头部契约 + 结构核对）、`metric_catalog.py`（此前已
> 抽查无漂移）、`viewer/server.py`（结构 + gradsig 段）。
> 定位：训练调试工具链——dump 捕获、帧访问层、离线诊断、viewer 后端。

### 本轮新发现问题

- **P-DK-1（确认，语义陈旧——tanh 时代遗留）**：`dump_rollout.py`
  的核心量在 atanh 域计算：`eps_raw = atanh(a_sampled) − atanh(a_det)`，
  docstring 明示 "a_det = tanh(mu) so atanh(a_det) recovers the raw
  mean"——这是 TanhGaussian 时代的建模前提。当前策略族（TruncNorm
  全部 8 个变体）的 deterministic action 就是 mean 本身，采样噪声
  住在线性 σ 域而非 atanh 域：atanh 残差是对线性残差的非线性
  翘曲（|a|→1 处发散放大），其 acf/频谱/saturated 统计仍**可计算**
  但"这就是采样噪声"的解读不成立。`viewer/server.py:1625-1649`
  的 rollout 端点重复同一前提。建议：改为线性残差
  `a_sampled − a_det`（truncnorm 下即 σ·z 的直接量），或按
  distribution_kind 分派。
- **P-DK-2（确认，硬编码基类）**：`dump_capture.py`
  `_export_stochastic_policy`（L562-565）无条件读取
  `policies/_export_template.py`（base TruncNorm 模板）并在两个
  wrapper 模板里写死 `ExportedTruncNormPolicy`——对 mixture /
  state-σ / bounded-σ / pre-tanh 任一非基类家族，dump 的
  stochastic_policy 是错模板+对 model.pt，导出的 policy.py 与
  权重不匹配（加载时模板校验会报错——晚败型失败，不在捕获时报）。
  另：`dump` 对携带 reference/delta 配置的 spec 显式
  NotImplementedError（L572-581，fail-loud 正确）。
- **P-DK-3（记录，序列化脆弱）**：`_serialize_explore_factor` 用
  `inspect.getsource` + `str.replace` 改函数名内联 callable——
  多行签名/装饰器/引用非标量全局量会静默产出坏代码（只捕获
  int/float/str/bool 全局标量）。适用范围有限但文档有声明。
- **P-DK-4（记录）**：`dump_capture.py:65` 注解用了未 import 的
  `Tuple`——`from __future__ import annotations` 下无运行期影响，
  纯整洁度问题。
- **P-DK-5（记录）**：`_flatten_observer_outputs` 的 key 集取自
  首个 episode（L168-169）——后续 episode 出现额外 observer key 会
  静默不进 episodes.npz。

### 阴性结果（复核确认）

- `dump_request.py`：sentinel 原子 rename 一次性消费、损坏文件
  警告而不删不崩——契约与实现一致。
- `frame_access.py` 头部契约（行空间语义、列式懒加载、None 语义
  不造零、traj_map 回退重建）与其使用方一致。
- `_make_frame_ids`：obs 内容 hash 定位 trajectory 归属，匹配失败
  标 `flat:` 不静默。
- `capture_dump` manifest 全字段（git provenance、has_* 标志）与
  落盘文件名表和模块头声明一致。

### dumpkit 遗留建议

- S13：dump_rollout 的 atanh 域分析改为线性域（或按导出清单的
  distribution_kind 分派解释器）。
- S14：stochastic_policy 导出按 MANIFEST 的 exported_class 选择
  对应模板，或在 capture 时对非基类家族 fail-loud 而不写错文件。

---

## 剩余区域：obsolete / curriculum / experiments_ppo / policy / docs / examples / 散落文件（2026-02 补）

> 范围：前一轮未逐文件覆盖的收尾区域。方法：import 级验证 +
> 死引用 grep + 注册表实测。

### 本区域发现

- **P-OBS-1（确认，死代码档案）**：`baseline/framework/obsolete/` 11 个
  文件（v1 框架 `ppo_loop/ppo_trainer/sac_loop/sac_trainer/experiment`
  + 4 个 eval/probe 脚本）**不被任何活代码 import**（全仓 grep 无
  非自身引用）。属于有意留档的 v1 框架——但 `experiment.py` 的删除
  使下游产生了真实断链（见下条）。
- **P-OBS-2（确认，import 级死亡）**：`baseline/humanoid21/curriculum/`
  的实验注册表整个死：`curriculum/experiments/__init__.py` 和
  `base.py` 等 5 个文件 import 已删除的 `baseline.framework.experiment`
  （实测 `ModuleNotFoundError`）。curriculum/ 顶层脚本多数只依赖
  `rollout`（仍存活），但注册表和依赖它的 eval/probe 脚本（包括
  `obsolete/` 里的 `eval_all_levels.py` 等 4 个）全不可用。
  CLAUDE.md 标"legacy superseded"但现状比 legacy 更糟：**import 即崩**。
- **P-TST-1（确认，连带死亡）**：根目录 `tests/test_stage_seg_rewards.py`
  collection error——import 上述已死注册表。该测试守护的
  stage_seg 相位奖励 off-by-one 修复当前**无任何活测试守护**。
- **P-EXP-1（确认，工作正常）**：`experiments_ppo` 注册表实测
  30 个实验全部可发现；`_discover()` 只 glob 顶层 `exp_*.py`——
  `archive/`（28 个）和 `todo/`（21 个）被正确排除。
  `base.py`（583 行）精读过：`--set` 参数 coercion 类型正确
  （bool 先于 int 检查）、delta_factor/reference_horizon/delta_mode
  三个组合约束 fail-loud、self-play job 构建干净。质量良好。
- **P-POL-1 复核**：`policy/baseline/` 81 个快照 policy.py 全部
  import 已删除的 `tanh_gaussian_mlp`（实测 u10328 加载
  ModuleNotFoundError）；另有一个 `.pyc` 残留（untracked，无碍）。
  `policy/random`、`policy/humanoid21/standing` 蓝图实测可构建可
  act。
- **P-DOC-1（补充确认）**：`docs/ENVIRONMENT_zh.md` 与英文版同样
  描述"四面墙方形场地 + 4 台墙角相机"——实际是 `battle_v2` 圆形
  24 段墙（circular_v2）。中英双版本同步过期。
- **P-MISC-1（记录）**：`_debug/run01-03/`、`debug_approach/`（百余张
  调试 PNG）是历史调试产物直接堆在仓库根；`scripts/` 只有一个
  `migrate_feet_forces_norm.py`（一次性迁移脚本，已完成使命，
  建议在 ledger 标记为一次性工具）。
- **P-EXM-1（阴性）**：`examples/` 10 个脚本的共享 `_common.py`
  import 实测通过；仅 2 个文件 import `baseline.framework.rollout`
  （存活模块）。无 import 级断链（未做运行级验证——运行需要完整
  仿真环境，留作后续抽查项）。

### 本轮总结与处置建议汇总

第 2.5 轮（框架/工具代码精读）累计确认问题分类：

| 类别 | 数量 | 代表 |
|---|---|---|
| 实现 bug | 3 | recorder on_post_episode 漏调、ConstantForce/VideoRecorder sticky override、observer events 容器错位 |
| 语义陈旧 | 3 | dump_rollout atanh 域、dump_capture 硬编码 TruncNorm 模板、DAMAGE_TARGET_PARTS 死条目 |
| 注释/文档漂移 | ~8 | on_eval "0-based"、experiment.py 示例过期方法名、format_repro_command 不全、critic_mlp TanhGaussian 引用 |
| 死代码/残留 | ~10 | _TURB_DEBUG 插桩、last_events/SyncStats 无消费者、changed 死变量、numpy 兼容层零租户、obsolete 目录 |
| import 级断链 | 4 | curriculum 注册表、policy/baseline×81、tests/stage_seg、bench_rollout |
| 测试过时 | ~10 | dump_delta×4、prev_gvec、viewer _dump_gradsig、policies/todo 自测 |

**优先级建议（仅供裁决）**：P-POL-1 一行修复价值最高（81 个资产复活）；
三个实现 bug 各自独立可修；dump_rollout 的 atanh 修正影响所有
已产出 dump 的解读口径，建议优先定语义。

---

## Phase 4 — 全项目文档审计（2026-02 补）

> 范围：除 `runs/` 下自动生成的 REPRODUCE/RECORD_GUIDE 之外的全部
> ~90 份 Markdown。方法：逐份读 + 关键声明与代码对照（import/路径/
> CLI 参数/API 名实测）。
> 结论分四档：**现役契约文档**（准确、高价值）/ **历史记录**（过时但合理，
> 是时间轴证据）/ **脱节文档**（声称是现状但与代码不符，最危险）/
> **个人备忘**（无契约义务）。

### A. 现役契约文档（核实准确，价值高）

| 文档 | 核实内容 |
|---|---|
| `envs/humanoid21/DATASPEC.md` | static/derived state 全部键名、contacts_vec SoA 字段、96 维布局(42+13+2+39)、feet_forces 体重归一化、sqrt 速度压缩——逐一与 simulator.py 对上。**A 层契约样板** |
| `envs/humanoid21/CONTROLSPEC.md` | 归一化位置控制、Target=a*Scale+Ref、KP/KD 固化原则——与实现一致 |
| `envs/humanoid21/OBSERVATION_zh.md` | 96 维观测布局与 DATASPEC 互洽 |
| `envs/humanoid21/CONTACT_DESIGN.md` | battle_v2 接触参数的实测推导，含 v1 教训附录——高质量工程记录 |
| `envs/humanoid21/ACCEPTANCE_CRITERIA.md` | 文档本身写得好——但**它是"未兑现的验收单"**：实现达不到（P-H21-4），且没有任何测试真的断言这些指标 |
| `docs/RULE.md` + `_zh` | HP 规则与 plugins.py 完全对上（ATTACK_PARTS=hand/foot、DAMAGE_RULES head3.0/torso1.0、force_scale=100、dt=0.002、100HP、KO）——准确 |
| `docs/SUBMISSION.md` + `_zh` | `${DIR}` 策略打包语法与 PolicyBlueprint 实现一致（平台侧无法离线核实） |
| `baseline/framework/ppo/GUIDE.md` | 全部 hook 名与 experiment.py 实测一致（on_eval/post_update/exploration/build_jobs/...）；数据流描述与 loop.py 吻合——**当前最好的"怎么写实验"文档** |
| `baseline/framework/ppo/README.md` | 文档索引准确，CLI 快速上手可用 |
| `baseline/experiments_ppo/README.md` | 注册机制描述与 `_discover()` 一致（顶层 exp_*.py、archive/todo 排除）；flags 全核实（`--set`/`--param`/`--reset-update` 都存在）；唯一漂移：列了 `on_update()` hook，实际只有 `post_update` |
| `baseline/framework/ppo/DESIGN_*` 四份 + `policies/DESIGN_*` 六份 + `POLICY_SELECTION.md` | 设计决策记录，与现行 v2 框架/TruncNorm 家族一致；POLICY_SELECTION 有实测数据背书（48 run） |
| `dumpkit/CONTEXT.md` / `DATA_FLOW.md` | 能力地图准确（含 gen_updates[0]=post-update 的正确语义）；唯一问题：把 atanh 域 eps 当作"采样噪声"描述——继承 P-DK-1 的 tanh 时代前提 |
| `envs/batchframework/discuss.md` + `ROADMAP.md` | 治理级文档，定方向不定 API；M0-M6 PLAN/RESULTS 是对应的里程碑档案 |

### B. 脱节文档（声称是现状、实际对不上——危害最大）

| 文档 | 脱节点（已验证） | 危害 |
|---|---|---|
| **`policy/README.md`** | `BaseCombatPolicy` 类不存在；`load_policy()` 函数不存在；`act()` 签名错（实际返回 `(action, extra)` 元组且 `want_extra` kwarg）；`combatbench.policy` 包路径不存在；引用已删 `ParallelRunner`；示例 `policy.act(obs)` 返回值形态全错 | **全仓最危险的文档**：整篇描述的是上一代已删除的策略契约，按它写代码的 AI 会产出完全错误的接口 |
| **`envs/humanoid21/README.md`** | `rule_blueprint.yaml` 不存在（实际 `blueprint.yaml`）；CLI 参数 `--blueprint/--policy-a` 是上一代（实际 `--env-blueprint/--policy-a-blueprint` 且吃蓝图 YAML 路径而非 `module:Class?k=v`）；`SPEC.md` 链接死；目录结构表与命令自相矛盾；CombatScoringObserver "输出命中事件"声称（P-H21-2 实际恒空） | 该目录的快速开始**全部不可执行** |
| **`baseline/humanoid21/README.md`** | `curriculum/train.py`/`curriculum/framework/`/`analyze_*.py` 均不存在；rewards 表 9 项 vs 实际 22 文件；plugins 表 4 vs 10；blueprints 表 12 vs ~33；整个训练流程指向已死的 curriculum 注册表 | 目录地图严重失真 |
| **`baseline/humanoid21/curriculum/README.md`** | 描述 `framework/{config,ppo_trainer,training_loop}.py` + `train.py`——**整套 v1 框架已删**（在 obsolete/ 且无 README 指路），无任何 legacy 声明 | 读它等于学一个已删除的框架 |
| **`README.md`（根）** | `get_termination_flags()` 不存在（真实 API：`is_episode_over()`/`is_agent_active()`/`get_agent_termination()`）；`assets/` 标注"MuJoCo XML models"实为纯图片；`OBSERVABILITY.md` 链接死；快速开始策略能加载前会撞 P-POL-1；"125+ runs"实为 874；"开发自己策略"一节全部指向已死的 curriculum 路径，**完全不提 train.py/experiments_ppo/GUIDE.md 这条活路** | 项目门面文档指错了主路径 |
| **`CLAUDE.md`** | 同 `get_termination_flags`；环境结构表列 `runtime_plugin.py`（已删）等旧文件；"Gymnasium-compatible"措辞过强（gymnasium 仅用于 `spaces`，无 `gym.Env` 适配器）；curriculum 描述为"legacy"但没说它 import 即崩 | AI 入门文档半过时 |
| **`envs/framework/README.md` / `DESIGN.md`** | `get_termination_flags()` 示例代码不可运行；CONTEXT.md 大面积引用已删模块（此前条目已记） | 契约示例不可信 |
| **`examples/README.md`** | 自述"规划草案 v2，等待确认后落地"——实际 01-09 全部实现且比提案多 3 个；文档从未更新为实现文档 | 把既有成果描述成未决提案 |
| **`baseline/humanoid21/follow/README.md`** | 引用 `baseline/experiments_v2/exp_follow.py`——目录已不存在 | 路径死 |
| **`baseline/humanoid21/end2end/README.md`** | 同上 `experiments_v2` 死路径 | 路径死 |
| **`baseline/humanoid21/blueprints/README.md`** | 整段是旧 CLI 调试命令（`--blueprint`/`--policy-a`/`curriculum.yaml` 均不存在） | 纯废文档 |
| **`envs/framework/RESET.md` §6/§3.2** | 规范表仍写 observer 钩子名 `on_reset`/`on_post_step`，但 G6 早已重命名为 `on_pre_episode`/`on_post_action_step`（代码已验证）；§7 状态栏声称"G1-G6 已落地"但 §3.4/§6 正文未同步 | 同一文档内部新旧并存 |
| **`envs/framework/SEED.md`** | `ParallelRunner` 段落、`_derive_seeds` 描述基于已删 runner；EpisodeRunner 侧的派生机制核实仍准确 | 半过时 |
| **`envs/framework/tests/README.md`** | 文件表列 5 个测试，实际 21 个文件 | 覆盖率描述失真 |
| **`envs/humanoid21/tests/README.md`** | 只描述 test_data_interfaces；7 个从不 assert 的 acceptance 测试的存在未提及 | 同上当 |
| **`ISSUES.md`** | 只登记 1 个问题（且已部分修复），与审计面差距悬殊——AUDIT.md 事实上接管了这个职责 | 残缺但可被 AUDIT 取代 |

### C. 历史记录/备忘（过时但定位合理，不需修）

`REVIEW_*`（untracked，自标"不纳入 Git"）、`V2_TRAINING_TIME_LOG`、`bootstrip.md` 各处、`curriculum/TRAINING_V1|V2`、`STANDUP_V2_TRAINING_HISTORY`、`balance_recover/` 六份、`end2end/` 四份、`FIXPLAN.md`、`policies/todo/*` 八份、`RESULTS_*`、`EXPERIMENT_LOG_*`、`obs_analysis/*`——
全部是诚实的时间戳档案。**问题不在内容而在准入**：它们和契约文档混排、
没有"历史档案"标记，AI/新人无法一眼区分"这是规范"还是"这是墓碑"。

### D. 缺失的文档（按缺口影响排序）

1. **准确的策略契约文档**——policy/README 全面失效后，"怎么写一个能提交的策略"没有任何正确文档。SUBMISSION.md 只说打包格式，不说 Policy/act 签名。这是面向参赛者的第一缺口。
2. **EnvBlueprint/PolicyBlueprint 的 YAML schema 文档**——蓝图是"复现别人实验"的核心杠杆，schema 只能从源码和样例逆向。
3. **round_runner/match_runner CLI 参考**——现有文档全是旧参数名。
4. **训练路径导览**——README 完全不提 `train.py`+`experiments_ppo`+`GUIDE.md` 这条活路；需要一个"我要训练/评估/调试"的任务→工具地图（DUMP/viewer/REPRODUCE 什么时候用）。
5. **rewards/plugins/blueprints 目录清单**——22 个 reward、10 个 plugin、33 个 blueprint 没有权威清单（humanoid21 README 表已失真）。
6. **Known-Issues 活跃登记**——ISSUES.md 名存实亡；需要一个"当前已知坑"活文档（AUDIT.md 暂代）。
7. **Determinism/seed 保证说明**——SEED.md 半过时；审计已证实进程内/跨进程 bit-identical，这个事实值得写成面向用户的承诺文档。
8. **dumpkit 用户向入门**——CONTEXT.md 是 AI 能力地图，缺"我训崩了，第一步看什么"的决策树式文档（DATA_FLOW 偏结构不偏流程）。
9. **batchframework 使用说明**——全是里程碑档案，没有"怎么跑一个 device 实验/什么时候别用"的用户向文档。
10. **`episode_options` 已支持 key 目录**——散落在 RESET.md §4.2 + 各插件 docstring，且 RESET 表还是规划时的"新增"标注意味着口径未更新。

### 处置建议（不动手写）

- S15：B 档脱节文档中 `policy/README.md`、`envs/humanoid21/README.md`、
  `baseline/humanoid21/README.md`+`curriculum/README.md`、根 README 五篇
  建议优先重写或加"已过时"横幅——它们都是"入口级"文档，错误成本最高。
- S16：历史文档统一打 archival 标记（一行 header 即可），把"规范/档案"
  二分显式化。
- S17：D 档缺口按 1→3→2→4 优先级补。

## [2026-10-01] envs/framework — S1–S8 建议执行记录

**对象**：`envs/framework/` 测试层 + `CONTEXT.md` + `episode_runner.py` docstring
**类别**：修复执行（用户对 AUDIT 第一批建议 S1–S8 拍板"按建议处理"）

**做了什么**：

- **S1**：删除 `tests/test_episode_runner.py`、`tests/test_parallel_runner.py`
  ——两者测试的是 runner 重构（`73fe8da3`）前的旧 API（`RolloutConfig`/
  `ObserverBinding`/`run_n_episodes`/`ParallelRunner`），新契约由新文件覆盖。
- **S2**：`test_policy.py` 重写——`act` 签名升级为
  `(obs, *, want_extra) -> (action, extra)`；删除 `TestCoerceAction`/
  `TestCallPolicy`（`coerce_action`/`call_policy` 已从 policy.py 移除）；
  `TestBuiltinPolicies` 适配 tuple 返回值。
- **S3**：`test_reset_chain.py`——`EpisodeRunner` 构造改为
  `policy_a=/policy_b=`；删 `run_n_episodes`/`options_fn` 测试（API 已删）；
  `TestI3` 改为经 `_ObsCaptureRecorder`（PostActionRecorder 子类）比对观测
  序列，因 `run_episode` 现返回 None。
- **S4**：`test_seed.py`——`EpisodeSeeds`→`_EpisodeSeeds`；删整节
  `TestDeriveBatchSeeds`（`_derive_batch_seeds`/`_parallel_derive_seeds` 已删，
  批量种子分配现属 `baseline/framework/rollout` 的 `Job.seed`）；
  `runner.policies`→`policy_a/policy_b`；两处 `result.seed` 改为
  `ctx.base_seed`（run_episode 不再返回 result）。
- **S5**：删 `TestRoundRunnerVideoSavePath` 中 3 个针对已删
  `_merge_video_path_into_options`/`videosave_path` 的测试；其中仍有效的
  `find_plugins` 测试挪入 `TestPerEpisodeOptionsOverride`。
- **S6**：`CONTEXT.md` 已在上一轮文档修复中重写（mental model/entry points/
  gotchas 全部对齐当前 API）。
- **S7**：`episode_runner.py` docstring L~46 `parallel_runner` 残留改为指向
  `baseline/framework/rollout` 的 `ParallelRollouter`；"subclass" 误述改为
  "composes"。
- **S8**：新增 `test_episode_runner_behaviors.py`（11 个测试）：
  `run_episode` 返回 None；duck-type 校验（无 `act` 拒收、非 Policy 子类
  鸭型放行 P0-6）；`post_termination_action="hold"` 重放最后动作且停调
  `act`；`"policy"` 模式继续调用；pre-episode 已终止 agent 仍获一次 act
  （锁存"a_active 恒从 True 起步"设计）；`want_extras` 透传到 act 与
  recorder `action_extras`；`EpisodeRunner.close` 不关 runtime。

**结果/证据**：`pytest envs/framework/tests/ -x -q` →
**216 passed, 0 failed, 0 collection errors**（修复前 159 pass / 3 fail /
5 err）。`tests/README.md` 文件清单已同步更新。

**备注**：顺带发现 `run_episode` 的 hold 分支中 `last_action_a is None`
RuntimeError 实际不可达（`a_active` 恒从 True 起步，首轮必然先调
`act`）——属防御性死代码，已用测试锁存当前语义，未改实现。

## [2026-10-01] P-FW-1 残留清理：README_zh.md

**对象**：`README_zh.md`
**类别**：修复执行（用户要求查 P-FW-1 是否还有未修到的）

**做了什么**：

- 复查 `grep -rn get_termination_flags`：6 处原文档中 5 处已修，
  **README_zh.md:59 漏修**（上一轮只改了英文版）。已按英文版同款改法
  修复为 `is_episode_over()`/`is_agent_active()`/`get_agent_termination()`。
- 顺带发现 README_zh 还有与英文版同一批的滞后点，一并对齐：
  L80 课程段改为指向 `experiments_ppo/README.md` + `ppo/GUIDE.md`
  （curriculum 标注归档）；快速开始 Python 段补 P-POL-1 失效 import
  提示；目录结构补 `framework/`+`experiments_ppo/`、rewards 改 20+、
  runs 改 800+、assets 描述修正；"开发自己的策略"文档清单中
  curriculum/OBSERVABILITY 死链替换为当前活文档。
- `CAPABILITY_LEDGER.md` Almost-Done 清单中 4 条已办事项标记 DONE。

**结果/证据**：`grep -rn get_termination_flags` 现仅剩 3 类有意保留的
命中：CONTEXT.md 的"该 API 已移除"提示、AUDIT.md 审计记录、
CAPABILITY_LEDGER 历史条目。

## [2026-10-01] P-FW-3 旧 Policy 契约残留清理

**对象**：examples/ + `envs/framework/policy.py` docstring + `policy/` 注释
**类别**：修复执行（用户拍板"按实际情况修复"）

**做了什么**：

- `examples/04_collect_rollouts.py`：`act_with_extras()` 改为新契约
  `act(obs, *, want_extra) -> (action, extra)`；`policies={}` 构造改
  `policy_a/policy_b`；`run_episode` 补 `want_extras=True`（否则 extras
  不会被请求）；`base_seed + ep` 种子算术改为 `SeedSequence.spawn`
  （对齐 SEED.md）。
- `examples/02_scripted_baseline.py`：SinusoidPolicy `act` 改 tuple 签名；
  `policies={}` 改 `policy_a/policy_b`；recorder `action_extras` 补默认 None。
- `examples/03_training_aids.py`：两处 `policies={}` 改 kwarg；种子算术
  同步改 SeedSequence.spawn。
- `examples/05_replay_and_inspect.py`：`policies={}` 改 kwarg；live 重跑
  种子改为 `SeedSequence(base_seed).spawn(ep_index+1)[ep_index]`，与 04
  的派生方式保持一致（spawn 按索引确定性，不依赖总数）。
- `examples/07_curriculum_recipe.py` `_RandomPolicy.act` 改 tuple 签名。
- `examples/08_rollout_collection.py`：`policies={}` 改 kwarg。
- `envs/framework/policy.py` docstring `load_policy` → `PolicyBlueprint`。
- `policy/random/policy.py`、`policy/humanoid21/standing/policy.py` 注释中
  `load_policy` → `PolicyBlueprint`/`file:` 加载。

**验证**：全仓库 grep 确认无 `call_policy`/`coerce_action`/`act_with_extras`/
`policies={` 残留（batchframework probe 的局部同名 `load_policy` 函数、
`_EpisodeSeeds.policies` 内部字段为同名不同物）；rollout 层与 gating 调试
脚本全部已是新契约；`pytest envs/framework/tests/` 217 全过；
9 个修改文件语法检查通过。

**有意不动**：
- `envs/framework/REVIEW_SUMMARY.md`——gitignore 的冻结复盘文档，历史
  记录不改写（其测试清单表行提及 `coerce_action`/`call_policy` 属当时的
  事实描述）。
- `baseline/framework/ppo/policies/todo/checkpoint.py`——归档代码注释。

## [2026-10-01] P-FW-4 `ctx._simulator` 沙箱旁路 —— 使用者排查 + 封堵方案

**对象**：`envs/framework/context.py:178`（`SimContext._simulator`）
**类别**：排查 + 方案（用户确认"不是逃生口"）

### 使用者排查（全仓 grep `ctx._simulator` / `accessor._simulator`）

- **生产代码/实验/蓝图：零使用**。注册表内 30 个 experiments_ppo、
  rewards/plugins/blueprints、rollout 层、batchframework——无一处通过
  `ctx._simulator` 取裸后端。
- **唯一使用群**：`baseline/humanoid21/balance_recover/gating/debug_*.py`
  共 9 个调试脚本（debug_obs_compare / debug_time_track2 /
  debug_snapshot_compare / debug_step_diverge / debug_deep_snapshot /
  debug_ctrl_compare / debug_lifecycle / debug_obs_compare 等），
  全是 balance_recover 调试战役期间的一次性诊断插件，拿
  `ctx._simulator` 是为了读 `sim.data`（`qacc_warmstart`/`solver_niter`/
  `ncon`/raw contacts）——**IDataAccessor 契约外的 MuJoCo 内部状态**，
  属于合理的"框架没提供的诊断面"需求，不是恶意绕过。
- **框架自身零消费**：`SimContext._simulator` 赋值后无任何框架代码读取
  （`_RuntimeCore`/`EnvRuntime` 各自持有 simulator 引用），是纯粹的
  暴露面。
- `ReadOnlySimContext`（observer/recorder 侧）无此字段、
  `_AccessorView`/`_MutatorView` 已名字混淆——**唯一未封堵的洞就是
  `SimContext._simulator` 这一个属性**。

### 封堵方案（待拍板）

推荐 **方案 A：直接删除 `SimContext._simulator`**（最干净，符合"不是
逃生口"的判定）：

1. `context.py:178` 删 `self._simulator = simulator`（无内部消费，直接删，
   连名字混淆都不用做）。
2. 9 个 gating 调试脚本改为**harness 层持引用**：它们都自己 build
   runtime，插件构造后可显式注入——
   `plugin.sim = runtime.simulator`（EnvRuntime 有公开 `simulator` 属性），
   插件钩子内用 `self.sim` 代替 `ctx._simulator`。每个脚本 ~3 行改动，
   语义不变（诊断插件本来就在沙箱外受信，只是引用路径改到正确的层）。
3. 新增审计测试 `test_audit_simulator_reach.py`：
   断言插件钩子里 `ctx._simulator`/`ctx.accessor._simulator`/
   `ctx.mutator._simulator`（只读钩 mutator=None 时）均不可达。
4. DESIGN.md/CONTEXT.md 沙箱小节补一句"SimContext 不携带后端引用"。

备选 **方案 B：名字混淆存 `self.__sim`**——与 `_AccessorView` 同款的
"显式 opt-out" 约定。若将来确实需要受控的诊断级透传再开这个口子；
但按"不是逃生口"的判定，A 更贴切（不留门）。

关联：P-FW-9（`_MutatorView` 缓存引用跨钩子不失效）是同类沙箱问题的
另一面——A 方案落地后建议一并处理（grant/revoke 改为生成新 view 或
view 内加 epoch 校验）。

## [2026-10-01] P-FW-4 `ctx._simulator` 沙箱旁路 —— 封堵实施（方案 A）

**对象**：`envs/framework/context.py` + `balance_recover/gating/debug_*.py`
**类别**：修复执行（用户拍板方案 A）

### 数量更正

前一条排查记录写"9 个调试脚本"，实为笔误——列出的文件名去重后是 **7 个**
（`debug_obs_compare` 被列了两次）。实际迁移的文件为：
`debug_obs_compare` / `debug_time_track2` / `debug_snapshot_compare` /
`debug_step_diverge` / `debug_deep_snapshot` / `debug_ctrl_compare` /
`debug_lifecycle`。

### 实施内容

1. **`context.py`**：删除 `SimContext.__init__` 里的
   `self._simulator = simulator`，原位留注释说明"刻意不携带裸后端引用"
   及诊断场景的正确路径（harness 层持 `runtime.simulator`）。
   框架内 `self._simulator` 零消费，无需其他改动。
2. **7 个 gating 调试脚本**：统一改为显式注入模式——
   插件 `__init__` 加 `self.sim = None`（注明 P-FW-4 来源），
   `env_bp.build(debug_plugins=[p])` 之后 `p.sim = runtime.simulator`，
   钩子内 `ctx._simulator` → `self.sim`。
   `debug_lifecycle.py` 里一处 `ctx.accessor._simulator` 探测性代码
   （`hasattr` 守卫、结果未被使用）一并删除。
3. **新测试** `envs/framework/tests/test_audit_simulator_reach.py`：
   - 探针插件（`require_mutator=True`）+ 探针 observer + 探针 recorder
     遍历全部钩子，断言 `ctx._simulator` / `ctx.accessor._simulator` /
     `ctx.mutator._simulator` 均不可达（observer 侧 `ReadOnlySimContext`
     连 `mutator` 字段都没有）。
   - 反向测试：只读钩（`on_post_action_step`）里 `ctx.mutator is None`
     且无任何 `ctx` 路径能拿到后端 → 写物理无门。
4. **文档**：`DESIGN.md` BasePlugin 边界加禁忌条目；
   `CONTEXT.md` SimContext 段落注明"不携带裸后端引用"。
   `CAPABILITY_LEDGER.md` P-FW-4 标 DONE（P-FW-9 单列仍为开放项）。

### 验证

- `PYTHONPATH=. pytest envs/framework/tests/ -q` → **219 passed**
  （217 + 新测试 2 个）。
- 全仓 grep `ctx._simulator` / `ctx.accessor._simulator` /
  `ctx.mutator._simulator`：除注释/新测试自身/审计记录外 **零命中**。
- 7 个脚本 `python3 -m py_compile` 全部通过。

### 残留边界（如实登记）

- 名字混淆的 `accessor._AccessorView__sim` / `mutator._MutatorView__sim`
  仍可达——这是 `_AccessorView`/`_MutatorView` 实现自带的"显式 opt-out"
  约定（需刻意拼名），属文档化逃逸面而非漏洞；若需绝对封闭要改 Python
  属性模型，超出本轮范围。
- **P-FW-9 未随本次关闭**：缓存的 `_MutatorView` 引用在 revoke 后仍可用
  （`self._mutator_view` 单例 + `ctx.mutator=None` 只断 ctx 不断已握引用），
  待独立修复（view 加 epoch/吊销校验）。

注：7 个 gating 调试脚本均被 `.gitignore`（`debug_*` 规则）排除，
属工作区一次性诊断工具——迁移改动不进 commit，但已在本地生效，
保证将来运行时不会因 `ctx._simulator` AttributeError 而挂掉。

## [2026-10-01] P-FW-5 静默失败 → fail-loud 修复

**对象**：`envs/framework/backend.py`、`envs/batchframework/backend.py`、
`envs/framework/common_plugins.py`
**类别**：修复执行（用户指示"改成 fail loud"）

### 实施内容

1. **`IDataMutator.apply_external_force`**（framework/backend.py:71）：
   默认实现 `pass` → `raise NotImplementedError`。docstring 改写为
   "可选能力 ≠ 可选调用"——后端不支持外力时调用方必须立刻失败，
   不允许静默丢扰动让实验在无外力下无声跑完。
   - 子类排查：`Humanoid21Simulator`（simulator.py:1179）与
     `ReplaySimulator`（raise ReplayReadOnlyError）均有正确实现；
     conftest `MockSimulator` 未实现——继承新默认即自动 fail-loud，
     框架测试无一路径静默依赖它。
   - 调用方排查：disturbance_plugins / host_compat / validation_*
     全部面向已实现后端，无误伤。
2. **`IBatchDataMutator.apply_external_force`**（batchframework/backend.py:235）：
   同款 `pass` 默认实现，同步改 `raise NotImplementedError`。
   `MjxHumanoid21Simulator`/`WarpHumanoid21Simulator` 均已实现。
3. **`VideoRecorderPlugin.on_post_episode`**（common_plugins.py）：
   - 删兜底 `except Exception: print("Error saving video...")`——
     ffmpeg 非零退出、管道断裂、IO 错等一律向外抛（strict runtime
     即中止 episode 并带原始栈）。
   - cv2 兜底路径：ffmpeg 缺失时 `import cv2` 失败 →
     `RuntimeError("无法保存视频…ffmpeg 与 opencv-python 均不可用")`
     替代原"Warning: cannot save video"静默丢帧。
   - 顺带补一个同级静默点：`cv2.VideoWriter` **打开失败本身不报错**
     （`isOpened()==False` 时 write 全静默丢弃）——加显式检查并抛错。
   - ffmpeg→cv2 降级链保留：那是有真实替代方案的合理 fallback。

4. **新测试** `test_audit_fail_loud.py`（5 个用例）：
   - 未实现后端的 `apply_external_force` 直接调用 + 经 `_MutatorView`
     转发均抛 `NotImplementedError`；
   - 无编码器（ffmpeg FileNotFoundError + cv2 ImportError）→ RuntimeError；
   - ffmpeg 非零退出 → RuntimeError 不再被吞；
   - cv2 VideoWriter 打开失败 → RuntimeError。

### 验证

- `PYTHONPATH=. pytest envs/framework/tests/ -q` → **224 passed**
  （219 + 新测试 5 个）。
- `git status` 仅涉及上述 4 个源文件 + 1 个新测试文件。

## [2026-10-01] P-FW-6 过期测试处置 —— 状态复核结案

**对象**：`envs/framework/tests/`
**类别**：复核 + 收尾（用户指示"顺应新实现，没必要可删"）

### 复核结论：P-FW-6 各子项均已由 S1–S8 落地，本次复核确认

| 原问题 | 处置 | 现状 |
|---|---|---|
| `test_episode_runner.py` / `test_parallel_runner.py` collection error | 已删——测的是已删 API（`RolloutConfig`/`ObserverBinding`/`run_n_episodes`/`ParallelRunner`），无迁移价值 | 文件不存在，README 有删除说明 |
| `test_policy.py` 测旧契约（`call_policy`/`coerce_action`/`act(obs)→array`） | 重写为 `(action, extra)` 契约 | 通过 |
| `test_reset_chain.py` 引已删符号 | 改 `policy_a/policy_b` 构造，TestI3 改 recorder 比对，`run_n_episodes` 用例删 | 通过（I1–I6 不变式仍被守护） |
| `test_seed.py` 引 `_EpisodeSeeds` 旧名/批量派生 | 改 `_EpisodeSeeds`，批量种子用例删（归 rollout 层） | 通过 |
| `TestRoundRunnerVideoSavePath` 3 个 fail | 删失效部分，`find_plugins` 有效用例保留 | 通过 |
| 薄 runner 新行为无覆盖 | `test_episode_runner_behaviors.py`（11 用例）：返回 None、hold/policy、want_extras、duck-type、close 语义 | 通过 |
| `tests/README.md` 清单滞后 | 重写为全量文件表 + 删除说明 | 本次又校正一处：`test_audit_*` 4 个 → 6 个（补 `simulator_reach`/`fail_loud` 两个回归探针） |

### 验证

- `pytest envs/framework/tests/ -q` → **224 passed / 0 failed /
  0 collection errors**。
- `CAPABILITY_LEDGER.md` 三行过时备注（"untested"/"stale test file"）
  已清除。

## [2026-10-01] P-FW-7 文档过期清单修复 + REVIEW_SUMMARY.md 处置

**对象**：`DESIGN.md` / `SEED.md` / `RESET.md` / `policy.py` /
`examples/README.md` / `examples/03_training_aids.py` /
`envs/framework/REVIEW_SUMMARY.md`
**类别**：修复执行（用户指示 + REVIEW_SUMMARY 删除裁决）

### 文档修复

- `DESIGN.md:151`：`ctx.termination_proposals: List[str]` → 实际字段
  `agent_termination_proposals: Dict[str, List[str]]` +
  `agent_terminated: Dict[str, bool]`，补 `request_termination(reason,
  agent_id=None)` 的唯一写入口语义（None=全局 / 指定 id=单方）。
- `SEED.md` 四处：L11 `run_n_episodes` → `ParallelRollouter` 的
  `Job.seed` / 直接 `run_episode(seed=...)`；L56-60 禁止条款改引
  `_resolve_seed` + `ctx.base_seed`（`EpisodeResult` 已删）；
  `EpisodeSeeds` → `_EpisodeSeeds`（内部 dataclass 不落盘）；
  "EpisodeResult 记录"节改为 `ctx.base_seed`。
- `RESET.md` 两处（历史 gap 记录）：`run_n_episodes(options_fn=...)`
  补"已删、批量 options 走 `Job.episode_options`"的更正注记，
  与既有 `ParallelRunner` 注记风格一致。
- `policy.py:5` 模块 docstring：`ParallelRunner` → `ParallelRollouter`
  worker。
- `examples/README.md` 04 示例描述：`ParallelRunner`/`RolloutConfig`
  (`store_extras`)/`EpisodeResult` 全部改为实际实现
  （EpisodeRunner 顺序跑 + `want_extras` + `SeedSequence.spawn`）。
- `examples/03_training_aids.py` docstring：`run_n_episodes(options_fn)`
  → `run_episode(options=...)` / `Job.episode_options`。
- `episode_runner.py:46`（S7 已修）、`CONTEXT.md`（S6 已重写）复核无误。

### REVIEW_SUMMARY.md → 已删除

按用户裁决"问题已解决则删"。逐项核对 §11 记录的 8 条技术债：

| 条目 | 状态 |
|---|---|
| 1. 文档滞后（CONTEXT/DESIGN/CLAUDE） | ✅ 已解决（CONTEXT S6 重写、DESIGN 本次修、CLAUDE 无 runtime_plugin） |
| 2. 钩子命名三套不统一 | 已由 RESET.md §6 承载（长期设计建议） |
| 3. BaseObserverPlugin 空别名 | 有意保留的兼容别名，observer_plugin.py:106-110 自文档 |
| 4. 录制体积 footgun | CONTEXT.md:165 + CAPABILITY_LEDGER 已载 |
| 5. 单 env 无向量化 | 架构路线说明，batchframework 为对应轨道 |
| 6. _AccessorView 沙箱非绝对 | context.py:69 docstring + P-FW-4 残留边界条目 |
| 7. process_token 不含 metrics/events | observer_plugin.py:283-287 内联注释（framework/C2） |
| 8. match_runner seed 派生 | 原文即判定"符合 SEED.md 规范"，非问题 |

无独有未承载信息 → 删除。注：文件被 `.gitignore`（`REVIEW_*.md`）
排除，删除不进 commit；`REVIEW_OVERVIEW.md` 索引中该行已标注
"已删（见 AUDIT.md P-FW-7）"，顺带标出另外两个失效引用
（baseline/framework、experiments_v2——后者目录已不存在）。

### 验证

- 活跃文档/代码中 `run_n_episodes`/`ParallelRunner`/`parallel_runner`
  残留：仅剩 RESET.md 更正注记、SEED.md "已删除"陈述、
  tests/README.md 删除说明——全为有意保留的历史注记。
- `pytest envs/framework/tests/ -q` → **224 passed**。

## [2026-10-01] P-FW-8 `EnvBlueprint.load` 缺 `${DIR}` 替换 —— 补齐（判为遗漏）

**对象**：`envs/framework/blueprint.py` /
`envs/framework/parameterized_blueprint.py` / `envs/framework/policy.py`
**类别**：修复执行（用户裁决"是遗漏，无影响直接改"）

### 影响面核查（改前）

- 全仓 env blueprint YAML **零处**使用 `${DIR}`（该特性原本只对
  policy 侧开放）→ 加替换是纯增量，`replace()` 对不含该变量的文本
  零副作用，**无回归面**。
- env blueprint 确有路径类字段可受益：`state_bank_path`、
  `policy_blueprint_path`、gating model dir——此前只能经
  `materialize()` override 传绝对路径，蓝图不可自迁移。

### 实施内容

1. `_DIR_VAR`/`_substitute_dir` 从 `policy.py` 上移到 `blueprint.py`
   （policy.py 本就 import blueprint 方向无环——policy 不 import
   blueprint，parameterized_blueprint import blueprint，新依赖安全）。
2. `EnvBlueprint.load`：解析前对原文做 `${DIR}` → YAML 父目录替换；
   经 `from_yaml` 自动物化 parameterized 文档的路径同样被覆盖。
3. `ParameterizedEnvBlueprint.load`：同款替换（直接 load 入口）。
4. `policy.py`：删本地实现，import 共享 `_substitute_dir`；
   注释保留 `${DIR}` 语义说明并注明 env 侧同样适用。
5. `test_blueprint.py` 新 `TestDirSubstitution`（3 用例）：
   `EnvBlueprint.load` 替换 config 内 `${DIR}`；
   `ParameterizedEnvBlueprint.load` 替换参数默认值里的 `${DIR}`；
   不含 `${DIR}` 的文件原样通过。

### 验证

- `pytest envs/framework/tests/` → **227 passed**。
- `envs/humanoid21/blueprint.yaml`（parameterized）load+materialize 正常。
- 三个模块独立 import 无环。

## [2026-10-01] P-H21-3 `NonFallConstraintPlugin` 死插件删除

**对象**：`envs/humanoid21/plugins.py`、`README.md`、`README_zh.md`
**类别**：修复执行（用户裁决"插件删掉"）

### 处置

- 删除 `NonFallConstraintPlugin` 类（plugins.py:22-110，89 行）：
  按旧版 `static_data['robot_info']`/`norm_params` schema 写的遗留，
  当前 schema 两键皆无 → 恒 no-op 且不报错，属 fail-silent 典型。
- 连带清理：模块 docstring 条目、`from scipy... import Rotation as R`
  （该 import 全文件仅此类使用；`json`/`os`/`time` 仍被
  `CombatScoringPlugin` 使用保留）。
- `README.md`/`README_zh.md` 插件实例清单：
  `NonFallConstraintPlugin` → `FrozenRobotPlugin`（同文件现存插件）。
- `envs/humanoid21/REVIEW_SUMMARY.md`（gitignored 历史复盘）保留不动——
  它记录的是复盘当时的事实。
- `CAPABILITY_LEDGER.md` 该行状态 UNSUPPORTED → REMOVED。

### 验证

- 引用排查：删除后全仓 `NonFallConstraint` 仅剩 gitignored
  REVIEW_SUMMARY 与本审计记录；**无蓝图、无代码、无测试引用**。
- `pytest envs/framework/tests/ envs/humanoid21/tests/ -q` →
  **274 passed**（227 + 47）。
- `python3 -c "import envs.humanoid21.plugins"` 正常，
  导出 `CombatScoringPlugin`/`FrozenRobotPlugin`。

## [2026-10-01] P-H21-4 假验收测试处置：降级为手动脚本 + 两个假测试改真 assert

**对象**：`envs/humanoid21/tests/test_acceptance.py` → `acceptance_check.py`、
`test_observation_symmetry.py`、`test_data_interfaces.py`、
`test_videos/`、`ACCEPTANCE_CRITERIA.md`、`tests/README.md`
**类别**：修复执行（用户裁决：降级为手动脚本 + 附带项全做）

### 处置

1. **`test_acceptance.py` → `envs/humanoid21/acceptance_check.py`**
   （git mv）：五个 `test_*` 函数改名 `measure_*`（pytest 不再收集），
   删 pytest fixture/import、删死 import（matplotlib 全文件零使用），
   `sys.path` 修正为新位置，视频输出目录改 `acceptance_videos/`，
   `__main__` 手动入口保留。文件头注明实测现状——该文件是 670 行
   的 PD 质量测量仪器，作为唯一复测工具保留而非删除。
2. **`test_videos/`（5 个 mp4）删除**：更正前审计条目——这些 mp4
   **从未入库**（`.gitignore` 的 `*.mp4` 规则一直覆盖），只是本地
   生成产物，本次清理本地目录。`acceptance_videos/` 同样被 `*.mp4`
   规则覆盖。
3. **`test_observation_symmetry.py`**：`return True/False` →
   `assert not tests_failed`（实测当前通过，等于白捡一条真回归）。
   `__main__` 入口同步改直接调用。
4. **`test_data_interfaces.py::test_static_data`**：内部本就全是
   真 assert——只删尾部 `return sim`（PytestReturnNotNoneWarning
   消除）。更正前审计"虚过"表述：该用例的断言一直有效，只是尾部
   return 触发警告。
5. **`ACCEPTANCE_CRITERIA.md`**：顶部加现状标注——§1/§2/§3 三项
   当前结构性不达标（目标契约性质，复测走 acceptance_check.py）。
6. **`tests/README.md`**：文件表去掉 test_acceptance 行、补移动
   说明；`test_audit_stale_contacts` 标注改为"已修复回归"（配合
   用户同步做的探针→回归改写）。

### 验证

- `pytest envs/humanoid21/tests/` → **42 passed, 0 warnings**
  （原 47：-5 个假绿收集项；真实覆盖不降反升——symmetry 现在会真挂）。
- `acceptance_check.py` py_compile 通过；不在 pytest 收集路径。

## [2026-10-01] P-H21-5 `envs/humanoid21/README.md` 残留失真修复

**对象**：`envs/humanoid21/README.md`
**类别**：修复执行（逐项核对 CLI/返回结构/文件清单后）

### 逐项核对结果

| 审计条目 | 现状 |
|---|---|
| `rule_blueprint.yaml` → `blueprint.yaml` | ✅ 前一轮 B 档修复已改 |
| CLI 参数 `--env-blueprint` 等 | ✅ 已对 `--help` 输出逐参数核实，全部真实 |
| `termination_reasons: ['timeout']`（list） | **本次修**：实际是 per-agent dict `{'robot_a': [...], 'robot_b': [...]}`（round_runner.py:117-120），示例已改 |
| `SPEC.md` 引用 | ✅ 前一轮已删 |
| 目录结构漏 `disturbance_plugins.py`/`meta.py`/XML | ✅ 前一轮已补；**本次补**：第三个 XML `battle_circular_v2.xml` 未列，且它是当前场地（`simulator.py:35 ARENA_XML`）——蓝图节"battle_v2.xml"更正为 circular_v2，目录树三者并立标注 v1/v2 为历史保留 |
| plugins.py"约束/冻结"描述 | 本次改"冻结（FrozenRobotPlugin）"——NonFallConstraint 已删（P-H21-3） |

### 验证

- `round_runner` 快速开始命令**端到端实跑通过**（seed=42，双方
  timeout，输出 dict 与新文档一致）。
- `policy/blueprints/random.yaml`、`policy/blueprints/humanoid21/standing.yaml`
  均存在；`RandomCombatPolicy(scale=...)` 签名属实。

## [2026-10-01] P-H21-6 资产盘点处置：CLAUDE.md 场地清单修正

**对象**：`CLAUDE.md`（envs/humanoid21 目录条目）
**类别**：修复执行（用户指示"按实际情况修复"；REVIEW_SUMMARY 属
git 外文档，不管）

### 逐项核对

- `CLAUDE.md` 两处（L11 assets 注释 + L20 目录树）只列 v1/v2 →
  已改为 `battle_circular_v2.xml` 为当前场地、v1/v2 标注
  "历史保留/迁移工具用"。
- `battle_v1/v2.xml` 的旧代码引用**核实为合法而非过期**：
  `scripts/migrate_feet_forces_norm.py` 是 v1 时代数据的迁移工具，
  `mocap/retarget*.py` 明确提取 battle_v1 的关节结构——它们的
  注释本就声明面向 v1，正确。
- `envs/humanoid21/README.md` 已在 P-H21-5 修复（三个 XML 并立、
  circular_v2 标当前）。
- `obs_analysis/`、`pose_images/`：非代码档案资产，保留合理，不动。
- `envs/humanoid21/REVIEW_SUMMARY.md`：git 外考古文档，按指示不管。

### 验证

- 活跃文档中 `battle_*` 提及仅剩正确表述（README/CLAUDE）。
- 文档类改动，无代码影响。

## [2026-10-08] P-H21-7 小问题集合处置

**对象**：`envs/humanoid21/simulator.py`、`plugins.py`、`acceptance_check.py`、
`observer_plugins.py`、`envs/framework/recorder.py`、`envs/framework/backend.py`
**类别**：修复执行（用户复核确认逐项建议，完成后暂不提交、人工 review）

### 逐项处置

1. **`MUJOCO_GL` 硬覆盖** → 改 `os.environ.setdefault`（与下一行
   PYOPENGL_PLATFORM 一致；嵌入方预设值不再被静默盖掉）。
2. **`reset(seed)` 未消费** → `self._seed = seed` 落地保存（当前
   确定性仿真无 RNG 消费者，保存以供溯源/未来随机初始化），
   docstring 如实注明"seed 不驱动任何随机源"。
3. **`get_sensor_data()` 恒 `{}`** → 契约内 stub，不动。
4. **渲染失败返回全黑帧** → `get_broadcastview_image` 失败返回
   `None`（+warning），签名改 `Optional[np.ndarray]`。调用方梳理：
   `VideoRecorderPlugin` 本就有 `frame is not None` 守卫（跳帧）；
   `recorder._write_image` 补 None → RuntimeError（拒绝写伪帧，
   fail-loud）；`observer_plugins.get_visualization_image` 补
   None → RuntimeError（替代原模糊的 shape ValueError）；
   `acceptance_check` 5 处视频帧采集补跳帧守卫；其余下游
   （visualize_samples/render_state_images/generate_pose_images）
   为一次性成像脚本，None 下游即炸属可接受 fail-loud。
   `framework/backend.py` docstring 补"失败应返回 None 而非伪帧"。
5. **`arena_radius = 3.44` 硬编码** → `__init__` 时从
   `model.geom_size[ground_geom_id][0]` 派生（无 ground geom 退回
   3.44），相机代码用 `self._arena_radius`。换 arena XML 自动跟随。
6. **`waist_upper`/`waist_lower` 死枚举** → `DAMAGE_TARGET_PARTS`
   收敛为 `{'head','torso'}`（`_get_part_category` 本就只有这两种
   受击产出）；命中分支的 `in ('torso','waist_upper','waist_lower')`
   简化为 `== 'torso'`。
7. **`min(force,1200)` 上限未文档化** → 提为类常量
   `MAX_DAMAGE_FORCE = 1200.0`，类 docstring 伤害公式同步写明。
8. **`while pop()` 清 events** → 已被用户的 EventJournal 改造
   顺带移除（journal append-only，pop 不再存在）。

### 验证

- `pytest envs/framework/tests/ envs/humanoid21/tests/` → 277 passed。
- 实跑冒烟：构造+reset(seed=42)+render 一帧 OK，`arena_radius`
  从模型正确读出 3.44。

## [2026-10-08] P-TF-1 复核结案：dump_delta 测试夹具已同步新语义

**对象**：`baseline/framework/ppo/tests/test_dump_delta.py`
**类别**：复核——已解决，无代码改动

- 用户执行 fix 前实测：**9 passed / 0 failed**。审计记录时的
  4 个失败已不存在。
- 修复时点：夹具在 `17d031fd`（delta reference-delta 重构，
  用户本人的并行开发线）中同步到新语义——现为
  `export_updates=(0,1,2,3,4)` + 断言 `gen_updates == [4,3,2,1]`，
  与审计建议的 `(1,2,3,4)`→`[4,3,2,1]` 语义等价且覆盖更全
  （多保留一档 u_0 基线参照）。
- `compute_delta` 的 row0=post-update u_N 语义经审计复核为有意
  改动（`85738c03` commit message 有正确性论证），实现正确。

## [2026-10-08] P-TF-2 复核结案：RNG roundtrip 测试夹具已适配

**对象**：`baseline/framework/ppo/tests/test_trainer.py::test_checkpoint_rng_state_roundtrip`
**类别**：复核——已解决，无代码改动

- 夹具已改为 `np.arange(sum(p.numel() for p in actor.parameters()))`
  （L3394-3398，注释即引用 `9af767fc` 的 stale-length 守护），
  正是审计建议的修法。不确定是用户哪次提交顺带修的，但当前
  HEAD 已是适配后状态。
- 裸 `pytest` 复跑时**仍报失败**，但失败点是
  `torch.randperm(64, device="cuda")` → **CUDA OOM**（GPU0 被
  在跑训练占满 23.8/24.5G），与断言无关。`CUDA_VISIBLE_DEVICES=3`
  复跑 → **1 passed**。属测试环境敏感而非逻辑失败；该测试设计为
  cuda 可用时顺带验证 cuda RNG 恢复，OOM 属环境不可控，不改代码。
