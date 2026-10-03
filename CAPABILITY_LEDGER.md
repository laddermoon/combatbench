# CombatBench Capability Ledger

Companion to `REGULARIZATION.md` §5–§6. This is the persistent audit state:
every capability discovered during the per-directory README pass gets one row.
Entries are written incrementally — presence means "discovered", not "verified
working" unless the Evidence column says so.

Status values (§5): STABLE / USABLE / WIP / LEGACY / UNSUPPORTED / OUT-OF-SCOPE.

## Status Overview

| Status | Count | Meaning |
|---|---|---|
| STABLE | 5 | Contract clear, tested, documented |
| USABLE | 11 | Works with documented limits |
| WIP | 0 | In progress, not usable yet |
| LEGACY | 1 | Works but superseded |
| UNSUPPORTED | 1 | Not usable / explicitly rejected |
| OUT-OF-SCOPE | 1 | Excluded from this round (SAC) |

## Capabilities

| Capability | Domain | Status | Evidence | Docs | Gaps / Notes |
|---|---|---|---|---|---|
| `EnvRuntime` + plugin lifecycle (accessor/mutator, 6 hooks, per-agent termination) | envs/framework | STABLE | pytest: permission/dispatch/lifecycle suites pass (157 total) | README, DESIGN.md, RESET.md | — |
| Observer pipeline (`BaseObserverPlugin` + dispatcher) | envs/framework | STABLE | pytest: observer dispatch/ordering suites pass | README | — |
| Recorder/replay (`PostActionRecorder`, `BaseFrameRecorder` v2, `ReplaySimulator`) | envs/framework | STABLE (caveat) | pytest: recorder lifecycle + replay suites pass | README | ~50KB/step derived_state footgun; no provenance metadata; **AUDIT P-FW-2**: mid-physics termination → terminal frame has stale observer_outputs + `step_XXXXX` filename collision (test-proven) |
| `EpisodeRunner` (thin episode loop) | envs/framework | USABLE | test_episode_runner_behaviors.py covers hold/want_extras/duck-type | README | — |
| `RoundRunner` / `MatchRunner` (round + match eval, CLI) | envs/framework | USABLE | CLI functional; core tests pass | README, CLAUDE.md | — |
| `EnvBlueprint` / `ParameterizedEnvBlueprint` (env as YAML) | envs/framework | USABLE | test_blueprint.py passes | README | — |
| `Policy` ABC / `PolicyBlueprint` / `ParameterizedPolicyBlueprint` | envs/framework | USABLE | blueprint round-trip works; test_policy.py on new `(action, extra)` contract | README, policy/README.md | — |
| `recorder_viewer` web viewer | envs/framework | USABLE | CLI + bundled viewer.html | README | — |
| `ParallelRunner` (process pool) | envs/framework | LEGACY | — | gone | Removed in 73fe8da3; superseded by `baseline/framework/rollout` |
| SAC training path (`baseline/framework/sac/`, `experiments_sac/`) | training | OUT-OF-SCOPE | — | — | Declared immature; not inventoried this round |

| `Humanoid21Simulator` (MuJoCo backend, 96-dim obs, normalized PD, broadcast cam) | envs/humanoid21 | STABLE (caveats) | 43 tests pass; blueprint round-trip | DATASPEC/CONTROLSPEC/OBSERVATION_zh | **P-H21-1** stale contacts cache (feet_forces lag); seed arg unused; render failure → black frame |
| `CombatScoringPlugin` (per-substep damage, KO, score log) | envs/humanoid21 | STABLE | exercised by matches/experiments | plugins.py docstring | — |
| `CombatScoringObserver` | envs/humanoid21 | USABLE (bug) | — | — | **P-H21-2**: reads `metrics['events']` (never written) → events/step_hit_events/step_damage_taken always empty |
| ~~`NonFallConstraintPlugin`~~ | envs/humanoid21 | REMOVED | — | — | **P-H21-3** resolved: dead plugin deleted (was silent no-op on stale static_data schema); README lists `FrozenRobotPlugin` instead |
| `FrozenRobotPlugin` | envs/humanoid21 | USABLE | — | — | — |
| Disturbance family (12 classes: RandomPush/InitPerturb/Wind/HeadStrike/RandomFallen/Impulse/ConstantForce/HeightLimit/StateBank/…) | envs/humanoid21 | USABLE | referenced by live exp_standup*/exp_step experiments | — | per-plugin maturity varies; not individually tested |
| `Humanoid21BalanceAnalysisObserver` (CoM/ankle support analysis + plan-view render) | envs/humanoid21 | USABLE | test_balance_analysis.py | — | heavy compute; visualization path |
| `blueprint.yaml` (parameterized rules: initial_distance/max_steps) | envs/humanoid21 | USABLE | — | — | README calls it `rule_blueprint.yaml` (stale name) |
| Arena XMLs | envs/humanoid21 | mixed | — | CONTACT_DESIGN.md | `battle_circular_v2.xml` = live default; `battle_v1`/`battle_v2` = legacy refs only |

## Almost-Done List (priority candidates to finish)

- ~~`envs/framework/tests/` stale tests~~ — **DONE**: dead-API test files
  removed, survivors migrated to new contracts, thin-runner behaviors pinned
  by `test_episode_runner_behaviors.py` (S1–S8); currently 224 pass /
  0 fail / 0 collection errors (AUDIT P-FW-6).
- ~~`envs/framework/CONTEXT.md` stale~~ — **DONE**: rewritten against current
  API (S6).
- ~~`episode_runner.py` docstring `parallel_runner` ref~~ — **DONE** (S7).
- ~~`get_termination_flags()` in docs~~ — **DONE**: all 6 stale references
  replaced with `is_episode_over()`/`is_agent_active()`/`get_agent_termination()`
  (README.md + README_zh.md + CLAUDE.md + envs/framework README/CONTEXT/DESIGN;
  remaining hits are intentional audit-record mentions).
- ~~`ctx._simulator` sandbox bypass~~ — **DONE**: attribute deleted from
  `SimContext`; 7 gating debug scripts now take an explicit
  `plugin.sim = runtime.simulator` injection; seal pinned by
  `test_audit_simulator_reach.py` (AUDIT P-FW-4). Mutator-view lifetime
  (P-FW-9) is a separate still-open issue.

## Explicitly Unavailable List

- SAC path — out of scope this round, do not build experiments on it.

---

## Round 2.5 精读增补（框架/工具层逐文件审计后）

| 能力 | 路径 | 状态 | 依据 |
|---|---|---|---|
| PPO 实验注册表 | `baseline/experiments_ppo/` | STABLE | 30 实验实测可发现；archive/todo 正确排除；base.py 质量良好 |
| Dump 捕获 | `ppo/dumpkit/dump_capture.py` | USABLE | 工件齐全；但 stochastic 导出硬编码 TruncNorm 基类模板 |
| Dump 帧访问层 | `dumpkit/frame_access.py` | STABLE | 懒加载/None 语义契约与实现一致 |
| Dump delta 诊断 | `dumpkit/dump_delta.py` | USABLE | 语义自洽（row0=post-update），但 4 个测试仍期望旧语义 |
| Dump rollout 诊断 | `dumpkit/dump_rollout.py` | USABLE(语义陈旧) | atanh 域前提是 tanh 时代建模，对 TruncNorm 家族解读失真 |
| Dump viewer | `dumpkit/viewer/` | USABLE | 功能可用；test_viewer 引用了不存在的 `_dump_gradsig` |
| 已训策略快照 | `policy/baseline/` (81 个) | BROKEN-可修复 | 全部 import 已删除模块；一行路径修复即可复活 |
| Curriculum 注册表 | `humanoid21/curriculum/experiments/` | DEAD | import 已删 `baseline.framework.experiment`，即崩 |
| v1 框架档案 | `baseline/framework/obsolete/` | ARCHIVE | 11 文件无活引用，纯历史留档 |
| Numpy 批量兼容层 | `batchframework/batch_plugin.py` | DORMANT | 769 行基础设施零租户插件 |
