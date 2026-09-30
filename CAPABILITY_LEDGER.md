# CombatBench Capability Ledger

Companion to `REGULARIZATION.md` §5–§6. This is the persistent audit state:
every capability discovered during the per-directory README pass gets one row.
Entries are written incrementally — presence means "discovered", not "verified
working" unless the Evidence column says so.

Status values (§5): STABLE / USABLE / WIP / LEGACY / UNSUPPORTED / OUT-OF-SCOPE.

## Status Overview

| Status | Count | Meaning |
|---|---|---|
| STABLE | 3 | Contract clear, tested, documented |
| USABLE | 6 | Works with documented limits |
| WIP | 0 | In progress, not usable yet |
| LEGACY | 1 | Works but superseded |
| UNSUPPORTED | 0 | Not usable / explicitly rejected |
| OUT-OF-SCOPE | 1 | Excluded from this round (SAC) |

## Capabilities

| Capability | Domain | Status | Evidence | Docs | Gaps / Notes |
|---|---|---|---|---|---|
| `EnvRuntime` + plugin lifecycle (accessor/mutator, 6 hooks, per-agent termination) | envs/framework | STABLE | pytest: permission/dispatch/lifecycle suites pass (157 total) | README, DESIGN.md, RESET.md | — |
| Observer pipeline (`BaseObserverPlugin` + dispatcher) | envs/framework | STABLE | pytest: observer dispatch/ordering suites pass | README | — |
| Recorder/replay (`PostActionRecorder`, `BaseFrameRecorder` v2, `ReplaySimulator`) | envs/framework | STABLE | pytest: recorder lifecycle + replay suites pass | README | ~50KB/step derived_state footgun; no provenance metadata |
| `EpisodeRunner` (thin episode loop) | envs/framework | USABLE | runs; new behaviors untested | README | hold/want_extras/duck-type untested; stale test file exists |
| `RoundRunner` / `MatchRunner` (round + match eval, CLI) | envs/framework | USABLE | CLI functional; core tests pass | README, CLAUDE.md | 3 stale tests for removed `videosave_path` API |
| `EnvBlueprint` / `ParameterizedEnvBlueprint` (env as YAML) | envs/framework | USABLE | test_blueprint.py passes | README | — |
| `Policy` ABC / `PolicyBlueprint` / `ParameterizedPolicyBlueprint` | envs/framework | USABLE | blueprint round-trip works | README, policy/README.md | stale test file for removed `call_policy`/`coerce_action` |
| `recorder_viewer` web viewer | envs/framework | USABLE | CLI + bundled viewer.html | README | — |
| `ParallelRunner` (process pool) | envs/framework | LEGACY | — | gone | Removed in 73fe8da3; superseded by `baseline/framework/rollout` |
| SAC training path (`baseline/framework/sac/`, `experiments_sac/`) | training | OUT-OF-SCOPE | — | — | Declared immature; not inventoried this round |

## Almost-Done List (priority candidates to finish)

- `envs/framework/tests/` — 157 pass / 3 fail / 5 collection errors; stale
  tests for deliberately-removed APIs. Fix = small mechanical pass
  (see REGULARIZATION_DETAIL.md S1–S5). Pending user decision.
- `envs/framework/CONTEXT.md` — stale throughout (describes removed
  `parallel_runner.py`/`runtime_plugin.py`/old runner API). Rewrite or merge
  into README. Pending user decision.
- `episode_runner.py` docstring references removed `parallel_runner` (line ~46).

## Explicitly Unavailable List

- SAC path — out of scope this round, do not build experiments on it.
