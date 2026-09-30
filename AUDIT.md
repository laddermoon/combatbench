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
