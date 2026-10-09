# CONTEXT

> 类型：指南

> AI-oriented context memo for this directory. Keep concise. Humans may edit freely;
> auto-curation will preserve hand-written notes.

## Purpose

Backend-agnostic engine底座 for the combatbench multi-agent fighting sim. Provides a
**capability-scoped plugin lifecycle** over any physics backend (MuJoCo / Isaac /
PyBullet) via the `BaseSimulator` contract, plus a read-only observer pipeline,
recording, and replay. Any training code consumes this through `EnvRuntime`.
Batch/process-parallel execution lives in `baseline/framework/rollout/` — this
package is deliberately single-process.

## Mental Model

- **`BaseSimulator`** (`backend.py`) — the physics backend contract. Implements
  `IDataAccessor` (5 read methods) + `IDataMutator` (set_core_state / set_action /
  apply_external_force) + lifecycle (`reset`, `physical_step`,
  `get_physical_frequency`). Nothing above this knows about MuJoCo etc.
- **`SimContext`** (`context.py`) — per-episode blackboard. Exposes `ctx.accessor`
  (always), `ctx.mutator` (granted per-plugin-per-hook; `None` when denied),
  `ctx.metrics` / `ctx.events` / `ctx.request_termination` /
  `ctx.episode_options` / `ctx.base_seed`. Deliberately carries **no raw
  simulator reference** — backends internals stay behind the accessor/mutator
  capability views; harness-side diagnostics hold `runtime.simulator` instead
  (P-FW-4 seal, pinned by `test_audit_simulator_reach.py`). `ctx.events` is an
  `EventJournal` — **append-only within an episode** (no pop/clear/remove;
  framework resets at `clear_episode_state` and bumps `epoch`). "This step's
  events" is a consumer-side cursor diff `(epoch, len)` — reference:
  `CombatScoringObserver`.
- **`BasePlugin`** (`plugin.py`) — world-rule unit. Writes physics only at writable
  hooks AND only if it declares `require_mutator=True`. Both conditions checked
  per-call in `_PluginManager.invoke`.
- **`BaseObserverPlugin`** (`observer_plugin.py`) — read-only policy-side unit
  (observations / rewards / debug views). Managed by the single
  `_ObserverDispatcherPlugin`, which owns the highest priority (`+1_000_000`) so
  its snapshots are always **fresh** for downstream plugins on the same hook.
- **`EnvRuntime`** (`env_runtime.py`) — the only public runtime entry. Takes
  `simulator`, `plugins`, `observer_plugins`, `recorders`. `step(action_a,
  action_b)` and `reset` return nothing; consumers pull via `get_observation()`
  / `get_observer_output(name)` / `is_episode_over()` /
  `get_agent_termination()`.
- **`PostActionRecorder` / `BaseFrameRecorder`** (`recorder.py`) — side-effect
  observers that persist the full `IDataAccessor` surface to a standard on-disk
  layout (`static.json` + per-step JSON + PNG) with `manifest_version=2`.
- **`ReplaySimulator`** (`replay.py`) — implements `BaseSimulator` on top of the
  recorder's layout. Lets observers/plugins/training code run against recordings
  unchanged. Mutators raise `ReplayReadOnlyError` except `set_action` (silent no-op
  for EnvRuntime compatibility; actions come from the recording).
- **`EpisodeRunner`** (`episode_runner.py`) — thin per-episode loop above
  `EnvRuntime`: derives seeds (`SeedSequence`), resets runtime + policies, then
  runs `policy.act → runtime.step` until `is_episode_over()`. Returns `None` —
  all episode data is pulled from attached recorders afterward. Handles
  per-agent termination (`post_termination_action`: `"policy"` keeps calling
  `act` after that agent terminated, `"hold"` replays its last action).
- **`RoundRunner`** (`round_runner.py`) / **`MatchRunner`** (`match_runner.py`) —
  compose `EpisodeRunner`: RoundRunner adds CLI/result-dict/video plumbing for a
  single round; MatchRunner runs N rounds with HP carry-over via
  `episode_options["initial_health_*"]`.

## Entry Points

- `backend.py` — `IDataAccessor`, `IDataMutator`, `BaseSimulator` contracts.
- `context.py` — `SimContext`, `ReadOnlySimContext`, termination API.
- `plugin.py` — `BasePlugin` + `require_mutator` permission flag.
- `observer_plugin.py` — `BaseRuntimeUnit`, `BaseObserverPlugin`,
  `_ObserverDispatcherPlugin` (priority `+1_000_000`).
- `env_runtime.py` — `EnvRuntime` + internal `_RuntimeCore` + `_PluginManager`.
- `policy.py` — canonical `Policy` ABC (`act` / `reset` / `to_blueprint`) +
  `PolicyBlueprint` / `ParameterizedPolicyBlueprint` loaders. The single source
  of truth for the policy contract.
- `episode_runner.py` — `EpisodeRunner` (+ `AGENT_IDS`).
- `round_runner.py` — `RoundRunner` (+ `__main__` CLI).
- `match_runner.py` — `MatchRunner` (+ `__main__` CLI).
- `recorder.py` / `replay.py` / `recorder_viewer.py` — record-and-replay trio.
- `blueprint.py` / `parameterized_blueprint.py` — `EnvBlueprint` YAML loading
  and `${param}` materialization.
- `DESIGN.md` / `README.md` / `RESET.md` / `SEED.md` — human-facing design docs;
  this file intentionally does not duplicate them.

## How to Use

Run tests from `things/combatbench`:

```bash
PYTHONPATH=. python3 -m pytest envs/framework/tests/ -q
```

Minimal consumer skeleton (real example in `README.md`):

```python
from envs.framework import EnvRuntime, BaseObserverPlugin
runtime = EnvRuntime(
    simulator=MySimulator(),
    plugins=[...],                              # world rules
    observer_plugins={"obs_a": MyObs()},        # read-only outputs
    recorders=[BaseFrameRecorder(output_dir=...)],  # optional
    phy_steps_per_action=10,
)
runtime.reset()
while not runtime.is_episode_over():
    runtime.step(action_a, action_b)
    obs_a, obs_b = runtime.get_observation()
```

Per-episode driver (recommended — handles seed derivation / resets):

```python
from envs.framework import EnvRuntime, EpisodeRunner
runtime = EnvRuntime(simulator=MySimulator(), ...)
runner = EpisodeRunner(runtime=runtime, policy_a=policy_a, policy_b=policy_b)
runner.run_episode(seed=42, options={"initial_health_a": 80})
# episode data lives in the attached recorders; nothing is returned
```

Single round / full match (with video):

```python
from envs.framework import EnvBlueprint, PolicyBlueprint, RoundRunner
bp = EnvBlueprint.load("envs/humanoid21/blueprint.yaml")
with RoundRunner(
    blueprint=bp,
    policy_a=PolicyBlueprint.load("policy/blueprints/random.yaml").build(),
    policy_b=PolicyBlueprint.load("policy/blueprints/humanoid21/standing.yaml").build(),
) as runner:
    result = runner.run(seed=42)   # {steps, termination_reasons, health_a, health_b, seed}
```

Batch / multi-process rollout lives in `baseline/framework/rollout`
(`ParallelRollouter`, `Job`) — `Job.seed` + `Job.episode_options` carry the
per-episode inputs; each worker internally runs `EpisodeRunner`.

Replay a recording:

```python
from envs.framework import EnvRuntime, ReplaySimulator
replay = ReplaySimulator("/path/to/recording_root")
runtime = EnvRuntime(simulator=replay, phy_steps_per_action=1, ...)
```

## Conventions & Gotchas

- **Permission enforcement is per-plugin-per-call**. `_PluginManager.invoke`
  regrants/revokes `ctx.mutator` before every plugin call based on
  `allow_mutator(hook) and plugin.require_mutator`. A plugin that forgets to
  override `require_mutator` silently gets `ctx.mutator is None` even on writable
  hooks. This is **intentional** (least privilege); do not work around it.
- **Observer dispatcher runs FIRST**. `priority = +1_000_000`; the sort in
  `_PluginManager` is `reverse=True`. Downstream plugins (termination / reward)
  read fresh observer output. Do not re-order.
- **Hooks and their writability** (pinned by tests in
  `tests/test_permission_control.py` and `tests/test_plugin_dispatch.py`):
  `on_pre_episode` / `on_pre_action_step` / `on_pre_phy_step` / `on_post_phy_step`
  are writable; `on_post_action_step` / `on_post_episode` are read-only. `set_*`
  calls on a read-only hook go through `ctx.mutator`, which is `None` → raises.
- **`EnvRuntime.step` / `reset` return nothing**. Pull observations via
  `get_observation()`, observer outputs via `get_observer_output(name)`,
  termination via `is_episode_over()` / `is_agent_active(agent_id)` /
  `get_agent_termination()`. (No `get_termination_flags()` — that API was
  removed; several old docs still reference it.)
- **`EpisodeRunner.run_episode` returns `None`** — read episode data from
  recorders; the runner deliberately aggregates nothing.
- **`_RuntimeCore` / `_PluginManager` / `_ObserverDispatcherPlugin` are private**.
  They are not re-exported from `__init__.py`; do not build against them.
- **Recorder schema is versioned**. `MANIFEST_VERSION=2` includes `derived_state`
  / `sensor_data` / `action` / `static.json`. `ReplaySimulator` rejects v1 by
  default (`strict_manifest_version=True`). Schema changes must bump version.
- **Recording size footgun**. `save_accessor_state=True` (the default) persists
  `derived_state` which is ~50 KB per step on humanoid21. Long episodes × many
  steps-per-file JSON layout turn filesystems into a swamp. Turn off what you
  don't need, or migrate to an `.npz` sidecar (see `recorder.py` docstring).
- **Replay caveats**: one `physical_step()` == one recorded frame, so
  `phy_steps_per_action` during replay must equal recorder stride (usually 1).
  `set_action` is a silent no-op on `ReplaySimulator`; the recorded action is
  authoritative. JSON round-trip loses dtype → arrays come back as `float32`.
- **Chinese comments / docstrings** are standard in this tree. Keep language
  consistent with surrounding code when editing.
- **`EpisodeRunner` hardcodes `robot_a` / `robot_b`**. This is intentional —
  the project is 1v1 combat, not generic MARL. Do not generalize to arbitrary
  agent counts without explicit discussion; downstream consumers key by these
  exact strings.
- **Observer output shape contract** (see `observer_plugin.py` docstring):
  observation plugins return a policy-ready value directly (no `(obs, info)`
  tuples); reward plugins return a scalar or a dict.
- **Seed propagation in `EpisodeRunner`**: `run_episode(seed)` derives a
  `_EpisodeSeeds` bundle (runtime / policy_a / policy_b / seedable plugins) via
  `SeedSequence.spawn`; `seed=None` resolves to a concrete uint32 published on
  `ctx.base_seed`. Batch seed assignment is the Job builder's job (rollout layer).
- **`RoundRunner` owns the runtime lifecycle** — `runtime.close()` on
  `runner.close()` / context exit. `EpisodeRunner` does NOT close the runtime —
  caller owns lifecycle.
- **Per-episode options**: `run_episode(options=...)` / `runtime.reset(options=)`
  publish on `ctx.episode_options`; plugins read them in `on_pre_episode`.
  Environment-only keys — policy knobs don't belong here.

## Open Questions / Notes for AI

- No provenance metadata in recordings yet (seed / policy hash / code version /
  training step). If a future task asks "which run produced this recording?",
  that's a known gap — add to `static.json` in the recorder.
- Per-step JSON layout is the current bottleneck candidate for long rollouts.
  Binary sidecar migration is explicitly left as a follow-up and is designed to
  be backward-compatible (new keys only).
- `Gym` / `SB3` adapters are intentionally **outside** this package; do not add
  them here. See DESIGN.md §7.
- Known gaps from the audit (see root `AUDIT.md`): abandoned-episode recorder
  `on_post_episode` is not invoked (P3-1); `VideoRecorderPlugin` episode-level
  `output_path` override permanently mutates the plugin (P3-2); `_MutatorView`
  lifetime is not enforced — a cached mutator reference stays usable on
  read-only hooks (P-FW-9).

<!-- USER NOTES (auto-curator will not rewrite below this line) -->
