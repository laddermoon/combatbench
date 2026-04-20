# CONTEXT

> AI-oriented context memo for this directory. Keep concise. Humans may edit freely;
> auto-curation will preserve hand-written notes.

## Purpose

Backend-agnostic engine底座 for the combatbench multi-agent fighting sim. Provides a
**capability-scoped plugin lifecycle** over any physics backend (MuJoCo / Isaac /
PyBullet) via the `BaseSimulator` contract, plus a read-only observer pipeline,
recording, and replay. Any training code consumes this through `EnvRuntime`.

## Mental Model

- **`BaseSimulator`** (`backend.py`) — the physics backend contract. Implements
  `IDataAccessor` (5 read methods) + `IDataMutator` (set_core_state / set_action /
  apply_external_force) + lifecycle (`reset`, `physical_step`,
  `get_physical_frequency`). Nothing above this knows about MuJoCo etc.
- **`SimContext`** (`context.py`) — per-episode blackboard. Exposes `ctx.accessor`
  (always), `ctx.mutator` (granted per-plugin-per-hook; `None` when denied),
  `ctx.metrics` / `ctx.events` / `ctx.request_termination`.
- **`BasePlugin`** (`plugin.py`) — world-rule unit. Writes physics only at writable
  hooks AND only if it declares `require_mutator=True`. Both conditions checked
  per-call in `_PluginManager.invoke`.
- **`BaseObserverPlugin`** (`runtime_plugin.py`) — read-only policy-side unit
  (observations / rewards / debug views). Managed by the single
  `_ObserverDispatcherPlugin`, which owns the highest priority (`+1_000_000`) so
  its snapshots are always **fresh** for downstream plugins on the same hook.
- **`EnvRuntime`** (`env_runtime.py`) — the only public runtime entry. Takes
  `simulator`, `plugins`, `observer_plugins`, `recorders`. `step(action_a, action_b)`
  and `reset` return nothing; consumers pull via `get_observer_output(name)`.
- **`PostActionRecorder` / `BaseFrameRecorder`** (`recorder.py`) — side-effect
  observers that persist the full `IDataAccessor` surface to a standard on-disk
  layout (`static.json` + per-step JSON + PNG) with `manifest_version=2`.
- **`ReplaySimulator`** (`replay.py`) — implements `BaseSimulator` on top of the
  recorder's layout. Lets observers/plugins/training code run against recordings
  unchanged. Mutators raise `ReplayReadOnlyError` except `set_action` (silent no-op
  for EnvRuntime compatibility; actions come from the recording).

## Entry Points

- `backend.py` — `IDataAccessor`, `IDataMutator`, `BaseSimulator` contracts.
- `context.py` — `SimContext`, `ReadOnlySimContext`, termination API.
- `plugin.py` — `BasePlugin` + `require_mutator` permission flag.
- `runtime_plugin.py` — `BaseRuntimeUnit`, `BaseObserverPlugin`,
  `_ObserverDispatcherPlugin` (priority `+1_000_000`).
- `env_runtime.py` — `EnvRuntime` + internal `_RuntimeCore` + `_PluginManager`.
- `recorder.py` / `replay.py` — record-and-replay pair. See their module docstrings.
- `DESIGN.md` / `README.md` — human-facing architecture doc; this file intentionally
  does not duplicate them.

## How to Use

Run tests from `things/combatbench`:

```bash
python3 -m pytest envs/framework/tests/ -q
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
while runtime.is_episode_active:
    runtime.step(action_a, action_b)
    obs = runtime.get_observer_output("obs_a")
```

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
- **`EnvRuntime.step` / `reset` return nothing**. Pull observer outputs via
  `get_observer_output(name)`, shared info via `get_shared_info()`, termination
  via `get_termination_flags()`.
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

## Open Questions / Notes for AI

- No provenance metadata in recordings yet (seed / policy hash / code version /
  training step). If a future task asks "which run produced this recording?",
  that's a known gap — add to `static.json` in the recorder.
- Per-step JSON layout is the current bottleneck candidate for long rollouts.
  Binary sidecar migration is explicitly left as a follow-up and is designed to
  be backward-compatible (new keys only).
- `Gym` / `SB3` adapters are intentionally **outside** this package; do not add
  them here. See DESIGN.md §7.

<!-- USER NOTES (auto-curator will not rewrite below this line) -->
