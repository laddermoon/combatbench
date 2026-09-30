# envs/framework — Backend-agnostic simulation framework

The engine underneath CombatBench: a backend-agnostic, capability-scoped
plugin lifecycle over any physics backend (MuJoCo today), plus a read-only
observer pipeline, recording, replay, and episode/round/match runners.

Everything above this directory — training, evaluation, policies, the online
platform — talks to physics through `EnvRuntime`. It knows nothing about
MuJoCo, and plugins know nothing about each other.

```python
from envs.framework import EnvRuntime, TimeoutPlugin, VideoRecorderPlugin

runtime = EnvRuntime(
    simulator=my_simulator,            # any BaseSimulator (see envs/humanoid21)
    plugins=[TimeoutPlugin(max_steps=600)],          # world rules (may write physics)
    observer_plugins={"robot_a_reward": MyReward()}, # read-only outputs
    recorders=[BaseFrameRecorder(output_dir="rec")], # optional persistence
    phy_steps_per_action=25,
)
runtime.reset(seed=42)
runtime.step(action_a, action_b)       # returns nothing — pull data out:
obs_a, obs_b = runtime.get_observation()
reward = runtime.get_observer_output("robot_a_reward")
terminated, truncated = runtime.get_termination_flags()
```

For a full episode with policies attached, use `EpisodeRunner`
(`runner.run_episode(seed=..., options={...})`); for combat evaluation use
`RoundRunner` / `MatchRunner` (blueprint-driven, CLI included).

## What's here (the map)

**Contracts** — the framework is these interfaces; everything else is plumbing:

| File | Provides |
|---|---|
| `backend.py` | `IDataAccessor` (5 read methods), `IDataMutator` (write: set state / set action / apply force), `BaseSimulator` — implement these to port a new physics backend |
| `context.py` | `SimContext` — per-episode blackboard shared by all plugins (`metrics`, `events`, `episode_options`, `base_seed`, per-agent `request_termination`); `ReadOnlySimContext` for observers/recorders |
| `plugin.py` | `BasePlugin` — world-rule unit with 6 lifecycle hooks |
| `observer_plugin.py` | `BaseObserverPlugin` — read-only unit (obs / reward / debug); `CompositeObserver`; the internal `_ObserverDispatcherPlugin` that batch-drives all observers |

**Runtime + runners:**

| File | Provides |
|---|---|
| `env_runtime.py` | `EnvRuntime` — the only public entry point (`reset`/`step`/pull APIs); private `_RuntimeCore` + `_PluginManager` |
| `episode_runner.py` | `EpisodeRunner` — thin episode loop: `policy.act → runtime.step`, seed derivation, per-agent active tracking. **Returns `None`** |
| `round_runner.py` | `RoundRunner` — blueprint → runtime → one combat round; CLI `python -m envs.framework.round_runner` |
| `match_runner.py` | `MatchRunner` — best-of-N match, HP carry-over, KO; CLI included |

**Serialization:**

| File | Provides |
|---|---|
| `blueprint.py` | `EnvBlueprint` — simulator class + plugins + observers + knobs as YAML; `build()` → fresh `EnvRuntime` |
| `parameterized_blueprint.py` | `ParameterizedEnvBlueprint` — blueprint with `${name}` knobs materialized at build time |
| `policy.py` | `Policy` ABC (`act(obs, want_extra) -> (action, extra)`), `PolicyBlueprint`, `ParameterizedPolicyBlueprint` |

**Recording / replay:**

| File | Provides |
|---|---|
| `recorder.py` | `PostActionRecorder` ABC + `BaseFrameRecorder` (per-step PNG+JSON on disk, `manifest_version=2`) + `EpisodeBufferRecorder` (in-memory) |
| `replay.py` | `ReplaySimulator` — `BaseSimulator` backed by a recording; observers/plugins rerun on recorded data unchanged |
| `recorder_viewer.py` | `python -m envs.framework.recorder_viewer <rec_dir>` — web viewer for recordings |
| `common_plugins.py` | `TimeoutPlugin`, `VideoRecorderPlugin` |

**Docs & tests:** `DESIGN.md` (architecture spec, zh), `RESET.md` (reset-chain
contract + invariants I1–I6), `SEED.md` (seeding contract), `REVIEW_SUMMARY.md`
(internal review), `CONTEXT.md` (**stale — see Known Issues**),
`tests/` (pytest, `conftest.py` has `MockSimulator` + test plugins).

## Core details (for AI / developers)

**Capability security.** `ctx.accessor` is always available; `ctx.mutator` is
granted per-plugin-per-call by `_PluginManager.invoke` only when
`allow_mutator(hook) and plugin.require_mutator`. Forgetting
`require_mutator=True` silently gives `ctx.mutator is None` — intentional
least-privilege, not a bug.

| Hook | Timing | Mutator |
|---|---|---|
| `on_pre_episode` | after reset | ✓ |
| `on_pre_action_step` | before each action | ✓ |
| `on_pre_phy_step` / `on_post_phy_step` | around each physics substep | ✓ |
| `on_post_action_step` | after each action step | ✗ read-only |
| `on_post_episode` | after episode ends | ✗ read-only |

**Observer dispatch.** One `_ObserverDispatcherPlugin` (priority `+1_000_000`)
runs all `BaseObserverPlugin`s first on each hook — downstream world plugins
always see fresh observer output. Do not re-order.

**EpisodeRunner is thin by design.** `run_episode()` returns `None`; it does
NOT capture trajectories, extract rewards, or aggregate results — attach a
`PostActionRecorder` to the runtime (`runtime.attach_recorder(...)`) and read
its state after the run. `post_termination_action` = `"policy"` (keep calling
`act` for terminated agents) or `"hold"` (repeat last action). Policies are
duck-typed (`hasattr(act)`) at construction, not `isinstance`-checked — exported
policies inline a minimal stub.

**Termination is per-agent.** `ctx.request_termination(reason, agent_id=None)`;
`agent_id=None` terminates both. Episode ends when *all* agents are terminated
(`is_episode_over()`); `is_agent_active(agent_id)` tracks each side.

**Seed chain (SEED.md / RESET.md).** `run_episode(seed=None)` resolves to a
concrete uint32 via `secrets.randbits`; `base_seed` → `SeedSequence.spawn` →
distinct child seeds for runtime / each policy / each seedable plugin (plugins
that override `set_episode_seed`). Order: plugin seeds → `runtime.reset`
(publishes `ctx.base_seed`, `ctx.episode_options`) → `policy.reset`. Same
`base_seed` ⇒ same derived seeds ⇒ reproducible episode. Mid-episode `reset()`
fires `on_post_episode` once with reason `"abandoned"`.

**Blueprint boundaries.** Blueprints capture simulator + world plugins +
observers + knobs — NOT recorders, video plugins, or anything marked
`BLUEPRINT_EXCLUDE`. Serializable classes opt in via `to_blueprint()` /
`from_blueprint()`.

**Recorder/replay contract.** `BaseFrameRecorder` writes `manifest_version=2`
layout (`static.json` + `step_NNNNN.{png,json}`); `save_accessor_state=True`
persists ~50 KB/step `derived_state` on humanoid21 — filesystem footgun.
`ReplaySimulator` rejects v1 manifests; `physical_step()` == one recorded
frame so `phy_steps_per_action` must equal recorder stride; mutators raise
`ReplayReadOnlyError` except `set_action` (silent no-op — recorded action is
authoritative); JSON round-trip returns `float32`.

**Hardcoded `robot_a`/`robot_b`.** `AGENT_IDS` is fixed 1v1 — downstream
consumers key by these strings; do not generalize.

**Private internals.** `_RuntimeCore`, `_PluginManager`,
`_ObserverDispatcherPlugin` are not public API — do not build against them.

## Known issues (as of 2026-10 regularization audit)

- `CONTEXT.md` is **stale** — describes removed `parallel_runner.py` /
  `runtime_plugin.py` and the old fat-runner API (`ObserverBinding`,
  `RolloutConfig`, `run_n_episodes`). This README's micro section is the
  current truth until CONTEXT.md is rewritten.
- Test suite has stale files: `pytest envs/framework/tests/ -q` →
  **157 pass / 3 fail / 5 collection errors** (tests of deliberately-removed
  APIs). Tracked in `REGULARIZATION_DETAIL.md` — fixes pending user decision.
- `episode_runner.py` docstring still references `parallel_runner` (removed).
- No provenance in recordings (seed/policy hash/code version) — known gap.
