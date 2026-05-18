# Training-rollout inspection

Run the latest trained curriculum policy under a **training-faithful** rollout
config and dump every step to disk via `BaseFrameRecorder`, so the resulting
data can be used offline to verify observer correctness (e.g. is
`MultiSignalRewardObserver` computing the right `r1 + r2 + r3`?).

This is **not** a new script — it's a single CLI invocation of
`envs/humanoid21/run_round.py`, which now accepts `--plugin` / `--observer` /
`--recorder` injection (added in the same series of changes).

## The launch command

```bash
cd /data1/mono/things/combatbench

# Path to the policy you want to inspect. Use the curriculum run's best snapshot.
POLICY=baseline/humanoid21/runs/curriculum_20260513_181212/policy

# Output directory for BaseFrameRecorder. One subdir is created per episode.
OUT=baseline/humanoid21/runs/curriculum_20260513_181212/inspect/$(date +%Y%m%d_%H%M%S)

python3 -m envs.humanoid21.run_round \
  --policy-a "$POLICY" \
  --policy-b "$POLICY" \
  --duration 10 \
  --control-frequency 20 \
  --damage-scale 100.0 \
  --plugin   "baseline.humanoid21.common:ImbalanceTerminationPlugin?agent_id=robot_a" \
  --observer "robot_a_reward=baseline.humanoid21.common:MultiSignalRewardObserver?agent_id=robot_a&default_weights=[1,1,1]" \
  --observer "robot_b_reward=baseline.humanoid21.common:MultiSignalRewardObserver?agent_id=robot_b&default_weights=[1,1,1]" \
  --recorder "envs.framework.recorder:BaseFrameRecorder?output_dir=$OUT&quiet=true"
```

Run it from the `combatbench/` directory (the script auto-adds the project
root to `PYTHONPATH`).

## What each flag mirrors from the trainer

| Flag | Training-rollout counterpart | Source |
|------|------------------------------|--------|
| `--policy-a` / `--policy-b` (same path) | `state_dicts={"robot_a": actor_sd, "robot_b": actor_sd}` — self-play with one actor on both sides | `curriculum.py` collector calls |
| `--duration 10`, `--control-frequency 20` | `MATCH_DURATION_SECONDS=10.0`, `CONTROL_FREQUENCY=20` → `CURRICULUM_MAX_STEPS=200` | `common.py` |
| `--damage-scale 100.0` | `CURRICULUM_DAMAGE_SCALE=100.0` | `common.py` |
| Default `CombatScoringPlugin` (from `run_round.py`) | Training also uses `CombatScoringPlugin` with `damage_scale=100`. **Caveat:** training sets `initial_health_a=initial_health_b=1e9` (`CURRICULUM_NO_KO_HEALTH`); the script uses the default `100.0`. Damage per episode in this run averages ~0.2 (scaled), well below 100, so KO does not fire — see `r3_dealt` in `curriculum_monitor.py` output. | `make_curriculum_runtime_for` |
| `--plugin ImbalanceTerminationPlugin?agent_id=robot_a` | `ImbalanceTerminationPlugin(target)` — only the **target agent** has the termination plugin | `make_curriculum_runtime_for` |
| `--observer robot_{a,b}_reward=MultiSignalRewardObserver?agent_id=...&default_weights=[1,1,1]` | `observer_plugins={"robot_a_reward": MultiSignalRewardObserver("robot_a"), "robot_b_reward": MultiSignalRewardObserver(...)}`. `default_weights=[1,1,1]` mimics stage-3 weights (the trainer normally injects this per-episode via `ctx.episode_options["reward_weights"]`). | `make_curriculum_runtime_for` |
| `make_env` defaults (added automatically) | `Humanoid21Observer("robot_a")`, `Humanoid21Observer("robot_b")` — match training one-to-one | `make_env` and `make_curriculum_runtime_for` both add them |
| `--recorder BaseFrameRecorder?...` | Not part of training (training uses `RolloutCollector` which streams batches in memory). This is the addition for inspection. | new |

## Known minor deltas from training (none affect observer-output correctness)

* **Simulator init distance** — training uses `MujocoCombatSimulator(initial_distance=3.0)` plus a per-episode random override on `[1.5, 3.5]` via `make_standing_options_fn`; `run_round.py` keeps the simulator's class default `2.0`. The observer output depends only on the resulting sim state, not on how that state was initialized.
* **Episode-options `reward_weights`** — training pushes weights through `ctx.episode_options` each episode; this script bakes them into the observer's `default_weights`. The observer treats both code paths identically (see `MultiSignalRewardObserver.on_pre_episode`).
* **Initial health** — see the table caveat above.

If any of these matter for a future analysis, expose them as additional CLI
flags on `run_round.py` and update this doc — do not fork a separate runner.

## Output layout (per `BaseFrameRecorder`)

```
<OUT>/
  index.json
  episode_00000/
    static.json                 # model/sensor metadata (episode-invariant)
    manifest.json               # ordered list of step records
    step_00000.json             # per-step payload (see below)
    step_00000.png              # broadcast view image
    step_00001.json
    step_00001.png
    ...
```

Each `step_NNNNN.json` contains:

```json
{
  "episode_step": 0,
  "physics_step": 0,
  "observer_outputs": {
    "robot_a_obs":    [...],            // Humanoid21Observer obs vector
    "robot_b_obs":    [...],
    "robot_a_reward": 0.02,             // MultiSignalRewardObserver scalar
    "robot_b_reward": 0.02
  },
  "core_state":   { "robot_a": {...}, "robot_b": {...} },
  "derived_state":{ "torso_distance": ..., "robot_robot_contacts": [...], ... },
  "sensor_data":  {...},
  "action":       {...}
}
```

## Caveats / not yet captured

`BaseFrameRecorder` dumps accessor-level data (`core_state`, `derived_state`,
`sensor_data`, `action`) **plus** `observer_outputs`. It does **not** dump
`ctx.metrics` (e.g. `health_a` / `damage_taken_a` written by
`CombatScoringPlugin`) or `ctx.events` (the per-hit events). If you need the
HP / damage stream for cross-checking `NetDamageRewarder`'s `r3`, recompute it
offline from `derived_state.robot_robot_contacts` and the rules in
`envs/humanoid21/plugins.py::CombatScoringPlugin`. If this becomes routine,
add a `BlackboardSnapshotRecorder` (small subclass of `PostActionRecorder`)
that writes `metrics` / `events` next to each step.

## Sanity check

A 2-second smoke run was verified before this doc was written:

```bash
ls <OUT>/episode_00000/ | head
# manifest.json  static.json  step_00000.json  step_00000.png  step_00001.json  ...

python3 -c "import json,sys; d=json.load(open(sys.argv[1])); \
  print(sorted(d['observer_outputs'].keys()))" \
  <OUT>/episode_00000/step_00010.json
# ['robot_a_obs', 'robot_a_reward', 'robot_b_obs', 'robot_b_reward']
```
