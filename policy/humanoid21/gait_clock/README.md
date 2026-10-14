# GaitClockAdapterPolicy

Deploys a step-lineage policy (step / balance / follow / attack_prep /
strike / fight_step — trained under `GaitClockSimulator`, `obs_dim=99`)
on the **canonical 96-dim platform environment** without requiring the
platform or opponents to know about the clock.

The adapter accepts the standard 96-dim observation, appends the 3-dim
gait clock (`cmd_L`, `cmd_R`, window progress) generated from its own
action counter, and delegates to the wrapped 99-dim policy.

## Deploy blueprint

Drop a blueprint next to the exported artifact (`policy.py` +
`model.pt`), e.g. `runs/<run>/policy/deploy_blueprint.yaml`:

```yaml
version: 1
cls: "policy.humanoid21.gait_clock.policy:GaitClockAdapterPolicy"
config:
  gait_period: 40      # must match the training env
  counter_step: 2      # set_action calls per frame (1 per agent)
  inner:
    version: 1
    cls: "file:${DIR}/policy.py:ExportedStateMixtureBoundedStdTruncNormPolicy"
    config: {}
```

Then run e.g.:

```bash
PYTHONPATH=. python3 -m envs.framework.round_runner \
  --env-blueprint envs/humanoid21/blueprint.yaml \
  --policy-a-blueprint runs/<run>/policy/deploy_blueprint.yaml \
  --policy-b-blueprint policy/blueprints/random.yaml \
  --seed 7
```

## Gotcha: `counter_step`

`GaitClockSimulator._action_step` counts `set_action` *calls*, not env
frames — the runtime calls it once per agent, so the clock advances
**2 per frame** in a 2-agent env (verified empirically). Keep
`counter_step: 2` for standard combat; it must match how many agents
submit actions each frame in the training env.

Tests: `envs/framework/tests/test_gait_clock_adapter.py`.
