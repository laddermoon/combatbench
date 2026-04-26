# `baseline/common/` — Training-side building blocks

A set of reusable, environment-agnostic *points* (components) that you
wire into your own training script (the *line*). The deep-dive design
rationale lives in [`baseline/DESIGN.md`](../DESIGN.md); this README is
just enough to write a working PPO loop in 50 lines.

## Module map

| Submodule | Public surface | Role |
|---|---|---|
| `policies/` | `TanhGaussianMLPPolicy`, `CriticMLP`, `TorchPolicyAdapter`, checkpoint helpers | actor / critic backbones + framework `Policy` adapter + checkpoint IO |
| `rollout/`  | `RolloutBatch`, `RolloutCollector`, `RolloutSampler` | data contract + episode-driver wrapper + minibatch iterator |
| `normalize/` | `RunningMeanStd`, `ObservationNormalizer`, `ReturnNormalizer` | obs / reward running-stats normalization (PPO standard) |
| `algos/`    | `compute_gae`, `compute_returns_to_go`, `compute_grpo_advantages`, `ppo_loss` | advantage estimators + clipped surrogate loss |
| `eval/`     | `PolicyEvaluator`, `bootstrap_ci`, `head_to_head_winrate` | episode-metric stats + bootstrap CIs |

## Minimal PPO recipe

```python
import torch
from torch.optim import Adam

from baseline.common.algos import compute_gae, ppo_loss
from baseline.common.policies import (
    CriticMLP, TanhGaussianMLPPolicy, TorchPolicyAdapter,
)
from baseline.common.rollout import RolloutCollector, RolloutSampler
from envs.framework import EnvRuntime  # your runtime factory

OBS_DIM, ACTION_DIM, HIDDEN = 31, 21, 256
device = "cuda"

actor  = TanhGaussianMLPPolicy(OBS_DIM, ACTION_DIM, HIDDEN).to(device)
critic = CriticMLP(OBS_DIM, HIDDEN).to(device)
optim  = Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

# One adapter instance per agent — collector reuses it across iterations.
shared_adapter = TorchPolicyAdapter(actor=actor, critic=critic, device=device)

collector = RolloutCollector(
    runtime_factory=lambda: build_runtime(),  # your EnvRuntime factory
    policy_factories={
        "robot_a": lambda: shared_adapter,
        "robot_b": lambda: shared_adapter,    # self-play
    },
    capture_agents=("robot_a",),              # only learn from one side
    store_extras=True,                         # populate log_probs / values
)

for iteration in range(NUM_ITERATIONS):
    # 1) Collect a batch of episodes with current weights.
    batches = collector.collect(n=32, base_seed=iteration)
    rollouts = batches["robot_a"]

    # 2) Compute advantages / returns episode-by-episode.
    adv_eps, ret_eps = [], []
    for b in rollouts:
        last_v = 0.0 if b.terminated else float(critic_value(b.final_obs))
        adv, ret = compute_gae(b.rewards, b.values, last_value=last_v)
        adv_eps.append(adv); ret_eps.append(ret)

    # 3) Build minibatch sampler.
    sampler = RolloutSampler.from_batches(
        rollouts,
        extras={"advantages": adv_eps, "returns": ret_eps},
        minibatch_size=256, mode="concat", seed=iteration,
    )

    # 4) PPO update — K epochs, each a fresh shuffle.
    for epoch in range(4):
        for mb in sampler:
            obs = mb["obs"].to(device)
            actions = mb["actions"].to(device)
            log_probs_new = actor.log_prob(obs, actions)   # your impl
            values_new    = critic(obs)
            entropy       = actor.entropy(obs)             # optional

            out = ppo_loss(
                log_probs_old=mb["log_probs"].to(device),
                log_probs_new=log_probs_new,
                advantages=mb["advantages"].to(device),
                values_old=mb["values"].to(device),
                values_new=values_new,
                returns=mb["returns"].to(device),
                entropy=entropy,
                clip_range=0.2, value_coef=0.5, entropy_coef=0.01,
            )
            optim.zero_grad(); out.loss.backward(); optim.step()

    # 5) Periodic evaluation with bootstrap CI.
    if iteration % 10 == 0:
        evaluator = PolicyEvaluator(
            runtime_factory=lambda: build_runtime(),
            policy_factories={"robot_a": lambda: shared_adapter,
                              "robot_b": lambda: build_baseline_opponent()},
        )
        report = evaluator.evaluate(n=100, base_seed=99,
                                    bootstrap_samples=1000)
        print(report)  # MetricStats __repr__ prints mean ± std + 95% CI
        evaluator.close()
```

## What lives where (depth gates)

Each component in this directory passes the [DESIGN.md §2 depth gate](../DESIGN.md#2-depth-gate-the-shallowness-test):

* **non-trivial implementation** — running stats, GAE recursion, PPO
  surrogate, bootstrap CI all have correctness corners and numerical
  pitfalls (RNG control, terminated-vs-truncated bootstrap,
  population vs. sample variance, percentile vs. BCa CI);
* **reusable across baselines and envs** — every public class is
  dim-parameterized at construction (`obs_dim` / `action_dim` /
  `hidden_dim` are ctor args, never module constants);
* **thin where upstream is solid** — `RolloutCollector` is a 70-line
  wrapper around `EpisodeRunner` and never reimplements the episode
  loop; `RolloutSampler` only does the
  variable-length → fixed-shape conversion that the upstream
  `EpisodeRunner` cannot do generically.

Things explicitly *not* in `common/` (see DESIGN.md §4): an
`OptionsSchedule` / curriculum library (recipe lives in
[`examples/07_curriculum_recipe.py`](../../examples/07_curriculum_recipe.py)
instead), an offline-RL replay reader (no offline baseline yet), a
`workers/pool.py` (`ParallelRunner` already is the pool).

## Backward compatibility with `humanoid21/`

Pre-existing `baseline/humanoid21/standing_*.py` scripts are untouched.
Their `from baseline.common.policies import ...` surface still imports
the same names; the only structural change is that
`Critic` in `baseline/humanoid21/base.py` is now a thin alias of
`CriticMLP`, and the checkpoint IO helpers have moved from
`tanh_gaussian_mlp.py` into `checkpoint.py` (re-exported for
back-compat). All `standing_v0.py` / `standing_v1.py` / etc. continue
to run bit-equal under the same seeds.

New code should import directly from the canonical locations:

```python
from baseline.common.policies import CriticMLP, TanhGaussianMLPPolicy, TorchPolicyAdapter
from baseline.common.policies.checkpoint import export_actor_policy_artifacts
```

## Tests

Each subpackage ships with `test_*.py` covering its contract; run
the whole batch with:

```bash
python -m pytest baseline/common/ envs/framework/tests/ -q
```

346+ tests covering `RolloutBatch` invariants, `RolloutCollector`
multi-agent rollout, `RolloutSampler` concat / pad / shuffle,
running-stats numerics, GAE termination semantics, PPO
clipping / KL / EV diagnostics, and bootstrap-CI bracketing.
