# Curriculum Training Plan (humanoid21)

This document describes how to train, monitor, and validate the unified
three-stage curriculum policy via `baseline/humanoid21/curriculum.py`.

## Design summary

- **Single trainer, three stages, one model.** Reward is the weighted sum
  of three signals collected every step:
  - `r1` — `CrossSupportBalanceRewarder` (cross-support balance)
  - `r2` — `OpponentRelationRewarder` (attribution-safe approach + heading)
  - `r3` — `NetDamageRewarder` (per-step net damage delta)
- **Per-step reward** = `w1 * r1_scale * r1 + w2 * r2_scale * r2 + w3 * r3_scale * r3`,
  with a sparse `-1.0` terminal fall penalty injected post-rollout on
  every imbalance-terminated trajectory (matches `stage1.py` exactly).
- **Stage weights** (`CurriculumStageGate.STAGE_WEIGHTS`):
  - Stage 1 → `(1, 0, 0)` — r1 only
  - Stage 2 → `(1, 1, 0)` — r1 + r2
  - Stage 3 → `(1, 1, 1)` — r1 + r2 + r3
  - Lower-stage rewards stay active in higher stages — anti-forgetting
    safeguard.
- **Magnitude balance** (per-episode contributions, all roughly O(1)):
  - r1: `0.02 * cross_support_sum` ∈ [-0.1, 0]; plus `-1` terminal penalty
  - r2: `0.02 * r2_sum` ∈ [0, +4] once approach is partially learned
  - r3: `0.05 * net_damage_sum` ∈ [-2, +5] for active fights

## Stage 1 alignment with `stage1.py`

The training signal in Stage 1 is **bit-identical** to `stage1.py`:

- Runtime: `make_curriculum_runtime_for(target)` — only the **target
  agent** has an `ImbalanceTerminationPlugin`, so the episode terminates
  iff THAT agent falls (matches `make_stage1_runtime_for`).
- Per-step reward (Stage 1 weights `(1,0,0)`):
  `r1_scale * r1` = `0.02 * cross_support_reward` =
  `cross_support_reward_scale * cross_support_reward` from `stage1.py`.
- Terminal penalty: `_inject_terminal_fall_penalty` (literally copied
  from `stage1.py`).
- Trainer alternates `target_agent` per rollout via the same
  `_agent_from_rollout_seed` function and discards non-target
  trajectories before PPO update.

## Eval-driven stage gate

Single-shot, stateless classifier — no hysteresis, no dwell, no fixed
transition graph. After every `eval_interval` updates we run a
deterministic eval batch, summarize it, and call
`gate.assign_from_eval(eval_summary)`:

```
len_ratio       = eval mean_length / max_steps
final_in_zone   = fraction of eval episodes whose LAST step is BOTH
                  in the [dist_min, dist_max] band AND has heading
                  angle within heading_max_angle_deg of the opponent

if len_ratio < 1.0:                        stage = 1   # balance not mastered (<200 steps)
elif final_in_zone < 0.5:                  stage = 2   # balance OK, but most episodes don't end inside the zone
else:                                      stage = 3   # both criteria met → combat
```

Any single eval can move the stage from 1 → 3 (or 3 → 1) — the gate
follows whatever the deterministic policy currently deserves.

## Launch (from scratch)

```bash
cd /data1/mono/things/combatbench
bash baseline/humanoid21/launch_curriculum.sh
```

This:

- Trains from a freshly-initialized actor + critic (no resume).
- Uses `CUDA_VISIBLE_DEVICES=1` (override via `CUDA=N`).
- Writes log + pid + LATEST_RUN under `baseline/humanoid21/logs/`.
- Saves the best-eval checkpoint to
  `baseline/humanoid21/runs/<RUN_NAME>/policy/`.

Foreground runs and CLI overrides:

```bash
python3 -u baseline/humanoid21/curriculum.py \
    --run-name my_run \
    --max-updates 5000 \
    --episodes-per-update 2048 \
    --rollout-workers 32
```

## Monitoring

Use `curriculum_monitor.py` to summarize the latest log:

```bash
python3 baseline/humanoid21/curriculum_monitor.py             # auto-pick newest log
python3 baseline/humanoid21/curriculum_monitor.py -w 50       # 50-update window
python3 baseline/humanoid21/curriculum_monitor.py path/to.log
```

The monitor reports:

- **Last update**: stage, weights, reward, length, term rate, in_range,
  KL, gate reason.
- **Rolling means** (over the last `-w` updates): `mean_length`,
  `term_rate`, `in_range`, `final_in_zone`, per-component rewards, KL.
- **Trends**: last-quartile minus first-quartile of the rolling window
  for length/term/in_range/r1/r2/r3.
- **Stage history**: chronological list of stage transitions.
- **Best eval so far**: update, stage, eval_length, eval_reward.
- **Health verdicts**: alive / balance_progress / ppo_stable /
  stage_advance.

## What "normal" training looks like

Numbers below assume the default config (`episodes_per_update=2048`,
`max_steps=200`, `eval_interval=5`).

Stage 1 (early, updates 1–~30):
- `len` rising from ~20 to >100; `term_rate` falling from ~1.0 to <0.8.
- `r1` rising from very negative (e.g. -0.8) toward 0.
- `term_pen` per-episode mean falling toward 0 as terminations drop.
- `eval_length` rising; gate stays at stage 1.
- KL ≤ 0.1; `policy_loss` near zero or slightly negative.

Stage 1 → 2 transition:
- A single eval reaches `eval_length=200` (full horizon) → gate flips
  to stage 2 in the next update; weights become `(1, 1, 0)`.

Stage 2 (updates ~30–~150):
- `r2` becomes positive and trending up; `in_range` and `final_in_zone`
  trending up.
- `r1` should NOT regress significantly. If `term_rate` jumps back to
  near 1.0 (i.e. forgot how to balance), the next eval will demote the
  gate back to stage 1 automatically.

Stage 2 → 3 transition:
- Eval shows `eval_length=200` AND `eval_final_in_zone ≥ 0.5` → gate
  flips to stage 3; weights become `(1, 1, 1)`.

Stage 3:
- `r3` non-zero and trending up (more damage dealt than taken).
- `r1`/`r2` stay healthy; if not, gate demotes to a lower stage.

## Definitely-bad signals

- **KL > 0.5 sustained** — PPO step too aggressive. Investigate the
  rollout/optimizer config (target_kl, learning_rate).
- **`len` plateau at <200 with `r1` saturated** — terminal penalty
  signal isn't dominant; check `term_pen` per-episode mean.
- **`term_rate` swings wildly between 0 and 1** between consecutive
  evals — something destabilized the policy (cold critic? bad batch?).
- **Stage flip-flop every eval** — the eval batch is too small or the
  policy is genuinely on the boundary; tighten `pass_len_ratio` /
  `pass_final_in_zone`, or make `eval_episodes` bigger.
- **`r3` stays at exactly 0 in stage 3 for many updates** — agents
  aren't connecting, opponent may be standing still or out of range.
  Check that the in_range and approach signals are still positive.

## Stop / resume

Stop:
```bash
kill $(cat baseline/humanoid21/logs/<RUN_NAME>.pid)
```

Resume from a saved best-eval actor checkpoint (critic still
re-initialized — different reward = different value function):
```bash
python3 -u baseline/humanoid21/curriculum.py \
    --resume-from baseline/humanoid21/runs/<RUN_NAME>/policy/model.pt \
    --run-name resume_$(date +%s)
```

Note: resume reloads the actor only; expect a brief value-loss spike
in the first few updates while the critic catches up.
