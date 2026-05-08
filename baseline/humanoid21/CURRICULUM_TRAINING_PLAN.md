# Curriculum Training Plan

A practical plan for running and watching `curriculum.py` — the unified
three-stage trainer for humanoid21 combat. Pair this document with
`curriculum_monitor.py` for the actual monitoring commands.

## 1. What the trainer does (recap)

`curriculum.py` runs ONE PPO training loop with ONE policy network.
The reward is a weighted sum of three signals computed by
`MultiSignalRewardObserver`:

```
reward = w1 * r1_balance + w2 * r2_approach + w3 * r3_net_damage
```

The active weights `(w1, w2, w3)` come from `CurriculumStageGate`,
which advances stages based on rollout statistics:

| Stage | Weights         | Promotion criterion                                                                        |
|-------|-----------------|--------------------------------------------------------------------------------------------|
| 1     | (1, 0, 0)       | balance only                                                                              |
| 2     | (1, 1, 0)       | term_rate ≤ 5% AND mean_len/max ≥ 95% (rolling window, with ≥ min_dwell updates in stage) |
| 3     | (1, 1, 1)       | stage 1 still passing AND in_range_ratio ≥ 80%                                            |

Demotions use a hysteresis FAIL band (term_rate ≥ 15% OR len_ratio ≤ 85%
forces stage → 1) — this is the catastrophic-forgetting safeguard.

Lower-stage rewards never get fully turned off once a higher stage opens.

## 2. Resume strategy

We start from the best stage1 cross-support balance checkpoint:

```
baseline/humanoid21/runs/stage1_20260430_093352/policy/model.pt
  algorithm = ppo_stage1_terminal_fall_penalty
  update    = 960
  best_eval_length = 200.0   (perfect survival, max_steps=200)
```

`--resume-from` loads **actor weights only**; the critic is fresh (the
reward is different, so a stale value head would actively hurt PPO).
`log_std` is also fresh because the previous variant didn't have a
trainable log_std parameter — that's expected.

Trade-off: the first ~5–10 updates show a transient as PPO re-fits the
critic. Watch for `value_loss` to come down within ~10 updates; if it
keeps climbing, that's the leading indicator of policy collapse.

## 3. Run command

Single run name so resuming / monitoring is unambiguous:

```bash
RUN_NAME="curriculum_resumed_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="baseline/humanoid21/logs"
mkdir -p "$LOG_DIR"

nohup python3 baseline/humanoid21/curriculum.py \
    --resume-from baseline/humanoid21/runs/stage1_20260430_093352/policy/model.pt \
    --run-name "$RUN_NAME" \
    > "$LOG_DIR/${RUN_NAME}.log" 2>&1 &

echo "$!" > "$LOG_DIR/${RUN_NAME}.pid"
```

Defaults from `CurriculumConfig`:
* `episodes_per_update = 2048`, `max_updates = 10000`
* `rollout_workers = max(1, min(64, ncpu // 2))` — on the 192-core box ⇒ 64
* `eval_interval = 5`, `eval_episodes = 16`
* `learning_rate = 3e-4`, `gamma = 0.99`, `gae_lambda = 0.95`
* `gate_window = 3`, `gate_min_dwell = 5`

## 4. What "normal" looks like

For each update the trainer prints one line:

```
update=  17 stage=1 weights=(1.0, 0.0, 0.0) reward=-0.7234 len= 88.50
        term=0.625 in_range=0.412 r1=-0.7 r2=-12.5 r3=+0.0
        policy_loss=+0.012 value_loss=+0.04 kl=0.083 gate_reason='no-op'
        | eval_reward=-0.6 eval_length=110.0  [new_best]
```

Health expectations (all rolling means over the last 20 updates):

| Phase                  | Update range | mean_length        | term_rate          | in_range          | KL                |
|------------------------|--------------|--------------------|--------------------|-------------------|-------------------|
| transient (new critic) | 0–10         | drops then recovers| may spike          | ~ 0               | up to ~0.3 OK     |
| stage 1 settling       | 10–60        | climbs > 150       | trending < 0.20    | ~ 0               | <  0.1            |
| stage 1 → 2 promotion  | 60–200       | mean_length ≥ 190  | ≤ 0.05             | starts to grow    | <  0.05           |
| stage 2 in-range learn | 200–800      | stays ≥ 190        | ≤ 0.05             | climbs to ≥ 0.80  | <  0.05           |
| stage 2 → 3 promotion  | 800–???      | stays ≥ 190        | ≤ 0.05             | ≥ 0.80            | <  0.05           |

The exact update numbers are **estimates** — the gate is data-driven, so
slow learners just sit at lower stages longer. Hysteresis can cause
brief regressions; that is by design.

### Definitely-bad signals (kill and investigate)

| Symptom                                        | Likely cause                                 | Action                                                  |
|------------------------------------------------|----------------------------------------------|---------------------------------------------------------|
| `kl > 0.5` for 3+ updates                      | clip_eps too lax / lr too high / value head reload bug | reduce `learning_rate` to 1e-4, restart                 |
| `value_loss` climbing monotonically            | value-target/reward scale mismatch           | check `r{1,2,3}_scale` / weight scaling                 |
| `mean_length` decreasing for 30+ updates       | reward gaming / reward bug                   | inspect `r1`/`r2`/`r3` traces, suspect scale            |
| `stage` flapping 1↔2↔1↔2 within 10 updates     | hysteresis band too tight                    | widen `fail_term_rate` / `fail_len_ratio`               |
| Frozen `update=N` line ≥ 5 min                 | rollout worker hung / process dead           | check `nvidia-smi`, `ps`, restart                       |

### Borderline signals (watch closely)

* `r3_mean ≈ 0` after stage 3 opens → both robots are passive (no hits land).
  The fix is self-play with frozen-past-self opponents (out of scope for this
  iteration — both agents currently train mirror-symmetric, so they tend to
  converge to similar passive policies).
* `in_range` plateauing below 0.6 in stage 2 → opponent-relation reward isn't
  strong enough; consider raising `r2_scale` (env var `CURRICULUM_R2_SCALE`).

## 5. Monitoring commands

The monitor parses the log file (no PID/no live IPC needed) and prints
a one-shot status report. Run it any time:

```bash
# Auto-pick newest log under baseline/humanoid21/logs/
python3 baseline/humanoid21/curriculum_monitor.py

# Specific log + larger rolling window
python3 baseline/humanoid21/curriculum_monitor.py \
    baseline/humanoid21/logs/curriculum_resumed_20260508_191600.log -w 50
```

The report has four sections:

1. **Last update** — most recent single line (sanity check).
2. **Rolling means** — over the last `--window` updates (default 20).
3. **Trends** — last-quartile minus first-quartile of the rolling
   window. Lets you read the slope at a glance.
4. **Health verdicts** — `[PASS]` / `[WARN]` / `[FAIL]` / `[INIT]`
   for `alive`, `balance_progress`, `ppo_stable`, `stage_advance`.

For continuous watching (e.g. on a separate tmux pane):

```bash
watch -n 30 'python3 baseline/humanoid21/curriculum_monitor.py'
```

For raw tail:

```bash
tail -f baseline/humanoid21/logs/<run>.log
```

For the actual list of new-best checkpoints saved:

```bash
grep '\[new_best\]' baseline/humanoid21/logs/<run>.log | tail -20
```

Best policy artifacts (single rolling slot — overwritten on each new
best) live at:

```
baseline/humanoid21/runs/<run-name>/policy/model.pt   # state_dict + meta
baseline/humanoid21/runs/<run-name>/policy/policy.py  # loader stub
```

## 6. Decision points

Roughly every couple of hours, run the monitor and ask:

1. **Is the trainer alive?** — `[PASS] alive` verdict.
2. **Is balance regressing?** — if `term_rate` is climbing or
   `mean_length` is dropping for ≥ 30 updates, the new reward is
   actively harming the resumed policy. Stop, reduce `r2_scale` /
   `r3_scale`, restart.
3. **Has the gate advanced yet?** — first stage 1 → 2 transition
   should appear within ~200 updates if the resumed actor was good.
   If not, the gate thresholds may be too strict for the actual
   policy quality — raise `pass_term_rate` to 0.10.
4. **Is stage 2 making approach progress?** — `in_range` should grow
   beyond ~0.3 within 100 updates of stage 2 opening.
5. **Has stage 3 ever opened?** — if yes, watch `r3_mean`; persistent
   ~0 there means the opponent never gets hit (passive draw), which
   is the known weakness without self-play opponents.

## 7. Stop and resume

Stop:
```bash
kill "$(cat baseline/humanoid21/logs/<run-name>.pid)"
```

Resume from this run's best policy:
```bash
python3 baseline/humanoid21/curriculum.py \
    --resume-from baseline/humanoid21/runs/<run-name>/policy/model.pt \
    --run-name "<new-name>"
```
