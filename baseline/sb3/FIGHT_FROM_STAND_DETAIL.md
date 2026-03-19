# Fight From Stand Detail Log

## [2026-03-20 00:43] Initialize fight-from-stand implementation log

**Why:** The next task is to turn the verified standing controller into a reusable combat baseline, validate damage accounting plus opponent observations, and reach the first measurable combat milestone: at least one robot loses HP in a completed rollout.

**Command:**
```bash
# Pending
```

**Result:**
Created this detail log for the fight-from-stand workstream.

**Next step:** Record the combat-mechanics root-cause findings, then implement the minimum reusable scripts and environment changes needed to validate and train fight behavior.

## [2026-03-20 00:41] Found two root-cause bugs in the combat damage path

**Why:** Close-range rollouts under the standing controller still showed no `hit_records` or HP loss. Before spending time on fight training, the authoritative collision and damage path had to be verified.

**Command:**
```bash
# Inspect combatbench/core/collision.py, then run short close-contact random rollouts
# under the standing pose + fixed PD controller.
```

**Result:**
Two root-cause issues were identified:

1. `collision.py` only matched legacy robot geom suffixes such as `_a` / `_b`, while the active battle XML uses `_red` / `_blue`, so cross-robot contacts were being silently ignored.
2. Once collision detection started running, the code crashed on `mujoco.mj_contactVelocity`, which is not exposed by the installed MuJoCo Python package.

Both issues were fixed by (a) accepting the actual `_red` / `_blue` suffixes via each robot's authoritative `suffix`, and (b) estimating relative contact speed from the contacting bodies' linear velocities in `data.cvel`.

**Next step:** Re-run the close-contact probes and formally validate that opponent observations and HP accounting now behave correctly.

## [2026-03-20 00:57] Verified opponent observations and full-episode damage accounting

**Why:** The user explicitly required validation of both hostile-observation output and combat damage logic, with a measurable goal that at least one robot loses HP by the end of a rollout.

**Command:**
```bash
python3 -m combatbench.baseline.sb3.validate_fight_mechanics \
  --duration 8 \
  --control-frequency 20 \
  --initial-distance 0.6 \
  --seed 0
```

**Result:**
The new validation script passed the observation checks for the sampled steps and confirmed that every damage event matches the effective per-step score delta after HP-floor clipping. The rollout also satisfied the quantitative combat target: it finished with heavy mutual damage and a KO result, with final scores **`robot_a=0`** and **`robot_b=2`** and end reason **`Robot A HP reached zero (KO)`**. This proves that the environment now exposes correct opponent observations, applies combat damage, and can complete a fight where at least one robot loses HP.

**Next step:** Launch a fight smoke training run initialized from the verified standing checkpoint so combat learning starts from the stable standing controller rather than from scratch.

## [2026-03-20 00:56] Added reusable video-export and training documentation assets

**Why:** The user asked to record the successful video-export flow for later reuse and to add explicit training documentation for the standing phase.

**Command:**
```bash
# Added reusable scripts and docs under combatbench/ and combatbench/baseline/sb3/
```

**Result:**
Added the standalone `combatbench/run_policy_video.py` entrypoint, which reuses the proven `run_without_policy.py`-style EGL initialization and can export policy videos reliably. Added `baseline/sb3/STAND_TRAINING_GUIDE_zh.md` to document standing training, evaluation, 10-second validation, and video export. Added `baseline/sb3/FIGHT_FROM_STAND_SUMMARY.md` to capture the correct fight-from-stand workflow plus the main pitfalls and fixes.

**Next step:** Start a smoke fight-training run from the standing checkpoint and monitor whether the first evaluations already show meaningful combat behavior.

## [2026-03-20 01:02] Verified the reusable video-export entrypoints

**Why:** The initial `python -m combatbench.baseline.sb3.export_video ...` implementation still inherited a fragile render initialization path. The export flow needed a stable reusable entrypoint before it could be documented as the recommended solution.

**Command:**
```bash
python3 combatbench/run_policy_video.py \
  --mode shared_env \
  --model /data1/mono/things/combatbench/combatbench/baseline/sb3/runs/stand_env_residual_smoke_v3/best_model/best_model.zip \
  --phase stand \
  --duration 10 \
  --control-frequency 20 \
  --initial-distance 2.0 \
  --video /data1/mono/things/combatbench/combatbench/baseline/sb3/runs/stand_env_residual_smoke_v3/stand_success_10s_reexport.mp4 \
  --device cpu

python3 -m combatbench.baseline.sb3.export_video \
  --mode shared_env \
  --model /data1/mono/things/combatbench/combatbench/baseline/sb3/runs/stand_env_residual_smoke_v3/best_model/best_model.zip \
  --phase stand \
  --duration 10 \
  --control-frequency 20 \
  --initial-distance 2.0 \
  --video /data1/mono/things/combatbench/combatbench/baseline/sb3/runs/stand_env_residual_smoke_v3/stand_success_10s_reexport_wrapper.mp4 \
  --device cpu
```

**Result:**
Both reusable entrypoints now work. The standalone `run_policy_video.py` successfully re-exported the standing success video, and `baseline/sb3/export_video.py` was converted into a wrapper that delegates to the standalone script in a fresh subprocess. The wrapper path now also exports successfully, producing another full-length `200`-step standing clip.

**Next step:** Launch and monitor a fight-from-stand smoke training run initialized from the standing best checkpoint.

## [2026-03-20 01:00] Launched the first fight-from-stand smoke training run

**Why:** After validating that the environment can now produce real combat damage and that fight starts from the stable standing controller, the next step is to test whether the training pipeline itself is healthy when initialized from the standing checkpoint.

**Command:**
```bash
python3 -u -m combatbench.baseline.sb3.train \
  --phase fight \
  --timesteps 50000 \
  --run-name fight_from_stand_smoke_v1 \
  --init-model /data1/mono/things/combatbench/combatbench/baseline/sb3/runs/stand_env_residual_smoke_v3/best_model/best_model.zip \
  --device cpu \
  --match-duration 8 \
  --control-frequency 20 \
  --initial-distance 1.0 \
  --checkpoint-freq 10000 \
  --eval-freq 10000
```

**Result:**
The run launched successfully in the background. The initial training log shows normal environment construction and PPO rollout updates. The first visible rollout statistics are already meaningful for a combat phase smoke run: `ep_len_mean=134`, `ep_rew_mean=87.6` at `2048` timesteps.

**Next step:** Continue monitoring until the first evaluation at `10k` timesteps, then run direct fight evaluation if the early checkpoint looks promising.

## [2026-03-20 01:10] Observed early fight-smoke behavior and reached the trained-policy KO milestone

**Why:** Shared-env evaluation reward improving is useful, but the real question is whether the current fight checkpoint can produce actual HP loss when evaluated as an independent match.

**Command:**
```bash
python3 -m combatbench.baseline.sb3.evaluate \
  --mode match \
  --model /data1/mono/things/combatbench/combatbench/baseline/sb3/runs/fight_from_stand_smoke_v1/best_model/best_model.zip \
  --phase fight \
  --duration 8 \
  --control-frequency 20 \
  --initial-distance 1.0

python3 -m combatbench.baseline.sb3.evaluate \
  --mode match \
  --model /data1/mono/things/combatbench/combatbench/baseline/sb3/runs/fight_from_stand_smoke_v1/best_model/best_model.zip \
  --phase fight \
  --duration 8 \
  --control-frequency 20 \
  --initial-distance 0.6
```

**Result:**
The smoke run's shared-env evaluations at `10k`, `20k`, and `30k` timesteps all reached the full `160`-step horizon with mean reward around `115-116`, indicating stable fight rollouts under the wrapper. Direct independent match evaluation showed an important nuance:

- At `initial_distance=1.0`, the current `best_model` remains conservative and finishes as a draw with both robots staying at `100 HP`.
- At `initial_distance=0.6`, the same trained `best_model` produces real combat damage and reaches a KO in `52` steps, with final scores **`robot_a=50`** and **`robot_b=0`** and end reason **`Robot B HP reached zero (KO)`**.

This satisfies the user-facing quantitative fight milestone with a trained policy checkpoint, not just with the diagnostic random-action validation.

**Next step:** Decide whether to keep pushing for damage at the looser `1.0m` setup or to treat the `0.6m` close-contact configuration as the current minimum viable combat baseline.

## [2026-03-20 01:11] The 50k fight-from-stand smoke run finished cleanly

**Why:** Before closing the task, the smoke training run itself needed to finish cleanly and leave behind a consistent final artifact set.

**Command:**
```bash
# Inspect the tail of combatbench/baseline/sb3/runs/fight_from_stand_smoke_v1/train.log
```

**Result:**
The run completed successfully through `51,200/50,000` timesteps and saved `model_final.zip` under `combatbench/baseline/sb3/runs/fight_from_stand_smoke_v1/`. The last recorded shared-env evaluation at `50k` timesteps remained stable with `mean_ep_length=160` and `mean_reward=116.10`. The run also produced checkpoints at `10k/20k/30k/40k/50k` plus a `best_model.zip`, which is already sufficient for the current MVP combat baseline validated at `initial_distance=0.6`.

**Next step:** Commit and push the environment fixes, reusable scripts, documentation, and smoke-run artifacts.
