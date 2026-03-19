# Stand Training Detail Log

## [2026-03-19 17:46] Initialize stand training run

**Why:** The user asked to launch background stand training and verify that it is progressing normally.

**Command:**
```bash
# Pending
```

**Result:**
Created this detail log. No training command has been executed yet.

**Next step:** Create a concise summary skeleton and then start a background stand training run with monitored logs.

## [2026-03-19 17:47] First background launch attempt failed

**Why:** Start the stand PPO training run in the background and redirect logs to a file.

**Command:**
```bash
python3 -u -m combatbench.baseline.sb3.train --phase stand --timesteps 300000 --run-name stand_bg_smoke --device cpu --match-duration 10 --control-frequency 20 --checkpoint-freq 50000 --eval-freq 25000 > combatbench/baseline/sb3/runs/stand_bg_smoke/train.log 2>&1
```

**Result:**
The shell reported `No such file or directory` because the target log directory did not exist yet.

**Next step:** Create the run directory first, relaunch the background training, and then inspect the generated log.

## [2026-03-19 17:48] Relaunched background stand training successfully

**Why:** Restart the training after creating the target run directory so logs and checkpoints can be written normally.

**Command:**
```bash
python3 -u -m combatbench.baseline.sb3.train --phase stand --timesteps 300000 --run-name stand_bg_smoke --device cpu --match-duration 10 --control-frequency 20 --checkpoint-freq 50000 --eval-freq 25000 > combatbench/baseline/sb3/runs/stand_bg_smoke/train.log 2>&1
```

**Result:**
The background process launched successfully. The log showed normal environment initialization, PPO startup, and ongoing rollout/train metrics.

**Next step:** Monitor the training log until at least one evaluation pass is produced, then test the early best model to see whether the policy can really stand.

## [2026-03-19 17:50] Confirmed PPO training is progressing normally

**Why:** Verify that the background run is not stuck in initialization and that learning metrics trend in the right direction.

**Command:**
```bash
# Inspect train.log and background process status
```

**Result:**
Observed stable PPO updates around 220 FPS. Early training metrics improved from roughly `ep_len_mean=7.38, ep_rew_mean=-2.5` to `ep_len_mean=14.5, ep_rew_mean=3.58`. First evaluation at 25k timesteps reported `mean_ep_length=23` and `mean_reward=15`, with `New best mean reward!`.

**Next step:** Load `best_model/best_model.zip` and run a deterministic evaluation to inspect whether the robots can already maintain standing for a meaningful duration.

## [2026-03-19 17:52] Evaluated the first best standing model

**Why:** Determine whether the early best checkpoint already produces genuine standing behavior.

**Command:**
```bash
# Load best_model/best_model.zip and run 5 deterministic episodes in the wrapped stand environment
```

**Result:**
The checkpoint is clearly better than the untrained policy, but it is not yet "truly standing" in a strong sense. Across 5 deterministic episodes, the policy consistently lasted 24 control steps with average reward `15.30`, but all episodes ended with `both robots fell below the stability threshold`. The average minimum torso heights were about `0.928m` for both robots.

**Next step:** Keep the background training running. The run is healthy, but the current checkpoint still only achieves early-stage standing stabilization rather than sustained upright standing.

## [2026-03-19 18:35] Exported evaluation video using the direct environment loop

**Why:** The previous `evaluate.py -> MatchRunner` path appeared stuck during video export. The repository already had `run_without_policy.py`, which proved that the direct `CombatGymEnv(render_mode="rgb_array")` loop could generate videos correctly with `MUJOCO_GL=egl`.

**Command:**
```bash
# Run a one-off script that sets MUJOCO_GL=egl before imports,
# loads model_final.zip on CPU, steps CombatGymEnv directly, and saves mp4
```

**Result:**
Video export succeeded. The file `stand_bg_smoke_final_match.mp4` was written successfully with `100` frames. The rollout logs showed that both robots started near standing height, but by around step 20 torso heights had already dropped sharply, and the rest of the episode mostly consisted of low-height ground contact. This confirms the trained standing policy still fails to maintain upright posture for the full clip.

**Next step:** Review a few extracted key frames to summarize the qualitative behavior, then decide whether to improve the control/reward setup before longer fight training.

## [2026-03-19 21:54] Diagnosed the standing baseline as a control-formulation problem

**Why:** Before launching more PPO runs, verify whether the existing stand environment is physically learnable or whether the environment/controller formulation itself is the bottleneck.

**Command:**
```bash
# Run validate_env.py, zero-action rollouts in CombatGymEnv, and targeted PD/preset-pose probes
```

**Result:**
The main issue is not a crash in MuJoCo but an ill-posed control interface for standing. `CombatGymEnv` uses torque motors, and the old SB3 stand wrapper sent one torque vector that was then held constant across the whole control interval. With zero torque, both robots reliably collapsed within about `14` control steps. Directly PD-controlling toward all-zero joint angles also failed, which confirmed that the old formulation was effectively asking PPO to learn full-body balance from scratch through open-loop torque holds.

**Next step:** Rework the stand phase into a residual balance task: initialize the humanoids in a better standing pose, then interpret the policy output as residual joint targets and compute torques through a low-level PD controller at every physics substep.

## [2026-03-19 22:10] Implemented stand-specific residual PD control in the SB3 environment

**Why:** Convert the standing task from unstable direct torque control into a more learnable balance-control problem without changing the fight-phase interface.

**Command:**
```bash
# Edit combatbench/envs/combat_gym.py, baseline/sb3/selfplay_env.py, and baseline/sb3/rewards.py
```

**Result:**
`CombatGymEnv.step()` now supports an optional per-substep action callback. In `baseline/sb3/selfplay_env.py`, the `stand` phase now: (1) resets both robots into a standing reference pose, (2) masks opponent-related observation slices, (3) treats the policy action as a residual joint-target delta for `robot_a`, and (4) converts those targets into torques with a PD controller recomputed at every MuJoCo substep. The `stand` reward config was also retuned to remove fight-oriented shaping terms such as distance, facing, and damage.

**Next step:** Re-run environment validation and compare the new stand wrapper's zero-action rollout against the old baseline to ensure the fix creates a materially longer balancing window.

## [2026-03-19 22:16] Confirmed a much stronger zero-action standing baseline after the control fix

**Why:** Validate that the residual-PD stand environment meaningfully improves passive stability before spending more compute on PPO.

**Command:**
```bash
# Re-run validate_env.py and step the stand wrapper with zero residual action for up to 120 control steps
```

**Result:**
The new stand wrapper passed `validate_env.py`. More importantly, the zero-action rollout no longer collapsed almost immediately: `robot_a` stayed near full height with rewards around `1.44` for the opening segment and only crossed the fall threshold at about step `33`, versus the old torque-only baseline that fell around step `14`. Torso height remained around `1.27m` through the first dozen steps and only dropped below the new stability threshold at the termination boundary.

**Next step:** Launch a fresh stand PPO smoke run on top of the residual-PD formulation and check whether training can now push the balance horizon well beyond the passive `~33`-step baseline.

## [2026-03-19 22:18] Ran the first PPO smoke training on the residual-PD standing environment

**Why:** Test whether PPO can exploit the new stand formulation and improve beyond the passive residual-PD baseline.

**Command:**
```bash
python3 -u -m combatbench.baseline.sb3.train \
  --phase stand \
  --timesteps 100000 \
  --run-name stand_respd_smoke \
  --device cpu \
  --match-duration 10 \
  --control-frequency 20 \
  --initial-distance 2.5 \
  --checkpoint-freq 20000 \
  --eval-freq 10000
```

**Result:**
Training was stable and clearly better than the old torque baseline. The first evaluation at `10k` timesteps already reached `mean_ep_length = 38`, beating the passive `~33`-step baseline. The best evaluation during this run appeared at `30k` timesteps with `mean_ep_length = 42` and `mean_reward = 44.9`. After that, rollout statistics kept rising into the mid-40s, but deterministic evaluation drifted back down into the `36-39` step range, so the run learned useful balance corrections but did not yet cross into robust long-horizon standing.

**Next step:** Evaluate the saved `best_model` directly and inspect its action magnitudes to see whether the remaining instability comes from overly aggressive residual targets or from the reference pose / reward itself.

## [2026-03-19 22:31] Ablated residual action scale using the current best model

**Why:** Determine whether the residual target magnitudes are still too large, causing the policy to leave the stable basin even after the structural controller fix.

**Command:**
```bash
# Load stand_respd_smoke/best_model.zip and replay it while multiplying _stand_action_scale by
# [0.4, 0.6, 0.8, 1.0, 1.2]
```

**Result:**
This ablation gave the clearest tuning signal so far. With the original scale (`1.0`), deterministic evaluation was consistently about `42` steps. Shrinking the residual scale to `0.4x` improved the exact same policy to `50` steps, while enlarging it to `1.2x` degraded performance to `39` steps. The best model's raw actions also showed large activity on hip pitch channels (one dimension reached about `0.94` in absolute value), which supports the conclusion that the stand residuals were still too aggressive.

**Next step:** Reduce the default stand residual scale in `selfplay_env.py` and start a second smoke run on top of the smaller action envelope.

## [2026-03-19 23:20] Refactored the base environment to expose residual-position control

**Why:** The latest controller discussion converged on a cleaner abstraction: the simulation environment should expose only residual joint-position actions, while all impedance / PD logic, joint-limit clipping, and torque-limit clipping should live inside the environment instead of being split across wrappers.

**Command:**
```bash
# Edit combatbench/core/humanoid_robot.py, combatbench/envs/combat_gym.py,
# and combatbench/baseline/sb3/selfplay_env.py
```

**Result:**
`CombatGymEnv` now interprets external actions as normalized residual joint-position commands for both robots. Internally it maintains a fixed controller stack: `target_position = reference_position + action_scale * residual_action`, then clips targets to MuJoCo joint limits, computes torques with fixed PD gains, and clips the final control vector to the actuator control range before writing `data.ctrl`. `HumanoidRobot` now exposes joint-limit and actuator-limit helpers, and the SB3 wrapper no longer computes torques itself; instead it configures stand-mode reference poses and residual scales through the base environment.

**Next step:** Run environment-level validation and a direct saturation probe to verify that the new controller state, joint-limit clipping, and torque clipping all behave as intended before launching new stand training.

## [2026-03-19 23:22] Validated the new residual-position environment interface

**Why:** Before retraining, verify that the refactored environment still passes SB3 smoke checks and that the new internal controller really respects joint and actuator limits under saturated residual actions.

**Command:**
```bash
python3 -m combatbench.baseline.sb3.validate_env

# Run a direct CombatGymEnv probe with zero / +1 / -1 residual actions and inspect
# info['controller_state'] plus the written actuator controls.
```

**Result:**
The updated stand and fight wrappers both passed `validate_env.py` without crashes. The returned `info` now includes `controller_state`, and both phases stepped normally for the initial smoke rollout. A targeted direct-environment probe confirmed the control contract: with zero residuals, the target positions stayed exactly at the reference pose and actuator controls remained at zero; with saturated `+1` and `-1` residual actions, the reported target positions stayed inside the MuJoCo joint limits and the actual control values written to the actuators stayed inside the actuator `ctrlrange` (including clipping at `-1.0/1.0` when necessary).

**Next step:** Use this refactored environment as the new base interface and launch a fresh stand-training run on top of it, rather than continuing from the older wrapper-side torque-conversion formulation.

## [2026-03-19 23:32] Prepared a fresh stand smoke run on the refactored environment

**Why:** The environment interface has changed in a meaningful way, so the next experiment should be a clean-from-scratch standing smoke run on the new base-env residual-position controller rather than a continuation of older wrapper-side torque runs.

**Command:**
```bash
python3 -u -m combatbench.baseline.sb3.train \
  --phase stand \
  --timesteps 100000 \
  --run-name stand_env_residual_smoke \
  --device cpu \
  --match-duration 10 \
  --control-frequency 20 \
  --initial-distance 2.5 \
  --checkpoint-freq 20000 \
  --eval-freq 10000 > combatbench/baseline/sb3/runs/stand_env_residual_smoke/train.log 2>&1
```

**Result:**
Planned the new smoke run command and log path. The next step is to launch it and watch the first evaluation signal to make sure the new environment/controller stack trains normally.

**Next step:** Start the background run, then inspect the live log until the first `Eval num_timesteps=` block appears.

## [2026-03-19 23:36] Detected a control-semantics regression in the first fresh smoke run

**Why:** The first fresh run on the refactored base environment trained much worse than the earlier residual-PD baseline, so the new control path needed to be checked before trusting the result.

**Command:**
```bash
# Inspect stand_env_residual_smoke/train.log and compare the resulting rollout / eval metrics
# against the previous residual-PD standing baseline.
```

**Result:**
The first evaluation at `10k` timesteps was only `mean_ep_length=9` with `mean_reward=-1.25`, and even by `30k` timesteps the run had only improved to about `15` steps. That was far below the earlier wrapper-side residual-PD baseline, which strongly suggested a control-semantics regression rather than simple PPO variance. The root cause was that the new base environment only recomputed PD torques once per control step and then held that torque fixed over the whole interval, whereas the intended impedance behavior is to recompute torques at every MuJoCo physics substep.

**Next step:** Restore per-substep PD recomputation inside `CombatGymEnv.step()`, re-run environment validation, and discard / replace the already-running smoke experiment that used the wrong controller semantics.

## [2026-03-19 23:37] Restored per-substep PD recomputation in the base environment

**Why:** Match the old successful stand wrapper semantics inside the refactored base environment so the internal controller is a true substep-level impedance controller instead of a held open-loop torque command.

**Command:**
```bash
# Edit combatbench/envs/combat_gym.py and re-run python3 -m combatbench.baseline.sb3.validate_env
```

**Result:**
`CombatGymEnv.step()` now reapplies the residual-position controller at every physics substep regardless of whether an explicit callback is used. After the fix, `validate_env.py` again showed the expected strong passive standing behavior: the stand smoke test opened with rewards around `1.438-1.442` for the first five control steps, which matches the previous healthy residual-PD baseline and confirms that the controller semantics are restored.

**Next step:** Stop the earlier invalid smoke run and launch a fresh clean training run on the corrected base-environment controller.

## [2026-03-19 23:39] Restarted stand smoke training on the corrected base-environment controller

**Why:** The previous `stand_env_residual_smoke` run used the wrong held-torque semantics and should not be trusted. A clean rerun was needed after restoring per-substep PD recomputation.

**Command:**
```bash
# Stop the previous stand_env_residual_smoke process and start:
python3 -u -m combatbench.baseline.sb3.train \
  --phase stand \
  --timesteps 100000 \
  --run-name stand_env_residual_smoke_v2 \
  --device cpu \
  --match-duration 10 \
  --control-frequency 20 \
  --initial-distance 2.5 \
  --checkpoint-freq 20000 \
  --eval-freq 10000 > combatbench/baseline/sb3/runs/stand_env_residual_smoke_v2/train.log 2>&1
```

**Result:**
The corrected run immediately returned to the expected healthy range. Early rollout stats stabilized around `ep_len_mean≈32` and `ep_rew_mean≈32`. The first evaluation at `10k` timesteps reached `mean_ep_length=33` and `mean_reward=33.78`, which is consistent with the previously healthy residual-PD baseline and confirms that the base-environment controller refactor is now functionally aligned with the older successful stand setup.

**Next step:** Let this corrected smoke run continue and monitor whether later evaluations move beyond the passive `~33`-step baseline instead of regressing back toward the invalid low-step behavior.

## [2026-03-19 23:47] Identified the next bottleneck as fixed-controller softness

**Why:** The corrected rerun was improving (`33 -> 36 -> 44` eval steps by `30k`) but remained far from the `200`-step / `10s` goal, so the next question was whether the remaining bottleneck was still the passive controller rather than PPO itself.

**Command:**
```bash
# 1) Replay the current best_model with residual scale multipliers.
# 2) Probe zero-action passive standing under different fixed Kp/Kd settings.
```

**Result:**
Two useful signals emerged. First, the current best model performed best at about `0.8x` of the current residual action scale, slightly better than the existing `1.0x`. Second, the stronger signal came from the passive-controller sweep: the default `Kp=4.0, Kd=0.4` baseline stood for only about `33` steps with zero action, while `Kp=8.0, Kd=0.4` raised the passive baseline to **`53` steps**. This strongly suggests the stand problem is still too "soft" at the low-level controller, and that increasing fixed stiffness is a higher-value next move than simply waiting for more PPO updates on the old controller.

**Next step:** Promote fixed stand gains and the slightly smaller residual scale into the actual stand environment defaults, then launch a fresh training run on top of the improved passive stability basin.

## [2026-03-19 23:49] Tuned the stand-mode fixed controller defaults for the next run

**Why:** Encode the promising controller settings directly into the stand environment so that the next smoke run starts from a better passive basin instead of relying on ad hoc overrides.

**Command:**
```bash
# Edit combatbench/envs/combat_gym.py and combatbench/baseline/sb3/selfplay_env.py,
# then re-run validate_env.py plus a zero-action stand rollout.
```

**Result:**
Added a public `set_controller_gains()` interface to `CombatGymEnv`, and switched the stand wrapper to fixed `Kp=8.0`, `Kd=0.4`, with a `0.8x` residual action-scale multiplier. `validate_env.py` still passed, and the direct zero-action stand rollout now lasts **`53` control steps**, up from the old `33`-step passive baseline.

**Next step:** Commit these controller-tuning changes, stop the older weaker-controller training run, and start a new smoke run on the stronger stand defaults.

## [2026-03-19 23:58] Found a controller setting that reaches the full 10-second horizon

**Why:** Even after the `v3` run improved to `57` steps at `10k` and `63` steps at `20k`, the `200`-step target was still far away. The next step was to check whether a stronger fixed controller would let the already-learned policy exploit a much larger stability basin.

**Command:**
```bash
# 1) Sweep passive zero-action standing over stronger Kp/Kd values.
# 2) Replay stand_env_residual_smoke_v3/best_model.zip under candidate gains and action scales.
```

**Result:**
The passive sweep showed further headroom: `Kp=16.0, Kd=0.2` reached `64` zero-action steps, versus `53` for the then-current defaults. More importantly, replaying the current `stand_env_residual_smoke_v3` best model under stronger fixed gains produced a breakthrough: both `Kp=12.0, Kd=0.2` and `Kp=16.0, Kd=0.2` reached the full **`200` control-step horizon** over repeated deterministic episodes. To keep the controller less aggressive while preserving the success, the stand defaults were promoted to **`Kp=12.0`, `Kd=0.2`** with the existing `0.8x` residual action-scale multiplier.

**Next step:** Re-run environment validation and formally evaluate the current best model under the promoted stand defaults to confirm stable `10s` standing across multiple episodes.

## [2026-03-20 00:00] Verified 10-second standing with the promoted stand controller defaults

**Why:** Confirm that the promoted default controller configuration is not just a one-off probe result, but actually solves the user's target task when used through the normal evaluation entrypoint.

**Command:**
```bash
python3 -m combatbench.baseline.sb3.validate_env

python3 -m combatbench.baseline.sb3.evaluate \
  --mode selfplay \
  --model /data1/mono/things/combatbench/combatbench/baseline/sb3/runs/stand_env_residual_smoke_v3/best_model/best_model.zip \
  --phase stand \
  --episodes 5 \
  --duration 10 \
  --control-frequency 20
```

**Result:**
Validation passed. The formal self-play evaluation then achieved the goal cleanly: **5/5 deterministic episodes** lasted the full `200` control steps (`10.0s`) with `winner=draw` and `reason=Time limit reached (10.0s), draw`. Average episode length was exactly `200.0`, confirming that the standing controller + current learned policy now satisfy the `10秒不倒` requirement.

**Next step:** Stop the now-obsolete background training process, commit the final promoted stand defaults and experiment log, and use this configuration as the new standing baseline for any later fine-tuning or transfer to fight training.

## [2026-03-20 00:25] Started exporting a standing-success video with the direct env loop

**Why:** The user asked for a qualitative success video. The repository already has a proven `run_without_policy.py` pattern that uses `CombatGymEnv(render_mode="rgb_array")` and `save_video()`, so the safest path is to reuse that rendering/export path with the verified standing model.

**Command:**
```bash
# Run a one-off script that sets MUJOCO_GL=egl before imports,
# loads stand_env_residual_smoke_v3/best_model.zip on CPU,
# steps the stand environment deterministically, and saves an mp4.
```

**Result:**
The one-off direct environment rollout succeeded. Using `stand_env_residual_smoke_v3/best_model.zip`, the deterministic stand episode lasted the full `200` control steps and ended with `Time limit reached (10.0s), draw`. The video was written to `combatbench/baseline/sb3/runs/stand_env_residual_smoke_v3/stand_success_10s.mp4` with `100` recorded frames. A follow-up OpenCV probe confirmed the file is readable and valid: `1280x720`, `10 FPS`, `100` frames, and the first frame decoded successfully.

**Next step:** Commit the updated export log and provide the video path to the user for inspection.
