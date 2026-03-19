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
