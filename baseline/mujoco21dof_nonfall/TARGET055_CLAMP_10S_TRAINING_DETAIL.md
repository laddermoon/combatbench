# Target 0.55 Clamp 10s Training Detail

## [2026-03-24 11:19 CST] Record previous experiment outcome

**Why:** The previous `target_distance=0.4` episode-uniform run produced actionable conclusions that should be written down before changing defaults.

**Command:**
```bash
edit baseline/mujoco21dof_nonfall/THOUGHTS_AND_EXP.md
```

**Result:**
Appended the 2026-03-24 experiment log summarizing the `0.4m` episode-uniform + `SubprocVecEnv` run, including the conclusion that the method is effective but `0.4m` is too aggressive for stage-1 and should move to `0.55m`.

**Next step:** Update the environment and reward path to penalize clamp usage.

## [2026-03-24 11:21 CST] Add clamp-count penalty support

**Why:** The current policy can exploit non-fall clamp assistance instead of learning more natural forward locomotion.

**Command:**
```bash
edit envs/combat_gym.py
edit baseline/mujoco21dof_nonfall/env_wrapper.py
edit baseline/mujoco21dof_nonfall/reward.py
edit baseline/mujoco21dof_nonfall/train_sb3.py
```

**Result:**
Implemented per-step and per-episode clamp counting in `CombatGymEnv`, forwarded robot A clamp counts through the single-agent wrapper metrics and episode stats, added `clamp_penalty_scale` to `DistanceStageRewardConfig`, and exposed the setting through `train_sb3.py`.

**Next step:** Validate the updated training path with a smoke run.

## [2026-03-24 11:23 CST] Smoke test the new 10s configuration

**Why:** Confirm that `target_distance=0.55`, `match_duration=10`, and clamp-penalized episode-uniform reward all work together before launching the full run.

**Command:**
```bash
python3 baseline/mujoco21dof_nonfall/train_sb3.py --run-name smoke_target055_clamp_10s --curriculum-stage distance_stage1 --distance-stage-reward-mode episode_uniform --distance-stage-reward-power 2.0 --distance-stage-target-distance 0.55 --distance-stage-clamp-penalty-scale 0.002 --opponent standing --eval-opponent standing --initial-distance 2.0 --match-duration 10 --control-frequency 20 --total-timesteps 400 --n-envs 2 --n-steps 200 --batch-size 400 --learning-rate 1e-4 --ent-coef 0.0005 --target-kl 0.01 --checkpoint-freq 1000 --eval-freq 1000 --eval-episodes 1 --device cpu --train-vec-env subproc --subproc-start-method spawn
```

**Result:**
Smoke training completed successfully. The rollout length matched the 10-second episode (`ep_len_mean=200`) and the updated reward path ran without errors.

**Next step:** Launch the formal multi-env training run.

## [2026-03-24 11:24 CST] Launch formal training run

**Why:** Start the next stage-1 experiment with the new target distance and clamp penalty under the parallelized env setup.

**Command:**
```bash
python3 baseline/mujoco21dof_nonfall/train_sb3.py --run-name distance_stage1_episode_uniform_target055_clamp10s_409k_nenv64_subproc_cpu --curriculum-stage distance_stage1 --distance-stage-reward-mode episode_uniform --distance-stage-reward-power 2.0 --distance-stage-target-distance 0.55 --distance-stage-clamp-penalty-scale 0.002 --opponent standing --eval-opponent standing --initial-distance 2.0 --match-duration 10 --control-frequency 20 --total-timesteps 409600 --n-envs 64 --n-steps 200 --batch-size 12800 --learning-rate 1e-4 --ent-coef 0.0005 --target-kl 0.01 --checkpoint-freq 51200 --eval-freq 25600 --eval-episodes 3 --device cpu --train-vec-env subproc --subproc-start-method spawn
```

**Result:**
Training launched successfully. Run directory: `baseline/mujoco21dof_nonfall/runs/distance_stage1_episode_uniform_target055_clamp10s_409k_nenv64_subproc_cpu_20260324_112359`.

**Next step:** Check the first rollout and first eval metrics.

## [2026-03-24 11:24 CST] Verify initial rollout and eval metrics

**Why:** Confirm that the longer episode length and larger rollout still produce healthy throughput and stable PPO updates.

**Command:**
```bash
check background command 749
```

**Result:**
The first rollout completed with `ep_len_mean=200`, `ep_rew_mean≈-22.9`, and `fps≈1393`. The first eval at `25600` timesteps reported `mean_reward≈-17.5`, `approx_kl≈0.0063`, and `clip_fraction≈0.0178`, which indicates the new run is stable so far.

**Next step:** Commit the code and documentation changes while the training continues in the background.
