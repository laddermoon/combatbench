# Subproc VecEnv Training Detail

## [2026-03-24 10:43 CST] Read repo and project context

**Why:** Follow project workflow rules before modifying combatbench training code.

**Command:**
```bash
read /data1/mono/CLAUDE.md
read /data1/mono/things/combatbench/CLAUDE.md
```

**Result:**
Loaded repo-wide and project-specific guidance. Confirmed this session is scoped to `things/combatbench` and code changes must be committed after completion.

**Next step:** Inspect current PPO training script and vectorized environment setup.

## [2026-03-24 10:43 CST] Inspect current training setup and running job

**Why:** Confirm current bottleneck and determine what needs to change before restarting training.

**Command:**
```bash
read baseline/mujoco21dof_nonfall/train_sb3.py
ps -ef | grep 'distance_stage1_episode_uniform_200k_nenv64_cuda' | grep -v grep
```

**Result:**
Confirmed training currently uses `DummyVecEnv` for the training environment and that the previous run is still active as PID `781996`.

**Next step:** Switch training env to `SubprocVecEnv`, validate startup, then stop the old run and restart.

## [2026-03-24 10:44 CST] Implement `SubprocVecEnv` support in training script

**Why:** Replace the single-process `DummyVecEnv` bottleneck with multi-process environment stepping while keeping the evaluation path compatible with the PPO callbacks.

**Command:**
```bash
edit baseline/mujoco21dof_nonfall/train_sb3.py
```

**Result:**
Added CLI flags for `--train-vec-env` and `--subproc-start-method`, introduced `build_train_vec_env()` and `build_eval_vec_env()`, and switched multi-env training to `SubprocVecEnv` while preserving `DummyVecEnv` for single-env fallback.

**Next step:** Validate that the new parallel env path works with episode-uniform reward rewriting.

## [2026-03-24 10:45 CST] Run smoke test for `SubprocVecEnv` training

**Why:** Ensure the new multi-process training path is runnable before terminating the long-running job.

**Command:**
```bash
python3 baseline/mujoco21dof_nonfall/train_sb3.py --run-name smoke_subproc_episode_uniform --curriculum-stage distance_stage1 --distance-stage-reward-mode episode_uniform --distance-stage-reward-power 2.0 --opponent standing --eval-opponent standing --initial-distance 2.0 --distance-stage-target-distance 0.4 --match-duration 1 --control-frequency 10 --total-timesteps 20 --n-envs 4 --n-steps 10 --batch-size 40 --learning-rate 1e-4 --ent-coef 0.0 --target-kl 0.02 --checkpoint-freq 1000 --eval-freq 1000 --eval-episodes 1 --device cpu --train-vec-env subproc --subproc-start-method spawn
```

**Result:**
Training completed successfully with `SubprocVecEnv`, but SB3 emitted a warning because the training env was `SubprocVecEnv` while the eval env remained `DummyVecEnv`.

**Next step:** Align eval env construction to remove the warning and rerun a quick smoke test.

## [2026-03-24 10:46 CST] Align eval env type and rerun smoke test

**Why:** Remove avoidable warnings and confirm that the final training/eval env combination is stable.

**Command:**
```bash
edit baseline/mujoco21dof_nonfall/train_sb3.py
python3 baseline/mujoco21dof_nonfall/train_sb3.py --run-name smoke_subproc_episode_uniform_evalmatch --curriculum-stage distance_stage1 --distance-stage-reward-mode episode_uniform --distance-stage-reward-power 2.0 --opponent standing --eval-opponent standing --initial-distance 2.0 --distance-stage-target-distance 0.4 --match-duration 1 --control-frequency 10 --total-timesteps 20 --n-envs 2 --n-steps 10 --batch-size 20 --learning-rate 1e-4 --ent-coef 0.0 --target-kl 0.02 --checkpoint-freq 1000 --eval-freq 1000 --eval-episodes 1 --device cpu --train-vec-env subproc --subproc-start-method spawn
```

**Result:**
The updated script ran successfully without the previous vec-env type warning. Episode-uniform reward logging remained intact.

**Next step:** Stop the old `DummyVecEnv` training job and restart the full run with `SubprocVecEnv`.

## [2026-03-24 10:47 CST] Stop old training run and restart with `SubprocVecEnv`

**Why:** Replace the slow single-process training job with the new multi-process configuration.

**Command:**
```bash
kill 781996
python3 baseline/mujoco21dof_nonfall/train_sb3.py --run-name distance_stage1_episode_uniform_200k_nenv64_subproc_cpu --curriculum-stage distance_stage1 --distance-stage-reward-mode episode_uniform --distance-stage-reward-power 2.0 --opponent standing --eval-opponent standing --initial-distance 2.0 --distance-stage-target-distance 0.4 --match-duration 5 --control-frequency 20 --total-timesteps 200000 --n-envs 64 --n-steps 100 --batch-size 6400 --learning-rate 1e-4 --ent-coef 0.0005 --target-kl 0.01 --checkpoint-freq 20000 --eval-freq 10000 --eval-episodes 3 --device cpu --train-vec-env subproc --subproc-start-method spawn
```

**Result:**
Stopped the old run and launched the new parallel training job successfully. The new run directory is `baseline/mujoco21dof_nonfall/runs/distance_stage1_episode_uniform_200k_nenv64_subproc_cpu_20260324_104746`.

**Next step:** Check initial rollout throughput to confirm the optimization is effective.

## [2026-03-24 10:48 CST] Check initial throughput of restarted training

**Why:** Confirm that replacing `DummyVecEnv` with `SubprocVecEnv` materially improved environment stepping throughput.

**Command:**
```bash
check background command 651
```

**Result:**
The first rollout completed successfully at about `1531 fps`, with `6400` timesteps collected in the first iteration (`64 envs x 100 steps`). This is a clear improvement over the prior single-process setup.

**Next step:** Commit the code and documentation changes.
