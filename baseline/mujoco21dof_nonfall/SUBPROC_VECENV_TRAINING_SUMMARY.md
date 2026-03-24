# Subproc VecEnv Training Summary

## The Correct Path

1. **Switch training env construction to `SubprocVecEnv`**
   - Action: update `baseline/mujoco21dof_nonfall/train_sb3.py` to build training envs through `build_train_vec_env()` and set `--train-vec-env subproc`
   - Expected: when `n_envs > 1`, training uses multi-process stepping instead of single-process `DummyVecEnv`

2. **Keep eval env compatible with the training vec-env type**
   - Action: add `build_eval_vec_env()` and construct eval env with the same vec-env family when training uses `SubprocVecEnv`
   - Expected: evaluation runs without SB3 vec-env type mismatch warnings

3. **Validate with a small smoke run**
   - Action: run a short `distance_stage1` + `episode_uniform` smoke command with `--train-vec-env subproc --subproc-start-method spawn`
   - Expected: PPO starts successfully, logs `episode_uniform` metrics, and exits cleanly

4. **Restart the real training run with multi-process env stepping**
   - Action: stop the old `DummyVecEnv` job and launch:
     - `python3 baseline/mujoco21dof_nonfall/train_sb3.py --run-name distance_stage1_episode_uniform_200k_nenv64_subproc_cpu --curriculum-stage distance_stage1 --distance-stage-reward-mode episode_uniform --distance-stage-reward-power 2.0 --opponent standing --eval-opponent standing --initial-distance 2.0 --distance-stage-target-distance 0.4 --match-duration 5 --control-frequency 20 --total-timesteps 200000 --n-envs 64 --n-steps 100 --batch-size 6400 --learning-rate 1e-4 --ent-coef 0.0005 --target-kl 0.01 --checkpoint-freq 20000 --eval-freq 10000 --eval-episodes 3 --device cpu --train-vec-env subproc --subproc-start-method spawn`
   - Expected: the first rollout should complete with much higher throughput than the previous single-process setup

## Pitfalls & Solutions

### Issue: `DummyVecEnv` became the main bottleneck at `n_envs=64`
- **Symptom:** the first rollout took a very long time to appear and overall throughput was poor
- **Root cause:** `DummyVecEnv` steps all environments serially in one process
- **Solution:** switch the training env to `SubprocVecEnv` so environment stepping is parallelized across subprocesses

### Issue: training and eval env types mismatched after the first parallelization change
- **Symptom:** SB3 emitted a warning that training env and eval env were not of the same type
- **Root cause:** training used `SubprocVecEnv` while eval still used `DummyVecEnv`
- **Solution:** add `build_eval_vec_env()` and use the matching vec-env type for eval when training runs with subprocesses

### Issue: GPU was not the right optimization target for this workload
- **Symptom:** SB3 warned that PPO with `MlpPolicy` is usually not faster on GPU
- **Root cause:** the bottleneck here is mostly MuJoCo environment stepping, not neural network compute
- **Solution:** restart the formal run on CPU after introducing `SubprocVecEnv`, which addresses the actual throughput bottleneck
