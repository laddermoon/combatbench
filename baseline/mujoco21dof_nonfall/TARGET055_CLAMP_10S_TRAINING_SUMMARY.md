# Target 0.55 Clamp 10s Training Summary

## The Correct Path

1. **Record the previous `0.4m` experiment conclusion first**
   - Action: append the prior run’s findings to `baseline/mujoco21dof_nonfall/THOUGHTS_AND_EXP.md`
   - Expected: the decision to move the stage-1 target from `0.4m` to `0.55m` is documented before the next iteration starts

2. **Expose clamp counts from the environment**
   - Action: update `envs/combat_gym.py` to track non-fall clamp counts per control step and per episode, and expose them in `info['non_fall_mode']['clamp_counts']`
   - Expected: the training wrapper can measure how much the policy relies on the clamp hack

3. **Turn clamp usage into a reward penalty**
   - Action: update `baseline/mujoco21dof_nonfall/env_wrapper.py`, `reward.py`, and `train_sb3.py` so `robot_a` clamp counts flow into `DistanceStageRewardConfig` and subtract from reward via `clamp_penalty_scale`
   - Expected: heavier clamp usage produces a larger penalty during training

4. **Lengthen stage-1 episodes to 10 seconds and resize PPO rollout settings**
   - Action: launch training with `--match-duration 10 --control-frequency 20 --n-envs 64 --n-steps 200 --batch-size 12800 --total-timesteps 409600`
   - Expected: each rollout covers one full 10-second episode per env, preserving the episode-uniform reward assumption

5. **Run the formal `SubprocVecEnv` training job**
   - Action: start `distance_stage1_episode_uniform_target055_clamp10s_409k_nenv64_subproc_cpu`
   - Expected: the run starts cleanly, the first rollout logs `ep_len_mean=200`, and the first eval appears at `25600` timesteps

## Pitfalls & Solutions

### Issue: `0.4m` was too small for the stage-1 distance target
- **Symptom:** the best previous policy approached steadily but plateaued around `0.57m~0.66m` in a 20-second rollout instead of entering `0.4m`
- **Root cause:** the stage-1 objective pushed the policy into a “close enough but not truly contact-ready” local optimum
- **Solution:** raise the stage-1 target distance to `0.55m`, which is still attack-ready but easier to reach consistently

### Issue: the policy could exploit non-fall clamp support as a locomotion hack
- **Symptom:** forward progress can be produced partly by leaning into the clamp limits instead of learning more natural stepping behavior
- **Root cause:** the environment enforced the clamp but did not expose or penalize clamp usage during reward computation
- **Solution:** count clamp events in `CombatGymEnv` and subtract a reward term proportional to clamp usage with `--distance-stage-clamp-penalty-scale`

### Issue: 5-second episodes were too short to reveal whether the policy would continue closing distance
- **Symptom:** the 5-second evaluation suggested the policy stalled near `1.0m`, but a 20-second evaluation showed it could keep moving closer
- **Root cause:** the short horizon hid slower but meaningful approach behavior
- **Solution:** extend stage-1 episodes to 10 seconds and expand rollout/batch settings so PPO still sees full episodes in each update
