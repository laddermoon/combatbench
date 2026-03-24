# GRPO Distance-Stage Curriculum Summary

## The Correct Path

1. **Stop the previous validated no-clamp-first run**
   - Action: stop the old process and verify the latest eval file in the run directory
   - Expected: the previous run stops around `eval_26880000.json`, preserving the best model and eval artifacts

2. **Confirm the previous run already solved the no-clamp + close-approach objective**
   - Action: inspect `eval_25600000.json` and `eval_26880000.json`
   - Expected:
     - `mean_episode_clamp_count=0.0`
     - `mean_final_distance` near or below the `0.55m` target
     - conclusion: the current best model is a good resume point

3. **Implement the new three-level curriculum reward**
   - Action: update `reward.py`, `env_wrapper.py`, `grpo.py`, `train_sb3.py`, and `train_grpo.py`
   - Expected:
     - clamp remains the top-level objective
     - if an episode never enters `0.6m`, optimize approach only
     - if a whole collected batch enters `0.6m` without clamp, switch to rewarding `damage_dealt`
     - `best_model.pt` can still be resumed with `--resume-from`

4. **Run a smoke resume training before the full budget**
   - Action:
     ```bash
     CUDA_VISIBLE_DEVICES=0 python3 baseline/mujoco21dof_nonfall/train_grpo.py \
       --run-name grpo_distance_stage1_curriculum_smoke_resume \
       --resume-from baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_target055_clamp10deg_noclampfirst_penalty1000_40m_nenv64_g8_cuda_20260324_131627/best_model/best_model.pt \
       --curriculum-stage distance_stage1 \
       --distance-stage-reward-mode episode_curriculum \
       --distance-stage-reward-power 2.0 \
       --distance-stage-target-distance 0.55 \
       --distance-stage-clamp-penalty-scale 1000 \
       --distance-stage-prioritize-no-clamp \
       --distance-stage-close-enough-distance 0.6 \
       --distance-stage-attack-damage-reward-scale 1000 \
       --opponent standing \
       --eval-opponent standing \
       --initial-distance 2.0 \
       --match-duration 5 \
       --control-frequency 20 \
       --total-timesteps 6400 \
       --n-envs 8 \
       --episodes-per-update 8 \
       --group-size 8 \
       --minibatch-size 800 \
       --update-epochs 2 \
       --learning-rate 1e-4 \
       --ent-coef 0.0005 \
       --target-kl 0.02 \
       --checkpoint-freq 6400 \
       --eval-freq 3200 \
       --device cuda \
       --train-vec-env subproc \
       --subproc-start-method spawn \
       --non-fall-pitch-limit-deg 10 \
       --non-fall-roll-limit-deg 10
     ```
   - Expected:
     - training and eval complete without runtime errors
     - eval can already report `curriculum_attack_enabled=True` once all eval episodes are close enough and clamp-free

5. **Launch the full-budget resumed training**
   - Action:
     ```bash
     CUDA_VISIBLE_DEVICES=0 python3 baseline/mujoco21dof_nonfall/train_grpo.py \
       --run-name grpo_distance_stage1_curriculum06_attack_resume_40m_nenv64_g8_cuda \
       --resume-from baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_target055_clamp10deg_noclampfirst_penalty1000_40m_nenv64_g8_cuda_20260324_131627/best_model/best_model.pt \
       --curriculum-stage distance_stage1 \
       --distance-stage-reward-mode episode_curriculum \
       --distance-stage-reward-power 2.0 \
       --distance-stage-target-distance 0.55 \
       --distance-stage-clamp-penalty-scale 1000 \
       --distance-stage-prioritize-no-clamp \
       --distance-stage-close-enough-distance 0.6 \
       --distance-stage-attack-damage-reward-scale 1000 \
       --opponent standing \
       --eval-opponent standing \
       --initial-distance 2.0 \
       --match-duration 5 \
       --control-frequency 20 \
       --total-timesteps 40960000 \
       --n-envs 64 \
       --episodes-per-update 64 \
       --group-size 8 \
       --minibatch-size 6400 \
       --update-epochs 4 \
       --learning-rate 1e-4 \
       --ent-coef 0.0005 \
       --target-kl 0.02 \
       --checkpoint-freq 2560000 \
       --eval-freq 1280000 \
       --device cuda \
       --train-vec-env subproc \
       --subproc-start-method spawn \
       --non-fall-pitch-limit-deg 10 \
       --non-fall-roll-limit-deg 10
     ```
   - Expected:
     - run directory: `baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_curriculum06_attack_resume_40m_nenv64_g8_cuda_20260324_182043`
     - early updates show the resumed model already near the `0.6m` threshold
     - later evals determine whether the policy can stay clamp-free and start converting proximity into damage

## Pitfalls & Solutions

### Issue: `kill` returned non-zero when stopping the previous run
- **Symptom:** `kill 1448445` did not report success.
- **Root cause:** the process likely exited between checks or the shell command raced with process state changes.
- **Solution:** verify with `ps -p <pid>` and check that the run directory stops producing new eval files.

### Issue: episode-level rewards leaked through `info["reward_terms"]`
- **Symptom:** even when nonterminal `reward` was set to `0.0`, GRPO still accumulated intermediate shaping terms.
- **Root cause:** `GRPORolloutCollector.extract_step_reward()` prefers `reward_terms` over the scalar `reward`, so zeroing only `reward` was insufficient.
- **Solution:** in `env_wrapper.py`, zero `reward_terms` too when suppressing nonterminal rewards for `episode_uniform` and `episode_curriculum`.

### Issue: the new curriculum objective is batch-level, not just episode-level
- **Symptom:** the intended rule was "only reward attack when all collected episodes have reached within `0.6m`", but the environment computes rewards one episode at a time.
- **Root cause:** batch-level curriculum gating cannot be expressed by a single environment step alone.
- **Solution:** keep environment metrics/episode stats minimal, then recompute episode returns inside `GRPORolloutCollector` and `evaluate_grpo_actor` using the whole collected batch.

### Issue: resume compatibility could have been broken by the new reward mode
- **Symptom:** a new reward pipeline sometimes implies changing checkpoint format or actor architecture.
- **Root cause:** if architecture or checkpoint layout had changed, `best_model.pt` could not be resumed safely.
- **Solution:** preserve the actor/checkpoint format and use `load_grpo_checkpoint()` directly; smoke-test `--resume-from best_model.pt` before launching the full run.
