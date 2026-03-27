## [2026-03-25 16:40] Restore combatbench working tree via proxy clone

**Why:** The original `things/combatbench` working tree disappeared, leaving only a broken gitlink state. Training and code validation could not continue until the repository contents were restored.

**Command:**
```bash
export http_proxy=http://192.168.16.76:18000
export https_proxy=http://192.168.16.76:18000
git clone https://github.com/laddermoon/combatbench.git /data1/mono/things/combatbench
```

**Result:**
`combatbench` was restored successfully under `/data1/mono/things/combatbench`.

**Next step:** Re-apply the self-play rollout and action-regularized GRPO changes to the freshly restored working tree.

## [2026-03-25 16:45] Re-apply self-play GRPO and action-regularization changes

**Why:** The restored repository was clean, so the earlier self-play rollout and action penalty modifications had to be reapplied before any smoke or full training could proceed.

**Command:**
```bash
Applied code changes in:
- baseline/__init__.py
- baseline/mujoco21dof_nonfall/env_wrapper.py
- baseline/mujoco21dof_nonfall/train_sb3.py
- baseline/mujoco21dof_nonfall/grpo.py
- baseline/mujoco21dof_nonfall/train_grpo.py
```

**Result:**
Implemented again:
- `SelfPlaySymmetricEnv` for symmetric two-sided rollout collection
- self-play training env switch in `train_sb3.py`
- GRPO rollout splitting into A/B trajectories
- action magnitude / action delta multiplicative regularization with neutral coefficient baseline `1.0`
- GRPO CLI flags and metric logging
- compatibility guard in `baseline/__init__.py` for the missing legacy `baseline.sb3` module

**Next step:** Run syntax checks and a short smoke training from scratch because the previous GRPO checkpoint artifacts are no longer available locally.

## [2026-03-25 16:53] Smoke-train self-play GRPO from scratch with conservative action penalties

**Why:** The old GRPO checkpoint artifacts were not available after restoring the repo, so the quickest validation path was to smoke-test the new training stack from scratch before launching a full-budget run.

**Command:**
```bash
python3 baseline/mujoco21dof_nonfall/train_grpo.py \
  --run-name grpo_distance_stage1_selfplay_regularized_smoke_fresh \
  --curriculum-stage distance_stage1 \
  --distance-stage-reward-mode episode_curriculum \
  --distance-stage-reward-power 2.0 \
  --distance-stage-target-distance 0.55 \
  --distance-stage-clamp-penalty-scale 1000 \
  --distance-stage-prioritize-no-clamp \
  --distance-stage-close-enough-distance 0.6 \
  --distance-stage-attack-damage-reward-scale 1000 \
  --rollout-self-play \
  --action-magnitude-loss-coef 2 \
  --action-delta-loss-coef 2 \
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
  --train-vec-env dummy \
  --non-fall-pitch-limit-deg 10 \
  --non-fall-roll-limit-deg 10
```

**Result:**
The smoke run completed successfully:
- run directory: `baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_selfplay_regularized_smoke_fresh_20260325_165306`
- final model and best model were both written successfully
- no runtime issues in self-play rollout, evaluation, checkpointing, or tensorboard logging
- observed `mean_loss_multiplier` stayed around `1.78x-1.80x`, which is much more reasonable than the previously tested `5/5` setting

Observed behavior in this short fresh-start run:
- policy is still far from solving the task from scratch
- clamp counts remain high
- eval reward stays strongly negative
- but the full training pipeline is stable and suitable for a long run

**Next step:** Commit the code changes, then launch the full-budget self-play GRPO training from scratch using the same `2/2` action-penalty coefficients.

## [2026-03-25 16:56] Commit source changes and launch the full-budget fresh training run

**Why:** The smoke run validated the code path, so the next step was to preserve the implementation in git and start the actual long training run from scratch.

**Command:**
```bash
git add baseline/__init__.py \
  baseline/mujoco21dof_nonfall/env_wrapper.py \
  baseline/mujoco21dof_nonfall/grpo.py \
  baseline/mujoco21dof_nonfall/train_grpo.py \
  baseline/mujoco21dof_nonfall/train_sb3.py \
  baseline/mujoco21dof_nonfall/GRPO_SELFPLAY_REGULARIZED_DETAIL.md
git commit -m "Add self-play GRPO rollout and action regularization"
git push

CUDA_VISIBLE_DEVICES=0 python3 baseline/mujoco21dof_nonfall/train_grpo.py \
  --run-name grpo_distance_stage1_selfplay_regularized_fresh_40m_nenv64_g8_cuda \
  --curriculum-stage distance_stage1 \
  --distance-stage-reward-mode episode_curriculum \
  --distance-stage-reward-power 2.0 \
  --distance-stage-target-distance 0.55 \
  --distance-stage-clamp-penalty-scale 1000 \
  --distance-stage-prioritize-no-clamp \
  --distance-stage-close-enough-distance 0.6 \
  --distance-stage-attack-damage-reward-scale 1000 \
  --rollout-self-play \
  --action-magnitude-loss-coef 2 \
  --action-delta-loss-coef 2 \
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

**Result:**
- local commit succeeded: `a533515` (`Add self-play GRPO rollout and action regularization`)
- `git push` failed with HTTP `403` because the current GitHub identity `conggova` does not have write permission to `laddermoon/combatbench`
- the full training run started successfully in the background
- run directory: `baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_selfplay_regularized_fresh_40m_nenv64_g8_cuda_20260325_165629`
- early training is healthy:
  - `Update 1` at `6400` timesteps
  - `Update 4` at `25600` timesteps
  - `mean_loss_multiplier` remains stable around `1.785-1.788`

**Next step:** Monitor the long training run, later update the final SUMMARY, and resolve the remote push permission issue separately if needed.

## [2026-03-25 23:05] Export a video with the latest saved GRPO checkpoint and inspect frames

**Why:** The training run had progressed to `7.68M` timesteps with a fresh checkpoint and eval record, so the next step was to visually inspect the learned behavior rather than relying only on scalar metrics.

**Command:**
```bash
python3 baseline/mujoco21dof_nonfall/eval_grpo_policy.py \
  --model-path baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_selfplay_regularized_fresh_40m_nenv64_g8_cuda_20260325_165629/checkpoints/grpo_attacker_7680000.pt \
  --opponent standing \
  --episodes 1 \
  --seed 0 \
  --device cuda \
  --video baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_selfplay_regularized_fresh_40m_nenv64_g8_cuda_20260325_165629/videos/eval_7680000_vs_standing.mp4 \
  --summary-json baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_selfplay_regularized_fresh_40m_nenv64_g8_cuda_20260325_165629/videos/eval_7680000_vs_standing.json \
  --match-duration 5 \
  --control-frequency 20 \
  --initial-distance 2.0 \
  --non-fall-mode \
  --non-fall-pitch-limit-deg 10 \
  --non-fall-roll-limit-deg 10 \
  --damage-scale 100

ffmpeg -y -i baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_selfplay_regularized_fresh_40m_nenv64_g8_cuda_20260325_165629/videos/eval_7680000_vs_standing.mp4 \
  -vf "select='eq(n,0)+eq(n,36)+eq(n,72)+eq(n,120)',scale=960:-1" -vsync 0 \
  baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_selfplay_regularized_fresh_40m_nenv64_g8_cuda_20260325_165629/videos/frames_7680000/frame_%03d.png
```

**Result:**
- video exported successfully: `videos/eval_7680000_vs_standing.mp4`
- summary exported successfully: `videos/eval_7680000_vs_standing.json`
- sampled frames show the robot still keeps a very similar bent-arm standing pose across the episode, with little visible forward locomotion
- the scalar results match the visual impression:
  - no damage dealt
  - draw after full `5s`
  - end distance around `1.88m`

**Next step:** Continue training and later compare newer checkpoints to see whether the policy eventually converts the no-clamp behavior into real forward approach.
