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
