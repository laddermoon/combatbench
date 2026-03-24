## [2026-03-24 18:00] Stop previous 10deg no-clamp-first GRPO run

**Why:** The current experiment had already validated the core hypothesis, so the next step was to stop the long run, record the result, and switch to the new three-level curriculum reward.

**Command:**
```bash
kill 1448445 && sleep 1 && ps -p 1448445 -o pid=,stat=,cmd=
```

**Result:**
`kill` returned non-zero, but follow-up `ps -p 1448445 -o pid=,stat=,cmd=` produced no output, and the run directory stopped growing after `eval_26880000.json`. The training process was no longer running.

**Next step:** Extract the best/latest eval results and write the completed findings into `THOUGHTS_AND_EXP.md`.

## [2026-03-24 18:05] Inspect best eval of the completed 10deg no-clamp-first run

**Why:** Needed to confirm the stopping point, best checkpoint, and whether the run had already solved the no-clamp + approach objective.

**Command:**
```bash
python3 - <<'PY'
import json
from pathlib import Path
run_dir = Path('/data1/mono/things/combatbench/baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_target055_clamp10deg_noclampfirst_penalty1000_40m_nenv64_g8_cuda_20260324_131627/eval')
best = None
for path in sorted(run_dir.glob('eval_*.json')):
    data = json.loads(path.read_text())
    reward = float(data['mean_reward'])
    if best is None or reward > best[1]:
        best = (path.name, reward, float(data['mean_episode_clamp_count']), float(data['mean_final_distance']))
print({'best_file': best[0], 'mean_reward': best[1], 'mean_clamp': best[2], 'mean_final_distance': best[3]})
PY
```

**Result:**
Best eval was `eval_26880000.json` with `mean_reward=-542.76171875`, `mean_clamp=0.0`, and `mean_final_distance=0.4796849489212036`.

**Next step:** Update experiment notes and design the new curriculum reward that builds on this best model.

## [2026-03-24 18:10] Implement three-level curriculum reward and resume support

**Why:** The validated run showed the policy can get close without clamp, but it still drifts away and attacks weakly. The next experiment should preserve no-clamp priority, then optimize reaching within `0.6m`, and only then reward offense.

**Command:**
```bash
Applied code changes in:
- baseline/mujoco21dof_nonfall/reward.py
- baseline/mujoco21dof_nonfall/env_wrapper.py
- baseline/mujoco21dof_nonfall/grpo.py
- baseline/mujoco21dof_nonfall/train_sb3.py
- baseline/mujoco21dof_nonfall/train_grpo.py
- baseline/mujoco21dof_nonfall/THOUGHTS_AND_EXP.md
```

**Result:**
Implemented `episode_curriculum` reward mode with the following hierarchy:
1. If clamp happens, ignore all other reward terms and apply only clamp penalty.
2. If an episode never enters `0.6m`, optimize approach via `max(min_distance - 0.6, 0)`.
3. If an entire collected batch reaches `0.6m` without clamp, switch to rewarding `damage_dealt` only.

Also confirmed that `best_model.pt` uses the same checkpoint format and can be used directly with `--resume-from`.

**Next step:** Run syntax checks and a smoke resume run before launching the full-budget training.

## [2026-03-24 18:15] Validate curriculum reward implementation

**Why:** Needed to ensure the new reward ordering really matches the intended lexicographic objective before spending another 40.96M-step budget.

**Command:**
```bash
python3 -m py_compile baseline/mujoco21dof_nonfall/reward.py baseline/mujoco21dof_nonfall/env_wrapper.py baseline/mujoco21dof_nonfall/grpo.py baseline/mujoco21dof_nonfall/train_sb3.py baseline/mujoco21dof_nonfall/train_grpo.py

python3 - <<'PY'
import sys
sys.path.insert(0, '/data1/mono/things')
from combatbench.baseline.mujoco21dof_nonfall.reward import DistanceStageRewardConfig, compute_distance_stage_curriculum_returns
from combatbench.baseline.mujoco21dof_nonfall.grpo import load_grpo_checkpoint
cfg = DistanceStageRewardConfig(reward_mode='episode_curriculum', clamp_penalty_scale=1000.0, close_enough_distance=0.6, distance_reward_scale=10.0, distance_reward_power=2.0, attack_damage_reward_scale=1000.0)
scenarios = [
    {'episode_clamp_count': 2, 'episode_min_horizontal_distance': 0.4, 'episode_damage_dealt': 1.0},
    {'episode_clamp_count': 0, 'episode_min_horizontal_distance': 0.8, 'episode_damage_dealt': 0.0},
    {'episode_clamp_count': 0, 'episode_min_horizontal_distance': 0.5, 'episode_damage_dealt': 0.2},
]
rewards, _, attack_enabled = compute_distance_stage_curriculum_returns(scenarios, cfg)
print({'mixed_rewards': rewards, 'mixed_attack_enabled': attack_enabled})
close_batch = [
    {'episode_clamp_count': 0, 'episode_min_horizontal_distance': 0.5, 'episode_damage_dealt': 0.2},
    {'episode_clamp_count': 0, 'episode_min_horizontal_distance': 0.55, 'episode_damage_dealt': 0.1},
]
rewards2, _, attack_enabled2 = compute_distance_stage_curriculum_returns(close_batch, cfg)
print({'close_rewards': rewards2, 'close_attack_enabled': attack_enabled2})
actor, checkpoint = load_grpo_checkpoint('/data1/mono/things/combatbench/baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_target055_clamp10deg_noclampfirst_penalty1000_40m_nenv64_g8_cuda_20260324_131627/best_model/best_model.pt', device='cpu')
print({'loaded_algorithm': checkpoint.get('algorithm'), 'hidden_sizes': checkpoint['model_config']['hidden_sizes']})
PY
```

**Result:**
Syntax check passed. The reward smoke check produced:
- mixed batch: `[-2000.0, -0.4000000000000003, 0.0]`, `attack_enabled=False`
- all-close batch: `[200.0, 100.0]`, `attack_enabled=True`
- `best_model.pt` loaded successfully as a GRPO checkpoint.

**Next step:** Run a short smoke training resumed from `best_model.pt`, then launch the full 40.96M-step training if the smoke run is healthy.

## [2026-03-24 18:18] Smoke resume run with `episode_curriculum`

**Why:** Before starting another expensive long run, verify that the new curriculum reward can actually resume from the previous `best_model.pt`, finish a short training loop, run eval, and save checkpoints without runtime issues.

**Command:**
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

**Result:**
The smoke run succeeded end-to-end.

Key signals from the run:
- `resume-from best_model.pt` worked
- training loop completed normally
- eval/checkpoint saving worked
- at `6400` timesteps, eval reported:
  - `mean_episode_clamp_count=0.00`
  - `mean_episode_min_horizontal_distance=0.413`
  - `mean_final_distance=0.413`
  - `curriculum_attack_enabled=True`
  - `best_eval_reward=0.0`

Smoke run directory:
- `baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_curriculum_smoke_resume_20260324_181835`

**Next step:** Commit the curriculum-reward implementation and launch the full-budget resumed GRPO training with the same `40.96M` budget as the previous run.

## [2026-03-24 18:20] Launch full-budget resumed curriculum GRPO training

**Why:** The smoke run showed that the new three-level curriculum reward works and can resume from the previous no-clamp best model. The next step is the real `40.96M`-step training run.

**Command:**
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

**Result:**
Training launched successfully in the background.

Run directory:
- `baseline/mujoco21dof_nonfall/runs/grpo_distance_stage1_curriculum06_attack_resume_40m_nenv64_g8_cuda_20260324_182043`

Initial health check from `Update 1`:
- `Total timesteps=6400`
- `Mean episode return=-906.250`
- `Mean episode clamp count=0.906`
- `Mean episode damage dealt=0.0046`
- `Mean episode min horizontal distance=0.494`
- `Mean final distance=0.497`
- `Curriculum attack enabled=False`

This is a healthy start: the resumed policy is already close to the `0.6m` threshold, still occasionally clamps, and has not yet stably entered the batch-level attack-reward phase.

**Next step:** Record a concise summary document and continue monitoring future eval points.
