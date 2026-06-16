# Combat/Fight Training Summary

## The Correct Path

1. **Verify Environment Configurations**
   - Ensure `@/data1/mono/things/combatbench/baseline/humanoid21/blueprints/fight_env.yaml` contains `CombatScoringPlugin` and net `NetDamageRewarder` under observers.
   - Verify that `@/data1/mono/things/combatbench/baseline/humanoid21/blueprints/fight_mixed.yaml` correctly references `FightMixedPolicy` with primary, follow fallback (`/data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_follow_20260615_211441/policy_exports/u10295/policy_blueprint.yaml`), and recover fallback blueprints.

2. **Trigger Training Run**
   - Launch training from the project root directory (`/data1/mono/things/combatbench`) with python's PPO train script:
     ```bash
     PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --v2 --experiment fight --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_follow_20260615_211441/checkpoints/checkpoint_u10294.pt &> fight.log &
     ```

3. **Monitor Output Log**
   - Stream the training logs to watch episode rewards, gating switches, and learning stability:
     ```bash
     tail -f fight.log
     ```

## Pitfalls & Solutions

### Issue: Switching Test Failed due to Zero-Observation Safety Falls
- **Symptom:** During policy unit testing with `obs = np.zeros(96)`, the policy kept staying in or switching to `'recover'` instead of `'fight'`.
- **Root cause:** An all-zero observation is extremely out-of-distribution for the Gating MLP model, causing it to predict `p_safe < 0.65` and immediately trigger the safety shield.
- **Solution:** Temporarily set `policy.threshold = 0.0` in the distance state machine test to bypass safety gating and isolate/verify the distance-based hysteresis transitions cleanly.
