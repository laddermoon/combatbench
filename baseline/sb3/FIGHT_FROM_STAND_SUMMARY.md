# Fight From Stand Summary

## The Correct Path

1. **Fix combat collision detection so cross-robot contacts are actually visible to the damage pipeline**
   - Action: update `combatbench/core/collision.py` to accept the real robot geom suffixes (`_red` / `_blue`) in addition to legacy suffixes
   - Expected: inter-robot collisions begin appearing in `info['collisions']` and can generate `hit_records`

2. **Validate opponent observations against the authoritative robot states**
   - Action: `python3 -m combatbench.baseline.sb3.validate_fight_mechanics --duration 8 --control-frequency 20 --initial-distance 0.6 --seed 0`
   - Expected: the script prints `observation_check=passed` for the sampled steps without assertion failures

3. **Validate damage accounting with HP-floor clipping**
   - Action: use the same `validate_fight_mechanics.py` command
   - Expected: each damage event passes the score-delta check, and the final output shows at least one robot has HP below `100`

4. **Make fight start from the proven standing controller instead of the zero-reference controller**
   - Action: in `baseline/sb3/selfplay_env.py`, configure fight reset to reuse the standing reference pose, fixed gains, and residual action scale for both robots
   - Expected: fight and stand now share the same stable control semantics, making stand-to-fight initialization meaningful

5. **Use a reusable direct video-export entrypoint**
   - Action: `python3 combatbench/run_policy_video.py ...`
   - Expected: rollout videos can be exported reliably without the fragile module-based render path

6. **Launch fight smoke training from a standing checkpoint**
   - Action: `python3 -m combatbench.baseline.sb3.train --phase fight --init-model <stand-best-model> ...`
   - Expected: the fight pipeline trains from a stable standing prior instead of from scratch

7. **Evaluate the trained fight checkpoint in direct independent matches**
   - Action: `python3 -m combatbench.baseline.sb3.evaluate --mode match --model combatbench/baseline/sb3/runs/fight_from_stand_smoke_v1/best_model/best_model.zip --phase fight --duration 8 --control-frequency 20 --initial-distance 0.6`
   - Expected: the current smoke-trained checkpoint can already achieve real combat damage and a KO at the validated close-contact distance, satisfying the minimum viable combat baseline

## Pitfalls & Solutions

### Issue: Combat collisions were silently ignored
- **Symptom:** random or scripted close-range rollouts produced no `hit_records` and no HP loss
- **Root cause:** collision detection only looked for legacy geom suffixes such as `_a` / `_b`, while the actual XML geoms use `_red` / `_blue`
- **Solution:** detect both the real XML suffixes and legacy suffixes, using the robot's authoritative `suffix` field when available

### Issue: Collision processing crashed when damage logic finally started running
- **Symptom:** the rollout failed with `AttributeError: module 'mujoco' has no attribute 'mj_contactVelocity'`
- **Root cause:** the current MuJoCo Python package in this environment does not expose `mj_contactVelocity`
- **Solution:** compute contact-speed filtering from the contacting bodies' linear velocities (`data.cvel`) instead

### Issue: Damage validation initially reported false mismatches
- **Symptom:** `hit_records` existed, but validation claimed `damage_sum` and score delta were inconsistent
- **Root cause:** per-step damage can exceed the defender's remaining HP, and the score calculator clips HP at `0`
- **Solution:** validate against the effective score delta after HP-floor clipping, not the raw sum of negative damage entries

### Issue: Module-style video export kept crashing under headless rendering
- **Symptom:** `python -m combatbench.baseline.sb3.export_video ...` aborted with GLFW / OpenGL initialization errors
- **Root cause:** that startup path was more fragile for EGL initialization in this environment
- **Solution:** add a standalone `combatbench/run_policy_video.py` entrypoint that mirrors the already-proven `run_without_policy.py` initialization pattern

### Issue: Direct match evaluation initially disagreed with training behavior
- **Symptom:** the smoke-trained fight checkpoint looked healthy in shared-env evaluation, but direct `MatchRunner` evaluation still behaved like a no-damage draw
- **Root cause:** direct match evaluation was using bare `CombatGymEnv` defaults instead of the same standing-reference fight controller configuration used during training
- **Solution:** make `MatchRunner`, `evaluate.py`, and `run_policy_video.py` pass through the `phase` and reapply the same stand/fight controller configuration after reset

### Issue: The current smoke fight policy is distance-sensitive
- **Symptom:** at `initial_distance=1.0`, the current `best_model` still finishes as a full-HP draw, while at `initial_distance=0.6` it can KO the opponent quickly
- **Root cause:** the learned fight behavior is still a minimum viable close-contact policy, not yet a robust longer-range engage-and-strike policy
- **Solution:** treat `0.6m` as the validated MVP combat baseline for now, and only then iterate on reward / curriculum / distance settings if stronger approach behavior is needed
