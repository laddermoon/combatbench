# Stand Training Summary

## The Correct Path

1. **Validate the SB3 stand environment**
   - Action: `python3 -m combatbench.baseline.sb3.validate_env`
   - Expected: The wrapped environment resets, steps, and passes the SB3 env check.

2. **Launch a background stand-training run**
   - Action: `python3 -u -m combatbench.baseline.sb3.train --phase stand ...`
   - Expected: PPO starts logging rollout/train stats and writes outputs to `combatbench/baseline/sb3/runs/<run-name>/`.

3. **Monitor logs until stability is confirmed**
   - Action: inspect the training log periodically
   - Expected: No import errors, no NaN crashes, and PPO iteration metrics continue updating.

4. **Evaluate the produced model if the run finishes quickly**
   - Action: `python3 -m combatbench.baseline.sb3.evaluate --mode selfplay ...`
   - Expected: The trained policy survives longer and shows a meaningful standing tendency.

5. **Export a qualitative video using the direct env loop if needed**
   - Action: `python3 combatbench/run_policy_video.py --mode shared_env ...` (or `python3 -m combatbench.baseline.sb3.export_video ...`, which now delegates to the standalone script)
   - Expected: Video export succeeds through the reusable EGL-safe entrypoint and writes a full standing rollout mp4 for visual inspection.

## Pitfalls & Solutions

### Issue: Background launch failed on the first attempt
- **Symptom:** Shell redirection failed with `No such file or directory` for `train.log`
- **Root cause:** The run directory did not exist before redirecting stdout/stderr into the log file
- **Solution:** Create `combatbench/baseline/sb3/runs/<run-name>/` first, then relaunch the background command

### Issue: Early positive reward did not yet mean sustained standing
- **Symptom:** PPO metrics improved quickly and the first eval reward became positive, but deterministic rollouts still terminated after about 24 control steps
- **Root cause:** The early checkpoint learned partial stabilization, not robust long-horizon standing
- **Solution:** Keep the run going beyond the first eval/checkpoint milestone and judge success using deterministic rollout duration and minimum torso height, not reward alone

### Issue: The `evaluate.py -> MatchRunner` path appeared stuck for video export
- **Symptom:** The evaluation command looked stalled during video generation and was manually terminated before producing an mp4
- **Root cause:** That path was a poor fit for quick debugging here; the repository already had a proven direct environment loop in `run_without_policy.py`
- **Solution:** Reuse the `run_without_policy.py` pattern: set `MUJOCO_GL=egl` before imports, instantiate `CombatGymEnv(render_mode="rgb_array")` directly, then step the trained model and save the video

### Issue: The reusable module-style video export path was still fragile under headless rendering
- **Symptom:** `python -m combatbench.baseline.sb3.export_video ...` initially aborted with GLFW / OpenGL initialization errors
- **Root cause:** the module startup path was more sensitive to EGL initialization order in this environment
- **Solution:** add `combatbench/run_policy_video.py` as the primary standalone export entrypoint and make `baseline/sb3/export_video.py` delegate to it in a clean subprocess

### Issue: Visual inspection showed immediate collapse despite successful video export
- **Symptom:** In the exported video, both robots start in a standing pose but collapse within the opening segment and remain on the floor for most of the clip
- **Root cause:** The learned standing policy improved reward and short-horizon survival, but it still does not robustly maintain upright balance
- **Solution:** Treat this run as proof that the training loop and video export work, but not as a successful standing controller; improve the control formulation or reward design before relying on this baseline
