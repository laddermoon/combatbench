# Fix Attacker Approach Semantics Summary

## The Correct Path

1. **Fix Action Composition in `SB3CombatPolicy`**
   - **File:** `baseline/sb3/policies.py`
   - **Action:** Modify `act()` method to conditionally apply `_apply_approach_base`.
   - **Logic:** Only apply the approach bias (e.g., `lean_forward`) if the current policy requests it AND the base policy does NOT already have it enabled. This prevents double application when stacking policies that use the same heuristic.

2. **Decouple Policy Initialization**
   - **File:** `baseline/sb3/policies.py`
   - **Action:** Remove recursive parameter passing in `__init__` when loading the `base_policy`.
   - **Reasoning:** Let the base policy load its own configuration from its `run_config.json`. This ensures the base policy behaves exactly as it was trained, rather than inheriting potentially incompatible settings from the wrapper policy.

3. **Ensure Reward Config Availability**
   - **File:** `baseline/sb3/rewards.py` & `baseline/sb3/__init__.py`
   - **Action:** Define and export `ATTACKER_APPROACH_REWARD_CONFIG`.
   - **Reasoning:** Prevents `ImportError` in `selfplay_env.py` and allows proper environment initialization for the `fight_attacker_approach` phase.

4. **Update Tooling for Attacker Phases**
   - **Files:** `tools/run_match.py`, `baseline/sb3/validate_env.py`
   - **Action:** Add support for `fight_attacker` and `fight_attacker_approach` phases.
   - **Benefit:** Enables direct debugging and smoke testing of attacker policies using standard CLI tools.

## Pitfalls & Solutions

### Issue: Robot Falling Immediately / Excessive Leaning
- **Symptom:** The robot falls forward immediately upon reset or behaves erratically, despite good training metrics.
- **Root Cause:** The "lean forward" bias (abdomen_y +0.6) was being applied twice: once by the base policy (Stage 1) and again by the wrapper policy (Stage 2), resulting in a massive +1.2 (clipped to 1.0) action that destabilized the robot.
- **Solution:** Implemented a check: `if self.approach_base_mode == "lean_forward" and base_policy_mode != "lean_forward": apply_bias()`.

### Issue: Recursive Configuration Overrides
- **Symptom:** Base policy behaving unlike its standalone version.
- **Root Cause:** Passing `**kwargs` or specific params like `approach_base_mode` to the base policy's constructor overrode its loaded metadata.
- **Solution:** Instantiate the base policy with minimal arguments (path, device, deterministic), allowing it to self-configure from its own metadata.

### Issue: Missing Reward Config
- **Symptom:** `ImportError` when running validation scripts.
- **Root Cause:** `selfplay_env.py` imported `ATTACKER_APPROACH_REWARD_CONFIG` but it wasn't defined in `rewards.py`.
- **Solution:** Added the missing configuration definition to `rewards.py`.
