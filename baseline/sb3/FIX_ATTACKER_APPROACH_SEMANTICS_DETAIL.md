# Fix Attacker Approach Semantics Detail Log

## 2026-03-20 13:00 Issue Identification

**Why:** The robot was observed to fall immediately or behave erratically during evaluation, despite training metrics looking okay. Suspected inconsistency between training and inference environments regarding the "lean_forward" heuristic.

**Observation:**
- In training (`selfplay_env.py`), the base policy's action is computed and then scaled. The environment wrapper *does not* re-apply the approach heuristic.
- In inference (`policies.py`), `SB3CombatPolicy.act` was applying `_apply_approach_base` *on top of* the base policy's action, effectively doubling the "lean_forward" bias if the base policy also had it enabled.

**Verification:**
Created a reproduction script checking `abdomen_y` action.
- Before fix: `0.9477` (approx `0.6` bias + `0.6` bias scaled? or just double application).
- Expected: Around `0.35` (residual `0.0` + base `0.6` * compensation `0.33` + bias `0.0`? No wait).
- Base policy (Stand) output: `~0.0`.
- Stage 1 (Approach) output: `~0.6` (bias).
- Stage 2 (Attacker) wraps Stage 1.
    - Stage 1 output: `0.6`.
    - Stage 2 compensation for abdomen_y: `0.33`.
    - Scaled Stage 1: `0.2`.
    - Stage 2 then adds bias `0.6`.
    - Total: `0.8` + residual.
    - This confirms the double application.

## 2026-03-20 13:45 Fix Implementation

**Command:**
Modified `combatbench/baseline/sb3/policies.py`.

**Change:**
Updated `SB3CombatPolicy.act` to check if `base_policy` already has the same `approach_base_mode`.
```python
            base_policy_mode = getattr(self.base_policy, "approach_base_mode", None)
            if self.approach_base_mode == "lean_forward" and base_policy_mode != "lean_forward":
                base_action = self._apply_approach_base(raw_obs, base_action, info)
```

**Refinement:**
Also removed recursive parameter passing in `__init__` to ensure base policies load their own config from `run_config.json` rather than inheriting potentially incorrect overrides from the wrapper.

## 2026-03-20 13:58 Import Error Resolution

**Error:** `ImportError: cannot import name 'ATTACKER_APPROACH_REWARD_CONFIG' from 'combatbench.baseline.sb3.rewards'`
**Cause:** `selfplay_env.py` imported this config, but it was missing from `rewards.py`.
**Fix:**
1. Added `ATTACKER_APPROACH_REWARD_CONFIG` to `baseline/sb3/rewards.py`.
2. Updated `baseline/sb3/__init__.py` to export it.

## 2026-03-20 14:20 Tooling Updates

**Action:** Updated `tools/run_match.py` and `baseline/sb3/validate_env.py`.
**Why:** To support `fight_attacker` and `fight_attacker_approach` phases in the CLI tools for easier debugging and validation.

**Changes:**
- `run_match.py`: Added CLI arguments for `--model`, `--model-b`, `--device`. Added phase handling for attacker phases.
- `validate_env.py`: Added smoke test for `fight_attacker_approach`.

## 2026-03-20 14:30 Verification

**Command:**
```bash
python3 -m combatbench.tools.run_match \
    --model /data1/mono/things/combatbench/combatbench/baseline/sb3/runs/fight_attacker_curriculum_v4_long/best_model/best_model.zip \
    --phase fight_attacker \
    --duration 2.0 \
    --device cpu
```

**Result:**
Match ran successfully without falling immediately.
Abdomen Y action verified as `~0.35` in script, confirming correct composition.

**Command:**
```bash
python3 -m combatbench.baseline.sb3.validate_env
```

**Result:**
All phases (stand, fight, fight_attacker, fight_attacker_approach) passed smoke tests.
