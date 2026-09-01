# SAC V2 Implementation Summary

## Overview

This document summarizes the SAC V2 framework implementation, validation
results, and current status.

## Implementation Status: COMPLETE (MVP)

All planned MVP components are implemented and validated:

### Framework Components (`baseline/framework/sac/`)

1. **`replay.py` — TaggedReplay buffer**
   - Trajectory-continuous storage with (traj_id, traj_step) tracking
   - Per-channel rewards, dones, actor_weights
   - n-step return computation via `sample_nstep()`
   - Tags and reward_features for stratification and relabeling
   - Thread-safe write interface (for future async rollout)
   - Circular capacity with overwrite
   - Buffer statistics for diagnostics

2. **`networks.py` — Multi-head Q critics**
   - `QTrunkHeads`: shared trunk + per-channel heads
   - `QTrunkGroup`: twin Q (Q1, Q2) + target networks per group
   - `MultiHeadQCritic`: manages all groups, auto-groups by gamma
   - Soft target updates
   - Checkpoint save/load

3. **`trainer.py` — SAC update kernel**
   - Per-channel n-step TD targets with per-channel gamma
   - Clipped double-Q (twin critics)
   - Automatic entropy temperature (alpha) tuning
   - Action-gradient normalization (primary mechanism)
   - Naive weighted Q sum (fallback)
   - `GradNormStats`: running RMS of per-channel gradient norms
   - Per-channel gradient share diagnostics

4. **`experiment.py` — ExperimentSAC interface**
   - `SACRewardChannel`: per-channel gamma, n_step, n_critics, trunk_group
   - `SACParams`: replay, batch, warmup, UTD, tau, alpha, grad_norm config
   - `CommonParamsSAC`: env_step-based clock (not update-based)
   - `DataSource`: multi-source declaration (self, opponent, pool, scripted)
   - `ReplayPlan`: stratification and freshness config
   - `TrajectorySlice`: SAC analog of PPO's Trajectory
   - `ExperimentSAC` ABC with all required methods

5. **`loop.py` — Training loop**
   - Synchronous collection
   - env_step-based clock
   - Warmup period before first gradient step
   - UTD-driven gradient step count
   - Divergence guardrails (Q magnitude, TD error, alpha collapse)
   - Per-channel diagnostics
   - Checkpoint/resume (model only, buffer re-warmups)
   - Video rendering
   - Machine-readable `__RAW_STATS__` logging

### Experiment Components (`baseline/experiments_sac/`)

6. **`base.py` — CombatExperimentSACBase**
   - Shared defaults for humanoid21 SAC experiments
   - Self-play job construction
   - Actor/Q-critic building

7. **`exp_sac_balance.py` — sac_balance experiment**
   - 2-channel validation: r_fall (survival) + r_cross (balance)
   - Per-step dones (terminated vs truncated)
   - φ²-gated actor weights for r_cross
   - Survival rate evaluation metric

### CLI Integration (`baseline/framework/train.py`)

8. **`--algo sac` dispatch**
   - SAC experiment registry (`baseline/experiments_sac/`)
   - SAC-specific smoke test parameters
   - SAC-specific config serialization
   - PPO-only flags (confidence) correctly gated

### Tests (`baseline/framework/sac/tests/`)

9. **`test_replay.py` — 9 tests, all passing**
   - Basic insertion and sampling
   - next_obs correctness (obs[t+1] vs last_obs)
   - Per-channel done semantics
   - Truncated (no done) vs terminated
   - n-step sampling
   - n-step done truncation
   - Circular overwrite
   - Buffer stats
   - Trajectory tracking (traj_id, traj_step)

10. **`test_trainer.py` — 7 tests, all passing**
    - QTrunkHeads forward pass
    - Multi-head Q critic grouping
    - Auto-grouping by gamma
    - sac_update_v2 runs without error
    - GradNormStats running statistics
    - sac_update_v2 with gradient normalization
    - Soft target update

## Validation Results

### Smoke Test (PASSED)
- 10K env_steps, 9.3K gradient steps, 59 rounds
- Alpha: 0.2 → 0.014 (auto-tuning working)
- Q losses: 3.0 → 0.01 (converging)
- Gradient shares: r_fall ~95%, r_cross ~5% (stable)
- Episode lengths: 25 → 29 (slight increase)
- No divergence detected
- Eval and video rendering worked

### Real Training (VALIDATED, early progress)
- Run: `sac_balance_real_v1`
- 7 rounds, ~5K env_steps, ~5K gradient steps in ~20 minutes
- Alpha: 0.2 → 0.056 (decreasing as expected)
- Q values: 0.2 → 5.7 (policy learning to accumulate reward)
- Actor loss: -2.9 → -20.0 (policy improving)
- Episode lengths: variable (14-27), early exploration phase
- No divergence detected
- Training was stopped manually after validating learning signal

## Key Design Decisions (see DECISIONS.md for details)

1. **N1 — Memory**: In-memory only, 500K default capacity, no disk persistence
2. **N2 — Relabel**: Full batch relabel with version tagging
3. **N3 — Q architecture**: Trunk groups by gamma, twin Q per group
4. **N4 — Async**: Synchronous MVP, thread-safe interface for future
5. **N5 — log_std**: Wide range (-10, 2), alpha controls exploration
6. **N6 — Sampling**: Shared batch, per-channel actor_weight masking
7. **N7 — Grad norm**: Primary actor loss mechanism, with fallback

## What's NOT Implemented (Phase 2+)

- Async rollout collection
- Per-channel sampling
- Buffer-based env reset
- Stratified retention
- Opponent-pool self-play data ingestion
- Multiple data sources (MVP uses single source)
- DroQ (Dropout + LayerNorm for Q networks)
- Disk persistence for replay buffer

## Files Created/Modified

### New files:
- `baseline/framework/sac/__init__.py`
- `baseline/framework/sac/DECISIONS.md`
- `baseline/framework/sac/experiment.py`
- `baseline/framework/sac/replay.py`
- `baseline/framework/sac/networks.py`
- `baseline/framework/sac/trainer.py`
- `baseline/framework/sac/loop.py`
- `baseline/framework/sac/tests/__init__.py`
- `baseline/framework/sac/tests/test_replay.py`
- `baseline/framework/sac/tests/test_trainer.py`
- `baseline/experiments_sac/__init__.py`
- `baseline/experiments_sac/base.py`
- `baseline/experiments_sac/exp_sac_balance.py`

### Modified files:
- `baseline/framework/train.py` (added `--algo sac` dispatch)

### Not committed (per user instruction):
All changes remain in the working tree for review.
