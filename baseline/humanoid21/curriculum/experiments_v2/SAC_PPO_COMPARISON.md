# SAC vs PPO Training Comparison — basic_balance_v2

## Experiment

- **Experiment**: `basic_balance_v2` (standing balance, fall detection only)
- **Environment**: `basic_balance_v2_env.yaml` (max_steps=600, ImbalanceTerminationPlugin)
- **Framework**: `framework_v2` (unified PPO/SAC interface)
- **Hardware**: GPU 1 (RTX 4090), 96 rollout workers
- **Goal**: Train until eval survived=1.0 (all 16 eval episodes stand for 200+ steps)

## Results

| Metric | PPO | SAC |
|--------|-----|-----|
| Convergence update | 455 | 55 |
| Total wall-clock time | ~107 min | ~32 min |
| Avg time per update | 14.1 s | 35.1 s |
| Episodes per update | 2048 | 2048 |
| Gradient steps per update | 4 epochs × ~50k samples | 2048 minibatches × 256 |
| Speedup | 1× | **3.3×** |

## Key Observations

1. **SAC converges 3.3× faster** in wall-clock time despite each update being slower (35s vs 14s).
2. **SAC needs far fewer updates** (55 vs 455) — off-policy data reuse is highly efficient.
3. **PPO has cheaper updates** but needs many more of them due to on-policy data limitations.
4. **SAC required reward scaling** (reward_scale=0.1) and fixed alpha=0.1 to prevent Q-function collapse.
5. **PPO was more "plug and play"** — default hyperparameters worked without tuning.

## SAC-specific Fixes Applied

1. **Reward scaling**: Added `reward_scale=0.1` to prevent Q-value explosion (combined rewards from 6 components can be large).
2. **Fixed alpha**: Disabled auto_alpha (auto-tuning drove alpha→0, causing policy collapse). Used fixed alpha=0.1.
3. **Target entropy**: Used `-0.5 * action_dim` instead of `-action_dim` (less aggressive entropy target).
4. **Alpha clamping**: Added `log_alpha.clamp_(-5.0, 2.0)` as safety net.
5. **Gradient step count**: Changed from `updates_per_step * transitions_added` (40k+ steps) to `min(updates_per_step * episodes_per_update, 10000)` (2048 steps).

## Training Curves

### PPO
- Updates 1-100: episode length 20→55 (slow progress)
- Updates 100-300: plateau at 50-60 steps
- Updates 300-380: rapid improvement 60→130 steps
- Update 380: first survival (0.125)
- Update 455: survived=1.0

### SAC
- Updates 1-20: episode length 20→35 (slow warmup)
- Update 25-30: jump to 67 steps
- Update 40: first survival (0.062)
- Update 45: 0.562 survival rate
- Update 55: survived=1.0

## Conclusion

SAC's off-policy data reuse makes it significantly more sample-efficient for this balance task, achieving the same goal in 1/3 the wall-clock time of PPO. However, SAC required careful hyperparameter tuning (reward scale, alpha, target entropy) to avoid instability, while PPO worked with default settings.
