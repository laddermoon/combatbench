# SAC V2 Implementation Decision Log

Chronological record of design decisions made during implementation.

---

## [2026-08-27] N1 — Memory budget & buffer storage structure

**Decision:** In-memory only, no disk persistence for replay buffer. Default capacity 500K transitions (configurable per experiment). On resume, buffer re-warmups from scratch — model weights are checkpointed, buffer is not.

**Rationale:**
- Machine has 1TB RAM. A 500K-transition buffer for `fight` (9 channels, obs=96, act=21) costs ~1GB. Trivial.
- Disk persistence of a 1GB+ buffer adds IO complexity and checkpoint bloat for marginal benefit (warmup is ~10K transitions, a few seconds of rollout).
- Thread-safe write interface designed from the start (for future async), but Phase 1 is synchronous.

**Per-transition storage layout:**
- `obs`: (obs_dim,) float32
- `action`: (action_dim,) float32
- `next_obs`: (obs_dim,) float32 — stored explicitly (not computed from trajectory) for O(1) sampling
- `done`: (n_channels,) bool — per-channel termination flag
- `reward`: (n_channels,) float32 — per-channel reward at this step
- `actor_weight`: (n_channels,) float32 — per-channel actor weight at this step
- `tags`: Dict[str, float32] — per-transition tags (phase, source, etc.)
- `reward_features`: Dict[str, float32] — raw features for relabeling (optional)
- `traj_id`: int32 — which trajectory this transition belongs to (for n-step continuity)
- `traj_step`: int32 — position within trajectory (for n-step continuity)

**Trajectory-segment storage:** Buffer stores transitions flat but tracks trajectory boundaries via (traj_id, traj_step) pairs. This enables n-step return computation without storing full trajectory arrays.

---

## [2026-08-27] N2 — Relabel strategy

**Decision:** Full batch relabel with version tagging. When `experiment.relabel()` is called (e.g. on curriculum advance), scan entire buffer, recompute rewards and actor_weights from stored `reward_features`. Tag each transition with a `relabel_version` so we can detect stale data.

**Rationale:**
- Full scan of 500K transitions takes <1 second (pure numpy).
- Simpler than lazy relabel — no per-sample computation overhead during training.
- Requires `reward_features` to be stored, which costs ~200 bytes/transition extra. Acceptable.
- If an experiment doesn't use relabeling, `reward_features` can be empty — zero overhead.

**Interface:** `experiment.relabel(features, tags, ctx) -> (rewards, actor_weights)` is optional. Default: no relabeling (returns None, buffer keeps original values).

---

## [2026-08-27] N3 — Q network trunk grouping

**Decision:** Experiments declare trunk groups explicitly via `SACRewardChannel.trunk_group`. Channels with the same `trunk_group` share a trunk network with per-channel heads. Default: auto-group by `gamma` (channels with same gamma share a trunk). Single-channel exclusive trunk allowed by setting a unique group name.

**Rationale:**
- Auto-grouping by gamma is semantically correct: gamma determines the effective time horizon, and channels with similar horizons benefit from shared representations.
- Explicit override allows semantic grouping (e.g. "all damage-related channels share a trunk").
- Escape hatch: a critical channel like `r_fall` can get its own trunk.

**Architecture per group:**
```
Trunk: Linear(obs+act, hidden) → ReLU → Linear(hidden, hidden) → ReLU
Head_c: Linear(hidden, 1)  # one per channel in the group
```
Twin Q: each group has two independent trunk+heads (Q1 and Q2 for clipped double-Q).
Target networks: deep copy of each Q, soft-updated.

**For the MVP (sac_balance, 2 channels):** Both channels share gamma=0.99, so they share one trunk with 2 heads. Total networks: 2 Q (trunk+2heads) + 2 target = 4 networks. Very lightweight.

---

## [2026-08-27] N4 — Async collection

**Decision:** Phase 1 is synchronous. `TaggedReplay` write interface is designed to be thread-safe (using a lock), but no async rollout in the initial implementation. Measure synchronous rollout/train ratio first, then decide.

**Rationale:**
- Async adds significant complexity (concurrent buffer writes, policy version tracking, staleness observability).
- Need baseline measurements before justifying the complexity.
- SAC's UTD ratio means training time dominates rollout time in many configs, so async's benefit may be smaller than expected.

---

## [2026-08-27] N5 — log_std_min vs auto-alpha

**Decision:** SAC experiments use a wide log_std range (-10, 2) by default. Exploration is controlled entirely by alpha. The `entropy_coef` field from `ExplorationSpec` is not used by SAC — alpha replaces it. `target_entropy` defaults to `-action_dim` (-21) but is configurable per experiment and can be scheduled.

**Rationale:**
- In SAC, alpha IS the exploration controller. Hard-clamping log_std fights alpha's调节.
- The existing PPO experiments' tight log_std bounds (-1.8 to -2.5) are PPO-specific hacks that don't transfer.

---

## [2026-08-27] N7 — Action gradient normalization

**Decision:** Implement as the primary actor loss mechanism, with a fallback to naive weighted Q sum. The fallback is controlled by a `SACParams` flag (`use_grad_norm=True/False`). Validation against the fallback happens in the `sac_balance` experiment.

**Implementation:**
- Every K steps (K=10 default), estimate `ŝ_c = running_RMS(||∂Q_c/∂a||)` on a subsample of the batch using `torch.autograd.grad`.
- Actor loss: `α·logπ - Σ_c w_c(s) · Q_c(s,a) / ŝ_c`
- Normalize `Σ_c w_c` to 1.0 so the effective Q scale is constant.
- Log per-channel gradient share as diagnostic.

**Fallback:** `use_grad_norm=False` → actor loss = `α·logπ - Σ_c w_c · Q_c` (naive weighted sum, matching V1 SAC).

---

## [2026-08-27] N6 — Per-channel sampling vs multi-head Q

**Decision for MVP:** Use shared batch sampling (all channels see the same batch). Per-channel sampling is a Phase 2 feature. The multi-head Q architecture still provides per-channel Q values from a single forward pass.

**Rationale:**
- Per-channel sampling with multi-head Q is architecturally conflicting (different batches can't share a trunk forward pass).
- For `sac_balance` (2 channels, both dense), per-channel sampling provides no benefit — both channels are active everywhere.
- For `sac_fight` (sparse damage channels), per-channel sampling matters more, but that's Phase 2.
- Shared batch + per-channel actor_weight masking (aw=0 frames don't contribute to that channel's Q loss) is the MVP approach.

---

## [2026-08-27] Architecture decision — package structure

```
baseline/framework/sac/
├── __init__.py
├── PLAN.md           (planning document)
├── DECISIONS.md      (this file)
├── replay.py         (TaggedReplay buffer)
├── networks.py       (MultiHeadQCritic, trunk+heads architecture)
├── trainer.py        (sac_update: per-channel n-step TD, auto-alpha, grad norm)
├── experiment.py     (ExperimentSAC ABC, SACParams, SACRewardChannel, data types)
├── loop.py           (train_sac: synchronous loop, env_step clock, diagnostics)
└── tests/
    ├── test_replay.py
    └── test_trainer.py
```

SAC experiments live in `baseline/experiments_sac/` (separate from V2 PPO experiments):
```
baseline/experiments_sac/
├── __init__.py       (registry, auto-discovery)
├── base.py           (CombatExperimentSACBase — shared combat defaults)
└── exp_sac_balance.py
```

---

## [2026-08-27] Implementation scope for first iteration

**In scope (MVP):**
1. TaggedReplay with trajectory-continuous storage, n-step targets, per-channel done
2. Multi-head Q critic (trunk + heads, twin Q, soft target update)
3. sac_update with per-channel n-step TD, clipped double-Q, auto-alpha
4. Action gradient normalization (with fallback)
5. ExperimentSAC interface with data_sources, build_slices, relabel, replay_plan
6. Synchronous training loop with env_step clock, diagnostics, divergence guardrails
7. train.py --algo sac dispatch
8. sac_balance experiment (2 channels, basic_balance env)
9. Unit tests for replay flattening, n-step, per-channel done
10. Smoke test + real training run

**Out of scope (Phase 2+):**
- Async rollout collection
- Per-channel sampling
- Buffer-based env reset
- Stratified retention (MVP uses uniform sampling with optional tag filtering)
- Opponent-pool self-play data ingestion
- Multiple data sources (MVP uses single source: learner rollout)
- DroQ (Dropout + LayerNorm for Q networks)

**Simplifications for MVP:**
- `data_sources()` returns a single `SelfRollout` source (learner's own rollout)
- `replay_plan()` returns uniform sampling (no stratification)
- `relabel()` not used (no curriculum in sac_balance)
- `tags` stored but not used for sampling in MVP
- `reward_features` stored but not used in MVP
- `core_state` not stored in MVP

---

## [2026-08-27] N8 — Training stability: alpha collapse & reward scale

**Context:** First real training runs of `sac_balance` revealed two critical
stability issues that required parameter tuning.

**Issue 1: Alpha collapse (v2 run)**
- With `target_entropy=-21` (= -action_dim) and `alpha_lr=3e-4`, alpha
  collapsed from 0.2 to 0.003 in <20 rounds (2000 grad steps/round).
- This caused policy collapse: episode lengths crashed from 36 to 8.
- Q values went from +3.7 to -5.6 in a death spiral.

**Fix:**
- `target_entropy`: -21 → -10 (less aggressive, allows earlier exploitation)
- `alpha_lr`: 3e-4 → 1e-4 (3x slower alpha convergence)
- `log_alpha_min`: -10 → -5 (alpha floor ≈ 0.007, prevents total collapse)
- `q_layer_norm`: False → True (stabilizes Q estimates)

**Issue 2: Q overestimation divergence (v6 run)**
- With `reward_scale=200`, the policy learned successfully (survived=14
  at 1.1M env steps, 27x more sample-efficient than PPO's 27M env steps).
- But Q losses grew from 200 to 1138, causing divergence at 1.55M env steps.
- The high reward scale made TD errors too large for stable Q learning.

**Fix:**
- `reward_scale`: 200 → 50 (4x reduction in TD error magnitude)
- `critic_learning_rate`: 3e-4 → 1e-4 (3x slower Q learning for stability)

**Key insight:** SAC with 1-step TD needs reward scaling for small per-step
rewards (~0.005), but the scale must be balanced against Q stability.
PPO doesn't need this because GAE naturally amplifies credit assignment.

---

## [2026-08-27] N9 — Training scale: matching PPO's env step budget

**Context:** PPO `basic_balance` requires ~27M env steps to first reach
survival_rate=1.0 (update 295, 1024 episodes/update, 96 workers).

**Decision:** SAC `sac_balance` configured with:
- `max_env_steps`: 10M (SAC should be more sample-efficient than PPO)
- `episodes_per_update`: 256 (PPO uses 1024, but SAC reuses data)
- `rollout_workers`: 96 (match PPO's parallelism)
- `utd_ratio`: 0.25 (1 grad step per 4 new transitions)
- `max_grad_steps_per_round`: 2000 (caps round time to ~52s)
- `replay_buffer_size`: 1M (allows long-term data reuse)
- `eval_interval`: 100K env steps

**Rationale:** SAC's off-policy data reuse should need fewer env steps
than PPO. The UTD ratio of 0.25 with a 1M buffer provides effective
data reuse of ~26x per transition (buffer_size / batch_size × rounds_in_buffer).
The grad step cap keeps wall-clock time reasonable (~52s/round).
