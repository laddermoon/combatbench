# Design: ② Low-Rank Covariance Gaussian

Reads `DESIGN_OVERVIEW.md` as a prerequisite. This is Stage 2.

New file: `baseline/common/policies/low_rank_gaussian_mlp.py`, class
`LowRankGaussianMLPPolicy(TanhSquashedPolicyBase)`.

## 1. What this family buys, and what it does not

Baseline and ① model the action distribution as a diagonal Gaussian in
raw space — every action dimension is independent given the state. This
family adds a rank-`k` correction to that covariance, so the *raw*
distribution can express pairwise correlations between action
dimensions. Concrete example: in a humanoid, hip and knee on the same
leg should generally flex together; a diagonal Gaussian cannot represent
"hip flex ⇒ likely knee flex", this family can.

**What it does *not* buy:** correlation in *action* space (post-tanh).
Tanh is applied elementwise after sampling and distorts the covariance
structure (it is not a linear map). The correlation we model is
correlation in joint-angle space *before* the squashing function maps
joint angles to the bounded action interval. This is the cheap option
from the original comparison table; modeling action-space correlation
directly is a materially larger undertaking and is **out of scope** for
this round. The per-family doc must say this explicitly so nobody later
misreads a learned raw-space correlation as an action-space correlation.

## 2. Parameterization

Covariance is `Σ = diag(σ²) + U Uᵀ`, with `U ∈ ℝ^{action_dim × k}`. This
is positive-definite for any `σ > 0` and any `U` — no PSD projection
needed, no Cholesky-of-covariance required, no eigendecomposition. This
is the whole reason to use the low-rank form instead of a full
`LLᵀ` Cholesky parameterization (which would need `action_dim²/2`
parameters and a careful parameterization to stay PD).

Outputs from the head, all functions of state:

| Output | Shape | Role |
|---|---|---|
| `mean` | `(B, action_dim)` | raw mean |
| `raw_log_std` | `(B, action_dim)` | per-dim diagonal log-std, bounded per ① §3 |
| `U_flat` | `(B, action_dim * k)` | low-rank factor, reshaped to `(B, action_dim, k)` |

Default `k = 4` for `action_dim = 21` (measured: 96,042 → 123,006
params, +27k — see `DESIGN_OVERVIEW.md` table). `k` is a constructor
kwarg, not a constant; it must travel in the export payload (§6).

`U` is **not** bounded. Bounding it would be a meaningful constraint on
expressible correlation strength, and there is no a-priori reason to
impose one. Instead, monitor its magnitude (§5) and rely on PPO's
trust-region mechanics (`target_kl`, `clip_eps`) to constrain per-update
movement. If `U` blows up in practice, the right response is a
weight-decay term or a norm penalty on `U`, not a hard clamp — clamping
would silently cap correlation strength and the policy would not be able
to tell that it's being capped.

## 3. Distribution: use `LowRankMultivariateNormal`, do not roll our own

`torch.distributions.LowRankMultivariateNormal(loc, cov_diag, cov_factor)`
exists in the installed torch (2.7.1) and implements exactly `Σ =
diag + U Uᵀ`. Use it. Specifically:

```python
dist = LowRankMultivariateNormal(
    loc=mean,
    cov_diag=effective_std.pow(2) + 1e-6,   # ε for PD margin
    cov_factor=U,                            # (B, action_dim, k)
)
raw_action = dist.rsample()                  # (B, action_dim)
raw_log_prob = dist.log_prob(raw_action)     # (B,)
```

Why not hand-roll `log_prob`:
- The closed form
  `log_prob = -0.5 * [k*log(2π) + logdet(Σ) + (x-μ)ᵀ Σ⁻¹ (x-μ)]`
  involves a Woodbury identity for `Σ⁻¹` and a Sylvester determinant
  `logdet(Σ) = logdet(diag) + logdet(I + Uᵀ diag⁻¹ U)`. Both are
  numerically delicate (the `diag⁻¹` term blows up if any σ → 0; the
  `I + Uᵀ diag⁻¹ U` can be ill-conditioned if `U` is large). The
  library implementation handles these with the standard
  `logsumexp`/Cholesky-based stable forms. Reimplementing invites the
  exact class of subtle numerical bug that the §5 normalization test is
  designed to catch — but a passing test only proves the bug isn't
  *triggered on the test inputs*, not that it doesn't exist. Trust the
  library; spend the effort on tests and diagnostics instead.

The `+ 1e-6` on `cov_diag` is a PD margin — it is **not** a replacement
for bounding σ. It is the same role as baseline's `1 - tanh² + 1e-6`
Jacobian epsilon: a guard against the exact-zero boundary, not a
substantive constraint on the distribution.

## 4. Sampling and `evaluate_actions`

`raw_sample` and `raw_log_prob` are both delegated to the
`LowRankMultivariateNormal` instance (§3). The base class applies tanh
and the Jacobian correction exactly as in ① — no family-specific change
to the tanh math. The fact that the raw distribution is no longer
diagonal does not affect the Jacobian term, because tanh is applied
elementwise and its Jacobian is diagonal.

`raw_mode` is just `mean` (the mode of a multivariate Gaussian is its
mean). Deterministic action = `tanh(mean)`.

Entropy: `LowRankMultivariateNormal.entropy()` exists and is closed-form
(it uses the same Sylvester determinant identity as `log_prob`).
Regularizer is `-entropy_coef * entropy.mean()`, same form as ① and
baseline. Do not approximate.

## 5. Temperature

The mathematically consistent choice is to scale the **whole covariance**
by `T²`, which means scaling **both** `σ` and `U` by `T`:

```python
# In set_exploration, store self._temperature (default 1.0).
# In _raw_sample / _raw_log_prob, build the distribution with:
effective_std = bounded_log_std.exp() * self._temperature
U_effective   = U * self._temperature
```

Why both, not just σ: temperature is supposed to scale "how random the
action is" — i.e. the entire spread of the distribution. If only σ is
scaled, then at high temperature the diagonal noise dominates and the
learned correlation structure is washed out (the distribution becomes
effectively diagonal); at low temperature the correlation structure
dominates but the diagonal exploration vanishes. That is not "more
random" vs "less random", it's "different shape" — which is not what
temperature is for. Scaling both preserves the *shape* of the
distribution and scales only its *spread*.

This is the choice the per-family doc is required to make explicit
(generic overview §"Detailed low-rank Gaussian design must cover"). The
alternative (scale σ only, or scale `U` only) is rejected for the reason
above; if a future experiment wants to manipulate correlation strength
specifically, that should be a separate exploration knob, not an
overloading of `temperature`.

## 6. Export

Constructor kwargs for round-trip: `obs_dim, action_dim, hidden_dim, k,
log_std_min, log_std_max`. The new one vs ① is `k` — and `k` is the
single most important field to get right, because a wrong `k` means a
wrong `U_flat` reshape on load, which means a silently wrong covariance
shape. The `strict=True` reload will catch a `k` mismatch as a shape
error on the head's weight matrix (the head's output dim is
`action_dim + action_dim + action_dim*k`, which depends on `k`), which
is exactly the loud failure we want.

`log_std_offset` (temperature) travels in the payload same as ①.

## 7. Stats

Add to `ActorEval.stats`:
- `std_mean_batch`, `std_min_batch`, `std_max_batch` — same as ①, from
  the diagonal part only. These are *not* the marginal standard
  deviations of the full distribution (which would be
  `sqrt(σ² + ||U_row||²)`); report those separately as
  `marginal_std_mean_batch` etc. if needed, but the diagonal-only stats
  are the ones comparable to ① and baseline.
- `U_frob_mean_batch` — `mean(||U||_F)` over the batch. Primary monitor
  for "is the low-rank part doing anything" and "is it blowing up".
- `U_frob_max_batch` — max over batch; pair with the mean to detect
  outlier states.
- `cov_trace_mean_batch` — `mean(trace(Σ)) = mean(sum(σ²) + ||U||_F²)`.
  This is the total variance of the raw distribution, the direct
  analogue of baseline's `sum(exp(log_std)²)` and the right number to
  compare across families when asking "how exploratory is the policy".
- `entropy` — closed-form, from the library.
- `std_squash_sat_frac` — same as ①, on the diagonal log-std only.

Do **not** report a single "correlation coefficient" — correlation is a
per-pair quantity and a single summary number would be misleading. If
pairwise correlation diagnostics are wanted later, add a
`top_corr_abs_mean_batch` (mean of the top-N |off-diagonal entries of the
normalized correlation matrix) as a scalar summary, but that's a
follow-up, not a Stage 2 requirement.

## 8. Risks specific to this family

- **`LowRankMultivariateNormal` shape pitfalls.** `cov_factor` must be
  `(B, action_dim, k)`, not `(B, k, action_dim)` — the API is not
  symmetric in the two factor dimensions despite the math being
  `U Uᵀ`. A transpose here passes the shape check, produces a
  different (wrong) covariance, and trains "fine" — caught only by the
  normalization test (§5 test 2 of overview) and the degenerate
  equivalence test (§9 below).
- **`U` drift to large values.** No clamp by design (§2). Mitigation is
  monitoring `U_frob_*` and adding a weight-decay / norm penalty *only
  if* the monitor shows runaway growth. Do not pre-emptively add the
  penalty — it would bias the learned correlation.
- **Ill-conditioning at small σ.** If `bounded_log_std` saturates against
  `log_std_min` (≈ exp(-4) ≈ 0.018) for some dims while `U` is
  non-trivial, `I + Uᵀ diag⁻¹ U` can have very large entries. The
  library handles this in `log_prob` via Cholesky, but `entropy` and
  `rsample` paths should be smoke-tested at the extreme. The `+1e-6` PD
  margin (§3) is the guard; do not increase it without re-running the
  normalization test, since a larger margin subtly changes the
  distribution.
- **Confusing raw-space and action-space correlation.** Already
  addressed in §1; restated here because it is the single most likely
  misinterpretation when reading the learned `U`.

## 9. Acceptance checklist for Stage 2

- [ ] Degenerate equivalence: set the head's `U_flat` weights to zero at
      init (and hold `k` whatever it is — `U=0` is well-defined for any
      `k`), `temperature=1` ⇒ `log_prob` matches ① (not baseline — ① is
      the closer reference since the σ-head is also state-dependent) to
      1e-6 on identical inputs. This isolates the low-rank math from the
      state-dependent-σ math.
- [ ] Normalization test passes on `action_dim=2, k=1`.
- [ ] Sample/score self-consistency passes.
- [ ] Gradient completeness: head's `U_flat` weights receive non-zero
      grad after `loss.backward()` on a batch with non-degenerate `U`
      (i.e. not the degenerate-equivalence config — that one *should*
      have zero grad on `U` by construction).
- [ ] Export round-trip with `strict=True`, including a deliberate
      `k`-mismatch case that is *expected* to raise (negative test —
      confirms the loud failure mode works).
- [ ] `act()` latency within 10× budget. `LowRankMultivariateNormal` is
      heavier than `Normal` but still O(action_dim · k) per step;
      expect well under 1ms.
- [ ] One full `basic_balance` regression run; `U_frob_mean_batch` and
      `cov_trace_mean_batch` logged and watched; no runaway growth.
