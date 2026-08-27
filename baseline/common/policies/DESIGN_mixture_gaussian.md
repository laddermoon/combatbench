# Design: ③ Mixture of Diagonal Gaussians (MoG)

Reads `DESIGN_OVERVIEW.md` as a prerequisite. This is Stage 3.

New file: `baseline/common/policies/mog_tanh_mlp.py`, class
`MoGTanhMLPPolicy(TanhSquashedPolicyBase)`.

## 1. What this family buys, and what it does not

Baseline, ①, and ② are all *unimodal* — for any state, the raw
distribution has exactly one mode at `mean`. This family is the first
that can represent **multiple modes** for a single state: "step left" and
"step right" can both be high-density regions of the action distribution
simultaneously, with a low-density region in between. This is the
motivating use case for the capability harness (`basic_balance_step` or
`follow_v2`) — a unimodal policy has to compromise between two good
options and put its mass in the (bad) middle; a mixture can put mass on
both good options directly.

**What it does *not* buy:** correlation between action dimensions within
a component. Each component is a *diagonal* Gaussian, like ①. If
intra-component correlation matters, that's ②'s job, not this family's.
Combining the two (mixture of low-rank Gaussians) is possible but is
**not in scope** for this round — it would multiply parameter count and
implementation risk without a clear task-motivated reason yet.

## 2. Parameterization

Per-state outputs from the head:

| Output | Shape | Role |
|---|---|---|
| `mixture_logits` | `(B, K)` | unnormalized component weights |
| `means` | `(B, K, action_dim)` | per-component raw means |
| `raw_log_stds` | `(B, K, action_dim)` | per-component diagonal log-stds, bounded per ① §3 |

Default `K = 3` for `action_dim = 21` (measured: 96,042 → 123,777 params,
+27.7k). `K = 5` is a constructor option (→ 145,879 params, +49.8k).
`K` is a constructor kwarg and **must** travel in the export payload
(§7) — a wrong `K` on reload means a wrong head-output-dim, which means
a wrong reshape, which means silently scrambled means/log_stds/logits.

`K` is chosen at construction time and is **fixed for the life of the
policy**. Adaptive-`K` (growing/shrinking components during training) is
out of scope; it's a research project in itself, not a policy family.

## 3. Distribution: pre-tanh mixture of diagonal Normals

Raw distribution (before tanh):

```
p_raw(a | s) = Σ_k π_k(s) · N(a | μ_k(s), diag(σ_k(s)²))
where  π_k = softmax(mixture_logits)[k]
```

Sampling (reparameterized):
```python
# Gumbel-max sample one component index per batch element, then rsample
# from that component. rsample gives gradients through μ_k and σ_k;
# the component selection itself is non-differentiable, which is correct
# — mixture weights are learned via the log_prob path (§4), not via the
# sampling path.
idx = gumbel_softmax_sample(mixture_logits)        # (B,), hard index
comp_mean   = means.gather(1, idx.view(-1,1,1).expand(-1,1,action_dim)).squeeze(1)
comp_std    = stds.gather(...)                      # same for std
raw_action  = comp_mean + comp_std * torch.randn_like(comp_mean)
```

Do **not** use the "sample all K components, weight by π" reparameterization
trick — it produces a sample that is *not* distributed as the mixture
(it's a per-component sample averaged by weights, which is a different
distribution) and breaks the sample/score consistency test (overview §5
test 3) immediately. The Gumbel-max path is the standard correct one.

## 4. `log_prob`: the order of operations matters

This is the single most error-prone part of this family. The correct
ordering:

1. Compute per-component raw log-prob: `comp_lp_k = N(a | μ_k, σ_k²).log_prob(a).sum(-1)`
   → shape `(B, K)`.
2. Add log mixture weights: `weighted = comp_lp_k + log_softmax(mixture_logits)` → `(B, K)`.
3. **Mixture log-prob in raw space**: `raw_log_prob = logsumexp(weighted, dim=-1)` → `(B,)`.
4. **Tanh Jacobian correction** (base class does this): `log_prob = raw_log_prob - log(1 - tanh(a)² + ε).sum(-1)`.

The Jacobian term is **added after** the `logsumexp`, not per-component.
Reason: the Jacobian term depends only on the (post-tanh) action being
scored, not on which component generated it. Adding it per-component
before `logsumexp` would be mathematically equivalent (it factors out of
the `logsumexp`), but in floating point it can shift the per-component
log-probs by different amounts if the components have very different
scales, changing which component dominates the `max` inside `logsumexp`
and thus changing the result. Adding it once, after, is both clearer and
more numerically consistent.

`log_prob` is differentiable through `means`, `raw_log_stds` (via the
bounded-σ squash), and `mixture_logits` (via `log_softmax`). The
gradient-completeness test (overview §5 test 4) must verify all three
heads receive gradient — a common bug is to accidentally `detach()` the
mixture logits "for stability", which silently freezes the component
weights.

## 5. Entropy: no closed form, use sampled estimate

The entropy of a Gaussian mixture has **no closed form** (it involves
`∫ p log p` over a sum of Gaussians, which has no analytic solution).
The tanh-squashed mixture entropy is even less tractable. Three options
were considered:

1. **Sampled entropy**: `H ≈ -mean(log_prob(rsample()))` over a batch.
   Differentiable through `log_prob` (which is differentiable through
   all heads per §4). This is the **chosen** option.
2. **Raw-space closed-form lower bound** (the `-Σ π_k log(π_k) + Σ π_k H_k`
   form, where `H_k` is the k-th component's diagonal-Gaussian entropy):
   this is the entropy of the mixture *only if the components don't
   overlap*, which they generally do, so it's an **over**-estimate and
   the bias is state-dependent. Rejected — a biased entropy regularizer
   biases the policy toward over-separated components, which is the
   opposite of what we want.
3. **No entropy regularizer**: rely on PPO's KL trust region alone.
   Rejected for now — the entropy term is what drives exploration, and
   dropping it is a bigger behavioral change than this round intends.

Chosen implementation:
```python
# In _regularizer_and_stats, with raw_action already rsampled:
sampled_log_prob = self._raw_log_prob(obs, raw_action)   # (B,), differentiable
entropy_estimate = -sampled_log_prob.mean()
regularizer = -self._entropy_coef * entropy_estimate     # negative because
                                                         # framework ADDS regularizer
                                                         # to actor loss, and we want
                                                         # to MAXIMIZE entropy
```

**Score-function vs pathwise:** this is a score-function (REINFORCE-style)
estimate of entropy — the gradient flows through `log_prob`'s dependence
on the parameters, not through the sample's dependence on the parameters
(the sample is treated as a fixed point for gradient purposes, even
though it was generated by `rsample`). This is the standard approach and
has higher variance than a pathwise estimate, but a pathwise estimate of
mixture entropy doesn't exist in a clean form. Variance is controlled by
the batch size (B ≈ 4000+ per update in this framework), which is large
enough that the estimate is usable. Watch the `entropy` stat's
per-update variance in the first training run; if it's noisy enough to
destabilize training, fall back to option 3 (no regularizer) for this
family and document the decision.

**Do not detach `raw_action`** in the `log_prob` call inside the
regularizer. The sample's *value* is treated as fixed (it's the point at
which we evaluate the density), but `log_prob`'s dependence on the
*parameters* (which determine the density at that point) must remain
differentiable. Detaching `raw_action` is fine (the sample locations are
not differentiated through); detaching anything inside `_raw_log_prob`
is not.

## 6. Temperature

Scale **component standard deviations** by `T`, do **not** touch
mixture logits:

```python
effective_stds = bounded_log_stds.exp() * self._temperature   # (B, K, action_dim)
# mixture_logits unchanged
```

Reason: temperature is "how random is the continuous noise", not "how
indifferent am I between components". Mixing those two semantics would
mean a single knob simultaneously makes the policy (a) more noisy within
each mode and (b) more uniform across modes — those are different
exploration behaviors and should not be coupled. If a future experiment
wants to control component-weight entropy specifically, that should be a
separate exploration knob (e.g. `mixture_temperature`), not an
overloading of `temperature`.

This decision is the one the overview's "Detailed MoG design must cover"
section requires us to make explicit. The rejected alternative (scale
both σ and logits by temperature-derived quantities) is rejected for the
reason above.

## 7. Export

Constructor kwargs for round-trip: `obs_dim, action_dim, hidden_dim, K,
log_std_min, log_std_max`. `K` is the critical field (§2). The
`strict=True` reload will catch a `K` mismatch as a shape error on the
head's weight matrix (head output dim = `K + K*action_dim + K*action_dim`,
which is linear in `K`), which is the loud failure we want.

`log_std_offset` (temperature) travels in the payload same as ① and ②.

## 8. Stats

Add to `ActorEval.stats`:
- `entropy` — the sampled estimate from §5. Comparable in *purpose*
  across families, but note it's a stochastic estimate here (vs a
  closed-form value for ①② and baseline) — its per-update variance will
  be higher. Watch this.
- `mixture_weight_entropy` — `mean(-Σ π_k log π_k)` over the batch, the
  entropy of the categorical over components. This is a closed-form
  diagnostic (not the regularizer), and is the right number to watch for
  component collapse (§9). Range: `[0, log K]`; `log K` = uniform weights,
  `0` = one component dominates.
- `mixture_weight_max_mean_batch` — mean over batch of
  `max_k softmax(logits)_k`. The collapse indicator: if this trends to
  1.0, one component is winning everywhere and the mixture has
  degenerated to ①. Pair with `mixture_weight_entropy`.
- `comp_std_mean_batch` — mean of `effective_stds` over all
  `B, K, action_dim` entries. The analogue of ①'s `std_mean_batch`, but
  averaged over components too.
- `std_squash_sat_frac` — same as ①, computed over all `K * action_dim`
  per-component log-stds.
- `effective_component_usage` — `mean over batch of (fraction of samples
  in this batch that came from component k)`, averaged over k. Should be
  ≈ `mean(π_k)` if sampling is correct; large divergence indicates a
  sampling bug (e.g. the §3 "weighted average" mistake). This is a
  self-consistency diagnostic computed from rollout `frame_modes` (which
  record which component was sampled) — preserve `frame_modes` threading
  per the overview's requirement.

## 9. Risks specific to this family

- **Component collapse.** One component's logits grow until
  `softmax → [1, 0, 0]`, and the mixture silently degenerates to ①
  (but with `K-1` wasted parameters and a worse conditioning of the
  head). Monitored by `mixture_weight_entropy` and
  `mixture_weight_max_mean_batch` (§8). Mitigations to consider *only if
  collapse is observed*: (a) a small L2 penalty on `mixture_logits`
  magnitude, (b) a small penalty on `1 - mixture_weight_entropy` added
  to the regularizer. Do not pre-emptively add these — they bias the
  learned mixture structure.
- **Logit domination via scale.** A component with very large σ can
  achieve high `log_prob` on many samples without being a "good" mode,
  just by being spread out. PPO's clipping limits per-update damage, but
  the policy can still drift toward "one wide component + narrow
  others". Monitored by `comp_std_mean_batch` per-component (if feasible
  to break out by component in stats — otherwise watch the overall
  stat). This is a softer failure mode than collapse; live with it
  unless it shows up in the capability-harness metrics.
- **`logsumexp` numerical instability.** Use `torch.logsumexp(weighted,
  dim=-1)` directly — do not hand-roll `max + log(sum(exp))`. The
  library version handles the `max` subtraction stably. With per-component
  log-probs that can be very negative (e.g. a far-away component scoring
  a sample from a different component), a hand-rolled version will
  underflow.
- **Sampling/scoring mismatch.** The §3 Gumbel-max sampling path and the
  §4 `logsumexp` scoring path are different code. A bug in either
  produces a sample distribution that doesn't match the scored
  distribution — the on-policy assumption breaks silently, exactly like
  the `log_std_min` export bug from the P0 refactor. The sample/score
  self-consistency test (overview §5 test 3) is the primary defense;
  `effective_component_usage` (§8) is the in-training secondary defense.
- **`K` mismatch on export.** Covered by `strict=True` (§7). The
  negative test (deliberately wrong `K` must raise) is in the acceptance
  checklist.

## 10. Acceptance checklist for Stage 3

- [ ] Degenerate equivalence: `K = 1` ⇒ `log_prob` matches ① to 1e-6 on
      identical inputs (mixture `logsumexp` over one component is a
      no-op, `log_softmax` over one logit is 0, so this reduces exactly
      to ①'s math — the test is verifying that reduction works).
- [ ] Normalization test passes on `action_dim = 2, K = 3`.
- [ ] Sample/score self-consistency passes — *and*
      `effective_component_usage ≈ mean(π_k)` within tolerance on the
      same samples (this is the sampling-path correctness check
      specific to MoG).
- [ ] Gradient completeness: all three heads (`mixture_logits`,
      `means`, `raw_log_stds`) receive non-zero grad after
      `loss.backward()`.
- [ ] Entropy regularizer gradient test: `regularizer.backward()` gives
      non-zero grad to at least one parameter in each of the three heads
      (the score-function estimate should reach all of them via
      `log_prob`'s dependence on each).
- [ ] Export round-trip with `strict=True`, including a deliberate
      `K`-mismatch negative test.
- [ ] `act()` latency within 10× budget. MoG sampling is `K` diagonal
      Gaussian samples + a Gumbel-max; expect ~2-3× baseline, well
      within budget.
- [ ] One full `basic_balance` regression run: `mixture_weight_entropy`
      and `mixture_weight_max_mean_batch` logged; confirm no collapse on
      this (unimodal) task. (Collapse on `basic_balance` would be
      expected and fine — it's a unimodal task — but the *rate* of
      collapse and the resulting `effective_component_usage` should be
      sane.)
- [ ] Capability harness A/B (`basic_balance_step` or `follow_v2`):
      compare MoG vs capacity-matched baseline (overview §7) on the
      task's primary metric. This is the stage's actual research
      question — does multimodality help where the task has a multimodal
      optimum? Pre-register the metric and a comparison threshold before
      running.
