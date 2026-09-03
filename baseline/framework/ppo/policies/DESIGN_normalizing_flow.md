# Design: ④ RealNVP Normalizing Flow

Reads `DESIGN_OVERVIEW.md` as a prerequisite. This is Stage 4, and is
**conditional** — per overview §8, only proceed if Stage 3's
capability-harness A/B doesn't already answer the "do we need more
expressiveness than a 3-component mixture" question. A flow is the
highest-risk, highest-cost family in this round; don't build it if a
mixture already captures the useful multimodality in these tasks.

New file: `baseline/common/policies/realnvp_tanh_mlp.py`, class
`RealNVPTanhMLPPolicy(TanhSquashedPolicyBase)`.

## 1. What this family buys, and what it does not

A normalizing flow can represent **arbitrary** continuous distributions
over the action space — not just mixtures of `K` Gaussians, but
distributions with skewed modes, curved high-density manifolds, holes,
rings, etc. In principle it's the most expressive family here.

**What it does *not* buy:** a tractable closed-form entropy, a cheap
`log_prob`, or a cheap sample. Every `log_prob` evaluation requires
running the **inverse** flow (action → base space); every sample
requires running the **forward** flow (base → action). With `L`
coupling layers, that's `L` neural-network evaluations per `log_prob`
*and* per sample, vs ①'s single head evaluation. This is why the
latency budget (overview §6) exists and why this family is conditional.

**Honest expectation:** for the combat tasks in this round, the
distributional structure is probably "a few modes with roughly Gaussian
shape" — which a mixture already captures. The flow's extra
expressiveness is most likely to pay off on tasks with *continuous*
multi-modality (e.g. a circular manifold of good actions), which the
current task suite may not have. Treat this family as a research
investment, not a guaranteed win.

## 2. Architecture

Two parts:

### 2.1 Base distribution

A diagonal Gaussian conditioned on state, exactly like ①:
```
base_mean, base_raw_log_std = base_head(trunk(obs)).split(action_dim, dim=-1)
base_log_std = bounded_squash(base_raw_log_std)   # per ① §3
base_dist = Normal(base_mean, base_log_std.exp())
```

The base distribution is the *starting* point of the flow; the flow's
job is to reshape it into something more expressive. If the flow is the
identity (§9 degenerate-equivalence test), this reduces exactly to ①.

### 2.2 Coupling layers (RealNVP)

`L` coupling layers (default `L = 4` for `action_dim = 21`). Each layer
`ℓ`:

1. **Mask** `m_ℓ ∈ {0,1}^{action_dim}` — alternating across layers
   (layer 0 masks dims 0,2,4,...; layer 1 masks dims 1,3,5,...; etc.).
   Masks are fixed buffers, not learned.
2. **Conditioner network** `s_ℓ, t_ℓ = conditioner_ℓ(trunk(obs), m_ℓ * x)`:
   - input: `trunk(obs)` (shared, state conditioning) concatenated with
     the masked half of the current flow state `m_ℓ * x`.
   - output: `s_ℓ ∈ ℝ^{action_dim}`, `t_ℓ ∈ ℝ^{action_dim}` — but only
     the entries corresponding to the *unmasked* half are used (the
     masked half is passed through unchanged).
   - `s_ℓ` is bounded: `s_ℓ = tanh(s_ℓ_raw) * scale_max` (default
     `scale_max = 1.0`). This bounds the per-layer Jacobian determinant
     contribution to `exp(±scale_max)`, preventing the "scale explosion"
     failure mode where one layer's `s` grows large and the flow's
     log-det dominates the log-prob.
3. **Forward transform** (used in sampling):
   ```
   y_masked       = m_ℓ * x                          # pass-through half
   y_unmasked     = (1 - m_ℓ) * (x * exp(s_ℓ) + t_ℓ) # transformed half
   x_next         = y_masked + y_unmasked
   ```
4. **Inverse transform** (used in `log_prob`):
   ```
   x_unmasked     = (1 - m_ℓ) * ((y - m_ℓ * y) * exp(-s_ℓ) - t_ℓ)
   x_masked       = m_ℓ * y
   x_prev         = x_masked + x_unmasked
   ```
   Note `s_ℓ, t_ℓ` depend on `m_ℓ * x_prev` (the masked half of the
   *input*), which in the inverse direction is `m_ℓ * y` (the masked
   half is unchanged by the forward pass, so it's the same in both
   directions). This is what makes RealNVP invertible: the conditioner's
   input is the unchanged half, which is known at inverse time without
   solving for it.

### 2.3 Conditioner networks

Each layer's conditioner is a small MLP: `Linear(obs_trunk_dim +
action_dim, hidden) → Tanh → Linear(hidden, 2 * action_dim)`, split into
`s_raw` and `t`. Use a **separate** conditioner per layer (not shared
weights across layers) — shared weights would couple the layers' behavior
and break the "each layer refines a different half" structure that
alternating masks give.

Parameter count: with `L = 4`, `hidden = 64` for conditioners, this is
the largest family (estimate: ~+40-60k over ①, exact count to be
measured at implementation). This is expected; a flow is parameter-heavy.

## 3. Forward and inverse: get the direction right

This is the single most error-prone part of this family. State the
directions explicitly:

- **Forward** = base → action. Used for **sampling**. Start with a base
  sample `z_0 ~ base_dist`, apply layers `0..L-1` forward, get `z_L`,
  then `raw_action = z_L`, then `action = tanh(raw_action)`.
- **Inverse** = action → base. Used for **`log_prob`**. Start with
  `raw_action = atanh(action)` (base class does this), apply layers
  `L-1..0` inverse (reverse order!), get `z_0`, then
  `log_prob = base_dist.log_prob(z_0).sum(-1) + flow_log_det + tanh_jacobian`.

The inverse must run layers in **reverse order**. A common bug is to run
them in forward order in the inverse path, which produces a number but
not the correct inverse (and the inverse-consistency test, overview §5
test 6, is exactly what catches this — `inverse(forward(x)) ≈ x` will
fail by a large margin).

## 4. `log_prob`: three log-det terms, get the signs right

```
log_prob(action) = base_log_prob(z_0)            # (B,)
                 + flow_log_det                  # (B,), sum over layers
                 + tanh_jacobian                 # (B,), from base class
```

where:

- `base_log_prob(z_0) = base_dist.log_prob(z_0).sum(-1)` — the density
  of the base distribution at the inverse-mapped point.
- `flow_log_det = Σ_ℓ log|det J_ℓ| = Σ_ℓ sum(s_ℓ * (1 - m_ℓ))` — the
  sum of the per-layer log-Jacobian-determinants. Each layer's Jacobian
  is diagonal (the masked half is identity, the unmasked half is
  `exp(s_ℓ)`), so the log-det is just the sum of `s_ℓ` over the
  unmasked dimensions. **Sign: positive.** The forward transform's
  Jacobian determinant is `exp(s_ℓ)` on the unmasked half, so its log
  is `s_ℓ` (not `-s_ℓ`). The change-of-variables formula for the
  *forward* direction (base → action) gives
  `log p_action(action) = log p_base(z_0) + log|det J_forward|`, so the
  flow log-det is **added**. A sign error here is the classic flow bug;
  the normalization test (overview §5 test 2) catches it (the density
  won't integrate to 1).
- `tanh_jacobian = -log(1 - tanh(raw_action)² + ε).sum(-1)` — from the
  base class, same as all other families. **Sign: negative** (it's a
  contraction, so the density on the bounded action space is *higher*,
  which means the log-prob correction is *positive* — wait, let me be
  precise: `log p_action(a) = log p_raw(atanh(a)) + log|d atanh(a)/d a|`,
  and `d atanh(a)/d a = 1/(1-a²)`, so `log|...| = -log(1-a²)`. With
  `a = tanh(z)`, `1-a² = 1-tanh(z)²`, so the correction is
  `-log(1-tanh(z)²)`. This is **added** to `log_prob`. The base class
  already does this correctly; restating here only because the flow
  adds another log-det term and it's easy to mis-sign one of the two.)

**Total sign convention:**
```
log_prob = base_log_prob + flow_log_det + tanh_jacobian
         = base_log_prob + Σ s_ℓ - Σ log(1 - tanh² + ε)
```
All three terms are **added**. The `tanh_jacobian` term is negative in
value (because `log(1-tanh²) < 0`), but it's added, not subtracted. Get
this wrong and the normalization test fails immediately.

## 5. Sampling and `evaluate_actions`

```python
def _raw_sample(self, obs):
    z = self.base_dist(obs).rsample()             # (B, action_dim)
    for layer in self.layers:                     # forward order
        z = layer.forward(z, obs)
    return z                                      # this is raw_action

def _raw_log_prob(self, obs, raw_action):
    z = raw_action
    log_det = 0
    for layer in reversed(self.layers):           # inverse order
        z, layer_ld = layer.inverse(z, obs)
        log_det = log_det + layer_ld
    base_lp = self.base_dist(obs).log_prob(z).sum(-1)
    return base_lp + log_det                       # raw_log_prob; base
                                                  # class adds tanh_jacobian
```

`raw_mode`: there is no closed-form mode of a flow. Use
`forward(base_mean)` — the image of the base distribution's mode. This
is *not* the true mode of the action distribution (the flow can move
mass away from the base mode), but it's a reasonable deterministic
action and is what the base class's `deterministic_action()` will use.
Document this limitation in the code comment; do not claim it's the
mode.

## 6. Entropy: no closed form, use sampled estimate

Like ③, the entropy of a tanh-squashed flow has no closed form. Use the
same sampled estimate:

```python
sampled_log_prob = self._raw_log_prob(obs, raw_action)   # differentiable
entropy_estimate = -sampled_log_prob.mean()
regularizer = -self._entropy_coef * entropy_estimate
```

Same score-function semantics as ③ §5: the sample's *value* is fixed for
gradient purposes, but `log_prob`'s dependence on the parameters (base
distribution params + all conditioner params) remains differentiable.
Variance is higher than ③ because the `log_prob` path now runs `L`
inverse layers, each contributing gradient — watch the `entropy` stat's
per-update variance.

An alternative considered: `H(action) ≈ H(base) + E[flow_log_det]`,
where `H(base)` is closed-form (diagonal Gaussian entropy, like ①) and
`E[flow_log_det]` is estimated by averaging `flow_log_det` over samples.
This is cheaper (no inverse pass needed for the entropy estimate —
`flow_log_det` is a byproduct of the forward sampling pass) but is
**only exact if the flow is volume-preserving on average**, which it
isn't in general (the `E[log|det J|]` term is not the same as
`log|det E[J]|` or any other tractable quantity). It's a biased
estimate. Rejected for the same reason as ③'s closed-form option: a
biased entropy regularizer biases the policy, and the direction of the
bias (toward flows with larger log-det, i.e. more "expanding" flows) is
not obviously desirable. Use the sampled estimate; if its variance
destabilizes training, fall back to no regularizer for this family and
document.

## 7. Temperature

Apply temperature to the **base distribution only**, not to the flow:

```python
base_std_effective = base_log_std.exp() * self._temperature
# flow conditioners unchanged
```

Reason: temperature is "how random is the *source* noise", and the flow
is a deterministic reshape of that noise. Scaling the base σ scales the
input scale to the flow, which scales the output scale (approximately —
the flow is not linear, so this isn't exact, but it's the right
*semantic*: "wider source noise → wider actions"). Scaling the flow's
`s_ℓ` or `t_ℓ` would change the *shape* of the transformation, not the
scale of the noise — that's a different kind of control and should not
be overloaded onto `temperature`.

Do **not** conflate temperature with the flow's `scale_max` bound (§2.2).
`scale_max` is a fixed architectural choice (controls the maximum
per-layer Jacobian contribution); temperature is a runtime exploration
knob. They are independent.

## 8. Export

Constructor kwargs for round-trip: `obs_dim, action_dim, hidden_dim,
num_layers, scale_max, log_std_min, log_std_max`. The new ones vs ① are
`num_layers` and `scale_max` — both are shape/architecture parameters
that determine the module structure. `strict=True` reload will catch a
`num_layers` mismatch (different number of conditioner modules →
different state-dict keys) and a `scale_max` mismatch won't be caught by
`strict=True` (it's a scalar, not a parameter) — so `scale_max` must be
in the payload `config` and used in the constructor on load, with an
explicit assertion that the loaded value matches the payload.

`log_std_offset` (temperature) travels in the payload same as the other
families.

## 9. Stats

Add to `ActorEval.stats`:
- `entropy` — the sampled estimate from §6. Same variance caveat as ③.
- `base_std_mean_batch`, `base_std_min_batch`, `base_std_max_batch` —
  same as ①'s stats, on the base distribution's σ. These are the
  direct analogue of ①'s stats and the right numbers to compare to ①
  when asking "how exploratory is the source noise".
- `flow_logdet_mean_batch`, `flow_logdet_std_batch` — mean and std of
  `flow_log_det` over the batch. Primary monitor for "is the flow doing
  anything" (`log_det ≈ 0` means the flow is approximately
  volume-preserving / identity) and "is it blowing up" (large `|log_det|`
  means the flow is strongly expanding or contracting volume, which is
  where numerical instability lives).
- `scale_sat_frac` — fraction of `s_ℓ_raw` values with `|tanh(s_ℓ_raw)|
  > 0.95` (i.e. operating near the `scale_max` bound). The flow analogue
  of `std_squash_sat_frac`; if this grows, the flow is trying to exceed
  its architectural scale bound and is being silently capped.
- `inverse_recon_err_mean_batch` — mean `||inverse(forward(z)) - z||`
  over the batch, computed in the forward pass (we have `z` from the
  base sample and can run the inverse on the forward output as a
  diagnostic). This is expensive (doubles the flow compute per step) so
  **gate it behind `want_stats=True`** — only compute in the
  rollout-batch `evaluate_actions(want_stats=True)` call, not in PPO
  minibatches. In normal operation this should be ~1e-6; if it spikes,
  the flow has lost invertibility (numerical breakdown in a coupling
  layer) and the policy's `log_prob` is no longer trustworthy.

## 10. Risks specific to this family

- **Forward/inverse direction error.** §3. Caught by inverse-consistency
  test (overview §5 test 6).
- **Sign error in flow log-det.** §4. Caught by normalization test
  (overview §5 test 2).
- **Scale explosion.** `s_ℓ` unbounded ⇒ one layer's `exp(s_ℓ)` grows ⇒
  `flow_log_det` dominates `log_prob` ⇒ the policy's gradient is
  dominated by "make `s_ℓ` bigger" rather than "fit the data". Mitigated
  by the `tanh * scale_max` bound (§2.2) and monitored by
  `flow_logdet_*` and `scale_sat_frac` (§9). If `scale_sat_frac` is high
  and training is unstable, reduce `scale_max` — but document the
  change, don't silently tune it.
- **Tanh / flow ordering.** The flow operates in **raw space** (pre-tanh),
  like all other families. Tanh is applied *after* the flow, and the
  tanh Jacobian is applied *after* the flow log-det. Getting this
  backwards (tanh-then-flow) would mean the flow is operating on the
  bounded action space, which changes its invertibility properties
  (tanh is not invertible at ±1) and breaks the math. The base class's
  ordering (raw distribution → tanh → Jacobian) handles this correctly
  as long as the flow is implemented as the raw distribution.
- **Invertibility breakage.** Even with correct math, floating-point
  error in `exp(s_ℓ)` / `exp(-s_ℓ)` can accumulate over `L` layers and
  make `inverse(forward(x))` diverge from `x`. Monitored by
  `inverse_recon_err_mean_batch` (§9). If this exceeds ~1e-4, the flow
  is numerically broken and the policy is not trustworthy.
- **Rollout latency.** This is the only family where the 10× latency
  budget (overview §6) is expected to bind. Each `act()` call runs `L`
  forward coupling layers; each `evaluate_actions` call runs `L` inverse
  layers. With `L = 4` and small conditioners, expect ~5-8× baseline
  (~0.7-1.1ms), within budget but not comfortably. **Measure before
  committing to `L = 4`** — if `L = 2` is within budget with more
  margin, prefer it; the expressiveness gain from `L = 4` over `L = 2`
  is unlikely to matter for these tasks (see §1's honest expectation).
- **Deployment export depending on repo-only imports.** The exported
  `policy.py` must import the flow class from
  `baseline.framework.ppo.policies.realnvp_tanh_mlp` — i.e. the deployment
  environment must have the combatbench repo importable. This is the
  same as ②③ (they also import their family class from the repo), but
  worth restating because the flow class is the most complex and the
  most likely to have a subtle import-time dependency (e.g. a helper
  function in a sibling module). The export round-trip test (overview
  §5 test 5) runs in a fresh process, which catches missing imports.

## 11. Acceptance checklist for Stage 4

- [ ] Degenerate equivalence: identity flow (set all `s_ℓ_raw` and
      `t_ℓ` weights to zero at init ⇒ each layer is the identity ⇒ the
      flow is the identity ⇒ `log_prob` matches ① to 1e-6 on identical
      inputs. This isolates the flow plumbing from the base-distribution
      math.
- [ ] Inverse consistency: `inverse(forward(z)) ≈ z` to 1e-5
      elementwise, on random `z` and random observations. This is the
      primary flow-specific correctness test.
- [ ] Normalization test passes on `action_dim = 2, L = 2`.
- [ ] Sample/score self-consistency passes.
- [ ] Gradient completeness: base head params + every conditioner's
      `s_ℓ_raw` and `t_ℓ` weights receive non-zero grad after
      `loss.backward()`.
- [ ] Exact `log_prob` check: on `action_dim = 2, L = 1`, hand-compute
      the log-prob for a fixed input and compare to the policy's output
      to 1e-6. (With `L = 1` the math is tractable by hand; this is the
      only family where a hand-computed check is feasible, and it's
      worth doing as a sign-error backstop.)
- [ ] Export round-trip with `strict=True`, including a deliberate
      `num_layers`-mismatch negative test.
- [ ] `act()` latency measured (1 thread, warmup excluded) and within
      10× budget. If over budget, reduce `L` and re-measure before
      proceeding.
- [ ] `inverse_recon_err_mean_batch` logged in a full `basic_balance`
      regression run; stays below 1e-4 throughout.
- [ ] Capability harness A/B (same as ③, same pre-registered metric):
      flow vs capacity-matched baseline. **Only run this if ③'s A/B
      didn't already answer the expressiveness question** (overview §8).
