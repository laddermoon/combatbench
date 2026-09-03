# Design: ① State-Dependent Diagonal Gaussian

Reads `DESIGN_OVERVIEW.md` as a prerequisite. This is Stage 1.

New file: `baseline/common/policies/state_gaussian_mlp.py`, class
`StateGaussianMLPPolicy(TanhSquashedPolicyBase)`.

## 1. What changes vs. baseline

Baseline (`TanhGaussianMLPPolicy`): mean is a function of state, `log_std`
is a single `(action_dim,)` `nn.Parameter` shared across all states.

This family: the head outputs `2 * action_dim` values — the first half is
`mean`, the second half is a raw per-state log-std, squashed into
`[log_std_min, log_std_max]` per §3. Every other architectural piece
(trunk width, depth, activation) is unchanged, so the comparison against
baseline isolates exactly one variable: state-dependence of σ.

## 2. Network

```
trunk: Linear(obs_dim, hidden) -> Tanh -> Linear(hidden, hidden) -> Tanh
head:  Linear(hidden, 2 * action_dim)
mean, raw_log_std = head(trunk(obs)).split(action_dim, dim=-1)
```

Same trunk shape as baseline (`hidden_dim=256` default) — this is
deliberate, so parameter delta (measured: 96,042 → 101,418, +5.4k) is
attributable only to the head, not a confounded trunk change.

## 3. Bounding σ: soft squash, not hard clamp — this is the central design decision

Baseline clamps a *global learnable parameter*:
`log_std = clamp(param, min, max)`. If that parameter saturates, it's a
global event and other gradients (from other action dims, or later
training) can still move it, because the parameter itself isn't
gradient-dead in general — only exactly at the clamp boundary is the
local gradient zero, and it's a single scalar so it's revisited by every
minibatch.

Here σ is a **function of state**. If a hard clamp is used and some
region of state space pushes `raw_log_std` past the boundary, every
sample from that region has **zero gradient through σ**, permanently (as
long as the network keeps predicting outside the bound for that region).
Regions the policy "already solved" are exactly the ones likely to drift
outside the bound (pressure to reduce σ there), so this failure mode is
correlated with training progress, not random — it will look like
progress right up until entire regions of state space silently lose
their exploration signal.

Use a smooth squash instead:

```python
def _bounded_log_std(self, raw_log_std: torch.Tensor) -> torch.Tensor:
    # raw_log_std: (B, action_dim), unconstrained.
    t = torch.tanh(raw_log_std)                      # (-1, 1), gradient everywhere
    return self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (t + 1.0)
```

This never has exactly-zero gradient in the interior representation
(`tanh` saturates but never hits its asymptote for finite input), so a
region can always be nudged back by a large enough gradient signal — no
permanent dead zones.

Diagnostic to add regardless of squash choice: fraction of batch with
`|raw_log_std| > 3` (i.e., operating in the flat part of tanh even if not
exactly saturated) — report as `std_squash_sat_frac` in stats. This is
the state-dependent analogue of baseline's `tanh_sat_frac` and should be
watched the same way.

## 4. Initialization: must reproduce baseline at step 0

If left to default init, the head's σ-half will emit arbitrary log-std
values that bear no relation to baseline's initial σ ≈ exp(-1) ≈ 0.368,
so update-1 stats would already diverge and the whole point of A/B
diffing against the 936-update baseline reference is lost.

Required initialization for the σ-half of the head:
- weight: zero (`nn.init.zeros_`)
- bias: constant such that, after the squash in §3, `effective_log_std ≈
  -1.0` everywhere. Solve for the bias `b`:
  `tanh(b) = 2*(-1.0 - log_std_min)/(log_std_max - log_std_min) - 1`,
  then `b = atanh(...)`. With defaults `log_std_min=-4, log_std_max=0`,
  this gives `b ≈ 0.848` (verify numerically at implementation time, not
  by hand).

With this init, at step 0 every state gets σ ≈ 0.368 = baseline's initial
σ, mean-half of the head is initialized the same way as baseline's mean
head (Linear default init, matching architecture), and update-1 stats
should match baseline's update-1 stats up to floating point — this is
the acceptance bar before moving past Stage 1's smoke test.

## 5. `set_exploration` / temperature

Apply temperature as an additive offset to `raw_log_std` *before* the
squash — i.e., same shape as baseline (`effective = squash(raw +
log(temperature))`), so at high temperature the σ saturates against
`log_std_max` (rather than exceeding it), matching baseline's semantics
of temperature as a bounded multiplier, not an unbounded one.

```python
def set_exploration(self, spec):
    if spec.temperature is not None:
        self._log_std_offset = float(np.log(spec.temperature))
    if spec.entropy_coef is not None:
        self._entropy_coef = float(spec.entropy_coef)
    return {"entropy_coef": self._entropy_coef,
            "temperature": float(np.exp(self._log_std_offset)),
            "log_std_min": self.log_std_min, "log_std_max": self.log_std_max}
```

## 6. `evaluate_actions` / entropy

Per-state entropy of a diagonal Gaussian has a closed form
(`0.5*log(2*pi*e) + log_std`, summed over action dims); use
`torch.distributions.Normal(mean, effective_std).entropy().sum(-1)` as in
baseline — this part is unchanged math, just evaluated with a
state-dependent σ instead of a shared one.

Regularizer: `-entropy_coef * entropy.mean()` — same form as baseline.
Note the `.mean()` is now averaging over states with genuinely different
σ, which is fine, but see §7 for why the resulting scalar means something
different from baseline's.

## 7. Stats: names must NOT collide with baseline's, because the semantics differ

Baseline's `std_mean` / `std_min` / `std_max` are computed from **21
scalar parameters** — a property of the policy, independent of which
states are in the batch. This family's analogous quantities are computed
from a **(B, 21) tensor** — a property of the batch. These are not the
same kind of quantity and comparing them directly across families (e.g.
plotting on the same chart) will produce a misleading comparison unless
this is understood.

Use distinguishable names in `ActorEval.stats`:
`std_mean_batch`, `std_min_batch`, `std_max_batch`,
`std_squash_sat_frac` (§3), plus `entropy` (comparable in *purpose* to
baseline's `entropy`, though it's now a batch expectation over a
state-dependent quantity rather than a fixed-parameter quantity — this
is actually the more meaningful reading of "entropy", baseline's is a
degenerate special case of it).

Also add `std_min_batch` monitoring: if this tracks near `log_std_min`'s
exp for a growing fraction of updates, that's the "solved regions losing
exploration" failure mode from §3 actually manifesting — treat it as an
actionable alarm, not just a metric.

## 8. Export

Constructor kwargs needed for round-trip (`export_config()`):
`obs_dim, action_dim, hidden_dim, log_std_min, log_std_max`. No
`log_std` parameter to carry separately (it's part of `state_dict` via
the head weights) — this is actually simpler than baseline's export,
which has to carry `log_std_min/max/offset` as separate scalar payload
fields because baseline's `log_std` is a bare parameter, not part of a
`Linear` layer's weight.

`log_std_offset` (temperature) still needs to travel in the payload the
same way it does for baseline, for the same reason: rollout sampling
must reproduce whatever `set_exploration` last configured on the
training-side actor.

## 9. Acceptance checklist for Stage 1

- [ ] Degenerate equivalence test passes (σ-head zeroed per §4 init,
      `temperature=1` ⇒ `log_prob` matches baseline to 1e-6 on identical
      `obs`/`actions`, using baseline's own mean-head weights copied in).
- [ ] Update-1 stats (`policy_loss`, `approx_kl`, `ep_len_mean`) match
      the `basic_balance` baseline run's update-1 stats bit-for-bit (this
      is a stronger, end-to-end version of the previous check, run
      through the actual training loop rather than a unit test).
- [ ] `std_squash_sat_frac` and `std_min_batch` logged and watched across
      a full `basic_balance` regression run; no runaway collapse.
- [ ] Export round-trip test passes with `strict=True`.
- [ ] `act()` latency within 10× budget (expect close to baseline's
      143µs since the trunk is unchanged and the head is only 2× wider).
