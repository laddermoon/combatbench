# Policy Families — Design Overview

Status: design phase, no implementation yet.
Scope: `baseline/common/policies/` only. PPO v2 framework
(`ppo_trainer_v2.py`, `ppo_loop_v2.py`, `experiment_v2.py`) is expected to
require **zero** changes — that is a hard acceptance criterion, not an
aspiration. If implementing a family forces a framework change, stop and
reconsider the family's design before touching the framework.

This document is the shared context for the four per-family design docs:

- `DESIGN_state_dependent_gaussian.md` (①)
- `DESIGN_low_rank_gaussian.md` (②)
- `DESIGN_mixture_gaussian.md` (③)
- `DESIGN_normalizing_flow.md` (④)

Read this file first; the per-family docs assume it.

## 0. Non-goals

- **`tanh_gaussian_mlp.py` is not modified.** It remains the baseline /
  regression reference. Every new family lives in a new file.
- No diffusion policy in this round.
- No change to `ppo_trainer_v2.py`, `ppo_loop_v2.py`, the `TrainablePolicy`
  protocol, `ActorEval`, or `ExplorationSpec` in `experiment_v2.py`. Those
  were designed (P0–P2, already committed) precisely so that new
  distribution families plug in without framework changes.
- `experiments_v2/base.py` gets exactly one new attribute
  (`actor_blueprint: str`) and one conditional guard in `build_actor()`
  (only set `log_std_min/max` on the actor if it exposes those attrs).
  That is the full extent of framework-adjacent changes.

## 1. Why a shared base class

All four families share the exact same tanh-squashing math and the exact
same `Policy` ABC / `TrainablePolicy` protocol glue. That code is easy to
get subtly wrong (see Pitfall list below), and getting it wrong once
means getting it wrong four times if each family reimplements it.

`tanh_squashed_base.py` (new file) will hold:

- `evaluate_actions()`, `sample_action()`, `deterministic_action()`,
  `act()`, `act_numpy()`, `set_deterministic()` — the tanh-squash +
  `Policy` ABC glue, identical in spirit to
  `TanhGaussianMLPPolicy` but generic over the underlying raw
  distribution.
- `to_blueprint()` / export wiring, generic over family (see §3).
- `set_exploration()` skeleton (temperature bookkeeping); each subclass
  only implements how temperature maps onto its own raw distribution
  parameters.

Subclasses implement four hooks, all in **raw space** (pre-tanh):

```python
def _raw_sample(self, obs: torch.Tensor) -> torch.Tensor:
    """(B, action_dim) rsample from the raw (pre-tanh) distribution."""

def _raw_log_prob(self, obs: torch.Tensor, raw_action: torch.Tensor) -> torch.Tensor:
    """(B,) log-density of raw_action under the raw distribution at obs."""

def _raw_mode(self, obs: torch.Tensor) -> torch.Tensor:
    """(B, action_dim) deterministic (mode) raw action."""

def _regularizer_and_stats(
    self, obs, raw_action, raw_log_prob, want_stats: bool,
) -> tuple[Optional[torch.Tensor], Optional[Dict[str, float]]]:
    """Family-owned entropy-like regularizer (already signed/scaled) and
    diagnostics dict. May return (None, None)."""
```

The base class owns the tanh Jacobian correction
(`-log(1 - tanh(raw)^2 + 1e-6)`, `.sum(-1)`) in exactly one place, so it
cannot be gotten wrong per-family.

**The base class is validated against `TanhGaussianMLPPolicy` before any
new family is built** — see §5, test 1. A trivial subclass that wraps a
single diagonal Gaussian must reproduce baseline numbers bit-for-bit
(well, to float precision) as a precondition for trusting the base class
with anything more complex.

## 2. Why baseline is not touched, and how equivalence is still checked

Requirement from the user: baseline stays as-is, is the regression
reference, and every new family is a genuinely new implementation. This
means the base class's tanh math is a **deliberate duplication** of
`TanhGaussianMLPPolicy`'s math, not a refactor of it.

Duplication risk is mitigated by the degenerate-equivalence test (§5,
test 1): configure the new base class to be mathematically a diagonal
Gaussian (same mean network, same log_std, temperature=1) and assert its
`log_prob` matches `TanhGaussianMLPPolicy.evaluate_actions()` to 1e-6.
This is the single most valuable test in the whole plan — it is a proof
by reduction that the shared plumbing (atanh, Jacobian, sum axis) is
correct, decoupled from any family-specific distribution math.

Do not "improve" the tanh math while duplicating it (e.g. switching to
the more numerically stable
`2*(log(2) - x - softplus(-2x))` form for `log(1-tanh(x)^2)`). Keeping it
bit-identical is what makes the equivalence test meaningful. A more
stable formulation can be added later as an explicit opt-in, tested
separately.

## 3. Export / checkpoint: new family-agnostic module, `checkpoint.py` untouched

`checkpoint.py` and `build_export_policy_code()` are hard-coded to
`TanhGaussianMLPPolicy` (fixed constructor kwargs, fixed class import,
`load_state_dict(strict=False)`). Reusing it for a new family without
changes means: the family's shape hyperparameters (e.g. mixture `K`,
low-rank `k`, flow depth) don't travel with the checkpoint, the export
side reconstructs the *wrong* module (defaults / wrong class), and
`strict=False` silently swallows the resulting missing/unexpected key
mismatch. Rollout would then run a different (partially-random) policy
than the one that was trained — the on-policy assumption breaks
silently, with no error and no log line, exactly like the `log_std_min`
export bug found during the P0 refactor.

New file: `export_generic.py`. One function:

```python
def build_generic_export_payload(
    actor: nn.Module,
    *,
    policy_class_path: str,   # e.g. "baseline.common.policies.mog_tanh_mlp:MoGTanhMLPPolicy"
    config: Dict[str, Any],   # full constructor kwargs, family-defined
    extra_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    ...

def export_generic_policy_artifacts(
    actor: nn.Module, policy_dir: Path, *,
    policy_class_path: str, config: Dict[str, Any],
    stochastic: bool, extra_payload=None,
) -> None:
    ...
```

Each family's `export_config()` method (part of the `TrainablePolicy`
surface each family adds for itself, not part of the framework protocol)
returns its full constructor kwargs. The generated `policy.py` does:

```python
module_path, cls_name = payload["policy_class_path"].split(":")
cls = import_module(module_path).__getattribute__(cls_name)
policy = cls(**payload["config"])
missing, unexpected = policy.load_state_dict(payload["state_dict"], strict=True)
```

`strict=True` is deliberate: for a new, actively-changing family, a
silent shape mismatch is strictly worse than a loud crash. Every
export/import path change must go through the round-trip test (§5, test
5) before it's trusted.

`checkpoint.py` itself is not edited. `tanh_gaussian_mlp.py` keeps using
it exactly as today.

## 4. `experiments_v2/base.py`: minimal, additive changes

Two changes, both additive and defaulting to current behavior:

1. New class attribute:
   ```python
   actor_blueprint: str = "init_policy.yaml"
   ```
   used in `build_actor()` instead of the hard-coded filename. Each new
   family gets its own `init_policy_<family>.yaml` under
   `humanoid21/blueprints/`. Existing experiments are unaffected (default
   value matches current hard-coded path).

2. In `build_actor()`, guard the two attribute-forcing lines:
   ```python
   if hasattr(actor, "log_std_min"):
       actor.log_std_min = float(self.log_std_min)
   if hasattr(actor, "log_std_max"):
       actor.log_std_max = float(self.log_std_max)
   ```
   so a family without a scalar `log_std_min/max` (e.g. the mixture,
   which clamps per-component) isn't forced to grow attributes it
   doesn't use just to satisfy this line.

No other framework-adjacent file changes are in scope for this round.

## 5. Cross-cutting test plan (`test_policy_families.py`, parametrized over family)

New distribution-family bugs share a signature: shapes are correct, the
policy trains, logs look normal, and only the distribution itself is
wrong (wrong log-prob normalization, sampling/scoring mismatch, wrong
axis for a `logsumexp`, a sign error in a Jacobian). None of that shows
up by "watching the reward curve go up." Every family must pass all of
the following before any training run is trusted:

1. **Degenerate equivalence.** Configure the family to its most
   degenerate form (weights zeroed / `U=0` / `K=1` / identity flow) and
   assert `log_prob` matches `TanhGaussianMLPPolicy.evaluate_actions()`
   to 1e-6 on the same inputs. This is the base-class correctness proof
   (§2) and must be re-run for every family, since each family's
   "degenerate form" wiring is itself new code.

2. **Normalization.** On a small `action_dim=2` instance, Monte-Carlo
   estimate `∫ p(a) da ≈ 1` (grid or importance sampling), tolerance
   ~2%. Catches wrong `logsumexp` axis, missing/duplicated Jacobian term,
   flow log-det sign errors.

3. **Sample/score self-consistency.** Draw samples from the policy,
   score them with `evaluate_actions`, and cross-check two ways:
   (a) `-mean(log_prob)` over samples ≈ MC entropy estimate;
   (b) on `action_dim=2`, compare a sample histogram against
   `exp(log_prob)` on a grid. This is the test that would have caught the
   original `log_std_min` export bug's failure *mode* (sampling
   distribution silently diverging from scoring distribution) if it had
   existed for the baseline.

4. **Gradient completeness.** After `loss.backward()` on a batch
   (`policy_loss - regularizer`), every `requires_grad=True` parameter
   has a non-`None`, non-all-zero `.grad`. Catches: MoG logits detached
   from the graph, a flow coupling layer not wired into `forward`, one
   half of a split head unused.

5. **Export round-trip.** Build → export via `export_generic.py` → load
   the generated `policy.py` in a fresh process → compare actions and
   log_prob at a fixed seed/obs against the in-memory actor. Must be
   bit-identical (deterministic path) since this is a `strict=True`
   `state_dict` reload of the same weights.

6. **Inverse consistency (flow only).** `inverse(forward(x)) ≈ x` to
   1e-5 elementwise.

7. **Latency budget.** `act()` single-step CPU latency (1 thread) ≤ 10×
   the measured baseline (~143µs → budget ~1.4ms). Rationale in §6.
   Measured with `torch.set_num_threads(1)`, JIT/warmup excluded (discard
   first ~50 calls).

Tests 1–5 are mandatory for every family. Test 6 is flow-only. Test 7 is
mandatory for every family but only expected to bind for the flow.

## 6. Why the latency budget is 10×, not "as fast as possible"

Measured on this machine: `TanhGaussianMLPPolicy.act()` single-step CPU
(1 thread) ≈ 143µs. A rollout update collects ~410k env steps across 96
workers ≈ 4270 steps/worker ≈ 0.61s of policy compute per worker, against
an observed total rollout wall time of ~16s per update — i.e. **policy
inference is ~4% of rollout time today**; the environment dominates.
That leaves roughly 10× headroom before policy compute becomes the
bottleneck (a 10×-slower policy turns ~16s rollout into ~21s, +33%,
tolerable). This is what makes the flow family (§ its own doc)
plausible at all instead of a non-starter. Any family blowing this
budget needs a profiling pass before being accepted, not after.

## 7. A/B methodology against the existing regression baseline

We recently confirmed (separate task) that the V2 framework is
bit-reproducible run-to-run for `basic_balance` up to update 936 on every
PPO-mechanical stat (`policy_loss`, `approx_kl`, `clip_frac`,
`grad_norm_actor`, etc. — only `entropy`'s measurement timing changed by
design). That gives a very clean baseline to diff against:

- **Regression harness:** `basic_balance`. Existing 936-update reference
  run first reached `survival_rate=1.0` at update 295. This is the
  cheapest discriminator for "did I break something" — it is a
  *balance* task, effectively unimodal, so it mainly validates that a
  new family hasn't regressed, not that multimodality helps.
- **Capability harness:** `basic_balance_step` (or `follow_v2`) —
  candidate tasks with a genuine bimodal optimum ("step left" vs "step
  right"), where MoG/flow's extra expressiveness could plausibly show up
  in the metrics. To be confirmed before Stage 3.
- **Capacity control:** new families have more parameters than baseline.
  Before crediting a metric improvement to "distributional expressiveness",
  run a control: baseline with `hidden_dim` increased to match the new
  family's parameter count (~96k → ~124k needs `hidden_dim≈300`). Compare
  new-family vs capacity-matched-baseline, not new-family vs baseline.
- **Fixed success criteria, decided up front, not post-hoc:** update at
  which `survival_rate` first reaches 1.0; `ep_len_mean` at update 500;
  the capability harness's own task metric.

## 8. Implementation order and rationale

**Stage 0 (infra, no new family):** `tanh_squashed_base.py`,
`export_generic.py`, `test_policy_families.py` skeleton (parametrized,
initially empty parameter list), `experiments_v2/base.py`'s two additive
changes, and a throwaway `_TanhGaussianRef` subclass used only to prove
the base class reproduces baseline (test 1) before any real family is
built on top of it.

**Stage 1 — ① state-dependent log_std.** Smallest change, highest
signal for effort, and it's the first real consumer of the Stage 0
infra, so problems in the shared base surface here first while the
blast radius is smallest.

**Stage 2 — ② low-rank Gaussian.** Closed-form via
`torch.distributions.LowRankMultivariateNormal` (confirmed available:
torch 2.7.1). Lowest implementation risk of the three remaining.

**Stage 3 — ③ mixture of Gaussians.** The first family with genuine
multimodality. Highest-value target for the capability harness.

**Stage 4 — ④ normalizing flow.** Highest implementation risk (bespoke
bidirectional math, three separate sign-sensitive log-det terms). Do
this last, and only if Stage 3's capability-harness A/B doesn't already
answer the "do we need more expressiveness" question — a flow is not
worth its complexity if a 3-component mixture already captures the
useful multimodality in these tasks.

Each stage's design and pitfalls are detailed in its own doc. Do not
start implementing a stage without reading its doc plus this one.

## 9. Open decisions (flagged in per-family docs, restated here for visibility)

- **② operates in pre-tanh (raw) space.** The low-rank covariance models
  joint-angle correlation *before* the tanh squash; tanh is applied
  elementwise afterward and will distort that correlation structure in
  action space. This is the cheap option from the original comparison
  table; modeling correlation in action space directly is a materially
  larger undertaking and is out of scope unless explicitly requested.
- **Capability harness choice** (`basic_balance_step` vs `follow_v2`) is
  provisional; confirm before Stage 3 A/B runs are executed.
- **Stage 4 is conditional** on Stage 3's results, per §8.
