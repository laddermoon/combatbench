"""Policy A/B variant of `standup_floor04`: pre-tanh normal actor.

  standup_floor04          : TruncatedNormalPolicy — σ is a global
                             nn.Parameter shared by all states
  standup_floor04_statesig : StateTruncatedNormalPolicy — σ = f(obs)
  standup_floor04_mixture  : MixtureTruncatedNormalPolicy — K=3
  standup_floor04_pretanh  : PreTanhNormalPolicy — diagonal Gaussian in
                             pre-tanh space, a = tanh(z), shared per-dim
                             σ (init e⁻¹), μ = raw MLP output (this run)

Same seed=42 and all other params as `standup_floor04`.  A/B pair
against train_standup_floor04_ppo_20260920_164819.

NOTE 1 — U is the action-space Rényi-2 (L2) width with a DIFFERENT
geometry than the truncated-normal L2 width: the tanh-Gaussian family
caps at U_max ≈ 0.985 and penalizes BOTH small σ and saturated |μ|.
The same uncertainty_floor=0.4 is a different-strength constraint
(see DESIGN_pre_tanh_normal.md §8-§9).

NOTE 2 — explore_factor here is a coverage-radial controller (moves
(mu, log σ) toward the max-coverage point), NOT a σ multiplier; e is
clamped to the policy distribution anyway for e=0 rollouts, but any
nonzero e means something different than in the other variants.

Watch `policy_stats`: `effective_uncertainty` / `latent_std_*`
(shared σ drift), `latent_mean_abs` (pre-tanh saturation pressure),
`near_boundary_probability`, `effective_unsafe_tail_max` (guard
margin — must stay ≪ 1e-12 or the run fails loudly by design).
"""
from __future__ import annotations

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04PreTanh(StandupFloor04):
    name = "standup_floor04_pretanh"

    actor_blueprint = "init_policy_pre_tanh_normal.yaml"


EXPERIMENT_CLASS = StandupFloor04PreTanh
