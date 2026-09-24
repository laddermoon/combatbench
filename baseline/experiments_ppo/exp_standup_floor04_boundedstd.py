"""Policy A/B variant of `standup_floor04`: bounded-σ truncated normal actor.

  standup_floor04           : TruncatedNormalPolicy — σ = exp(log_std),
                              a global nn.Parameter shared by all states
                              (σ unbounded except a ±20 log clamp)
  standup_floor04_statesig  : StateTruncatedNormalPolicy — σ = f(obs)
  standup_floor04_mixture   : MixtureTruncatedNormalPolicy — K=3
  standup_floor04_pretanh   : PreTanhNormalPolicy — tanh-Gaussian
  standup_floor04_boundedstd: BoundedStdTruncatedNormalPolicy — same
                              truncated normal on [-1,1], but
                              σ = exp(r_min + Δr·sigmoid(v)) with
                              σ ∈ (0.05, 2.0); explore_factor shifts the
                              raw sigmoid input additively (v + αe),
                              NOT σ multiplicatively (this run)

Same seed=42 and all other params as `standup_floor04`.  A/B pair
against train_standup_floor04_ppo_* runs.

NOTE 1 — σ bounds are the candidate defaults from
DESIGN_bounded_std_truncated_normal.md §1.1 (σ_min=0.05, σ_max=2.0);
init σ = e⁻¹ matches the baseline so e=0 starts at the same σ.

NOTE 2 — explore_factor is an additive shift on the raw sigmoid input
(v + αe), calibrated so the LOCAL d(log σ)/de at init matches the
legacy 3^e slope (α ≈ 1.1993).  It is NOT a σ multiplier: σ does not
triple at e=1, and the response saturates near the bounds.

NOTE 3 — bounded σ caps peak-U reachability: U_max(μ=0) ≈ 0.96 at
σ_max=2.0 (vs ~1.0 for the unbounded baseline).  uncertainty_floor=0.4
stays reachable, but interpret U cross-run accordingly.

Watch `policy_stats`: `raw_std_min/max` and `std_position_mean`
(sigmoid saturation drift), `std_lower/upper_saturation_frac`,
`log_std_sensitivity`, `exploration_sensitivity`, `effective_uncertainty`.
"""
from __future__ import annotations

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04BoundedStd(StandupFloor04):
    name = "standup_floor04_boundedstd"

    actor_blueprint = "init_policy_bounded_std_truncated_normal.yaml"


EXPERIMENT_CLASS = StandupFloor04BoundedStd
