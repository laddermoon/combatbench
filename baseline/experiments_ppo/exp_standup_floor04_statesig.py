"""Policy A/B variant of `standup_floor04`: state-dependent σ actor.

  standup_floor04          : TruncatedNormalPolicy — σ is a global
                             nn.Parameter shared by all states
  standup_floor04_statesig : StateTruncatedNormalPolicy — σ = f(obs)
                             via trunk + head(2·action_dim); log_std
                             half init w=0/b=-1 → σ ≡ e⁻¹ at init,
                             bit-identical to the baseline policy
                             (this run)

Same seed=42 and all other params as `standup_floor04`.  A/B pair
against train_standup_floor04_ppo_20260920_164819.

Watch `policy_stats.sigma_state_std` (batch-internal spatial variance of σ):
≈0 means the σ head stays constant (degenerate — behaves like the
baseline); >0 means state-dependent exploration is actually used.
"""
from __future__ import annotations

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04StateSig(StandupFloor04):
    name = "standup_floor04_statesig"

    actor_blueprint = "init_policy_state_truncated_normal.yaml"


EXPERIMENT_CLASS = StandupFloor04StateSig
