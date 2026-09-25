"""Policy A/B variant of `standup_floor04`: state-σ pre-tanh actor.

[STATUS: 未完成 / on hold] StatePreTanhNormalPolicy 族暂停开发，未做
正式训练验证。本实验保留供参考，不建议启动——见
DESIGN_pre_tanh_normal.md 的状态说明。


  standup_floor04_pretanh          : PreTanhNormalPolicy — shared σ
  standup_floor04_pretanh_statesig : StatePreTanhNormalPolicy — σ = f(obs)
                                     head output (init σ ≡ e⁻¹ ≡ shared
                                     version at init) (this run)

Same seed=42 and all other params as `standup_floor04`.  Same metric /
floor / explore caveats as `standup_floor04_pretanh` — see that
experiment's docstring and DESIGN_pre_tanh_normal.md.

Watch `policy_stats`: `sigma_state_std` (state-dependence of σ — the
metric that must grow if this variant is doing anything), plus the
shared pretanh set: `coverage_distance`, `latent_mean_abs`,
`near_boundary_probability`, `effective_unsafe_tail_max`.
"""
from __future__ import annotations

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04PreTanhStateSig(StandupFloor04):
    name = "standup_floor04_pretanh_statesig"

    actor_blueprint = "init_policy_state_pre_tanh_normal.yaml"


EXPERIMENT_CLASS = StandupFloor04PreTanhStateSig
