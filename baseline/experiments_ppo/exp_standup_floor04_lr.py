"""Adaptive-LR variant of `standup_floor04`: closed-loop step size.

Per-update actor LR rules, driven by the *previous* update's UpdateStats:

  - previous update early-stopped (``early_stop_kl > 0``)  → LR × 0.8
  - previous ``post_kl_max`` < 0.2 × ``target_kl``         → LR × 1.1

Early stop takes precedence — an update that hit the KL cap never
triggers growth.  Growth is capped at the base ``learning_rate``
(recovery to base, not unbounded step-size growth); decay is effectively
self-floored because a very small LR makes ``post_kl_max`` tiny, which
fires the growth rule.  Critic LR stays at its static value.

Same seed=42 and all other params as `standup_floor04`.
"""
from __future__ import annotations

from typing import Optional

from baseline.framework.ppo import UpdateStats
from baseline.framework.ppo.experiment import LRSpec

from .exp_standup_floor04 import StandupFloor04


class StandupFloor04LR(StandupFloor04):
    name = "standup_floor04_lr"

    # --- Adaptive actor LR ---
    lr_decay: float = 0.8        # applied after an early-stopped update
    lr_growth: float = 1.1       # applied when post_kl_max stays tiny
    lr_grow_frac: float = 0.2    # growth threshold, fraction of target_kl

    _prev_early_stop: bool = False
    _prev_post_kl_max: float = 0.0
    _cur_actor_lr: float = 0.0   # 0 = never adjusted → base LR

    def on_update(self, stats: UpdateStats, update: int):
        # Merge super's experiment metrics (online_success etc.) so the
        # exp.* charts keep working under this override.
        metrics = dict(super().on_update(stats, update) or {})
        if not stats.is_empty:
            self._prev_early_stop = stats.early_stop_kl > 0.0
            self._prev_post_kl_max = stats.post_kl_max
        return metrics or None

    def lr_schedule(self, update: int) -> Optional[LRSpec]:
        base = self.common_params().learning_rate
        cur = self._cur_actor_lr or base
        if self._prev_early_stop:
            new = cur * self.lr_decay
        elif self._prev_post_kl_max < self.lr_grow_frac * self.target_kl:
            new = min(cur * self.lr_growth, base)
        else:
            return None
        self._cur_actor_lr = new
        return LRSpec(actor_lr=new)


EXPERIMENT_CLASS = StandupFloor04LR
