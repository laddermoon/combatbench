"""Device-collector standup variant retuned to exploit large batches.

Rationale (see run analysis of standup_dev): with the CPU recipe the
per-update KL stays ~0.012, far below target_kl=0.05 — per-update
progress is bounded by gradient step size, not the trust region.
Large batches give lower-variance gradients, so this variant spends
that budget on a bigger step: 4x lr and 2x KL headroom.  KL
early-stop remains the safety net against overshoot.

This is a labeled device-tuned recipe — NOT R2 protocol-identical.
Quality must still be judged on the same eval metrics vs CPU refs.
"""

from .exp_standup import Standup


class StandupDevFast(Standup):
    name = "standup_devfast"

    # --- PPO tuning (retuned for big batch) ---
    learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    target_kl: float = 0.10
    # minibatch_size/update_epochs unchanged (4096 / 4).

    # --- Rollout schedule ---
    # 8x the CPU protocol; sized to fill 4 workers x 1024 rows.
    episodes_per_update: int = 4096
    eval_episodes: int = 64


EXPERIMENT_CLASS = StandupDevFast
