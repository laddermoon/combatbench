"""PPO clipped-surrogate loss (Schulman et al. 2017).

This module is intentionally *only* the loss + diagnostics. Optimizer
stepping, epoch loop, minibatch iteration, and gradient clipping all
live in the user's training script: keeping the loss stateless makes
it easy to drop into custom training loops, share between PPO and GRPO
(GRPO uses the same surrogate with a different advantage source —
:func:`baseline.framework.ppo.algos.compute_grpo_advantages`), and unit-test
the individual pieces.

Standard PPO recipe (matches CleanRL / SB3 conventions):

  ratio        = exp(log_probs_new - log_probs_old)
  pg1          = ratio * advantages
  pg2          = clip(ratio, 1 - clip, 1 + clip) * advantages
  policy_loss  = -min(pg1, pg2).mean()

  [optional] value clipping (Schulman recipe):
    v_unclipped = (values_new - returns) ** 2
    v_clipped   = (clip(values_new - values_old, -c, +c) + values_old - returns) ** 2
    value_loss  = 0.5 * max(v_unclipped, v_clipped).mean()
    [if value_clip is None: value_loss = 0.5 * v_unclipped.mean()]

  total_loss   = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

Diagnostics (returned alongside the loss, useful for early-stop /
logging):

  approx_kl     ≈ ((ratio - 1) - log(ratio)).mean()  [Schulman approximation]
  clip_fraction = fraction of samples where |ratio - 1| > clip_range
  explained_var = 1 - var(returns - values_new) / var(returns)

Design rules:
  * Inputs are torch tensors (training-side), not numpy. Advantages
    pre-normalization is the caller's job (``(adv - adv.mean()) /
    (adv.std() + eps)`` is the standard CleanRL trick — easy to do
    inside ``ppo_loss(..., normalize_advantages=True)`` so we offer it
    as a flag).
  * Entropy can be either a per-sample tensor of shape ``(B,)`` or
    ``None``. ``None`` skips the entropy term entirely (e.g. for
    deterministic actors during eval-style fine-tuning).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PPOLossOutput:
    """Bundled return value of :func:`ppo_loss`.

    All fields are scalar 0-D tensors except ``loss`` which carries
    grad. Detach diagnostics in the caller before logging.
    """

    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    explained_variance: torch.Tensor


def ppo_loss(
    *,
    log_probs_old: torch.Tensor,
    log_probs_new: torch.Tensor,
    advantages: torch.Tensor,
    values_old: torch.Tensor,
    values_new: torch.Tensor,
    returns: torch.Tensor,
    entropy: Optional[torch.Tensor] = None,
    clip_range: float = 0.2,
    value_clip: Optional[float] = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    normalize_advantages: bool = True,
    advantage_eps: float = 1e-8,
) -> PPOLossOutput:
    """Compute PPO clipped surrogate + value + entropy losses.

    Shape requirements (``B`` = minibatch size):
      log_probs_old / log_probs_new / advantages / values_old / values_new
      / returns / (entropy if not None) all ``(B,)``.
    """
    _validate_shapes(
        log_probs_old=log_probs_old,
        log_probs_new=log_probs_new,
        advantages=advantages,
        values_old=values_old,
        values_new=values_new,
        returns=returns,
        entropy=entropy,
    )

    advantages_used = advantages
    if normalize_advantages:
        # Use the *minibatch* mean / std — CleanRL convention. SB3 does
        # the same; some impls use rollout-level normalization, which
        # is a different choice. Document but don't over-parameterize.
        advantages_used = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + advantage_eps
        )

    log_ratio = log_probs_new - log_probs_old
    ratio = torch.exp(log_ratio)
    pg1 = ratio * advantages_used
    pg2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_used
    policy_loss = -torch.min(pg1, pg2).mean()

    if value_clip is None:
        value_unclipped = (values_new - returns) ** 2
        value_loss = 0.5 * value_unclipped.mean()
    else:
        v_unclipped = (values_new - returns) ** 2
        v_clipped = (
            torch.clamp(values_new - values_old, -value_clip, value_clip)
            + values_old
            - returns
        ) ** 2
        value_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

    if entropy is not None:
        entropy_mean = entropy.mean()
    else:
        entropy_mean = torch.zeros((), device=log_probs_new.device)

    total_loss = (
        policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
    )

    with torch.no_grad():
        # Schulman's bias-reduced KL approximation: see
        # http://joschu.net/blog/kl-approx.html
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (torch.abs(ratio - 1.0) > clip_range).float().mean()
        var_returns = returns.var(unbiased=False)
        explained_variance = (
            torch.tensor(float("nan"), device=returns.device)
            if var_returns.item() < 1e-12
            else 1.0 - (returns - values_new).var(unbiased=False) / var_returns
        )

    return PPOLossOutput(
        loss=total_loss,
        policy_loss=policy_loss.detach(),
        value_loss=value_loss.detach(),
        entropy=entropy_mean.detach(),
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        explained_variance=explained_variance,
    )


def _validate_shapes(**tensors: Optional[torch.Tensor]) -> None:
    shapes = {name: t.shape for name, t in tensors.items() if t is not None}
    if not shapes:
        return
    ref_name, ref_shape = next(iter(shapes.items()))
    if len(ref_shape) != 1:
        raise ValueError(
            f"PPO loss expects 1-D minibatch tensors; got {ref_name}.shape={ref_shape}."
        )
    for name, shape in shapes.items():
        if shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: {name}.shape={shape} != "
                f"{ref_name}.shape={ref_shape}."
            )
