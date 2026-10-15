"""Cut 7: per-sample gradient signal (the SAC GradSig).

For one critic_tick dump, compute the exact per-sample gradient of the
objective each sample contributed, then project it onto the realized
aggregate direction:

    l_i      = mask_i * objective_i / denom   (so sum_i l_i == the loss)
    g_i      = grad_theta(l_i)
    G        = sum_i g_i == grad_theta(loss)
    p_i      = <g_i, G_hat>                    (signed projection)
    cos_i    = <g_i, G_hat> / ||g_i||

Identity check: ``sum(p_i) == ||G||`` — reported as ``closure_err``.
SAC batches are small (B<=1024), so this loop is exact over 100% of the
batch — no subsampling disclaimer needed.

Two sides:
- ``actor``: grads w.r.t. theta_pre through expectation_samples on the
  captured frozen noise ``u_actor`` (critic frozen at phi_post — same
  order the trainer used).
- ``critic``: per-channel grads w.r.t. phi_pre of masked TD-MSE.
- ``channel_conflict``: aggregate per-channel Q-pull gradients and their
  cosines — do r_fall and r_cross push theta in the same direction?
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .experiment import SACRewardChannel


def _channels_of(spec: Mapping[str, Any]) -> Tuple[SACRewardChannel, ...]:
    return tuple(
        SACRewardChannel(**item) for item in spec.get("channels", [])
    )


def _dot(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> float:
    return float(
        sum((x.reshape(-1) * y.reshape(-1)).sum() for x, y in zip(a, b))
    )


def _norm(g: Sequence[torch.Tensor]) -> float:
    return float(
        sum(float((x.reshape(-1) ** 2).sum()) for x in g) ** 0.5
    )


def _scale(g: Sequence[torch.Tensor], s: float) -> List[torch.Tensor]:
    return [x * s for x in g]


def _sub(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [x - y for x, y in zip(a, b)]


def _top_rows(vals: np.ndarray, ids: np.ndarray, k: int = 15) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]]
]:
    order = np.argsort(vals)
    bottom = [
        {"sample_id": int(ids[i]), "p": float(vals[i])}
        for i in order[:k]
    ]
    top = [
        {"sample_id": int(ids[i]), "p": float(vals[i])}
        for i in order[::-1][:k]
    ]
    return top, bottom


def _percentiles(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(x.mean()),
        "q05": float(np.quantile(x, 0.05)),
        "q50": float(np.quantile(x, 0.5)),
        "q95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def per_sample_gradsig(
    dump_dir: Path,
    *,
    sides: Sequence[str] = ("actor", "critic"),
    store_all: bool = True,
) -> Dict[str, Any]:
    from .debugkit import _build_actor, _build_critic, load_dump
    from .trainer import (
        _RegularizerArgs,
        _forward_all_candidates,
        _resolve_u_kind,
        _state_regularizer,
    )

    dump = load_dump(Path(dump_dir))
    spec = dump["spec"]
    batch = dump["batch"]
    fwd = dump["forward"]
    pre, post = dump["trainer_pre"], dump["trainer_post"]
    sp = spec["sac_params"]
    channels = _channels_of(spec)
    mode = str(sp.get("regularizer_mode", "shannon"))
    ids = batch["sample_ids"].cpu().numpy()
    obs = batch["obs"].float()
    actions = batch["actions"].float()
    B = obs.shape[0]
    out: Dict[str, Any] = {
        "dump_dir": str(dump_dir),
        "critic_tick": dump["manifest"].get("critic_tick"),
        "batch_size": int(B),
        "regularizer_mode": mode,
        "sides": {},
    }

    channel_valid = batch["channel_valid"].bool()
    sample_weight = batch["sample_weight"].float()
    actor_mask = channel_valid.all(dim=1).float() * sample_weight
    actor_den = float(actor_mask.sum().item()) or 1.0

    # ---------------------------------------------------------------- actor
    if "actor" in sides:
        actor = _build_actor(spec, pre)
        critic_post = _build_critic(spec, channels, post)
        for p in actor.parameters():
            p.requires_grad_(True)
        actor_params = [
            p for p in actor.parameters() if p.requires_grad
        ]
        req = [(p, p.requires_grad) for p in critic_post.all_parameters()]
        for p, _ in req:
            p.requires_grad_(False)

        alpha = (
            fwd["alpha_pre"].float()
            if isinstance(fwd.get("alpha_pre"), torch.Tensor)
            else torch.tensor(float(sp.get("init_alpha", 1.0)))
        )
        u_actor = fwd["u_actor"].float()
        actor_weight = batch["actor_weight"].float()
        share = torch.tensor(
            [1.0 if ch.actor_weight_share else 0.0 for ch in channels],
            dtype=torch.float32,
        )
        is_shannon = mode == "shannon"
        u_kind = str(sp.get("u_kind", "native"))
        reg_lambda = float(sp.get("reg_lambda", 0.0))
        u_floor = float(sp.get("u_floor", 0.0))

        def actor_loss_i(i: int):
            o = obs[i:i + 1]
            na, nlp, w = actor.expectation_samples(o, u_actor[i:i + 1])
            q1 = torch.stack(
                [
                    _forward_all_candidates(
                        lambda x, a, ch=ch: critic_post.q1_forward(
                            x, a, ch.name),
                        o, na,
                    )
                    for ch in channels
                ], dim=-1,
            )
            q2 = torch.stack(
                [
                    _forward_all_candidates(
                        lambda x, a, ch=ch: critic_post.q2_forward(
                            x, a, ch.name),
                        o, na,
                    )
                    for ch in channels
                ], dim=-1,
            )
            qp = torch.stack((q1, q2), dim=-1)        # [1,K,M,C,2]
            wa = (
                actor_weight[i][None, None, None, :, None]
                * share[None, None, None, :, None]
            )                                          # [1,1,1,C,1]
            f_a = (qp * wa).sum(dim=3)                # [1,K,M,2]
            pair = f_a.argmin(dim=-1)
            f_sel = f_a.gather(-1, pair[..., None]).squeeze(-1)
            if is_shannon:
                term = (w * (alpha * nlp - f_sel)).sum(dim=(1, 2))
            else:
                term = (w * (-f_sel)).sum(dim=(1, 2))
                b_state = _state_regularizer(
                    actor, o,
                    _RegularizerArgs(
                        regularizer_mode=mode,
                        reg_lambda=reg_lambda,
                        u_floor=u_floor,
                        u_kind=u_kind,
                    ),
                )
                term = term + (-b_state)
            return (actor_mask[i] * term.sum()) / actor_den

        grads: List[List[torch.Tensor]] = []
        for i in range(B):
            g = torch.autograd.grad(
                actor_loss_i(i), actor_params, retain_graph=True,
                allow_unused=True,
            )
            grads.append([
                (x.detach().clone() if x is not None
                 else torch.zeros_like(p))
                for x, p in zip(g, actor_params)
            ])
        G = [
            sum(g[j] for g in grads) for j in range(len(actor_params))
        ]
        norm_G = _norm(G)
        G_hat = _scale(G, 1.0 / norm_G) if norm_G > 0 else G
        p_i = np.array([_dot(g, G_hat) for g in grads])
        norms_i = np.array([_norm(g) for g in grads])
        cos_i = np.divide(
            p_i, norms_i, out=np.zeros_like(p_i), where=norms_i > 0,
        )
        closure_err = abs(float(p_i.sum()) - norm_G)
        top, bottom = _top_rows(p_i, ids)
        out["sides"]["actor"] = {
            "norm_G": norm_G,
            "closure_err": closure_err,
            "p": _percentiles(p_i),
            "g_norm": _percentiles(norms_i),
            "cos": _percentiles(cos_i),
            "positive_share": float((p_i > 0).mean()),
            "top_push": top,
            "top_pull": bottom,
            "p_all": p_i.tolist() if store_all else None,
            "norm_all": norms_i.tolist() if store_all else None,
        }

        # ----- channel conflict: per-channel Q-pull gradients -----
        # g_c = grad_theta sum_i mask_i * sum_km w*( -w_c * Q_c,j* )
        # (shared entropy/reg terms excluded — they are mode-level, not
        # channel-level signals)
        conflict: Dict[str, Any] = {}
        if len(channels) > 1:
            g_per_ch: Dict[str, List[torch.Tensor]] = {}
            for ci, ch in enumerate(channels):
                if not ch.actor_weight_share:
                    continue

                def chan_loss(ci=ci):
                    total = None
                    for i in range(B):
                        o = obs[i:i + 1]
                        na, nlp, w = actor.expectation_samples(
                            o, u_actor[i:i + 1])
                        q1 = torch.stack(
                            [
                                _forward_all_candidates(
                                    lambda x, a, ch=ch:
                                    critic_post.q1_forward(x, a, ch.name),
                                    o, na)
                                for ch in channels
                            ], dim=-1)
                        q2 = torch.stack(
                            [
                                _forward_all_candidates(
                                    lambda x, a, ch=ch:
                                    critic_post.q2_forward(x, a, ch.name),
                                    o, na)
                                for ch in channels
                            ], dim=-1)
                        qp = torch.stack((q1, q2), dim=-1)  # [1,K,M,C,2]
                        wa = (
                            actor_weight[i][None, None, None, :, None]
                            * share[None, None, None, :, None]
                        )
                        f_a = (qp * wa).sum(dim=3)
                        pair = f_a.argmin(dim=-1)            # joint argmin
                        q_sel = qp[..., ci, :].gather(
                            -1, pair[..., None]).squeeze(-1)  # [1,K,M]
                        part = (
                            w * (-actor_weight[i, ci] * q_sel)
                        ).sum()
                        part = part * actor_mask[i] / actor_den
                        total = part if total is None else total + part
                    return total

                g = torch.autograd.grad(
                    chan_loss(), actor_params, allow_unused=True)
                g_per_ch[ch.name] = [
                    (x.detach().clone() if x is not None
                     else torch.zeros_like(p))
                    for x, p in zip(g, actor_params)
                ]
            names = list(g_per_ch)
            mat: List[List[Optional[float]]] = []
            for a_ in names:
                row = []
                for b_ in names:
                    na_, nb_ = _norm(g_per_ch[a_]), _norm(g_per_ch[b_])
                    row.append(
                        _dot(g_per_ch[a_], g_per_ch[b_]) / (na_ * nb_)
                        if na_ > 0 and nb_ > 0 else None
                    )
                mat.append(row)
            conflict = {
                "channels": names,
                "cos_matrix": mat,
                "grad_norms": {n: _norm(g_per_ch[n]) for n in names},
            }
        out["channel_conflict"] = conflict

        for p, was in req:
            p.requires_grad_(was)

    # ---------------------------------------------------------------- critic
    if "critic" in sides:
        critic_pre = _build_critic(spec, channels, pre)
        targets = fwd.get("targets") or {}
        critic_out: Dict[str, Any] = {}
        for c, ch in enumerate(channels):
            if ch.name not in targets:
                continue
            y = targets[ch.name].float()
            mask = (
                channel_valid[:, c].float() * sample_weight
            )
            msum = float(mask.sum().item()) or 1.0
            grp = critic_pre.groups[
                critic_pre.channel_to_group[ch.name]]
            params = (
                list(grp.q1.parameters()) + list(grp.q2.parameters())
            )
            # freeze everything except this channel's online q1/q2
            for p_ in critic_pre.all_parameters():
                p_.requires_grad_(False)
            for p_ in params:
                p_.requires_grad_(True)
            grads: List[List[torch.Tensor]] = []
            for i in range(B):
                o = obs[i:i + 1]
                a = actions[i:i + 1]
                q1 = critic_pre.q1_forward(o, a, ch.name)
                q2 = critic_pre.q2_forward(o, a, ch.name)
                li = mask[i] * (
                    (q1.squeeze() - y[i]) ** 2 + (q2.squeeze() - y[i]) ** 2
                ) / msum
                g = torch.autograd.grad(
                    li, params, retain_graph=True, allow_unused=True)
                grads.append([
                    (x.detach().clone() if x is not None
                     else torch.zeros_like(p))
                    for x, p in zip(g, params)
                ])
            G = [
                sum(g[j] for g in grads) for j in range(len(params))
            ]
            norm_G = _norm(G)
            G_hat = _scale(G, 1.0 / norm_G) if norm_G > 0 else G
            p_i = np.array([_dot(g, G_hat) for g in grads])
            norms_i = np.array([_norm(g) for g in grads])
            cos_i = np.divide(
                p_i, norms_i, out=np.zeros_like(p_i), where=norms_i > 0)
            top, bottom = _top_rows(p_i, ids)
            critic_out[ch.name] = {
                "norm_G": norm_G,
                "closure_err": abs(float(p_i.sum()) - norm_G),
                "p": _percentiles(p_i),
                "cos": _percentiles(cos_i),
                "positive_share": float((p_i > 0).mean()),
                "top_push": top,
                "top_pull": bottom,
                "p_all": p_i.tolist() if store_all else None,
            }
        out["sides"]["critic"] = critic_out

    return out
