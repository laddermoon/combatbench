"""Realized-exploration diagnostic (rollout stage).

For a dump captured at update ``u``, the rollout policy is
``policy_exports/u{u-1:05d}`` (``policy_exports/uNNNNN`` is the
*post-update-NNNNN* policy — rollout at update N consumes the version
produced by update N-1).  We replay its deterministic ``act()`` on the
episode's stored observations to obtain ``a_det(t)`` — the policy's
*intended* action at every frame.

The viewer then derives the realized exploration step in raw
(pre-tanh) space, where the sampling noise actually lives::

    eps_raw(t, d) = atanh(a_sampled(t, d)) - atanh(a_det(t, d))

``a_det = tanh(mu)`` so ``atanh(a_det)`` recovers the raw mean;
``atanh(a_sampled)`` recovers ``mu + sigma*z + noise_shift`` — the
exploration input as it truly entered the environment.

Only **trained** agents are evaluated (same rule as ``dump_delta``):
an agent counts as trained iff ``traj_map`` lists a trajectory sourced
from ``(episode, agent)``.

Output layout::

    <dump_dir>/rollout/episode_{pos:05d}/
        det.npz    # a_det.{agent}: (T, action_dim)
        meta.json  # {update, rollout_update, episode_pos, agents,
                   #  n_frames, action_dim, stats: {agent: {...}}}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from baseline.framework.ppo.dumpkit.dump_delta import (
    _load_exported_policy,
    trained_agents,
)

#: Atanh-domain clamp — sampled actions are tanh outputs, so |a| can
#: hit exactly 1.0 in float32.  Values at/over this bound are
#: "saturated": eps is then a lower bound, not an exact residual.
ATANH_CLAMP = 1.0 - 1e-6


def _atanh(a: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(a, -ATANH_CLAMP, ATANH_CLAMP))


def rollout_stats(
    eps_raw: np.ndarray,
    sat: np.ndarray,
) -> Dict[str, Any]:
    """Aggregate per-agent exploration stats from ``eps_raw`` (T, D).

    Returns JSON-safe scalars plus small arrays for the viewer:
    ``acf`` (lags 1..20) and ``spec`` (rfft power of the per-dim-mean
    spectrum, AC part only — index 0 excluded).
    """
    eps = np.asarray(eps_raw, dtype=np.float64)
    T, D = eps.shape
    eps_norm = np.linalg.norm(eps, axis=1)  # (T,)

    # Lag-k autocorrelation, averaged over dims (NaN-safe: a dim with
    # zero variance contributes nothing).
    max_lag = min(20, T - 1)
    acf: List[float] = []
    x = eps - eps.mean(axis=0, keepdims=True)
    var = (x * x).sum(axis=0)  # (D,)
    for k in range(1, max_lag + 1):
        num = (x[k:] * x[:-k]).sum(axis=0)  # (D,)
        r = np.divide(num, var, out=np.full(D, np.nan), where=var > 0)
        acf.append(float(np.nanmean(r)) if np.isfinite(r).any() else 0.0)

    # Mean power spectrum over dims (AC part), plus a low-frequency
    # share: bins whose period >= T/10 frames (≤10 cycles per episode).
    spec = None
    lf_ratio = None
    if T >= 8:
        pw = np.abs(np.fft.rfft(x, axis=0)) ** 2  # (F, D)
        spec_mean = pw[1:].mean(axis=1)  # drop DC — (F-1,)
        spec = [float(v) for v in spec_mean]
        n_bins = len(spec_mean)
        k_low = max(1, n_bins // 10)  # periods ≥ T/10
        tot = float(spec_mean.sum())
        lf_ratio = float(spec_mean[:k_low].sum() / tot) if tot > 0 else None

    peak = int(np.argmax(eps_norm)) if T else 0
    return {
        "mean_norm": float(eps_norm.mean()) if T else 0.0,
        "peak_frame": peak,
        "peak_norm": float(eps_norm[peak]) if T else 0.0,
        "rho1": acf[0] if acf else 0.0,
        "lf_ratio": lf_ratio,
        "sat_frac": float(np.asarray(sat).mean()),
        "acf": acf,
        "spec": spec,
    }


def compute_rollout(
    dump_dir: Path,
    episode_pos: int,
    *,
    log=print,
) -> Path:
    """Replay the rollout policy's deterministic act() on one episode.

    Writes ``<dump_dir>/rollout/episode_{pos:05d}/`` and returns it.
    """
    from baseline.framework.ppo.dumpkit.frame_access import DumpDataset

    dump_dir = Path(dump_dir)
    ds = DumpDataset(dump_dir)
    update = int(ds.manifest["update"])
    traj_map = ds.traj_map

    if episode_pos < 0 or episode_pos >= len(traj_map):
        raise IndexError(
            f"episode {episode_pos} out of range (0..{len(traj_map) - 1})"
        )
    agents = trained_agents(traj_map, episode_pos)
    if not agents:
        raise ValueError(
            f"episode {episode_pos} has no trained agents in traj_map"
        )

    rollout_update = update - 1
    export_dir = (
        dump_dir.parent.parent / "policy_exports" / f"u{rollout_update:05d}"
    )
    if not export_dir.is_dir():
        raise FileNotFoundError(
            f"rollout policy export missing: {export_dir}"
        )
    pol = _load_exported_policy(export_dir)
    log(
        f"[rollout] episode {episode_pos}: rollout policy u{rollout_update}, "
        f"trained agents = {agents}"
    )

    out: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {
        "update": update,
        "rollout_update": rollout_update,
        "episode_pos": episode_pos,
        "agents": agents,
        "stats": {},
    }

    ev = ds.episodes[episode_pos]
    for agent in agents:
        obs = np.asarray(ev.col(f"obs.{agent}"), dtype=np.float32)
        a_sampled = np.asarray(ev.col(f"actions.{agent}"), dtype=np.float32)
        T = obs.shape[0]
        pol.reset()
        a_det = np.stack(
            [np.asarray(pol.act(obs[t])[0], dtype=np.float32)
             for t in range(T)],
            axis=0,
        )  # (T, action_dim)
        out[f"a_det.{agent}"] = a_det

        # Eager stats so the coverage list and the page share numbers.
        eps_raw = _atanh(a_sampled) - _atanh(a_det)
        sat = np.abs(a_sampled) >= ATANH_CLAMP * 0.9999
        meta["stats"][agent] = rollout_stats(eps_raw, sat)
        meta.setdefault("n_frames", T)
        meta.setdefault("action_dim", int(a_det.shape[1]))
        log(
            f"[rollout]   {agent}: T={T} dim={a_det.shape[1]} "
            f"mean||eps||={meta['stats'][agent]['mean_norm']:.4f} "
            f"rho1={meta['stats'][agent]['rho1']:.3f}"
        )

    out_dir = dump_dir / "rollout" / f"episode_{episode_pos:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "det.npz", **out)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[rollout] wrote {out_dir}")
    return out_dir
