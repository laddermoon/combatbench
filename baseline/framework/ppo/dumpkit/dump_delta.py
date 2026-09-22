"""Policy-drift delta diagnostic (stage-0, diagnostics only).

See ``baseline/framework/ppo/TODO_reference_policy_delta_exploration.md``.

For a dump captured at update ``u``, the rollout policy is
``policy_exports/u{u:05d}`` (exported at the start of update ``u``).
For each generation ``g = 1..K`` we load ``policy_exports/u{u-g:05d}``
and evaluate the deterministic ``act()`` of every generation's policy
on the episode's stored observations.  The per-frame action-space
displacement ``Δ_g(t) = a_0(t) - a_g(t)`` is the quantity of interest —
the action-space drift that update ``u-g .. u`` of training produced,
evaluated on the states the current policy actually visited.

Only **trained** agents are evaluated: an agent counts as trained iff
``traj_map`` contains trajectories sourced from ``(episode, agent)`` —
trajectories are the training units, so this is experiment-agnostic
and does not assume self-play.

Episode actions are deliberately unused — they are stochastic samples
(noise dominates drift); both sides of Δ must be deterministic.

Output layout::

    <dump_dir>/delta/episode_{pos:05d}/
        delta.npz   # actions.{agent}: (G+1, T, action_dim)
                    # gen_updates: (G+1,) int — row 0 is the rollout
                    #   policy, rows 1.. are the reference generations
        meta.json   # {update, episode_pos, agents, gen_updates,
                    #  missing_updates, n_frames, action_dim}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

MAX_GENS = 10


def _load_exported_policy(export_dir: Path):
    """Load a ``policy_exports/uNNNNN/`` directory as a Policy."""
    from envs.framework.policy import PolicyBlueprint

    manifest_path = export_dir / "MANIFEST.json"
    policy_py = export_dir / "policy.py"
    model_pt = export_dir / "model.pt"
    if not manifest_path.exists() or not policy_py.exists() or not model_pt.exists():
        raise FileNotFoundError(
            f"incomplete policy export at {export_dir} "
            "(need MANIFEST.json + policy.py + model.pt)"
        )
    exported_class = json.loads(manifest_path.read_text())["exported_class"]
    bp = PolicyBlueprint.from_dict({
        "cls": f"file:{policy_py}:{exported_class}",
        "config": {"model_path": str(model_pt)},
    })
    return bp.build()


def trained_agents(traj_map: List[Dict[str, Any]], ep_pos: int) -> List[str]:
    """Agent ids that produced trajectories in this episode."""
    ep = traj_map[ep_pos]
    return [t["agent_id"] for t in ep.get("trajectories", [])]


def compute_delta(
    dump_dir: Path,
    episode_pos: int,
    gens: int = 3,
    *,
    log=print,
) -> Path:
    """Compute per-generation deterministic actions for one episode.

    Writes ``<dump_dir>/delta/episode_{pos:05d}/`` and returns that dir.
    """
    from baseline.framework.ppo.dumpkit.frame_access import DumpDataset

    dump_dir = Path(dump_dir)
    if not 1 <= gens <= MAX_GENS:
        raise ValueError(f"gens must be in [1, {MAX_GENS}], got {gens}")

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
    log(f"[delta] episode {episode_pos}: trained agents = {agents}")

    run_dir = dump_dir.parent.parent
    exports_root = run_dir / "policy_exports"

    # Load policy generations: gen 0 = rollout policy (u), then u-1..u-gens.
    # Missing exports are skipped but recorded — never silently dropped.
    policies: List[Any] = []
    gen_updates: List[int] = []
    missing: List[int] = []
    for g in range(0, gens + 1):
        u_ref = update - g
        export_dir = exports_root / f"u{u_ref:05d}"
        if u_ref < 1 or not export_dir.is_dir():
            missing.append(u_ref)
            continue
        policies.append(_load_exported_policy(export_dir))
        gen_updates.append(u_ref)
    if len(policies) < 2:
        raise FileNotFoundError(
            f"need at least 2 policy generations; found {len(policies)} "
            f"(missing: {missing})"
        )
    log(
        f"[delta] update={update} gens loaded: {gen_updates}"
        + (f"  missing: {missing}" if missing else "")
    )

    out: Dict[str, np.ndarray] = {"gen_updates": np.asarray(gen_updates, dtype=np.int64)}
    meta: Dict[str, Any] = {
        "update": update,
        "episode_pos": episode_pos,
        "gen_updates": gen_updates,
        "missing_updates": missing,
        "agents": agents,
    }

    ev = ds.episodes[episode_pos]
    for agent in agents:
        obs_arr = ev.col(f"obs.{agent}")
        if obs_arr is None:
            raise KeyError(f"obs.{agent} not in episodes.npz")
        obs = np.asarray(obs_arr, dtype=np.float32)
        T, obs_dim = obs.shape
        per_gen = []
        for pol in policies:
            pol.reset()
            rows = [
                np.asarray(pol.act(obs[t])[0], dtype=np.float32)
                for t in range(T)
            ]
            per_gen.append(np.stack(rows, axis=0))  # (T, action_dim)
        actions = np.stack(per_gen, axis=0)  # (G+1, T, action_dim)
        act_dim = actions.shape[2]
        out[f"actions.{agent}"] = actions
        log(f"[delta]   {agent}: T={T} obs_dim={obs_dim} action_dim={act_dim}")
        meta.setdefault("n_frames", T)
        meta.setdefault("action_dim", act_dim)

    out_dir = dump_dir / "delta" / f"episode_{episode_pos:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "delta.npz", **out)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[delta] wrote {out_dir}")
    return out_dir
