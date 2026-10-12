"""Held-out evaluation for SAC runs (P6-FINAL-1).

Loads a run's final checkpoint, exports a deterministic policy blueprint,
and evaluates it on held-out episode seeds that never overlapped training
eval seeds (``seed + seed_offset + i`` per the frozen P6 protocol).

Usage:

    PYTHONPATH=. python3 -m baseline.framework.sac.held_out_eval \
        --run baseline/runs/<run> \
        [--checkpoint checkpoints/checkpoint_sXXXXXXXX | --policy policy] \
        [--episodes 64] [--seed-offset 900000] [--workers 32] \
        [--out held_out_eval.json]

By default the newest checkpoint's actor is used. ``--policy`` instead
evaluates an exported policy dir (e.g. the run's ``policy/`` best-of-run
artifact — R2-W51, the same model.pt that would ship). The result JSON
records the evaluated artifact's identity (kind/fingerprint/hash), the
exact episode seeds, per-episode/per-agent detail rows, and the code
version — the verdict is auditable down to a single agent's episode.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch


def _load_experiment(run_dir: Path):
    """Rebuild the experiment with the exact kwargs recorded in config.json."""
    from baseline.experiments_sac import get_sac_experiment

    cfg = json.loads((run_dir / "config.json").read_text())
    exp = cfg["experiment"]
    cp = exp["common_params"]
    sp = exp["sac_params"]
    knobs = exp.get("knobs", {})
    kwargs: Dict[str, Any] = {
        "seed": cp["seed"],
        "target_entropy": sp["target_entropy"],
        "q_layer_norm": sp["q_layer_norm"],
        "reward_scale": sp["reward_scale"],
        "q_hidden_dim": sp["q_hidden_dim"],
    }
    kwargs.update(knobs)
    return get_sac_experiment(exp["name"], **kwargs), cfg


def _newest_checkpoint(run_dir: Path) -> Path:
    """Newest *complete* checkpoint bundle (manifest present, no tmp dirs)."""
    cks = sorted(
        p for p in (run_dir / "checkpoints").iterdir()
        if p.is_dir()
        and p.name.startswith("checkpoint_s")
        and "." not in p.name[len("checkpoint_s"):]  # skip atomic-save tmp dirs
        and (p / "manifest.json").exists()
    )
    if not cks:
        raise FileNotFoundError(
            f"no complete checkpoints under {run_dir}/checkpoints"
        )
    return cks[-1]


def _resolve_artifact_path(path: Path, run_dir: Path) -> Path:
    """Resolve a CLI artifact path, preferring run-relative for relatives.

    A bare relative path can accidentally match an unrelated dir under
    the CWD (the repo's own ``policy/`` did, hiding the intended
    ``<run>/policy``) — run-relative wins when it exists.
    """
    if path.is_absolute():
        return path
    run_rel = run_dir / path
    if run_rel.exists() or not path.exists():
        return run_rel
    return path


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_version(run_dir: Path) -> Dict[str, Any]:
    snap = run_dir / "code_snapshot.json"
    if not snap.exists():
        return {"available": False}
    data = json.loads(snap.read_text(encoding="utf-8"))
    return {
        "available": True,
        "branch": data.get("branch"),
        "commit": data.get("commit") or data.get("commit_hash"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument(
        "--checkpoint", type=Path, default=None,
        help="checkpoint dir; relative paths resolve under --run",
    )
    ap.add_argument(
        "--policy", type=Path, default=None,
        help="exported policy dir (e.g. <run>/policy) — evaluates the "
             "published best-of-run artifact instead of a checkpoint",
    )
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--seed-offset", type=int, default=900_000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    if args.policy and args.checkpoint:
        ap.error("--policy and --checkpoint are mutually exclusive")

    experiment, cfg = _load_experiment(run_dir)
    train_seed = int(cfg["experiment"]["common_params"]["seed"])
    base_seed = train_seed + args.seed_offset

    policy: Dict[str, Any]
    if args.policy:
        # Evaluate the published artifact itself — the same model.pt the
        # run's policy/ dir ships, not a checkpoint re-export.
        from envs.framework.policy import PolicyBlueprint

        policy_dir = _resolve_artifact_path(Path(args.policy), run_dir)
        bp = PolicyBlueprint.load(policy_dir / "policy_blueprint.yaml")
        meta_path = policy_dir / "meta.json"
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists() else {}
        )
        model_path = policy_dir / "model.pt"
        policy = {
            "kind": meta.get("kind", "policy_export"),
            "artifact": str(policy_dir),
            "fingerprint": meta.get("policy_fingerprint"),
            "model_sha256": _sha256(model_path) if model_path.exists() else None,
            "eval_origin": meta.get("eval"),
        }
        jobs = experiment.build_jobs(
            bp, base_seed, args.episodes,
            collection_round=0, run_id=f"{run_dir.name}_heldout",
            deterministic=True,
        )
    else:
        if args.checkpoint:
            ckpt = _resolve_artifact_path(Path(args.checkpoint), run_dir)
        else:
            ckpt = _newest_checkpoint(run_dir)

        from .checkpoint import load_checkpoint_bundle

        device = torch.device("cpu")
        bundle = load_checkpoint_bundle(ckpt, warm_start=True)
        actor = experiment.build_actor(device)
        actor.load_state_dict(bundle.trainer_state["actor_state_dict"])
        actor.eval()
        policy = {
            "kind": "checkpoint_actor",
            "artifact": ckpt.name,
            "fingerprint": (
                actor.policy_fingerprint()
                if hasattr(actor, "policy_fingerprint") else None
            ),
        }

        with tempfile.TemporaryDirectory() as tmp:
            bp = actor.to_blueprint(tmp, stochastic=False)
            jobs = experiment.build_jobs(
                bp, base_seed, args.episodes,
                collection_round=0, run_id=f"{run_dir.name}_heldout",
                deterministic=True,
            )

    from .collection import create_rollouter

    with create_rollouter(num_workers=args.workers) as rollouter:
        episodes = rollouter.collect(jobs)
    result = experiment.on_eval(episodes, env_step=0)

    # Evidence rows (R2-W50): per-episode/per-agent detail behind the
    # aggregate — absent for experiments without eval_episode_details.
    details = experiment.eval_episode_details(episodes)
    episode_rows: Any = details if details else (
        "unavailable — experiment does not implement eval_episode_details"
    )

    out = {
        "run": run_dir.name,
        "experiment": cfg["experiment"]["name"],
        "policy": policy,
        "train_seed": train_seed,
        "held_out_base_seed": base_seed,
        "episode_seeds": sorted({int(ep.episode_seed) for ep in episodes}),
        "episodes_requested": args.episodes,
        "episodes_collected": len(episodes),
        "env_blueprint_hashes": sorted(
            {str(ep.blueprint_hash) for ep in episodes}
        ),
        # Verdict counts agents (2 per episode); same-episode agents are
        # correlated, not independent samples — report honestly.
        "denominator": "agents",
        "agent_rows": len(details) if details else None,
        "correlation_note": (
            "both agents share each episode; agent rows are not "
            "statistically independent"
        ),
        "eval": result.get("info", {}),
        "is_new_best": bool(result.get("is_new_best", False)),
        "episode_details": episode_rows,
        "code_version": _code_version(run_dir),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False, default=str)
    print(text)
    out_path = Path(args.out) if args.out else run_dir / "held_out_eval.json"
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"[held_out_eval] written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
