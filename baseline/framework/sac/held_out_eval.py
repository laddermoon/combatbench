"""Held-out evaluation for SAC runs (P6-FINAL-1).

Loads a run's final checkpoint, exports a deterministic policy blueprint,
and evaluates it on held-out episode seeds that never overlapped training
eval seeds (``seed + seed_offset + i`` per the frozen P6 protocol).

Usage:

    PYTHONPATH=. python3 -m baseline.framework.sac.held_out_eval \
        --run baseline/runs/<run> \
        [--checkpoint checkpoints/checkpoint_sXXXXXXXX] \
        [--episodes 64] [--seed-offset 900000] [--workers 32] \
        [--out held_out_eval.json]

By default the newest checkpoint in the run is used.  The result JSON is
printed to stdout and optionally written into the run directory.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument(
        "--checkpoint", type=Path, default=None,
        help="checkpoint dir; relative paths resolve under --run",
    )
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--seed-offset", type=int, default=900_000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_absolute() and not ckpt.exists():
            ckpt = run_dir / ckpt
    else:
        ckpt = _newest_checkpoint(run_dir)

    experiment, cfg = _load_experiment(run_dir)
    train_seed = int(cfg["experiment"]["common_params"]["seed"])
    base_seed = train_seed + args.seed_offset

    from .checkpoint import load_checkpoint_bundle

    device = torch.device("cpu")
    bundle = load_checkpoint_bundle(ckpt, warm_start=True)
    actor = experiment.build_actor(device)
    actor.load_state_dict(bundle.trainer_state["actor_state_dict"])
    actor.eval()

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

    out = {
        "run": run_dir.name,
        "checkpoint": ckpt.name,
        "experiment": cfg["experiment"]["name"],
        "train_seed": train_seed,
        "held_out_base_seed": base_seed,
        "episodes": args.episodes,
        "eval": result.get("info", {}),
        "is_new_best": bool(result.get("is_new_best", False)),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False, default=str)
    print(text)
    out_path = Path(args.out) if args.out else run_dir / "held_out_eval.json"
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"[held_out_eval] written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
