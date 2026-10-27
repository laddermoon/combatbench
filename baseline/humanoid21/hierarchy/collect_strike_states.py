"""Collect strike-start initial states from a trained guard policy.

Runs the S2 guard policy in the standup-guard env (random fallen start ->
stands -> jab_l guard at ~2 m separation) and records every frame where
BOTH robots are standing (torso z >= stand_h).  Saved core states are
directly re-injectable via ``mutator.set_core_state``.

v1: no extra disturbance (per user spec — pipeline bring-up first).

Usage:
    PYTHONPATH=. python3 baseline/humanoid21/hierarchy/collect_strike_states.py \
        --policy-yaml baseline/runs/hier_s2_guard_96d_s1/videos/video_policy_a.yaml \
        --episodes 48 \
        --out-dir baseline/humanoid21/hierarchy/assets/initial_states/strike_start_v1 \
        --merged strike_start_states.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-yaml", required=True,
                    help="Policy blueprint yaml (e.g. a run's "
                         "videos/video_policy_a.yaml)")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=70000)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--stand-h", type=float, default=1.10)
    ap.add_argument("--sample-per-ep", type=int, default=24,
                    help="Random states kept per episode in the merged set")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--merged", default="strike_start_states.npz")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))

    from envs.framework.blueprint import EnvBlueprint  # noqa: E402
    from envs.framework.policy import PolicyBlueprint  # noqa: E402
    from envs.framework.round_runner import RoundRunner  # noqa: E402
    from envs.framework.parameterized_blueprint import (  # noqa: E402
        ParameterizedEnvBlueprint,
    )
    from baseline.humanoid21.plugins.strike_state_recorder import (  # noqa: E402
        StrikeStartStateRecorder,
    )

    env_yaml = (root / "baseline" / "humanoid21" / "hierarchy" /
                "blueprints" / "hier_s1_guard_96d_env.yaml")
    env_bp = ParameterizedEnvBlueprint.load(env_yaml).materialize(
        max_steps=args.max_steps)

    pol_bp = PolicyBlueprint.load(args.policy_yaml)
    pol_a = pol_bp.build()
    pol_b = pol_bp.build()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = StrikeStartStateRecorder(str(out_dir), stand_h=args.stand_h)
    runner = RoundRunner(env_bp, pol_a, pol_b, recorders=[rec])

    for i in range(args.episodes):
        res = runner.run(seed=args.seed0 + i)
        print(f"[ep {i}] steps={res['steps']} "
              f"terms={res['termination_reasons']}", flush=True)
    runner.close()

    # ---- merge: concat all episode files, subsample per episode ----
    rng = np.random.default_rng(0)
    merged: dict[str, list] = {}
    steps_all: list = []
    metas = []
    for f in sorted(out_dir.glob("ep_*.npz")):
        d = np.load(f, allow_pickle=False)
        n = len(d["steps"])
        take = min(args.sample_per_ep, n)
        idx = np.sort(rng.choice(n, size=take, replace=False))
        steps_all.append(d["steps"][idx])
        for k in d.files:
            if k == "steps":
                continue
            if k == "meta_json":
                metas.append(json.loads(str(d["meta_json"])))
                continue
            merged.setdefault(k, []).append(d[k][idx])
    if not merged:
        print("no standing states collected", flush=True)
        sys.exit(1)
    payload = {k: np.concatenate(v) for k, v in merged.items()}
    payload["steps"] = np.concatenate(steps_all)
    payload["meta_json"] = np.asarray(json.dumps({
        "episodes": len(metas),
        "n_states": int(len(payload["steps"])),
        "policy_yaml": args.policy_yaml,
        "env": "hier_s1_guard_96d_env",
        "fields": ["root_pos", "root_rot", "root_vel_local",
                   "root_angular_vel_local", "joint_pos_norm",
                   "joint_vel_norm"],
        "note": "both-robot standing core states, re-injectable via "
                "mutator.set_core_state({robot: state_dict})",
    }))
    out_path = out_dir / args.merged
    np.savez(out_path, **payload)
    print(f"saved {len(payload['steps'])} states -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
