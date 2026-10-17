#!/usr/bin/env python3
"""Render one eval video per skill with the latest trained policy.

Both robots run the *same* skill in each video (consistent with
training episodes).  Videos land in a timestamped subdir of
``baseline/humanoid21/hierarchy/videos/``.

Usage:
    PYTHONPATH=/data1/mono/things/combatbench \
      python3 -B baseline/humanoid21/hierarchy/render_skill_videos.py \
        [--run-dir baseline/runs/skill_train_s2b_u1635d] \
        [--export baseline/runs/.../policy_exports/u03721] \
        [--out videos_dir] [--seed 1234] [--skills jab_l,jab_r]

``--run-dir`` picks the newest ``policy_exports/u*`` automatically;
``--export`` pins an explicit export dir.  Default run dir = newest
``skill_train_*`` run under baseline/runs/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from baseline.experiments_ppo.exp_skill_train_mbs import (  # noqa: E402
    SKILLS,
    SkillTrainMBS,
    _skill_len,
    _skill_rows,
)

VIDEOS_ROOT = Path(__file__).resolve().parent / "videos"
RUNS_ROOT = REPO / "baseline" / "runs"


def _latest_export(run_dir: Path) -> Path:
    """Newest ``policy_exports/uNNNNN`` under a run dir."""
    exp_root = run_dir / "policy_exports"
    cands = sorted(
        (p for p in exp_root.glob("u*/MANIFEST.json")),
        key=lambda p: int(p.parent.name[1:]),
    )
    if not cands:
        raise SystemExit(f"no policy exports under {exp_root}")
    return cands[-1].parent


def _latest_run() -> Path:
    cands = sorted(
        (p for p in RUNS_ROOT.glob("skill_train_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not cands:
        raise SystemExit(f"no skill_train_* run under {RUNS_ROOT}")
    return cands[-1]


def _write_policy_bp(export_dir: Path, dest: Path) -> None:
    manifest = json.loads((export_dir / "MANIFEST.json").read_text())
    cls = f"file:{export_dir / 'policy.py'}:{manifest['exported_class']}"
    dest.write_text(f"version: 1\ncls: {cls}\nconfig: {{}}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--export", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: videos/<timestamp>)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skills", type=str, default=None,
                    help="comma list; default: all skills")
    ap.add_argument("--distance", type=float, default=None,
                    help="initial_distance (default: uniform like eval)")
    args = ap.parse_args()

    run_dir = args.run_dir or _latest_run()
    export_dir = args.export or _latest_export(run_dir)
    out_dir = args.out or (
        VIDEOS_ROOT / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    skills = (args.skills.split(",") if args.skills else list(SKILLS))
    exp = SkillTrainMBS()
    rng = __import__("numpy").random.default_rng(args.seed)

    pol_a = out_dir / "policy_a.yaml"
    pol_b = out_dir / "policy_b.yaml"
    _write_policy_bp(export_dir, pol_a)
    _write_policy_bp(export_dir, pol_b)

    print(f"[videos] run={run_dir.name} export={export_dir.name} "
          f"out={out_dir}", flush=True)

    ok, fail = 0, 0
    for i, skill in enumerate(skills):
        fam = SKILLS[skill]["fam"]
        n_steps = _skill_len(skill)
        env_pb = SkillTrainMBS._fam_env_pb(fam).materialize(
            max_steps=n_steps)
        env_yaml = out_dir / f"{skill}_env.yaml"
        env_pb.save(env_yaml)

        dist = (args.distance if args.distance is not None else
                float(rng.uniform(exp.init_distance_min,
                                  exp.init_distance_max)))
        rows = _skill_rows(skill)
        options = {
            "initial_distance": dist,
            "skills": {"robot_a": skill, "robot_b": skill},
            "move_schedule": {"robot_a": rows, "robot_b": rows},
        }
        opts_json = out_dir / f"{skill}_options.json"
        opts_json.write_text(json.dumps(options))

        mp4 = out_dir / f"{skill}.mp4"
        log = out_dir / f"{skill}.log"
        with open(log, "w") as lf:
            proc = subprocess.run(
                [sys.executable, "-m", "envs.framework.round_runner",
                 "--env-blueprint", str(env_yaml),
                 "--policy-a-blueprint", str(pol_a),
                 "--policy-b-blueprint", str(pol_b),
                 "--video", str(mp4),
                 "--seed", str(args.seed + i),
                 "--options-json", str(opts_json)],
                cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
        if proc.returncode == 0 and mp4.exists():
            ok += 1
            print(f"  [ok] {skill:<10s} -> {mp4.name}", flush=True)
        else:
            fail += 1
            print(f"  [FAIL] {skill:<10s} (see {log.name})", flush=True)

    (out_dir / "MANIFEST.json").write_text(json.dumps({
        "run_dir": str(run_dir), "export_dir": str(export_dir),
        "skills": skills, "seed": args.seed,
    }, indent=2))
    print(f"[videos] done: {ok} ok, {fail} failed -> {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
