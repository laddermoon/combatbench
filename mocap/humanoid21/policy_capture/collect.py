"""Batch driver for policy-rollout motion capture.

Two modes:

1. **Random mode** (default): the env blueprint's RandomPushPlugin applies
   random-direction pushes; ``--push-force/--push-duration/--push-interval``
   override its ranges without editing the yaml.

2. **Directed sweep mode** (``--sweep``): each episode gets ONE deterministic
   constant push — fixed force magnitude, fixed world-frame azimuth, fixed
   duration — applied to both robots at ``--push-at-step``.  The three axes
   are swept as a grid (one episode per combination)::

       PYTHONPATH=. python3 -m mocap.humanoid21.policy_capture.collect \
           --env-blueprint .../push_probe_base_env.yaml \
           --policy-a <wrapped_policy.yaml> \
           --sweep --sweep-force 70,95,120 \
           --sweep-azimuth 0,45,90,135,180,225,270,315 \
           --sweep-duration 0.5,1.0,1.5,2.0 \
           --push-at-step 60 \
           --out-dir /data1/dev/policy_capture/<run> --render 4

Random-mode usage::

    PYTHONPATH=. python3 -m mocap.humanoid21.policy_capture.collect \
        --env-blueprint .../push_probe_base_env.yaml \
        --policy-a <blueprint.yaml> \
        --episodes 8 --seed-base 1000 \
        --out-dir /data1/dev/policy_capture/<run_name> --render 2

Outputs per episode ``ep_<seed>_{robot_a,robot_b}.npz`` +
``ep_<seed>_meta.json`` (see recorder.py docstring), plus ``ep_<seed>.mp4``
for rendered episodes.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
from pathlib import Path

from envs.framework.blueprint import ClassSpec, EnvBlueprint
from envs.framework.common_plugins import VideoRecorderPlugin
from envs.framework.policy import PolicyBlueprint
from envs.framework.round_runner import RoundRunner

from mocap.humanoid21.policy_capture.recorder import TrajectoryRecorder

_DIRECTED_CLS = "envs.humanoid21.disturbance_plugins:DirectedPushPlugin"
_RANDOM_CLS_SUFFIX = "RandomPushPlugin"


def _parse_range(spec: str) -> tuple[float, float]:
    lo, hi = spec.split(":", 1)
    return float(lo), float(hi)


def _parse_list(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _patch_random_push(env_bp: EnvBlueprint, args) -> EnvBlueprint:
    """Override RandomPushPlugin knobs without touching the yaml file."""
    if args.push_force is None and args.push_duration is None \
            and args.push_interval is None:
        return env_bp
    plugins = []
    for spec in env_bp.plugins:
        if not spec.cls.endswith(_RANDOM_CLS_SUFFIX):
            plugins.append(spec)
            continue
        cfg = dict(spec.config)
        if args.push_force is not None:
            lo, hi = _parse_range(args.push_force)
            cfg["force_magnitude_min"], cfg["force_magnitude_max"] = lo, hi
        if args.push_duration is not None:
            cfg["push_duration_steps"] = int(args.push_duration)
        if args.push_interval is not None:
            lo, hi = _parse_range(args.push_interval)
            cfg["min_interval"], cfg["max_interval"] = int(lo), int(hi)
        plugins.append(dataclasses.replace(spec, config=cfg))
    return dataclasses.replace(env_bp, plugins=tuple(plugins))


def _directed_env(env_bp: EnvBlueprint, force: float, azimuth: float,
                  duration_steps: int, start_step: int,
                  target_body: str) -> EnvBlueprint:
    """Replace push plugins with a fixed DirectedPushPlugin per robot."""
    plugins = [
        spec for spec in env_bp.plugins
        if not spec.cls.endswith(_RANDOM_CLS_SUFFIX)
        and not spec.cls.endswith("DirectedPushPlugin")
    ]
    for rid in ("robot_a", "robot_b"):
        plugins.append(ClassSpec(cls=_DIRECTED_CLS, config={
            "target_robot": rid,
            "target_body": target_body,
            "start_step": start_step,
            "duration_steps": duration_steps,
            "force_magnitude": force,
            "azimuth_deg": azimuth,
        }))
    return dataclasses.replace(env_bp, plugins=tuple(plugins))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-blueprint", required=True)
    ap.add_argument("--policy-a", required=True, help="Policy blueprint yaml for robot_a")
    ap.add_argument("--policy-b", default=None, help="Blueprint yaml for robot_b (default: same as A)")
    ap.add_argument("--episodes", type=int, default=8,
                    help="Random mode: episode count. Sweep mode: ignored "
                         "(one episode per grid combination).")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--seed-stride", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--render", type=int, default=0,
                    help="Render mp4 for the first N episodes/combos")
    # random-mode overrides
    ap.add_argument("--push-force", default=None, metavar="MIN:MAX",
                    help="Random mode: override push force range (N)")
    ap.add_argument("--push-duration", type=int, default=None,
                    help="Random mode: override push length (action steps; 1 = 50ms)")
    ap.add_argument("--push-interval", default=None, metavar="MIN:MAX",
                    help="Random mode: override push interval range (action steps)")
    # directed sweep mode
    ap.add_argument("--sweep", action="store_true",
                    help="Directed sweep mode: one constant push per episode")
    ap.add_argument("--sweep-force", default="100",
                    help="Comma list of force magnitudes (N)")
    ap.add_argument("--sweep-azimuth", default="0,45,90,135,180,225,270,315",
                    help="Comma list of world-frame azimuth angles (deg)")
    ap.add_argument("--sweep-duration", default="0.5,1.0,1.5,2.0",
                    help="Comma list of push durations (seconds)")
    ap.add_argument("--push-at-step", type=int, default=60,
                    help="Action step where the directed push starts "
                         "(default 60 = 3s settle)")
    ap.add_argument("--push-body", default="torso",
                    help="Target body for directed pushes")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override episode length (action steps)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_bp = EnvBlueprint.load(args.env_blueprint)
    if args.max_steps is not None:
        base_bp = dataclasses.replace(base_bp, max_steps=int(args.max_steps))
    policy_a = PolicyBlueprint.load(args.policy_a).build()
    policy_b = (PolicyBlueprint.load(args.policy_b).build()
                if args.policy_b else policy_a)

    provenance = {
        "env_blueprint": str(Path(args.env_blueprint).resolve()),
        "policy_a": str(Path(args.policy_a).resolve()),
        "policy_b": str(Path(args.policy_b).resolve()) if args.policy_b else None,
    }

    fps = 500.0 / base_bp.phy_steps_per_action  # humanoid21 DT=0.002

    # ---- build the episode job list ----
    jobs = []
    if args.sweep:
        forces = _parse_list(args.sweep_force)
        azimuths = _parse_list(args.sweep_azimuth)
        durations_s = _parse_list(args.sweep_duration)
        for force, az, dur_s in itertools.product(forces, azimuths, durations_s):
            dur_steps = max(1, int(round(dur_s * fps)))
            jobs.append({
                "env_bp": _directed_env(base_bp, force, az, dur_steps,
                                        args.push_at_step, args.push_body),
                "meta": {"sweep": {
                    "force_n": force, "azimuth_deg": az,
                    "duration_s": dur_s, "duration_steps": dur_steps,
                    "start_step": args.push_at_step, "body": args.push_body,
                }},
            })
    else:
        env_bp = _patch_random_push(base_bp, args)
        for _ in range(args.episodes):
            jobs.append({"env_bp": env_bp, "meta": {"push_overrides": {
                "force": args.push_force, "duration": args.push_duration,
                "interval": args.push_interval,
            }}})

    print(f"[collect] {len(jobs)} episodes -> {out_dir}", flush=True)

    for i, job in enumerate(jobs):
        seed = args.seed_base + i * args.seed_stride
        video_plugin = None
        if i < args.render:
            video_plugin = VideoRecorderPlugin(
                fps=30, output_path=str(out_dir / f"ep_{seed}.mp4"))
        recorder = TrajectoryRecorder(
            output_dir=str(out_dir),
            phy_steps_per_action=base_bp.phy_steps_per_action,
            provenance=provenance,
            extra_meta=job["meta"],
        )
        with RoundRunner(
            blueprint=job["env_bp"],
            policy_a=policy_a,
            policy_b=policy_b,
            video_plugin=video_plugin,
            recorders=[recorder],
        ) as runner:
            result = runner.run(seed=seed)
        tag = job["meta"].get("sweep")
        desc = (f"force={tag['force_n']}N az={tag['azimuth_deg']}° "
                f"dur={tag['duration_s']}s") if tag else ""
        print(f"[{i+1}/{len(jobs)} seed={seed}] steps={result['steps']} "
              f"{desc} term={result['termination_reasons']}", flush=True)


if __name__ == "__main__":
    main()
