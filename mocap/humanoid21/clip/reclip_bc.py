"""Reclip→BC pretraining data pipeline.

Stage T1 samples complete CoreStates by running the frozen locomotion
policy inside ``CommandSimulator``.  Stage T3 solves one position-free
``pass`` TargetPath per sampled state.  Stage T4 filters the solver
manifest into accepted/rejected expert trajectories.
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as _R

from .core_clip import CONTROL_DT, CORE_FIELDS, SPEC_VERSION, CoreClip
from .objective_evaluator import ReclipObjectiveConfig
from .path_reclip import reclip_path_search
from .target_path import TargetPath

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY_EXPORT = (
    PROJECT_ROOT / "baseline" / "runs"
    / "train_skill_full_mbs_ppo_20261020_104315"
    / "policy_exports" / "u06689")
DEFAULT_TARGET_CLIP = (
    PROJECT_ROOT / "mocap" / "humanoid21" / "clip" / "data"
    / "reclip_bc" / "cmu_strike_target" / "target_clip.npz")

WARMUP_SKILLS = (
    "move_fwd", "move_back", "move_left", "move_right",
    "turn_left", "turn_right", "step",
)
_REPLAY_KEYS = (
    "joint_pos_err_mean_max", "joint_vel_err_mean_max",
    "root_pos_err_max", "root_rot_err_max",
    "root_vel_err_max", "root_angvel_err_max")
_WORKER_SIM = None


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"bad JSONL line {line_no} in {path}: {exc}") from exc
    return out


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _latest_results(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the last solver attempt for each ``init_id``.

    ``reclip_results.jsonl`` is an append-only attempt log.  Retried
    failures therefore appear multiple times; quality filtering and
    top-level summaries must report the final outcome per initial state,
    not inflate counts by retry attempts.
    """
    latest: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for i, row in enumerate(rows):
        key = row.get("init_id")
        key = f"__row_{i}" if key is None else str(key)
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[k] for k in order]


def _yaw_wxyz(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    r = _R.from_quat([q[1], q[2], q[3], q[0]])
    return float(r.as_euler("xyz")[2])


def load_export_policy(export_dir: Path):
    export_dir = Path(export_dir)
    manifest = json.loads(
        (export_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    cls = (f"file:{export_dir / 'policy.py'}:"
           f"{manifest['exported_class']}")
    from envs.framework.policy import PolicyBlueprint
    return PolicyBlueprint(cls=cls, config={}).build()


def warmup_schedule(skill: str, steps: int,
                    command_scale: float = 1.0) -> List[List[float]]:
    """Physical-unit v2 move_schedule row for one warmup skill."""
    from baseline.humanoid21.hierarchy.command_simulator import V_MAX, W_MAX
    n = max(1, int(steps))
    s = float(command_scale)
    moves = {
        "move_fwd": (V_MAX * s, 0.0),
        "move_back": (-V_MAX * s, 0.0),
        "move_left": (0.0, V_MAX * s),
        "move_right": (0.0, -V_MAX * s),
    }
    if skill in moves:
        return [[n, *moves[skill], 0.0]]
    if skill == "turn_left":
        return [[n, 0.0, 0.0, W_MAX * s]]
    if skill == "turn_right":
        return [[n, 0.0, 0.0, -W_MAX * s]]
    if skill == "step":
        return [[n, 0.0, 0.0, 0.0]]
    raise ValueError(f"unknown warmup skill {skill!r}")


def make_warmup_spec(index: int, base_seed: int,
                     skills: Sequence[str] = WARMUP_SKILLS,
                     steps_min: int = 0, steps_max: int = 30,
                     command_scale_min: float = 0.6,
                     command_scale_max: float = 1.0
                     ) -> Dict[str, Any]:
    """Deterministic per-index spec; safe for resume/regeneration."""
    if not skills:
        raise ValueError("skills must not be empty")
    if not (0 <= steps_min <= steps_max):
        raise ValueError("require 0 <= steps_min <= steps_max")
    rng = np.random.default_rng(int(base_seed) + int(index) * 7919)
    skill = str(skills[int(rng.integers(0, len(skills)))])
    steps = int(rng.integers(steps_min, steps_max + 1))
    scale = float(rng.uniform(command_scale_min, command_scale_max))
    return {
        "skill": skill,
        "steps": steps,
        "command_scale": scale,
        "schedule": warmup_schedule(skill, max(steps, 1), scale),
        "sim_seed": int(rng.integers(0, 2**31 - 1)),
        "policy_seed": int(rng.integers(0, 2**31 - 1)),
    }


def run_warmup_state(sim, policy, spec: Dict[str, Any],
                     robot: str = "robot_a",
                     sample_actions: bool = True,
                     phy_steps_per_action: int = 25
                     ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Run one CommandSimulator warmup and return (CoreState, obs57)."""
    other = "robot_b" if robot == "robot_a" else "robot_a"
    sim.reset(seed=int(spec["sim_seed"]),
              options={"move_schedule": {robot: spec["schedule"]}})
    if policy is not None and hasattr(policy, "reset"):
        policy.reset(seed=int(spec["policy_seed"]))
    zero = np.zeros(21, np.float32)
    obs57 = np.asarray(sim.get_observation()[robot], np.float32)[:57]
    for _ in range(int(spec["steps"])):
        obs = np.asarray(sim.get_observation()[robot], np.float32)
        if policy is None:
            action = zero.copy()
        elif sample_actions and hasattr(policy, "sample"):
            action, _ = policy.sample(obs)
        else:
            action, _ = policy.act(obs)
        sim.set_action({robot: np.asarray(action, np.float32),
                        other: zero})
        for _ in range(int(phy_steps_per_action)):
            sim.physical_step()
        obs57 = np.asarray(sim.get_observation()[robot],
                           np.float32)[:57]
    return sim.get_core_state()[robot], obs57


def _state_summary(state: Dict[str, np.ndarray],
                   obs57: np.ndarray) -> Dict[str, Any]:
    return {
        "root_z": float(state["root_pos"][2]),
        "root_yaw": _yaw_wxyz(state["root_rot"]),
        "root_vel_norm": float(np.linalg.norm(
            state["root_vel_local"])),
        "joint_pos_abs_mean": float(np.abs(
            state["joint_pos_norm"]).mean()),
        "foot_contact": np.asarray(obs57[-2:], np.float32).tolist(),
    }


def _write_init_shard(out_dir: Path, shard_idx: int,
                      states: Sequence[Dict[str, np.ndarray]],
                      metas: Sequence[Dict[str, Any]]) -> Path:
    shard_rel = Path("init_states") / f"shard_{shard_idx:06d}.npz"
    shard_path = out_dir / shard_rel
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    clip = CoreClip.from_states(states, {
        "spec_version": SPEC_VERSION,
        "clip_id": f"warmup_init_bank_{shard_idx:06d}",
        "source": {
            "type": "generated",
            "generator": "frozen_warmup_policy",
        },
        "semantics": "independent_core_states_not_a_trajectory",
        "state_count": len(states),
        "warmup_specs": metas,
    })
    clip.save(shard_path)
    return shard_rel


def summarize_init_manifest(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_skill: Dict[str, int] = {}
    root_z, yaw, speed, steps = [], [], [], []
    for r in records:
        spec = r.get("warmup", {})
        by_skill[spec.get("skill", "unknown")] = \
            by_skill.get(spec.get("skill", "unknown"), 0) + 1
        summ = r.get("state_summary", {})
        if "root_z" in summ:
            root_z.append(float(summ["root_z"]))
            yaw.append(float(summ["root_yaw"]))
            speed.append(float(summ["root_vel_norm"]))
        steps.append(int(spec.get("steps", 0)))
    return {
        "count": len(records),
        "by_skill": dict(sorted(by_skill.items())),
        "steps": {"min": min(steps) if steps else None,
                  "max": max(steps) if steps else None,
                  "mean": float(np.mean(steps)) if steps else None},
        "root_z": {"min": min(root_z) if root_z else None,
                   "max": max(root_z) if root_z else None,
                   "mean": float(np.mean(root_z)) if root_z else None},
        "root_yaw": {"min": min(yaw) if yaw else None,
                     "max": max(yaw) if yaw else None,
                     "mean": float(np.mean(yaw)) if yaw else None},
        "root_vel_norm": {
            "min": min(speed) if speed else None,
            "max": max(speed) if speed else None,
            "mean": float(np.mean(speed)) if speed else None},
    }


def generate_init_state_bank(out_dir: Path, count: int,
                             base_seed: int,
                             policy=None,
                             skills: Sequence[str] = WARMUP_SKILLS,
                             steps_min: int = 0,
                             steps_max: int = 30,
                             shard_size: int = 64,
                             robot: str = "robot_a",
                             sample_actions: bool = True,
                             command_scale_min: float = 0.6,
                             command_scale_max: float = 1.0,
                             sim=None) -> Dict[str, Any]:
    """Generate/resume a bank of independent warmup CoreStates."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "init_manifest.jsonl"
    existing = _read_jsonl(manifest_path)
    done = {str(r.get("init_id")) for r in existing}
    tasks = [i for i in range(int(count))
             if f"init_{i:06d}" not in done]
    own = sim is None
    if own:
        from baseline.humanoid21.hierarchy.command_simulator import (
            CommandSimulator)
        sim = CommandSimulator(cmd_version="v2",
                               strike_dyn_target=True)
    try:
        shard_idx = len(list((out_dir / "init_states").glob("*.npz"))) \
            if (out_dir / "init_states").exists() else 0
        for chunk_start in range(0, len(tasks), int(shard_size)):
            chunk = tasks[chunk_start:chunk_start + int(shard_size)]
            states, metas, rows = [], [], []
            for i in chunk:
                spec = make_warmup_spec(
                    i, base_seed, skills=skills,
                    steps_min=steps_min, steps_max=steps_max,
                    command_scale_min=command_scale_min,
                    command_scale_max=command_scale_max)
                state, obs57 = run_warmup_state(
                    sim, policy, spec, robot=robot,
                    sample_actions=sample_actions)
                states.append(state)
                metas.append(spec)
                rows.append({
                    "init_id": f"init_{i:06d}",
                    "sample_index": i,
                    "robot": robot,
                    "shard_index": shard_idx,
                    "frame_index": len(states) - 1,
                    "warmup": spec,
                    "state_summary": _state_summary(state, obs57),
                })
            shard_rel = _write_init_shard(
                out_dir, shard_idx, states, metas)
            for row in rows:
                row["shard_file"] = str(shard_rel)
            _append_jsonl(manifest_path, rows)
            print(f"[init-states] shard {shard_idx}: "
                  f"{len(states)} states -> {shard_rel}")
            shard_idx += 1
    finally:
        if own:
            sim.close()
    all_records = _read_jsonl(manifest_path)
    summary = summarize_init_manifest(all_records)
    summary.update({
        "base_seed": int(base_seed),
        "count_requested": int(count),
        "robot": robot,
        "skills": list(skills),
        "steps_min": int(steps_min),
        "steps_max": int(steps_max),
        "command_scale_min": float(command_scale_min),
        "command_scale_max": float(command_scale_max),
        "sample_actions": bool(sample_actions),
    })
    (out_dir / "init_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False)
        + "\n", encoding="utf-8")
    return summary


def load_init_records(init_dir: Path,
                      offset: int = 0,
                      limit: Optional[int] = None,
                      shard_index: int = 0,
                      num_shards: int = 1
                      ) -> List[Dict[str, Any]]:
    init_dir = Path(init_dir)
    rows = _read_jsonl(init_dir / "init_manifest.jsonl")
    seen = set()
    out = []
    for row_no, row in enumerate(rows):
        if row_no % int(num_shards) != int(shard_index):
            continue
        init_id = str(row.get("init_id"))
        if init_id in seen:
            continue
        seen.add(init_id)
        r = dict(row)
        r["init_dir"] = str(init_dir)
        out.append(r)
    if offset:
        out = out[int(offset):]
    if limit is not None:
        out = out[:int(limit)]
    return out


def load_bank_state(record: Dict[str, Any]) -> Dict[str, np.ndarray]:
    shard = Path(record["init_dir"]) / record["shard_file"]
    return CoreClip.load(shard).state(int(record["frame_index"]))


def target_pose(target_clip: Path, target_frame: int = -1
                ) -> Tuple[CoreClip, Dict[str, np.ndarray], int]:
    clip = CoreClip.load(target_clip)
    frame = clip.T - 1 if int(target_frame) < 0 else int(target_frame)
    return clip, clip.state(frame), frame


def build_target_path(start: Dict[str, np.ndarray],
                      target: Dict[str, np.ndarray],
                      source_clip_id: str,
                      s_ext: float = 0.1) -> TargetPath:
    return TargetPath.from_pose_pair(
        start, target, source_clip_id=source_clip_id,
        s_ext=s_ext, root_pos_mode="free")


def _worker_sim():
    global _WORKER_SIM
    if _WORKER_SIM is None:
        from .rederive_action import _mk_sim
        _WORKER_SIM = _mk_sim()
        atexit.register(_WORKER_SIM.close)
    return _WORKER_SIM


def solve_init_record(record: Dict[str, Any],
                      state: Dict[str, np.ndarray],
                      target: Dict[str, np.ndarray],
                      target_clip_id: str,
                      target_frame: int,
                      out_dir: Path,
                      budget_frames: int = 6,
                      ds_max: float = 0.5,
                      s_ext: float = 0.1,
                      post_frames: int = 1,
                      sweeps: int = 1,
                      n_samples: int = 8,
                      n_iters: int = 1,
                      seeds: Sequence[int] = (0,),
                      robot: str = "robot_a",
                      save_all: bool = False) -> Dict[str, Any]:
    """Solve one init record and save a replay-valid expert clip."""
    t0 = time.perf_counter()
    init_id = str(record["init_id"])
    sim = _worker_sim()
    sim.reset(seed=int(record.get("sample_index", 0)) + 1000003)
    path = build_target_path(
        state, target, source_clip_id=target_clip_id, s_ext=s_ext)
    objective = ReclipObjectiveConfig(
        kind="pass", time_budget_frames=int(budget_frames),
        ds_max=float(ds_max))
    try:
        res = reclip_path_search(
            state, path, objective, init_actions=None,
            seeds=seeds, budget_frames=budget_frames, robot=robot,
            sweeps=sweeps, n_samples=n_samples, n_iters=n_iters,
            post_frames=post_frames,
            clip_id=f"{init_id}_expert", sim=sim)
        clip_path = None
        status = str(res.evaluation.status)
        if save_all or status == "completed":
            clip_rel = Path("expert_clips") / f"{init_id}.npz"
            clip_path = Path(out_dir) / clip_rel
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            res.clip.meta.setdefault("reclip_bc", {})
            res.clip.meta["reclip_bc"].update({
                "init_id": init_id,
                "init_record": record,
                "target_clip_id": target_clip_id,
                "target_frame": int(target_frame),
                "policy_contract": "obs57 + zeros9 -> action21",
            })
            res.clip.save(clip_path)
            clip_path = str(clip_rel)
        a = np.asarray(res.actions, np.float32)
        root0 = np.asarray(state["root_pos"], np.float32)
        root1 = np.asarray(res.clip.states["root_pos"][-1], np.float32)
        return {
            "init_id": init_id,
            "init_record": record,
            "status": status,
            "event_frame": res.evaluation.event_frame,
            "event_time": (None if res.evaluation.event_frame is None
                           else float(res.evaluation.event_frame
                                      * CONTROL_DT)),
            "solver_score": float(res.evaluation.score),
            "solver_seed": int(res.solver_seed),
            "s_final": float(res.evaluation.progress[-1]),
            "s_max_seen": float(res.evaluation.progress.max()),
            "path_error_mean": float(
                res.evaluation.metrics["path_error_mean"]),
            "path_error_max": float(
                res.evaluation.metrics["path_error_max"]),
            "terminal_pose_error": float(
                res.evaluation.metrics["terminal_pose_error"]),
            "terminal_vel_error": float(
                res.evaluation.metrics["terminal_vel_error"]),
            "fall_count": int(res.evaluation.metrics["fall_count"]),
            "fall_frame": res.evaluation.metrics["fall_frame"],
            "replay_error": res.replay_metrics,
            "action_abs_mean": (None if len(a) == 0 else
                                float(np.abs(a).mean())),
            "action_saturation_frac": (None if len(a) == 0 else
                                       float((np.abs(a) >= 0.99)
                                             .mean())),
            "root_pos_drift": float(np.linalg.norm(root1 - root0)),
            "root_z_delta": float(root1[2] - root0[2]),
            "clip_path": clip_path,
            "clip_T": int(res.clip.T),
            "root_pos_mode": path.root_pos_mode,
            "target_clip_id": target_clip_id,
            "target_frame": int(target_frame),
            "budget_frames": int(budget_frames),
            "sweeps": int(sweeps),
            "n_samples": int(n_samples),
            "n_iters": int(n_iters),
            "post_frames": int(post_frames),
            "elapsed_s": float(time.perf_counter() - t0),
        }
    except Exception as exc:
        return {
            "init_id": init_id,
            "init_record": record,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "root_pos_mode": path.root_pos_mode,
            "target_clip_id": target_clip_id,
            "target_frame": int(target_frame),
            "elapsed_s": float(time.perf_counter() - t0),
        }


def _solve_task(task: Tuple[Dict[str, Any], Dict[str, np.ndarray],
                           Dict[str, np.ndarray], Dict[str, Any]]
                ) -> Dict[str, Any]:
    record, state, target, cfg = task
    return solve_init_record(
        record, state, target,
        target_clip_id=cfg["target_clip_id"],
        target_frame=cfg["target_frame"],
        out_dir=Path(cfg["out_dir"]),
        budget_frames=cfg["budget_frames"],
        ds_max=cfg["ds_max"],
        s_ext=cfg["s_ext"],
        post_frames=cfg["post_frames"],
        sweeps=cfg["sweeps"],
        n_samples=cfg["n_samples"],
        n_iters=cfg["n_iters"],
        seeds=cfg["seeds"],
        robot=cfg["robot"],
        save_all=cfg["save_all"])


def generate_experts(init_dir: Path, target_clip_path: Path,
                     out_dir: Path, **cfg) -> Dict[str, Any]:
    """Run/resume Reclip generation for selected init records."""
    init_dir, out_dir = Path(init_dir), Path(out_dir)
    manifest_path = out_dir / "reclip_results.jsonl"
    clip, target, frame = target_pose(target_clip_path,
                                      cfg.pop("target_frame", -1))
    records = load_init_records(
        init_dir, offset=cfg.pop("offset", 0),
        limit=cfg.pop("limit", None),
        shard_index=cfg.pop("shard_index", 0),
        num_shards=cfg.pop("num_shards", 1))
    prior = _read_jsonl(manifest_path)
    retry_failures = bool(cfg.pop("retry_failures", False))
    if retry_failures:
        done = {str(r.get("init_id")) for r in prior
                if r.get("status") == "completed"}
    else:
        done = {str(r.get("init_id")) for r in prior}
    records = [r for r in records if str(r["init_id"]) not in done]
    cfg.update({
        "target_clip_id": clip.clip_id,
        "target_frame": frame,
        "out_dir": str(out_dir),
    })
    tasks = [(r, load_bank_state(r), target, cfg) for r in records]
    results: List[Dict[str, Any]] = []
    workers = int(cfg.get("workers", 1))
    if workers > 1 and tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_solve_task, t) for t in tasks]
            for fut in as_completed(futs):
                row = fut.result()
                results.append(row)
                _append_jsonl(manifest_path, [row])
                print(f"[reclip-batch] {row['init_id']}: "
                      f"{row.get('status')}")
    else:
        for task in tasks:
            row = _solve_task(task)
            results.append(row)
            _append_jsonl(manifest_path, [row])
            print(f"[reclip-batch] {row['init_id']}: "
                  f"{row.get('status')}")
    all_rows = _read_jsonl(manifest_path)
    summary = summarize_reclip_results(_latest_results(all_rows))
    summary.update({
        "attempt_count": len(all_rows),
        "attempt_status_counts": _status_counts(all_rows),
        "target_clip_id": clip.clip_id,
        "target_frame": frame,
        "target_clip_path": str(target_clip_path),
        "selected_records": len(records),
        "new_results": len(results),
        "workers": workers,
        "solver": {k: cfg[k] for k in (
            "budget_frames", "ds_max", "s_ext", "post_frames",
            "sweeps", "n_samples", "n_iters", "seeds")},
    })
    (out_dir / "solver_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False)
        + "\n", encoding="utf-8")
    return summary


def _status_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        k = str(r.get("status", "missing"))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def summarize_reclip_results(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    done = [r for r in rows if r.get("status") == "completed"]
    events = [int(r["event_frame"]) for r in done
              if r.get("event_frame") is not None]
    terminal = [float(r["terminal_pose_error"]) for r in done
                if r.get("terminal_pose_error") is not None]
    path_err = [float(r["path_error_max"]) for r in done
                if r.get("path_error_max") is not None]
    elapsed = [float(r["elapsed_s"]) for r in rows
               if r.get("elapsed_s") is not None]
    return {
        "count": len(rows),
        "status": _status_counts(rows),
        "completed": len(done),
        "completed_rate": float(len(done) / max(1, len(rows))),
        "event_frame": {
            "min": min(events) if events else None,
            "max": max(events) if events else None,
            "mean": float(np.mean(events)) if events else None},
        "terminal_pose_error": {
            "min": min(terminal) if terminal else None,
            "max": max(terminal) if terminal else None,
            "mean": float(np.mean(terminal)) if terminal else None},
        "path_error_max": {
            "min": min(path_err) if path_err else None,
            "max": max(path_err) if path_err else None,
            "mean": float(np.mean(path_err)) if path_err else None},
        "elapsed_s": {
            "min": min(elapsed) if elapsed else None,
            "max": max(elapsed) if elapsed else None,
            "mean": float(np.mean(elapsed)) if elapsed else None},
    }


def _replay_ok(row: Dict[str, Any], tol: float) -> bool:
    err = row.get("replay_error") or {}
    return all(float(err.get(k, math.inf)) <= tol
               for k in _REPLAY_KEYS)


def filter_reclip_results(results_path: Path,
                          accepted_path: Path,
                          rejected_path: Path,
                          summary_path: Path,
                          replay_tol: float = 1e-5,
                          terminal_pose_max: Optional[float] = None,
                          path_error_max: Optional[float] = None,
                          action_saturation_max: Optional[float] = None
                          ) -> Dict[str, Any]:
    all_rows = _read_jsonl(Path(results_path))
    rows = _latest_results(all_rows)
    accepted, rejected = [], []
    for row in rows:
        reasons = []
        if row.get("status") != "completed":
            reasons.append("not_completed")
        if row.get("event_frame") is None:
            reasons.append("missing_event")
        if int(row.get("fall_count") or 0) != 0:
            reasons.append("fall")
        if not _replay_ok(row, replay_tol):
            reasons.append("replay_error")
        if not row.get("clip_path"):
            reasons.append("missing_clip")
        if terminal_pose_max is not None and \
                float(row.get("terminal_pose_error", math.inf)) \
                > terminal_pose_max:
            reasons.append("terminal_pose_error")
        if path_error_max is not None and \
                float(row.get("path_error_max", math.inf)) \
                > path_error_max:
            reasons.append("path_error")
        if action_saturation_max is not None and \
                row.get("action_saturation_frac") is not None and \
                float(row["action_saturation_frac"]) \
                > action_saturation_max:
            reasons.append("action_saturation")
        if reasons:
            r = dict(row)
            r["reject_reasons"] = reasons
            rejected.append(r)
        else:
            accepted.append(row)
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_path.write_text(
        "".join(json.dumps(_jsonable(r), ensure_ascii=False) + "\n"
                for r in accepted), encoding="utf-8")
    rejected_path.write_text(
        "".join(json.dumps(_jsonable(r), ensure_ascii=False) + "\n"
                for r in rejected), encoding="utf-8")
    by_skill: Dict[str, Dict[str, int]] = {}
    for r in rows:
        skill = ((r.get("init_record") or {}).get("warmup") or {}) \
            .get("skill", "unknown")
        m = by_skill.setdefault(skill, {"total": 0, "accepted": 0})
        m["total"] += 1
        if r in accepted:
            m["accepted"] += 1
    summary = {
        "total": len(rows),
        "attempt_count": len(all_rows),
        "attempt_status_counts": _status_counts(all_rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accept_rate": float(len(accepted) / max(1, len(rows))),
        "criteria": {
            "status": "completed",
            "fall_count": 0,
            "replay_tol": float(replay_tol),
            "terminal_pose_max": terminal_pose_max,
            "path_error_max": path_error_max,
            "action_saturation_max": action_saturation_max,
        },
        "status_counts": _status_counts(rows),
        "by_skill": dict(sorted(by_skill.items())),
        "solver": summarize_reclip_results(rows),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False)
        + "\n", encoding="utf-8")
    return summary


def _csv_ints(s: str) -> List[int]:
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def _csv_strings(s: Optional[str]) -> List[str]:
    if not s:
        return list(WARMUP_SKILLS)
    return [x.strip() for x in s.split(",") if x.strip()]


def _cmd_sample_inits(args) -> int:
    policy = None
    if not args.zero_policy:
        policy = load_export_policy(Path(args.policy_export))
    summary = generate_init_state_bank(
        Path(args.out_dir), count=args.count, base_seed=args.seed,
        policy=policy, skills=_csv_strings(args.skills),
        steps_min=args.steps_min, steps_max=args.steps_max,
        shard_size=args.shard_size, robot=args.robot,
        sample_actions=not args.deterministic,
        command_scale_min=args.command_scale_min,
        command_scale_max=args.command_scale_max)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_experts(args) -> int:
    summary = generate_experts(
        Path(args.init_dir), Path(args.target), Path(args.out_dir),
        target_frame=args.target_frame,
        offset=args.offset, limit=args.limit,
        shard_index=args.shard_index, num_shards=args.num_shards,
        retry_failures=args.retry_failures,
        budget_frames=args.budget_frames, ds_max=args.ds_max,
        s_ext=args.s_ext, post_frames=args.post_frames,
        sweeps=args.sweeps, n_samples=args.n_samples,
        n_iters=args.n_iters, seeds=_csv_ints(args.seeds),
        workers=args.workers, robot=args.robot,
        save_all=args.save_all)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_filter(args) -> int:
    summary = filter_reclip_results(
        Path(args.results), Path(args.accepted), Path(args.rejected),
        Path(args.summary), replay_tol=args.replay_tol,
        terminal_pose_max=args.terminal_pose_max,
        path_error_max=args.path_error_max,
        action_saturation_max=args.action_saturation_max)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="reclip-bc",
        description="warmup init sampling -> position-free Reclip -> "
                    "expert filtering")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample-inits",
                       help="sample frozen-warmup initial CoreStates")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy-export", default=str(DEFAULT_POLICY_EXPORT))
    p.add_argument("--zero-policy", action="store_true",
                   help="debug path: emit zero actions instead of "
                        "loading the frozen export")
    p.add_argument("--deterministic", action="store_true",
                   help="use policy.act instead of policy.sample")
    p.add_argument("--skills", default=None,
                   help="CSV subset of warmup skills")
    p.add_argument("--steps-min", type=int, default=0)
    p.add_argument("--steps-max", type=int, default=30)
    p.add_argument("--command-scale-min", type=float, default=0.6)
    p.add_argument("--command-scale-max", type=float, default=1.0)
    p.add_argument("--shard-size", type=int, default=64)
    p.add_argument("--robot", default="robot_a",
                   choices=("robot_a", "robot_b"))
    p.set_defaults(fn=_cmd_sample_inits)

    p = sub.add_parser("experts",
                       help="solve position-free pass experts for init bank")
    p.add_argument("--init-dir", required=True)
    p.add_argument("--target", default=str(DEFAULT_TARGET_CLIP))
    p.add_argument("--target-frame", type=int, default=-1)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--retry-failures", action="store_true")
    p.add_argument("--budget-frames", type=int, default=6)
    p.add_argument("--ds-max", type=float, default=0.5)
    p.add_argument("--s-ext", type=float, default=0.1)
    p.add_argument("--post-frames", type=int, default=1)
    p.add_argument("--sweeps", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--n-iters", type=int, default=1)
    p.add_argument("--seeds", default="0")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--robot", default="robot_a",
                   choices=("robot_a", "robot_b"))
    p.add_argument("--save-all", action="store_true",
                   help="also save budget_exhausted outputs")
    p.set_defaults(fn=_cmd_experts)

    p = sub.add_parser("filter",
                       help="split reclip_results.jsonl into accepted/"
                            "rejected expert lists")
    p.add_argument("--results", required=True)
    p.add_argument("--accepted", required=True)
    p.add_argument("--rejected", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--replay-tol", type=float, default=1e-5)
    p.add_argument("--terminal-pose-max", type=float, default=None)
    p.add_argument("--path-error-max", type=float, default=None)
    p.add_argument("--action-saturation-max", type=float, default=None)
    p.set_defaults(fn=_cmd_filter)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
