"""P6 planning-interface and same-budget comparison tests."""
from dataclasses import dataclass

import numpy as np
import pytest

from envs.humanoid21.simulator import Humanoid21Simulator
from mocap.humanoid21.clip.objective_evaluator import ReclipObjectiveConfig
from mocap.humanoid21.clip.target_path import TargetPath
from mocap.humanoid21.centroidal_dynamics.planning import (
    build_centroidal_plan,
    direct_reclip_solver,
    make_plan_context,
    run_same_budget_comparison,
)


@dataclass
class _Eval:
    status: str = "completed"
    score: float = 1.25
    event_frame: int = 1
    progress: np.ndarray = None
    metrics: dict = None

    def __post_init__(self):
        self.progress = np.array([0.2, 1.0])
        self.metrics = {
            "fall_count": 0,
            "path_error_max": 0.2,
            "terminal_pose_error": 0.1,
            "terminal_vel_error": 0.3,
        }


@dataclass
class _Clip:
    T: int = 3
    meta: dict = None

    def __post_init__(self):
        self.meta = {}


@dataclass
class _Result:
    clip: _Clip
    evaluation: _Eval
    actions: np.ndarray
    solver_actions: np.ndarray
    replay_metrics: dict
    solver_seed: int = 0


def _fake_result():
    return _Result(_Clip(), _Eval(), np.zeros((2, 21)),
                   np.zeros((4, 21)),
                   {"root_pos_err_max": 0.0})


@pytest.fixture()
def sim():
    s = Humanoid21Simulator()
    s.reset(seed=0)
    try:
        yield s
    finally:
        s.close()


def _context(sim):
    start = sim.get_core_state()["robot_a"]
    target = dict(start)
    target["joint_pos_norm"] = np.clip(
        start["joint_pos_norm"] + 0.01, -1.0, 1.0)
    path = TargetPath.from_pose_pair(start, target,
                                     source_clip_id="test",
                                     root_pos_mode="free")
    objective = ReclipObjectiveConfig(kind="pass", time_budget_frames=4,
                                      ds_max=0.5)
    return make_plan_context(sim, "robot_a", start, path, objective,
                             budget_frames=4)


def test_build_centroidal_plan_collects_state_loads_and_domain(sim):
    start = sim.get_core_state()["robot_a"]
    target = dict(start)
    path = TargetPath.from_pose_pair(start, target, root_pos_mode="free")
    objective = ReclipObjectiveConfig(kind="pass", time_budget_frames=4,
                                      ds_max=0.5)
    plan = build_centroidal_plan(sim, "robot_a", start, path, objective, 4)
    assert plan.domain_valid
    assert plan.initial_state.mass > 40.0
    assert plan.initial_loads.metadata["ncon"] > 0
    assert plan.target_summary["root_pos_mode"] == "free"
    assert plan.metadata["is_full_body_executable"] is False


def test_same_budget_comparison_normalizes_rows_and_resets(sim):
    ctx = _context(sim)
    root = sim._meta['robots']['robot_a']['root_qpos_adr']
    original = sim.data.qpos[root:root + 3].copy()
    seen = []

    def method_a(context):
        context.sim.data.qpos[root] += 1.0
        return _fake_result()

    def method_b(context):
        seen.append(context.sim.data.qpos[root:root + 3].copy())
        return _fake_result()

    out = run_same_budget_comparison(
        ctx, {"direct": method_a, "centroidal_stub": method_b})
    assert len(out["runs"]) == 2
    assert all(r["status"] == "completed" for r in out["runs"])
    assert all(r["budget_frames"] == 4 for r in out["runs"])
    assert all(r["executed_physics_steps"] == 50 for r in out["runs"])
    assert np.allclose(seen[0], original)
    assert out["plan"]["domain_valid"]
    assert out["runs"][0]["terminal_pose_error"] == pytest.approx(0.1)


def test_comparison_records_solver_exception(sim):
    ctx = _context(sim)

    def bad(context):
        raise RuntimeError("boom")

    out = run_same_budget_comparison(ctx, {"bad": bad})
    assert out["runs"][0]["status"] == "solver_exception"
    assert out["runs"][0]["exception_type"] == "RuntimeError"
    assert out["runs"][0]["completed"] is False


def test_direct_reclip_solver_factory_returns_callable():
    solver = direct_reclip_solver(seed=0, sweeps=1, n_samples=1,
                                  n_iters=1)
    assert callable(solver)
