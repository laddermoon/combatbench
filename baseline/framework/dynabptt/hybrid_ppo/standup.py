"""Self-only Standup environment: 57-dim self observation + no robot-robot collision.

Two changes relative to ``Humanoid21Simulator``:

1. ``get_observation`` returns the first 57 dims (proprioception 42 +
   root state 13 + feet forces 2), dropping the 39 opponent dims.  Index
   order verified against ``simulator._get_robot_view``: joint_pos(21),
   joint_vel(21), projected_gravity(3), height(1), linear_vel(3),
   ang_vel(3), feet_forces(2), arena_center_local(3).

2. Geom contype/conaffinity bits are rewritten once at construction so
   robot_a geoms (contype bit0) and robot_b geoms (contype bit1) can no
   longer collide with each other, while self- and env-collisions are
   preserved.  MuJoCo contact rule: pair collides iff
   ``(contype1 & conaffinity2) | (contype2 & conaffinity1)``.

   A geoms: contype|bit0 & ~bit1, conaffinity|bit0 & ~bit1
   B geoms: contype|bit1 & ~bit0, conaffinity|bit1 & ~bit0
   env geoms (ground/walls): |bits 0..1 both ways -> still collide with
   A, B, and each other.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import mujoco
import numpy as np

from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.humanoid21.simulator import Humanoid21Simulator

SELF_OBS_DIM = 57
STATE_DIM = 55
ACTION_DIM = 21
REWARD_COEF = 0.01  # r_t = 0.01 * phi_{t+1}

_BLUEPRINT_DIR = (
    Path(__file__).resolve().parents[3] / "humanoid21" / "blueprints"
)
ENV_BLUEPRINT_NAME = "standup_4stage_dense_v2_env.yaml"

CS_OBSERVER_NAME = "dynabptt_cs"
CS_OBSERVER_SPEC = {
    "cls": "baseline.framework.dynabptt.corestate:CoreStateObserver",
    "config": {},
}
REWARDER_NAMES = {"robot_a": "standing_balance_a", "robot_b": "standing_balance_b"}


class SelfOnlyStandupSimulator(Humanoid21Simulator):
    """Humanoid21 variant: self-obs only + inter-robot collision disabled."""

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)
        n_a, n_b = self._isolate_robot_geoms()
        if n_a == 0 or n_b == 0:
            raise RuntimeError(
                f"robot geom isolation found n_a={n_a}, n_b={n_b} — body "
                "naming convention ('*_a'/'*_b') may have changed"
            )

    def _isolate_robot_geoms(self) -> tuple:
        model = self.model
        n_a = n_b = 0
        for gid in range(model.ngeom):
            ctype = int(model.geom_contype[gid])
            caff = int(model.geom_conaffinity[gid])
            if ctype == 0 and caff == 0:
                continue  # non-collidable geom — leave untouched
            bid = int(model.geom_bodyid[gid])
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname.endswith("_a"):
                model.geom_contype[gid] = (ctype | 0b01) & ~0b10
                model.geom_conaffinity[gid] = (caff | 0b01) & ~0b10
                n_a += 1
            elif bname.endswith("_b"):
                model.geom_contype[gid] = (ctype | 0b10) & ~0b01
                model.geom_conaffinity[gid] = (caff | 0b10) & ~0b01
                n_b += 1
            else:
                model.geom_contype[gid] = ctype | 0b11
                model.geom_conaffinity[gid] = caff | 0b11
        return n_a, n_b

    def get_observation(self) -> Dict[str, np.ndarray]:
        obs = super().get_observation()
        out = {}
        for rid, o in obs.items():
            o = np.asarray(o, dtype=np.float32)
            if o.shape[-1] != 96:
                raise RuntimeError(
                    f"expected 96-dim raw observation for {rid}, got "
                    f"shape {o.shape} — observation layout changed"
                )
            out[rid] = o[:SELF_OBS_DIM].copy()
        return out


def make_standup_env_bp(max_steps: int = 200,
                        phy_steps_per_action: int = 25) -> EnvBlueprint:
    """Materialize the standup env blueprint patched for self-only training.

    In-memory derivation only — no YAML or shared file is modified.
    ``phy_steps_per_action`` overrides the control decimation
    (25 x 2ms = 50ms default); 10 gives a 20ms control step.
    """
    pb = ParameterizedEnvBlueprint.load(_BLUEPRINT_DIR / ENV_BLUEPRINT_NAME)
    env_bp = pb.materialize(max_steps=max_steps)
    d = env_bp.to_dict()
    d["runtime"]["phy_steps_per_action"] = int(phy_steps_per_action)
    d["simulator"]["cls"] = (
        "baseline.framework.dynabptt.hybrid_ppo.standup:"
        "SelfOnlyStandupSimulator"
    )
    d.setdefault("observer_plugins", {})[CS_OBSERVER_NAME] = dict(
        CS_OBSERVER_SPEC
    )
    return EnvBlueprint.from_dict(d)


def unflatten_core_state(vec: np.ndarray) -> Dict[str, np.ndarray]:
    """Inverse of ``flatten_core_state`` — 55-dim vector → named arrays."""
    from baseline.framework.dynabptt.corestate import STATE_KEYS

    sizes = {"root_pos": 3, "root_rot": 4, "root_vel_local": 3,
             "root_angular_vel_local": 3, "joint_pos_norm": 21,
             "joint_vel_norm": 21}
    out: Dict[str, np.ndarray] = {}
    i = 0
    for k in STATE_KEYS:
        n = sizes[k]
        out[k] = np.asarray(vec[i:i + n], dtype=np.float64).copy()
        i += n
    assert i == vec.shape[-1] == STATE_DIM
    return out


def phi_per_frame(episode, agent_id: str) -> np.ndarray:
    """Per-frame potential phi(s_{t+1}) for one agent, shape (T,)."""
    from baseline.framework.rollout import extract_per_step_field

    arr = extract_per_step_field(
        episode.observer_outputs, REWARDER_NAMES[agent_id], "potential",
        episode.num_frames,
    )
    if arr is None:
        raise KeyError(
            f"rewarder '{REWARDER_NAMES[agent_id]}' field 'potential' "
            "missing from episode.observer_outputs"
        )
    return np.asarray(arr, dtype=np.float32)


def episode_metrics(episode, agent_ids=("robot_a", "robot_b")) -> Dict[str, float]:
    """Eval metrics per agent: potential stats + success criteria.

    ``hold_success``: last 20 action frames all have phi >= 0.9.
    ``reach_success``: any frame has phi >= 0.9.
    ``final_success``: last frame phi >= 0.9.
    """
    n = int(episode.num_frames)
    out: Dict[str, float] = {}
    for aid in agent_ids:
        phi = phi_per_frame(episode, aid)
        tag = aid[-1]
        out[f"{tag}_max_pot"] = float(phi.max())
        out[f"{tag}_final_pot"] = float(phi[-1])
        tail = phi[-20:] if n >= 20 else phi
        out[f"{tag}_tail_mean_pot"] = float(tail.mean())
        out[f"{tag}_reach_success"] = float(phi.max() >= 0.9)
        out[f"{tag}_final_success"] = float(phi[-1] >= 0.9)
        out[f"{tag}_hold_success"] = float(np.all(tail >= 0.9))
    return out


def aggregate_metrics(episodes, agent_ids=("robot_a", "robot_b")) -> Dict[str, float]:
    """Mean of per-episode metrics across a deterministic eval batch."""
    acc: Dict[str, float] = {}
    n = 0
    for ep in episodes:
        m = episode_metrics(ep, agent_ids)
        for k, v in m.items():
            acc[k] = acc.get(k, 0.0) + v
        n += 1
    if n == 0:
        raise RuntimeError("aggregate_metrics: no episodes")
    return {k: v / n for k, v in acc.items()}
