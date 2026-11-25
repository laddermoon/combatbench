"""Balance-task spec for the SuperMimic-faithful BPTT replication.

Environment: basic_balance_v2_phi_dual_env.yaml patched to self-only
(SelfOnlyStandupSimulator slices 96-dim obs to the 57-dim self view and
disables A<->B collisions) + CoreStateObserver, control dt selectable.

Reward channel used inside imagination: r_t = W_FALL * 0.01 * phi(s_{t+1})
with phi computed ANALYTICALLY from the GenCoord rep —
phi = uprightness * z / standing_height = -g_b[2] * z / 1.28 — so the
imagined reward carries zero head error (the exact same trick the
official WorldModelEnv uses for its tracking terms).

Observations consumed by the actor inside imagination are composed from
(rep, aux): 55/57 dims are exact differentiable transforms of the core
state; only feet_forces [52:54] fall back to the learned obs head.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from envs.framework.blueprint import EnvBlueprint
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from baseline.framework.rollout import extract_per_step_field

from ..hybrid_ppo.gencoord import (R_GB, R_QJ, R_VB, R_VJ, R_WB, R_Z,
                                 quat_apply_inverse)
from ..hybrid_ppo.standup import (CS_OBSERVER_NAME, CS_OBSERVER_SPEC,
                                  SelfOnlyStandupSimulator)
from ..hybrid_ppo.transitions import AGENT_IDS, AlignedTraj

_BLUEPRINT_DIR = (
    Path(__file__).resolve().parents[3] / "humanoid21" / "blueprints"
)
ENV_BLUEPRINT_NAME = "basic_balance_v2_phi_dual_env.yaml"

PHI_OBSERVERS = {"robot_a": "height_phi_a", "robot_b": "height_phi_b"}
CROSS_OBSERVERS = {"robot_a": "cross_support_a", "robot_b": "cross_support_b"}
STANDING_HEIGHT = 1.28

W_FALL = 3.0          # r_fall actor weight (matches exp_basic_balance)
STEP_PHI_COEF = 0.01  # r_t = W_FALL * STEP_PHI_COEF * phi
MAX_STEP_REWARD = W_FALL * STEP_PHI_COEF * 1.0  # phi in [0,1]


def make_balance_env_bp(max_steps: int = 1500,
                        phy_steps_per_action: int = 10) -> EnvBlueprint:
    """Materialize the balance env patched for self-only training."""
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


def analytic_phi(rep: torch.Tensor) -> torch.Tensor:
    """phi(s) = uprightness * z / standing_height, from rep channels."""
    uprightness = -rep[..., R_GB][..., 2]
    return uprightness * rep[..., R_Z].squeeze(-1) / STANDING_HEIGHT


def compose_obs(rep: torch.Tensor, aux: torch.Tensor,
                feet: torch.Tensor) -> torch.Tensor:
    """Differentiable exact obs composition: (…,52)+(…,7)+(…,2) -> (…,57).

    Layout (self obs of the 96-dim simulator obs):
      [0:21]  joint_pos_norm          = rep.qj
      [21:42] sign(jv)*sqrt(|jv|)/2   (jv = rep.vj, norm units)
      [42:45] projected_gravity       = rep.g_b
      [45]    height                  = rep.z
      [46:49] linear_vel body         = rep.v_b
      [49:52] sign(w)*sqrt(|w|/2)     (w = rep.w_b, rad/s)
      [52:54] feet_forces             <- learned head (contact channel)
      [54:57] arena_center_local      = R^T @ (-root_pos)
    """
    jv = rep[..., R_VJ]
    jv_obs = torch.sign(jv) * torch.sqrt(jv.abs().clamp_min(1e-12)) / 2.0
    w = rep[..., R_WB]
    w_obs = torch.sign(w) * torch.sqrt(w.abs().clamp_min(1e-12) / 2.0)
    center = quat_apply_inverse(aux[..., 3:7], -aux[..., :3])
    return torch.cat([
        rep[..., R_QJ], jv_obs,
        rep[..., R_GB], rep[..., R_Z],
        rep[..., R_VB], w_obs,
        feet, center,
    ], dim=-1)


def extract_balance(episode, update: int,
                    agent_ids=AGENT_IDS) -> List[AlignedTraj]:
    """Balance episode -> per-agent AlignedTraj (stage const 1)."""
    cs_node = episode.observer_outputs.get(CS_OBSERVER_NAME)
    if cs_node is None:
        raise KeyError("dynabptt_cs observer missing")
    T = int(episode.num_frames)
    out: List[AlignedTraj] = []
    for aid in agent_ids:
        cs = cs_node.get(aid)
        obs = episode.observations.get(aid)
        act = episode.actions.get(aid)
        if cs is None or obs is None or act is None:
            raise KeyError(f"episode missing cs/obs/act for {aid}")
        B = int(episode.agent_frame_boundary[aid])
        if B < 2:
            continue
        phi = extract_per_step_field(
            episode.observer_outputs, PHI_OBSERVERS[aid], "phi", T)
        if phi is None:
            raise KeyError(f"height_phi observer missing for {aid}")
        rc = episode.observer_outputs.get(CROSS_OBSERVERS[aid])
        aux_rew = np.zeros(B, dtype=np.float32)
        if rc is not None:
            raw = rc.get("value") if isinstance(rc, dict) else None
            if raw is not None:
                aux_rew = np.asarray(raw[:B], dtype=np.float32)
        terminated = bool(
            episode.agent_termination_proposal_records.get(aid, ())
        ) and episode.agent_termination_proposal_records[aid][0][0] != \
            "timeout"
        done = np.zeros(B, dtype=np.float32)
        if terminated:
            done[B - 1] = 1.0
        out.append(AlignedTraj(
            cs=np.asarray(cs[:B], dtype=np.float32),
            obs=np.asarray(obs[:B], dtype=np.float32),
            act=np.asarray(act[:B], dtype=np.float32),
            phi=np.asarray(phi[:B], dtype=np.float32),
            stage=np.ones(B, dtype=np.float32),
            final_obs=np.asarray(
                episode.final_observation[aid], dtype=np.float32),
            terminated=terminated,
            ep_key=int(episode.base_seed),
            update=int(update),
            agent_id=aid,
            done=done,
            aux_rew=aux_rew,
        ))
    return out
