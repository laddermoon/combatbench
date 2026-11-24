"""standup (dynabptt) — dual-agent 4-stage standup from fallen poses,
self-only model.

Mirrors experiments_ppo/exp_standup.py reward semantics:

  - single channel r_potential = 0.01 * phi(t) per step (actor weight 1.0)
  - phi is the StandingBalance4StageRewarder "potential" field in [0,1]
  - NO early termination: RandomFallenStatePlugin starts robots fallen,
    episodes always run to the 200-step timeout, so the done head is
    unused (run with --no-done).

Unlike basic_balance this is NOT a knife-edge task: the dense potential
gives a smooth, directional return landscape — the intended first test
of whether the DynaBPTT model gradient carries real signal at all.

Blueprint: baseline/humanoid21/blueprints/standup_4stage_dense_v2_env.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy,
)

_BLUEPRINT_DIR = (
    Path(__file__).resolve().parent.parent
    / "humanoid21" / "blueprints"
)


class StandupDynaBPTT:

    name = "standup"

    agent_ids = ("robot_a", "robot_b")

    # --- Shapes ---
    state_dim = 55
    action_dim = 21
    obs_dim = 96
    actor_hidden_dim = 256
    critic_hidden_dim = 256
    wm_hidden = (1024, 1024, 1024)
    wm_loss_weights = {"s": 1.0, "obs": 1.0, "aux": 1.0, "done": 1.0}

    # --- Env ---
    env_blueprint_name = "standup_4stage_dense_v2_env.yaml"
    max_steps = 200
    init_distance_min = 1.5
    init_distance_max = 3.5

    # --- Channels ---
    # aux "phi" = the 4-stage potential; kept under the name "phi" so the
    # phi-dependent machinery (signal weighting, dagger phi-gate, clamps)
    # applies unchanged.
    aux_names = ("phi",)
    channels = (
        {"name": "r_potential", "gamma": 0.99},
    )

    per_step_phi_coef = 0.01
    w_potential = 1.0

    aux_clamps = {"phi": (0.0, 1.0)}

    # Potential returns are bounded: r = 0.01*phi in [0,0.01], gamma 0.99
    # -> return in [0, 1.0].
    channel_vrange = {"r_potential": (0.0, 1.0)}

    def env_bp(self):
        pb = ParameterizedEnvBlueprint.load(
            _BLUEPRINT_DIR / self.env_blueprint_name
        )
        return pb.materialize(max_steps=self.max_steps)

    def build_actor(self, device) -> TruncatedNormalPolicy:
        return TruncatedNormalPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=self.actor_hidden_dim,
            device=str(device),
        )

    def aux_keys(self, agent_id: str) -> Dict[str, tuple]:
        suf = "a" if agent_id == "robot_a" else "b"
        return {
            "phi": (f"standing_balance_{suf}", "potential"),
        }

    def channel_reward(self, name: str, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        if name == "r_potential":
            return self.per_step_phi_coef * aux["phi"]
        raise KeyError(f"unknown channel {name!r}")

    def channel_weight(self, name: str, aux: Dict[str, torch.Tensor]):
        if name == "r_potential":
            return self.w_potential
        raise KeyError(f"unknown channel {name!r}")


EXPERIMENT_CLASS = StandupDynaBPTT
