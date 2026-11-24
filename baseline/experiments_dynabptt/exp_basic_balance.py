"""basic_balance (dynabptt) — dual-agent standing balance, self-only model.

Mirrors experiments_ppo/exp_basic_balance.py reward semantics:

  - r_fall = 0.01 * phi(t) per step          (actor weight 3.0 fixed)
  - r_cross = CrossSupportBalanceRewarder    (actor weight 1.0 * phi^2)
  - termination: DualImbalanceTerminationPlugin ("imbalance", non-timeout)

World-model aux heads predict the raw observer outputs (phi, r_cross);
channel rewards + per-step actor weights are then computed analytically,
identical to the PPO recipe.

Blueprint: baseline/humanoid21/blueprints/basic_balance_v2_phi_dual_env.yaml
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


class BasicBalanceDynaBPTT:

    name = "basic_balance"

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
    env_blueprint_name = "basic_balance_v2_phi_dual_env.yaml"
    max_steps = 600
    init_distance_min = 1.5
    init_distance_max = 3.5

    # --- Channels (gamma per channel, like PPO RewardChannel) ---
    aux_names = ("phi", "r_cross")
    channels = (
        {"name": "r_fall", "gamma": 0.99},
        {"name": "r_cross", "gamma": 0.99},
    )

    # --- Reward constants (mirror PPO exp) ---
    per_step_phi_coef = 0.01
    w_fall = 3.0
    w_cross = 1.0

    # Plausible output ranges for aux heads inside imagination (see
    # train.improve): prevents the actor from arbitraging unbounded head
    # extrapolation on off-manifold imagined states.
    aux_clamps = {"phi": (0.0, 1.0)}

    # Plausible ranges for critic *values* at bootstrap consumption points
    # (J bootstrap + both TD targets). r_fall returns are bounded:
    # per-step r = 0.01*phi in [0, 0.01] with gamma 0.99 -> return in
    # [0, 1.0]. Without the clamp, self-referential TD inflation pushed
    # V_fall to ~2.7 and the actor learned to chase inflated bootstrap
    # states instead of balancing (observed ep_len regression u19->u29).
    channel_vrange = {"r_fall": (0.0, 1.0)}

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
            "phi": (f"height_phi_{suf}", "phi"),
            "r_cross": (f"cross_support_{suf}", None),
        }

    def channel_reward(self, name: str, aux: Dict[str, torch.Tensor]) -> torch.Tensor:
        if name == "r_fall":
            return self.per_step_phi_coef * aux["phi"]
        if name == "r_cross":
            return aux["r_cross"]
        raise KeyError(f"unknown channel {name!r}")

    def channel_weight(self, name: str, aux: Dict[str, torch.Tensor]):
        if name == "r_fall":
            return self.w_fall
        if name == "r_cross":
            return self.w_cross * aux["phi"] ** 2
        raise KeyError(f"unknown channel {name!r}")


EXPERIMENT_CLASS = BasicBalanceDynaBPTT
