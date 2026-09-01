"""Multi-head Q critic networks for SAC V2.

Channels with similar time horizons (gamma) share a trunk network with
per-channel heads. This reduces the network count from O(C) to O(G)
where G is the number of trunk groups, while preserving per-channel Q
values.

Architecture per group:
    Trunk: Linear(obs+act, hidden) → ReLU → Linear(hidden, hidden) → ReLU
    Head_c: Linear(hidden, 1)  # one per channel in the group

Twin Q: each group has two independent trunk+heads (Q1, Q2).
Target networks: deep copies, soft-updated.

See DECISIONS.md N3 for trunk grouping rationale.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .experiment import SACRewardChannel


# ---------------------------------------------------------------------------
# Q Trunk + Heads
# ---------------------------------------------------------------------------

class QTrunkHeads(nn.Module):
    """One trunk + multiple heads for a group of channels.

    The trunk takes (obs, action) concatenated and maps to a shared
    hidden representation. Each head maps the hidden representation to
    a scalar Q-value for one channel.

    Optionally uses LayerNorm in the trunk for training stability
    (recommended for high-DOF environments).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        channel_names: Tuple[str, ...],
        layer_norm: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.channel_names = tuple(channel_names)
        self.layer_norm = bool(layer_norm)

        layers: List[nn.Module] = [
            nn.Linear(obs_dim + action_dim, hidden_dim),
        ]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({
            ch: nn.Linear(hidden_dim, 1) for ch in channel_names
        })

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor,
        channel: str,
    ) -> torch.Tensor:
        """Q(s, a) for one channel. Returns (B,)."""
        x = torch.cat([obs, action], dim=-1)
        h = self.trunk(x)
        return self.heads[channel](h).squeeze(-1)

    def forward_all(
        self, obs: torch.Tensor, action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Q(s, a) for all channels in this group. Returns {ch: (B,)}."""
        x = torch.cat([obs, action], dim=-1)
        h = self.trunk(x)
        return {
            ch: self.heads[ch](h).squeeze(-1)
            for ch in self.channel_names
        }


# ---------------------------------------------------------------------------
# Multi-Head Q Critic — manages all groups, twin Q, and targets
# ---------------------------------------------------------------------------

class QTrunkGroup:
    """One trunk group: channels sharing a trunk.

    Contains Q1, Q2 (twin critics), and their target copies.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        channel_names: Tuple[str, ...],
        layer_norm: bool,
        device: torch.device,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.channel_names = tuple(channel_names)
        self.device = device

        self.q1 = QTrunkHeads(
            obs_dim, action_dim, hidden_dim, channel_names, layer_norm,
        ).to(device)
        self.q2 = QTrunkHeads(
            obs_dim, action_dim, hidden_dim, channel_names, layer_norm,
        ).to(device)

        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        self.q1_optimizer = torch.optim.Adam(
            self.q1.parameters(), lr=3e-4,  # overridden by caller
        )
        self.q2_optimizer = torch.optim.Adam(
            self.q2.parameters(), lr=3e-4,
        )

    def set_lr(self, lr: float) -> None:
        for pg in self.q1_optimizer.param_groups:
            pg["lr"] = lr
        for pg in self.q2_optimizer.param_groups:
            pg["lr"] = lr

    def soft_update(self, tau: float) -> None:
        with torch.no_grad():
            for p, pt in zip(
                self.q1.parameters(), self.q1_target.parameters()
            ):
                pt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
            for p, pt in zip(
                self.q2.parameters(), self.q2_target.parameters()
            ):
                pt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if "q1" in sd:
            self.q1.load_state_dict(sd["q1"])
        if "q2" in sd:
            self.q2.load_state_dict(sd["q2"])
        if "q1_target" in sd:
            self.q1_target.load_state_dict(sd["q1_target"])
        if "q2_target" in sd:
            self.q2_target.load_state_dict(sd["q2_target"])
        if "q1_optimizer" in sd:
            try:
                self.q1_optimizer.load_state_dict(sd["q1_optimizer"])
            except (ValueError, RuntimeError) as e:
                print(f"[checkpoint] Q1 optimizer load error: {e}", flush=True)
        if "q2_optimizer" in sd:
            try:
                self.q2_optimizer.load_state_dict(sd["q2_optimizer"])
            except (ValueError, RuntimeError) as e:
                print(f"[checkpoint] Q2 optimizer load error: {e}", flush=True)


class MultiHeadQCritic:
    """Manages all trunk groups for a SAC experiment.

    Groups channels by ``trunk_group`` (or auto-groups by gamma), builds
    twin Q + targets per group, and provides a unified interface for
    forward passes, updates, and target sync.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        channels: Tuple[SACRewardChannel, ...],
        hidden_dim: int,
        layer_norm: bool,
        critic_lr: float,
        device: torch.device,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.device = device
        self.channel_names = tuple(ch.name for ch in channels)

        # Group channels by trunk_group (or auto-group by gamma)
        groups: Dict[str, List[str]] = {}
        for ch in channels:
            group_key = ch.trunk_group if ch.trunk_group else f"gamma_{ch.gamma:.4f}"
            groups.setdefault(group_key, []).append(ch.name)

        self.group_keys = list(groups.keys())
        self.channel_to_group: Dict[str, str] = {}
        for gk, chs in groups.items():
            for ch in chs:
                self.channel_to_group[ch] = gk

        self.groups: Dict[str, QTrunkGroup] = {}
        for gk, chs in groups.items():
            grp = QTrunkGroup(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                channel_names=tuple(chs),
                layer_norm=layer_norm,
                device=device,
            )
            grp.set_lr(critic_lr)
            self.groups[gk] = grp

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def q1_forward(
        self, obs: torch.Tensor, action: torch.Tensor, channel: str,
    ) -> torch.Tensor:
        gk = self.channel_to_group[channel]
        return self.groups[gk].q1(obs, action, channel)

    def q2_forward(
        self, obs: torch.Tensor, action: torch.Tensor, channel: str,
    ) -> torch.Tensor:
        gk = self.channel_to_group[channel]
        return self.groups[gk].q2(obs, action, channel)

    def q1_target_forward(
        self, obs: torch.Tensor, action: torch.Tensor, channel: str,
    ) -> torch.Tensor:
        gk = self.channel_to_group[channel]
        return self.groups[gk].q1_target(obs, action, channel)

    def q2_target_forward(
        self, obs: torch.Tensor, action: torch.Tensor, channel: str,
    ) -> torch.Tensor:
        gk = self.channel_to_group[channel]
        return self.groups[gk].q2_target(obs, action, channel)

    def q1_forward_all(
        self, obs: torch.Tensor, action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Q1 for all channels. Uses one trunk forward per group."""
        results: Dict[str, torch.Tensor] = {}
        for gk, grp in self.groups.items():
            all_q = grp.q1.forward_all(obs, action)
            results.update(all_q)
        return results

    def q2_forward_all(
        self, obs: torch.Tensor, action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        results: Dict[str, torch.Tensor] = {}
        for gk, grp in self.groups.items():
            all_q = grp.q2.forward_all(obs, action)
            results.update(all_q)
        return results

    # ------------------------------------------------------------------
    # Target sync
    # ------------------------------------------------------------------

    def soft_update(self, tau: float) -> None:
        for grp in self.groups.values():
            grp.soft_update(tau)

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------

    def zero_grad_all(self) -> None:
        for grp in self.groups.values():
            grp.q1_optimizer.zero_grad()
            grp.q2_optimizer.zero_grad()

    def step_all(self) -> None:
        for grp in self.groups.values():
            grp.q1_optimizer.step()
            grp.q2_optimizer.step()

    def all_parameters(self):
        """Iterate all trainable Q parameters (for grad clip)."""
        for grp in self.groups.values():
            yield from grp.q1.parameters()
            yield from grp.q2.parameters()

    def q1_parameters(self):
        for grp in self.groups.values():
            yield from grp.q1.parameters()

    def q2_parameters(self):
        for grp in self.groups.values():
            yield from grp.q2.parameters()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            gk: grp.state_dict() for gk, grp in self.groups.items()
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        for gk, grp in self.groups.items():
            if gk in sd:
                grp.load_state_dict(sd[gk])
            else:
                print(f"[checkpoint] group '{gk}' not in checkpoint -> fresh init", flush=True)

    @property
    def n_networks(self) -> int:
        """Total Q networks (Q1 + Q2 per group, excluding targets)."""
        return 2 * len(self.groups)
