from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from envs.framework.policy import PolicyBlueprint
from baseline.humanoid21.curriculum.fight_mixed_policy import FightMixedPolicy


def test_fight_mixed_policy_blueprint_loading():
    # Load and build from the YAML blueprint directly to verify blueprint validity
    blueprint_path = REPO_ROOT / "baseline" / "humanoid21" / "blueprints" / "fight_mixed.yaml"
    assert blueprint_path.exists()
    
    pb = PolicyBlueprint.load(blueprint_path)
    assert pb.cls == "baseline.humanoid21.curriculum.fight_mixed_policy:FightMixedPolicy"
    
    # We can build it (primary and fallback defaults point to valid blueprints in combatbench)
    policy = pb.build()
    assert isinstance(policy, FightMixedPolicy)
    assert policy.active_mode == "fight"


def test_fight_mixed_policy_switching():
    # Create instance manually to mock/control properties
    blueprint_path = REPO_ROOT / "baseline" / "humanoid21" / "blueprints" / "fight_mixed.yaml"
    pb = PolicyBlueprint.load(blueprint_path)
    policy = pb.build()
    
    # Disable safety gating fallback for distance transition testing
    policy.threshold = 0.0
    
    # Mock observation: 96-dimensional array
    # Opponent rel_x, rel_y are at index 57, 58
    obs = np.zeros(96, dtype=np.float32)
    
    # 1. Close distance: within fight zone
    obs[57] = 0.5  # rel_x
    obs[58] = 0.5  # rel_y -> distance = sqrt(0.5) ~ 0.707m (<= 1.0m)
    
    # Run a step
    action, extra = policy.act(obs, want_extra=True)
    assert policy.active_mode == "fight"
    assert extra["gating_mode"] == 1.0
    
    # 2. Transition to Follow: distance > 1.3m
    obs[57] = 1.0
    obs[58] = 1.0  # distance = sqrt(2.0) ~ 1.414m (> 1.3m)
    action, extra = policy.act(obs, want_extra=True)
    assert policy.active_mode == "follow"
    assert extra["gating_mode"] == -1.0
    
    # 3. Stay in Follow: distance decreases to 1.1m (hysteresis, stays in follow since > 1.0m)
    obs[57] = 0.8
    obs[58] = 0.8  # distance = sqrt(1.28) ~ 1.13m (between 1.0m and 1.3m)
    action, extra = policy.act(obs, want_extra=True)
    assert policy.active_mode == "follow"
    assert extra["gating_mode"] == -1.0
    
    # 4. Transition back to Fight: distance <= 1.0m
    obs[57] = 0.6
    obs[58] = 0.6  # distance = sqrt(0.72) ~ 0.848m (<= 1.0m)
    action, extra = policy.act(obs, want_extra=True)
    assert policy.active_mode == "fight"
    assert extra["gating_mode"] == 1.0
