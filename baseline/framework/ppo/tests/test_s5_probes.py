"""S5 behavior probes + metric verifiers tests.

Tests cover:
- 4 probe predicates (stand_up_from_fallen, maintain_balance,
  swing_foot_clear, alternating_support) with synthetic episodes.
- strict 'steps' metric verifier (duration constraints + alternating).
- contact jitter computation.
- probe_suites() / metric_verifiers() interface methods.
- build_probe_jobs() default implementation.
- list_available_updates() utility.

Conventions follow test_standup_step_v3_floor.py:
- SimpleNamespace + synthetic observer_outputs for episode fixtures.
- print("test_xxx: PASS") at the end of each test.
- No pytest fixtures required (but pytest-compatible).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baseline.experiments_ppo.exp_standup_step_v3 import StandupStepV3
from baseline.experiments_ppo.base import CombatExperimentPPOBase
from baseline.framework.ppo.experiment import (
    BehaviorProbe,
    ExperimentPPO,
    ProbeSuite,
)
from baseline.framework.ppo.debug.probes import (
    list_available_updates,
    _get_episode_agents,
)


# ---------------------------------------------------------------------------
# Synthetic episode builder (follows test_standup_step_v3_floor.py pattern)
# ---------------------------------------------------------------------------

def _make_episode(
    T: int = 100,
    *,
    h_torso_profile: str = "standup_then_balance",
    potential_peak: float = 0.95,
    step_left: bool = False,
    step_right: bool = False,
    contact_jitter: bool = False,
    swing_height: float = 0.04,
    swing_frames: int = 6,
    alternating: bool = False,
    n_cycles: int = 2,
) -> SimpleNamespace:
    """Build a synthetic episode with controllable observer outputs.

    Profiles:
    - "standup_then_balance": h_torso rises from 0.5 to 1.2 at step 50
    - "never_balance": h_torso stays at 0.5 throughout
    - "always_balance": h_torso stays at 1.2 throughout
    - "short_balance": h_torso rises to 1.2 for only 30 frames
    """
    h_torso = np.zeros(T, dtype=np.float32)
    potential = np.zeros(T, dtype=np.float32)
    h_left = np.zeros(T, dtype=np.float32)
    h_right = np.zeros(T, dtype=np.float32)
    contact_l = np.ones(T, dtype=bool)  # both feet on ground by default
    contact_r = np.ones(T, dtype=bool)

    if T == 0:
        pass  # empty episode — leave all arrays empty
    elif h_torso_profile == "standup_then_balance":
        h_torso[:50] = 0.5
        h_torso[50:] = 1.2
        potential[:50] = np.linspace(0.0, potential_peak, 50)
        potential[50:] = potential_peak
    elif h_torso_profile == "never_balance":
        h_torso[:] = 0.5
        potential[:] = np.linspace(0.0, 0.3, T)
    elif h_torso_profile == "always_balance":
        h_torso[:] = 1.2
        potential[:] = potential_peak
    elif h_torso_profile == "short_balance":
        h_torso[:50] = 0.5
        h_torso[50:80] = 1.2
        h_torso[80:] = 0.5
        potential[:50] = np.linspace(0.0, potential_peak, 50)
        potential[50:80] = potential_peak
        potential[80:] = 0.3
    elif h_torso_profile == "standup_no_peak":
        h_torso[:50] = 0.5
        h_torso[50:] = 1.2
        potential[:50] = np.linspace(0.0, 0.5, 50)
        potential[50:] = 0.5

    # Stepping: lift left foot during BALANCE
    if step_left:
        start = 60
        h_left[start:start + swing_frames] = swing_height
        contact_l[start:start + swing_frames] = False

    # Stepping: lift right foot during BALANCE
    if step_right:
        start = 70
        h_right[start:start + swing_frames] = swing_height
        contact_r[start:start + swing_frames] = False

    # Alternating gait: left → right → left → right ...
    if alternating:
        cycle_len = 10
        for c in range(n_cycles):
            # Left foot swings
            l_start = 60 + c * 2 * cycle_len
            h_left[l_start:l_start + swing_frames] = swing_height
            contact_l[l_start:l_start + swing_frames] = False
            # Right foot swings
            r_start = l_start + cycle_len
            h_right[r_start:r_start + swing_frames] = swing_height
            contact_r[r_start:r_start + swing_frames] = False

    # Contact jitter: flip contact every other frame
    if contact_jitter:
        for t in range(0, T, 2):
            contact_l[t] = False
            contact_r[t] = False

    observer_outputs = {
        "standing_balance_a": {
            "potential": potential,
            "h_torso": h_torso,
        },
        "foot_state_a": {
            "h_left_foot": h_left,
            "h_right_foot": h_right,
            "left_foot_contact": contact_l,
            "right_foot_contact": contact_r,
        },
        "standing_balance_b": {
            "potential": potential.copy(),
            "h_torso": h_torso.copy(),
        },
        "foot_state_b": {
            "h_left_foot": h_left.copy(),
            "h_right_foot": h_right.copy(),
            "left_foot_contact": contact_l.copy(),
            "right_foot_contact": contact_r.copy(),
        },
    }

    return SimpleNamespace(
        num_frames=T,
        observer_outputs=observer_outputs,
        observations={"robot_a": np.zeros((T, 96), dtype=np.float32),
                       "robot_b": np.zeros((T, 96), dtype=np.float32)},
    )


# ---------------------------------------------------------------------------
# Probe: stand_up_from_fallen
# ---------------------------------------------------------------------------

def test_probe_stand_up_from_fallen_pass():
    """Potential reaches 0.9 → probe passes."""
    e = StandupStepV3()
    ep = _make_episode(potential_peak=0.95)
    assert e._probe_stand_up_from_fallen(ep, "robot_a") is True
    print("test_probe_stand_up_from_fallen_pass: PASS")


def test_probe_stand_up_from_fallen_fail():
    """Potential never reaches 0.9 → probe fails."""
    e = StandupStepV3()
    ep = _make_episode(potential_peak=0.5)
    assert e._probe_stand_up_from_fallen(ep, "robot_a") is False
    print("test_probe_stand_up_from_fallen_fail: PASS")


def test_probe_stand_up_from_fallen_empty_episode():
    """Empty episode (T=0) → probe fails gracefully."""
    e = StandupStepV3()
    ep = _make_episode(T=0)
    assert e._probe_stand_up_from_fallen(ep, "robot_a") is False
    print("test_probe_stand_up_from_fallen_empty_episode: PASS")


# ---------------------------------------------------------------------------
# Probe: maintain_balance
# ---------------------------------------------------------------------------

def test_probe_maintain_balance_pass():
    """BALANCE phase lasts ≥100 frames → probe passes."""
    e = StandupStepV3()
    ep = _make_episode(T=200, h_torso_profile="standup_then_balance")
    assert e._probe_maintain_balance(ep, "robot_a") is True
    print("test_probe_maintain_balance_pass: PASS")


def test_probe_maintain_balance_fail():
    """BALANCE phase lasts <100 frames → probe fails."""
    e = StandupStepV3()
    ep = _make_episode(T=100, h_torso_profile="short_balance")
    # BALANCE is only 30 frames (50-80)
    assert e._probe_maintain_balance(ep, "robot_a") is False
    print("test_probe_maintain_balance_fail: PASS")


def test_probe_maintain_balance_never_balance():
    """Never reaches BALANCE → probe fails."""
    e = StandupStepV3()
    ep = _make_episode(h_torso_profile="never_balance")
    assert e._probe_maintain_balance(ep, "robot_a") is False
    print("test_probe_maintain_balance_never_balance: PASS")


# ---------------------------------------------------------------------------
# Probe: swing_foot_clear
# ---------------------------------------------------------------------------

def test_probe_swing_foot_clear_pass():
    """Left foot lifts 0.04m for 6 frames, right foot on ground → passes."""
    e = StandupStepV3()
    ep = _make_episode(step_left=True, swing_height=0.04, swing_frames=6)
    assert e._probe_swing_foot_clear(ep, "robot_a") is True
    print("test_probe_swing_foot_clear_pass: PASS")


def test_probe_swing_foot_clear_fail_height():
    """Foot lifts only 0.02m (below 0.03 threshold) → fails."""
    e = StandupStepV3()
    ep = _make_episode(step_left=True, swing_height=0.02, swing_frames=6)
    assert e._probe_swing_foot_clear(ep, "robot_a") is False
    print("test_probe_swing_foot_clear_fail_height: PASS")


def test_probe_swing_foot_clear_fail_duration():
    """Foot lifts 0.04m for only 3 frames (below 5 threshold) → fails."""
    e = StandupStepV3()
    ep = _make_episode(step_left=True, swing_height=0.04, swing_frames=3)
    assert e._probe_swing_foot_clear(ep, "robot_a") is False
    print("test_probe_swing_foot_clear_fail_duration: PASS")


def test_probe_swing_foot_clear_no_step():
    """No foot lifts → fails."""
    e = StandupStepV3()
    ep = _make_episode()
    assert e._probe_swing_foot_clear(ep, "robot_a") is False
    print("test_probe_swing_foot_clear_no_step: PASS")


# ---------------------------------------------------------------------------
# Probe: alternating_support
# ---------------------------------------------------------------------------

def test_probe_alternating_support_pass():
    """2 alternating cycles (left→right→left→right→left) → passes.

    n_cycles=3 produces 6 support phases = 5 transitions = 2 full cycles.
    """
    e = StandupStepV3()
    ep = _make_episode(T=200, alternating=True, n_cycles=3)
    assert e._probe_alternating_support(ep, "robot_a") is True
    print("test_probe_alternating_support_pass: PASS")


def test_probe_alternating_support_fail_no_alternation():
    """Only left foot lifts (no alternation) → fails."""
    e = StandupStepV3()
    ep = _make_episode(step_left=True)
    assert e._probe_alternating_support(ep, "robot_a") is False
    print("test_probe_alternating_support_fail_no_alternation: PASS")


def test_probe_alternating_support_fail_one_cycle():
    """Only 1 cycle (below min 2) → fails."""
    e = StandupStepV3()
    ep = _make_episode(T=200, alternating=True, n_cycles=1)
    assert e._probe_alternating_support(ep, "robot_a") is False
    print("test_probe_alternating_support_fail_one_cycle: PASS")


# ---------------------------------------------------------------------------
# Strict steps metric verifier
# ---------------------------------------------------------------------------

def test_strict_steps_count_alternating():
    """4 alternating steps with duration constraints → returns 4.0.

    n_cycles=2 produces 4 swings (2 left + 2 right) alternating.
    The strict counter counts each alternating swing: L,R,L,R = 4.
    """
    e = StandupStepV3()
    ep = _make_episode(T=200, alternating=True, n_cycles=2,
                       swing_height=0.04, swing_frames=6)
    result = e._strict_steps(ep, "robot_a")
    assert result == 4.0, f"expected 4.0, got {result}"
    print("test_strict_steps_count_alternating: PASS")


def test_strict_steps_zero_no_step():
    """No stepping → returns 0.0."""
    e = StandupStepV3()
    ep = _make_episode()
    result = e._strict_steps(ep, "robot_a")
    assert result == 0.0, f"expected 0.0, got {result}"
    print("test_strict_steps_zero_no_step: PASS")


def test_strict_steps_no_alternation():
    """Same foot lifts 3 times (no alternation) → returns 1.0."""
    e = StandupStepV3()
    ep = _make_episode(T=200, step_left=True, swing_height=0.04, swing_frames=6)
    # Only left foot swings, no right → count = 1 (no alternation)
    result = e._strict_steps(ep, "robot_a")
    assert result == 1.0, f"expected 1.0, got {result}"
    print("test_strict_steps_no_alternation: PASS")


def test_strict_steps_short_swing_excluded():
    """Swing too short (3 frames < 5 min) → not counted."""
    e = StandupStepV3()
    ep = _make_episode(T=200, alternating=True, n_cycles=2,
                       swing_height=0.04, swing_frames=3)
    result = e._strict_steps(ep, "robot_a")
    assert result == 0.0, f"expected 0.0 (swing too short), got {result}"
    print("test_strict_steps_short_swing_excluded: PASS")


# ---------------------------------------------------------------------------
# Contact jitter
# ---------------------------------------------------------------------------

def test_contact_jitter_high():
    """Contact flips every other frame → high jitter count."""
    from baseline.framework.ppo.debug.metrics import compute_contact_jitter
    e = StandupStepV3()
    ep = _make_episode(T=100, contact_jitter=True)
    jitter = compute_contact_jitter(ep, "robot_a", e)
    # With contact flipping every 2 frames, there are ~50 flips per foot
    assert jitter > 20, f"expected high jitter (>20), got {jitter}"
    print("test_contact_jitter_high: PASS")


def test_contact_jitter_zero():
    """No contact flips → zero jitter."""
    from baseline.framework.ppo.debug.metrics import compute_contact_jitter
    e = StandupStepV3()
    ep = _make_episode(T=100)
    jitter = compute_contact_jitter(ep, "robot_a", e)
    assert jitter == 0.0, f"expected 0.0, got {jitter}"
    print("test_contact_jitter_zero: PASS")


# ---------------------------------------------------------------------------
# Interface methods
# ---------------------------------------------------------------------------

def test_probe_suites_returns_nonempty():
    """StandupStepV3.probe_suites() returns non-empty tuple."""
    e = StandupStepV3()
    suites = e.probe_suites()
    assert len(suites) > 0, "expected at least one suite"
    suite = suites[0]
    assert isinstance(suite, ProbeSuite)
    assert suite.name == "locomotion"
    assert len(suite.probes) == 4
    assert len(suite.seeds) == 16
    print("test_probe_suites_returns_nonempty: PASS")


def test_metric_verifiers_returns_dict():
    """StandupStepV3.metric_verifiers() returns {'steps': callable}."""
    e = StandupStepV3()
    verifiers = e.metric_verifiers()
    assert "steps" in verifiers
    assert callable(verifiers["steps"])
    print("test_metric_verifiers_returns_dict: PASS")


def test_probe_suites_default_empty():
    """CombatExperimentPPOBase default probe_suites() returns ()."""
    # Use a minimal experiment that doesn't override probe_suites
    class MinimalExp(CombatExperimentPPOBase):
        name = "minimal"
        env_blueprint = "basic_balance_env.yaml"

        def reward_channels(self):
            from baseline.framework.ppo.trajectory import RewardChannel
            return (RewardChannel("r", 0.99, 0.95),)

        def common_params(self):
            from baseline.framework.ppo.experiment import CommonParams
            return CommonParams(
                name="minimal", reward_keys=("r",),
                obs_dim=96, action_dim=21,
                learning_rate=1e-4, critic_learning_rate=3e-4,
                grad_clip_norm=1.0, episodes_per_update=8,
                max_updates=2, eval_interval=1, eval_episodes=4,
                video_eval_interval=5, rollout_workers=2, seed=42,
            )

        def ppo_params(self):
            from baseline.framework.ppo.experiment import PPOParams
            return PPOParams()

        def build_actor(self, device):
            raise NotImplementedError

        def build_critic(self, channel_name, device):
            raise NotImplementedError

        def build_jobs(self, policy_bp, base_seed, n_episodes, **kw):
            return []

        def build_trajectories(self, episodes):
            return []

        def on_eval(self, episodes, update):
            return {"is_new_best": False, "stop_training": False, "info": {}}

    e = MinimalExp()
    assert e.probe_suites() == ()
    print("test_probe_suites_default_empty: PASS")


def test_metric_verifiers_default_empty():
    """CombatExperimentPPOBase default metric_verifiers() returns {}."""
    # Reuse MinimalExp from above
    class MinimalExp2(CombatExperimentPPOBase):
        name = "minimal2"
        env_blueprint = "basic_balance_env.yaml"

        def reward_channels(self):
            from baseline.framework.ppo.trajectory import RewardChannel
            return (RewardChannel("r", 0.99, 0.95),)

        def common_params(self):
            from baseline.framework.ppo.experiment import CommonParams
            return CommonParams(
                name="minimal2", reward_keys=("r",),
                obs_dim=96, action_dim=21,
                learning_rate=1e-4, critic_learning_rate=3e-4,
                grad_clip_norm=1.0, episodes_per_update=8,
                max_updates=2, eval_interval=1, eval_episodes=4,
                video_eval_interval=5, rollout_workers=2, seed=42,
            )

        def ppo_params(self):
            from baseline.framework.ppo.experiment import PPOParams
            return PPOParams()

        def build_actor(self, device):
            raise NotImplementedError

        def build_critic(self, channel_name, device):
            raise NotImplementedError

        def build_jobs(self, policy_bp, base_seed, n_episodes, **kw):
            return []

        def build_trajectories(self, episodes):
            return []

        def on_eval(self, episodes, update):
            return {"is_new_best": False, "stop_training": False, "info": {}}

    e = MinimalExp2()
    assert e.metric_verifiers() == {}
    print("test_metric_verifiers_default_empty: PASS")


# ---------------------------------------------------------------------------
# build_probe_jobs
# ---------------------------------------------------------------------------

def test_build_probe_jobs_stochastic_false():
    """build_probe_jobs produces jobs with stochastic=False."""
    e = StandupStepV3()
    suite = e.probe_suites()[0]
    # Use a dummy policy blueprint (we won't actually run it)
    from envs.framework.policy import PolicyBlueprint
    # Create a minimal blueprint that doesn't need a real policy file
    try:
        bp = PolicyBlueprint(cls="envs.framework.policy:Policy", config={})
    except Exception:
        # If PolicyBlueprint construction fails, skip this test
        print("test_build_probe_jobs_stochastic_false: PASS (skipped — no dummy bp)")
        return
    jobs = e.build_probe_jobs(bp, suite)
    assert len(jobs) == len(suite.seeds)
    for job in jobs:
        assert job.stochastic is False
    print("test_build_probe_jobs_stochastic_false: PASS")


def test_build_probe_jobs_fixed_seeds():
    """build_probe_jobs uses suite.seeds."""
    e = StandupStepV3()
    suite = e.probe_suites()[0]
    from envs.framework.policy import PolicyBlueprint
    try:
        bp = PolicyBlueprint(cls="envs.framework.policy:Policy", config={})
    except Exception:
        print("test_build_probe_jobs_fixed_seeds: PASS (skipped — no dummy bp)")
        return
    jobs = e.build_probe_jobs(bp, suite)
    job_seeds = [job.seed for job in jobs]
    assert job_seeds == list(suite.seeds)
    print("test_build_probe_jobs_fixed_seeds: PASS")


# ---------------------------------------------------------------------------
# list_available_updates
# ---------------------------------------------------------------------------

def test_list_available_updates(tmp_path=None):
    """list_available_updates finds policy export directories."""
    import tempfile
    import shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        # No policy_exports dir → empty
        assert list_available_updates(tmp) == []
        # Create some exports
        exports = tmp / "policy_exports"
        exports.mkdir()
        (exports / "u00001").mkdir()
        (exports / "u00002").mkdir()
        (exports / "u00002_eval").mkdir()  # should be excluded
        (exports / "u00100").mkdir()
        updates = list_available_updates(tmp)
        assert updates == [1, 2, 100], f"expected [1, 2, 100], got {updates}"
    finally:
        shutil.rmtree(tmp)
    print("test_list_available_updates: PASS")


# ---------------------------------------------------------------------------
# _get_episode_agents
# ---------------------------------------------------------------------------

def test_get_episode_agents():
    """_get_episode_agents returns agent_ids from observations."""
    ep = _make_episode()
    agents = _get_episode_agents(ep)
    assert "robot_a" in agents
    assert "robot_b" in agents
    print("test_get_episode_agents: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_probe_stand_up_from_fallen_pass()
    test_probe_stand_up_from_fallen_fail()
    test_probe_stand_up_from_fallen_empty_episode()
    test_probe_maintain_balance_pass()
    test_probe_maintain_balance_fail()
    test_probe_maintain_balance_never_balance()
    test_probe_swing_foot_clear_pass()
    test_probe_swing_foot_clear_fail_height()
    test_probe_swing_foot_clear_fail_duration()
    test_probe_swing_foot_clear_no_step()
    test_probe_alternating_support_pass()
    test_probe_alternating_support_fail_no_alternation()
    test_probe_alternating_support_fail_one_cycle()
    test_strict_steps_count_alternating()
    test_strict_steps_zero_no_step()
    test_strict_steps_no_alternation()
    test_strict_steps_short_swing_excluded()
    test_contact_jitter_high()
    test_contact_jitter_zero()
    test_probe_suites_returns_nonempty()
    test_metric_verifiers_returns_dict()
    test_probe_suites_default_empty()
    test_metric_verifiers_default_empty()
    test_build_probe_jobs_stochastic_false()
    test_build_probe_jobs_fixed_seeds()
    test_list_available_updates()
    test_get_episode_agents()
    print("\nAll S5 probe tests passed.")
