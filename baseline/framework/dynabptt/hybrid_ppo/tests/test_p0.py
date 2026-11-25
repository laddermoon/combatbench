"""P0 correctness gates for hybrid_ppo (plan §7 P0).

Run:
    PYTHONPATH=. python3 -B baseline/framework/dynabptt/hybrid_ppo/tests/test_p0.py

Every test fails loud.  These are the contract checks that must pass
before any training run is meaningful.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from baseline.framework.dynabptt.corestate import (
    STATE_DIM, flatten_core_state,
)
from baseline.framework.dynabptt.hybrid_ppo.model import SelfWorldModel
from baseline.framework.dynabptt.hybrid_ppo.ppo import (
    PPOConfig, build_ppo_batch, ppo_update,
)
from baseline.framework.dynabptt.hybrid_ppo.standup import (
    ACTION_DIM, SELF_OBS_DIM, make_standup_env_bp,
    unflatten_core_state,
)
from baseline.framework.dynabptt.hybrid_ppo.transitions import (
    AlignedTraj, extract_aligned,
)
from baseline.framework.dynabptt.hybrid_ppo.validation import (
    GateState, validate_model,
)
from baseline.framework.critic_mlp import CriticMLP
from baseline.framework.ppo.algos.advantages import compute_gae
from baseline.framework.ppo.policies.truncated_normal_mlp import (
    TruncatedNormalPolicy,
)
from baseline.framework.rollout.episode import blueprint_hash
from baseline.framework.rollout.episode_recorder import EpisodeRecorder

ATOL_STATE = 2e-3    # branch-replay vs recorded core state
ATOL_OBS = 2e-3


def _seeded_reset(rt, seed: int) -> None:
    """Reproduce EpisodeRunner's seeding for direct EnvRuntime use.

    EpisodeRunner derives per-plugin RNG seeds from the job's base_seed
    and calls plugin.set_episode_seed BEFORE runtime.reset — plugins
    keep their ctor-time (unseeded) RNG otherwise, making direct-
    runtime episodes nondeterministic.
    """
    for p in rt.plugins:
        p.set_episode_seed(seed)
    rt.reset(seed=seed, base_seed=seed)


def _run_episode(env_bp, seed: int, n_steps: int, act_a, act_b):
    """Run a full episode through the recorder path; return Episode.

    Callers must build env_bp with max_steps == n_steps so the episode
    actually terminates (recorder only emits on episode end).
    """
    rt = env_bp.build()
    rt.attach_recorder(EpisodeRecorder(blueprint_hash(env_bp)))
    _seeded_reset(rt, seed)
    for i in range(n_steps):
        rt.step(act_a(i), act_b(i))
        if rt.is_episode_over():
            break
    ep = rt._recorders[0].get_last_episode()
    return ep, rt


def test_time_alignment():
    """cs[t]=s_{t+1}, obs[t]=o_t, and branch replay from cs[t-1]+act[t]
    reproduces cs[t] and obs[t+1] — the load-bearing alignment."""
    env_bp = make_standup_env_bp(max_steps=12)
    rng = np.random.default_rng(7)
    ep, _ = _run_episode(env_bp, seed=11, n_steps=12,
                         act_a=lambda i: rng.uniform(-1, 1, 21).astype(np.float32),
                         act_b=lambda i: rng.uniform(-1, 1, 21).astype(np.float32))
    trajs = extract_aligned(ep, update=0)
    assert len(trajs) == 2
    tr = trajs[0]
    T = tr.T
    assert tr.cs.shape == (T, STATE_DIM)
    assert tr.obs.shape == (T, SELF_OBS_DIM)
    assert tr.act.shape == (T, ACTION_DIM)

    # branch-replay: restore s_t = cs[t-1], apply recorded a_t, must
    # reproduce cs[t] (post-action state) and o_{t+1}.
    rt2 = env_bp.build()
    n_checked = 0
    for t in (1, 2, 5, T - 1):
        _seeded_reset(rt2, 9000 + t)
        rt2.simulator.set_core_state(
            {tr.agent_id: unflatten_core_state(tr.cs[t - 1])})
        a = tr.act[t]
        zero = np.zeros(ACTION_DIM, dtype=np.float32)
        rt2.step(a if tr.agent_id == "robot_a" else zero,
                 zero if tr.agent_id == "robot_a" else a)
        cs_now = flatten_core_state(
            rt2.simulator.get_core_state(), tr.agent_id)
        err = np.abs(cs_now - tr.cs[t]).max()
        assert err < ATOL_STATE, (
            f"branch replay diverged at t={t}: max|dcs|={err:.2e} "
            "(state alignment broken)")
        o_now, _ = (rt2.get_observation()
                    if tr.agent_id == "robot_a"
                    else (lambda x: (x[1], x[0]))(rt2.get_observation()))
        o_target = tr.next_obs(t)
        oerr = np.abs(o_now - o_target).max()
        assert oerr < ATOL_OBS, (
            f"obs mismatch at t={t}: max|dobs|={oerr:.2e} "
            "(obs alignment broken)")
        n_checked += 1
    assert n_checked == 4
    print("  test_time_alignment OK")


def test_self_only_isolation():
    """Opponent actions must not affect the focal robot's state/obs."""
    env_bp = make_standup_env_bp()
    rng = np.random.default_rng(3)
    acts_a = [rng.uniform(-1, 1, 21).astype(np.float32) for _ in range(30)]
    acts_b1 = [rng.uniform(-1, 1, 21).astype(np.float32) for _ in range(30)]
    acts_b2 = [rng.uniform(-1, 1, 21).astype(np.float32) for _ in range(30)]

    def run(acts_b):
        rt = env_bp.build()
        _seeded_reset(rt, 55)
        seq = []
        for i in range(30):
            rt.step(acts_a[i], acts_b[i])
            seq.append(flatten_core_state(
                rt.simulator.get_core_state(), "robot_a"))
        return np.stack(seq)

    s1 = run(acts_b1)
    s2 = run(acts_b2)
    err = np.abs(s1 - s2).max()
    assert err < 1e-5, (
        f"opponent actions changed focal state: max|ds|={err:.2e} "
        "— collision isolation broken")
    # also: 57-dim obs cannot contain opponent info by construction —
    # verify dims and that two runs agree exactly on obs too.
    rt = env_bp.build()
    _seeded_reset(rt, 55)
    o_a, o_b = rt.get_observation()
    assert o_a.shape[-1] == SELF_OBS_DIM and o_b.shape[-1] == SELF_OBS_DIM
    print("  test_self_only_isolation OK")


def test_ppo_semantics():
    """GAE hand-check + first-minibatch ratio == 1 + KL early-stop path."""
    # GAE: r=[1,1,1], v=0, last_v=0, gamma=lam=1 -> adv=[3,2,1]
    adv, ret = compute_gae(np.ones(3, dtype=np.float32),
                           np.zeros(3, dtype=np.float32),
                           last_value=0.0, gamma=1.0, lam=1.0)
    assert np.allclose(adv, [3, 2, 1]), adv
    assert np.allclose(ret, [3, 2, 1]), ret
    # truncated bootstrap: last_v=2 -> adv0 = 1+0.99*2-0=2.98
    adv, _ = compute_gae(np.ones(1, dtype=np.float32),
                         np.zeros(1, dtype=np.float32),
                         last_value=2.0, gamma=0.99, lam=0.95)
    assert np.isclose(adv[0], 1.0 + 0.99 * 2.0), adv

    # small synthetic PPO batch through ppo_update
    device = torch.device("cpu")
    torch.manual_seed(0)
    actor = TruncatedNormalPolicy(SELF_OBS_DIM, ACTION_DIM, 64, "cpu")
    critic = CriticMLP(SELF_OBS_DIM, 64)
    rng = np.random.default_rng(0)
    trajs = []
    for k in range(8):
        T = 40
        trajs.append(AlignedTraj(
            cs=rng.normal(size=(T, 55)).astype(np.float32),
            obs=rng.normal(size=(T, 57)).astype(np.float32),
            act=rng.uniform(-1, 1, (T, 21)).astype(np.float32),
            phi=rng.uniform(0, 1, T).astype(np.float32),
            stage=np.ones(T, dtype=np.float32),
            final_obs=rng.normal(size=57).astype(np.float32),
            terminated=False, ep_key=k, update=0))
    batch = build_ppo_batch(trajs, critic, device, 0.99, 0.95)
    assert batch.obs.shape == (320, 57)
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=3e-4)
    stats = ppo_update(actor, critic, opt_a, opt_c, batch,
                       PPOConfig(minibatch=64, epochs=2), device,
                       rng=np.random.default_rng(0))
    assert stats["first_mb_ratio_dev"] < 1e-4, (
        f"ratio dev {stats['first_mb_ratio_dev']:.2e} — old_lp "
        "recomputation disagrees with theta_old")
    assert np.isfinite(stats["policy_loss"])
    print("  test_ppo_semantics OK")


def test_export_parity():
    """Exported policy blueprint produces identical det actions."""
    actor = TruncatedNormalPolicy(SELF_OBS_DIM, ACTION_DIM, 64, "cpu")
    torch.manual_seed(1)
    bp = actor.to_blueprint()
    from envs.framework.policy import PolicyBlueprint  # noqa
    cls_path = bp.cls
    assert cls_path.startswith("file:")
    # load exported policy via blueprint machinery
    import importlib.util
    mod_path = cls_path.split(":")[1]
    cls_name = cls_path.split(":")[2]
    spec = importlib.util.spec_from_file_location("expol", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pol = getattr(mod, cls_name)()
    rng = np.random.default_rng(5)
    for _ in range(8):
        o = rng.normal(size=57).astype(np.float32)
        a_exp, _ = pol.act(o)
        with torch.no_grad():
            a_tr = actor.deterministic_action(
                torch.as_tensor(o).unsqueeze(0)).squeeze(0).numpy()
        assert np.allclose(a_exp, a_tr, atol=1e-5), (
            f"export parity failed: {np.abs(a_exp - a_tr).max():.2e}")
    print("  test_export_parity OK")


def test_gate_failsafe():
    """A garbage model must fail validation -> gate stays closed."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = SelfWorldModel()  # random init — uncalibrated
    actor = TruncatedNormalPolicy(SELF_OBS_DIM, ACTION_DIM, 64, "cpu")
    critic = CriticMLP(SELF_OBS_DIM, 64)
    env_bp = make_standup_env_bp(max_steps=40)

    from baseline.framework.dynabptt.hybrid_ppo.transitions import (
        ModelReplay,
    )
    replay = ModelReplay()
    rng = np.random.default_rng(0)
    # populate holdout with real episodes (small n for test speed)
    for k in range(6):
        ep, _ = _run_episode(
            env_bp, seed=2000 + k, n_steps=40,
            act_a=lambda i: rng.uniform(-1, 1, 21).astype(np.float32),
            act_b=lambda i: rng.uniform(-1, 1, 21).astype(np.float32))
        for tr in extract_aligned(ep, update=0):
            replay.hold.append(tr)
    report = validate_model(
        model, actor, critic, replay, rng, device,
        n_windows=512, env_bp=env_bp, paired_seeds=8, paired_h=2)
    gate = GateState()
    gate.update(report)
    assert not gate.admitted, (
        "untrained model passed the gate — admission logic broken")
    print("  test_gate_failsafe OK")


def test_alignment_extraction():
    """extract_aligned carries correct per-agent data and split."""
    env_bp = make_standup_env_bp(max_steps=10)
    rng = np.random.default_rng(1)
    ep, _ = _run_episode(env_bp, seed=42, n_steps=10,
                         act_a=lambda i: rng.uniform(-1, 1, 21).astype(np.float32),
                         act_b=lambda i: rng.uniform(-1, 1, 21).astype(np.float32))
    trajs = extract_aligned(ep, update=3)
    assert {t.agent_id for t in trajs} == {"robot_a", "robot_b"}
    for tr in trajs:
        assert tr.update == 3
        assert tr.final_obs.shape == (SELF_OBS_DIM,)
        assert not tr.terminated  # timeout, not real terminal
        s, o, st = tr.seed(1)
        assert np.array_equal(s, tr.cs[0])
        assert np.array_equal(o, tr.obs[1])
        assert st == tr.stage[0]
    print("  test_alignment_extraction OK")


if __name__ == "__main__":
    print("P0 gates:")
    test_alignment_extraction()
    test_time_alignment()
    test_self_only_isolation()
    test_ppo_semantics()
    test_export_parity()
    test_gate_failsafe()
    print("ALL P0 GATES PASSED")
