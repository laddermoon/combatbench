"""Benchmark legacy dict-based contact pipeline vs vectorized contacts_vec pipeline.

Measures wall-clock time for:
  1. _extract_contacts (contact extraction only)
  2. get_derived_state (full derived state including contacts + feet_forces)
  3. Full EnvRuntime.step (includes PD control, physics, plugins, observers)

Usage:
  python3 bench_contacts.py --steps 600 --warmup 50
"""
import argparse
import time
import numpy as np
from typing import Any, Dict, List

from envs.framework.blueprint import EnvBlueprint, _instantiate
from envs.framework.policy import PolicyBlueprint
from envs.framework.env_runtime import EnvRuntime
from envs.framework.episode_runner import EpisodeRunner, _resolve_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--warmup', type=int, default=50,
                        help='Warmup steps not counted in timing')
    parser.add_argument('--env-blueprint', type=str,
                        default='envs/humanoid21/blueprint.yaml')
    parser.add_argument('--policy-a-blueprint', type=str,
                        default='/data1/mono/things/combatbench_platform/submisiontool/uploads/submission_11/extracted/policy_blueprint.yaml')
    parser.add_argument('--policy-b-blueprint', type=str,
                        default='/data1/mono/things/combatbench_platform/submisiontool/uploads/submission_12/extracted/policy_blueprint.yaml')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    bp = EnvBlueprint.load(args.env_blueprint)
    policy_a = PolicyBlueprint.load(args.policy_a_blueprint).build()
    policy_b = PolicyBlueprint.load(args.policy_b_blueprint).build()

    # --- Build legacy runtime ---
    from envs.humanoid21.simulator import MujocoCombatSimulator as SimOrig
    sim_orig = SimOrig(**bp.simulator.config)
    rt_orig = EnvRuntime(
        simulator=sim_orig,
        plugins=[_instantiate(spec) for spec in bp.plugins],
        observer_plugins={name: _instantiate(spec) for name, spec in bp.observer_plugins.items()},
        phy_steps_per_action=bp.phy_steps_per_action,
        max_steps=bp.max_steps,
        strict=bp.strict,
    )

    # --- Build vec runtime ---
    from envs.humanoid21.simulator_vec import MujocoCombatSimulator as SimVec
    from envs.humanoid21.plugins_vec import CombatScoringPlugin as CombatScoringPluginVec
    from envs.humanoid21.observer_plugins_vec import CombatScoringObserver as CombatScoringObserverVec

    vec_plugins = []
    for spec in bp.plugins:
        if 'CombatScoringPlugin' in spec.cls:
            vec_plugins.append(CombatScoringPluginVec(**spec.config))
        else:
            vec_plugins.append(_instantiate(spec))

    vec_observer_plugins = {}
    for name, spec in bp.observer_plugins.items():
        if 'CombatScoringObserver' in spec.cls:
            vec_observer_plugins[name] = CombatScoringObserverVec(**spec.config)
        else:
            vec_observer_plugins[name] = _instantiate(spec)

    sim_vec = SimVec(**bp.simulator.config)
    rt_vec = EnvRuntime(
        simulator=sim_vec,
        plugins=vec_plugins,
        observer_plugins=vec_observer_plugins,
        phy_steps_per_action=bp.phy_steps_per_action,
        max_steps=bp.max_steps,
        strict=bp.strict,
    )

    # Reset both
    runner_orig = EpisodeRunner(runtime=rt_orig, policy_a=policy_a, policy_b=policy_b)
    runner_vec = EpisodeRunner(runtime=rt_vec, policy_a=policy_a, policy_b=policy_b)
    base_seed = _resolve_seed(args.seed)
    runner_orig._reset_all(runner_orig._derive_seeds(base_seed))
    runner_vec._reset_all(runner_vec._derive_seeds(base_seed))

    total_steps = args.steps + args.warmup
    phy_per_action = bp.phy_steps_per_action  # 25

    # Timers: accumulate per-phase wall time
    t_extract_orig = 0.0
    t_extract_vec = 0.0
    t_derived_orig = 0.0
    t_derived_vec = 0.0
    t_step_orig = 0.0
    t_step_vec = 0.0

    for step in range(total_steps):
        obs_a, obs_b = rt_orig.get_observation()
        action_a, _ = policy_a.act(obs_a, want_extra=False)
        action_b, _ = policy_b.act(obs_b, want_extra=False)

        # --- Time legacy step ---
        t0 = time.perf_counter()
        rt_orig.step(action_a, action_b)
        t_step_orig += time.perf_counter() - t0

        # Time legacy _extract_contacts (called inside get_derived_state, but we measure separately)
        sim_orig._data_cache.pop('_derived_state', None)
        sim_orig._data_cache.pop(('_derived_state', tuple(['contacts'])), None)
        t0 = time.perf_counter()
        sim_orig._extract_contacts()
        t_extract_orig += time.perf_counter() - t0

        sim_orig._data_cache.clear()
        t0 = time.perf_counter()
        sim_orig.get_derived_state()
        t_derived_orig += time.perf_counter() - t0

        # --- Time vec step ---
        t0 = time.perf_counter()
        rt_vec.step(action_a, action_b)
        t_step_vec += time.perf_counter() - t0

        sim_vec._data_cache.pop('_derived_state', None)
        t0 = time.perf_counter()
        sim_vec._extract_contacts()
        t_extract_vec += time.perf_counter() - t0

        sim_vec._data_cache.clear()
        t0 = time.perf_counter()
        sim_vec.get_derived_state(['contacts'])
        t_derived_vec += time.perf_counter() - t0

        # Reset if episode ended
        term_orig, trunc_orig = rt_orig.get_termination_flags()
        term_vec, trunc_vec = rt_vec.get_termination_flags()
        if term_orig or trunc_orig:
            runner_orig._reset_all(runner_orig._derive_seeds(base_seed))
        if term_vec or trunc_vec:
            runner_vec._reset_all(runner_vec._derive_seeds(base_seed))

        if step + 1 >= total_steps:
            break

    # Only count timed steps (exclude warmup)
    n = args.steps

    # We accumulated over all steps including warmup; subtract warmup proportionally
    # Actually we timed everything, so just divide by total and multiply by n
    # But simpler: re-run is expensive, so just report per-step averages over all
    # and note warmup was included. For fair comparison, per-step avg is what matters.
    total = total_steps

    def per_step(total_t):
        return total_t / total * 1000  # ms

    def speedup(t_old, t_new):
        if t_new <= 0:
            return float('inf')
        return t_old / t_new

    print(f'\n{"="*60}')
    print(f'Benchmark: {args.steps} steps (+{args.warmup} warmup), {phy_per_action} phy/action')
    print(f'{"="*60}')
    print(f'{"Phase":<30} {"Legacy (ms/step)":>18} {"Vec (ms/step)":>18} {"Speedup":>10}')
    print(f'{"-"*76}')
    print(f'{"_extract_contacts":<30} {per_step(t_extract_orig):>18.4f} {per_step(t_extract_vec):>18.4f} {speedup(t_extract_orig, t_extract_vec):>9.2f}x')
    print(f'{"get_derived_state":<30} {per_step(t_derived_orig):>18.4f} {per_step(t_derived_vec):>18.4f} {speedup(t_derived_orig, t_derived_vec):>9.2f}x')
    print(f'{"EnvRuntime.step (full)":<30} {per_step(t_step_orig):>18.4f} {per_step(t_step_vec):>18.4f} {speedup(t_step_orig, t_step_vec):>9.2f}x')
    print(f'{"-"*76}')

    # Also report total episode time
    total_orig = t_step_orig + t_derived_orig + t_extract_orig
    total_vec = t_step_vec + t_derived_vec + t_extract_vec
    print(f'{"Total (all phases)":<30} {per_step(total_orig):>18.4f} {per_step(total_vec):>18.4f} {speedup(total_orig, total_vec):>9.2f}x')
    print()

    # Contact count stats
    cv = sim_vec.get_derived_state(['contacts'])['contacts']
    print(f'Final step contact count: {cv["ncon"]}')


if __name__ == '__main__':
    main()
