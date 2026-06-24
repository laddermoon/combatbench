"""Compare legacy dict-based contact pipeline vs vectorized contacts_vec pipeline.

Side A: simulator.py + original plugins (legacy dict contacts)
Side B: simulator_vec.py + vec plugins (contacts_vec)

Both run with identical seeds and policies. We compare:
  1. Observations (include feet_forces computed from contacts)
  2. Health metrics (computed by CombatScoringPlugin from contacts)
  3. Termination flags
  4. Derived state fields (torso_distance, per-robot views)

Usage:
  python3 compare_contacts.py --seed 42
"""
import argparse
import numpy as np
from typing import Any, Dict, List

from envs.framework.blueprint import EnvBlueprint, _instantiate
from envs.framework.policy import PolicyBlueprint
from envs.framework.env_runtime import EnvRuntime
from envs.framework.episode_runner import EpisodeRunner, _resolve_seed


def _compare_robot_view(rid: str, rv1: Dict, rv2: Dict, step: int) -> bool:
    """Compare per-robot derived state view. Returns True if mismatch found."""
    keys1 = set(rv1.keys())
    keys2 = set(rv2.keys())
    # vec side doesn't have legacy contact keys, but per-robot views should match
    if keys1 != keys2:
        only1 = keys1 - keys2
        only2 = keys2 - keys1
        if only1 or only2:
            print(f'[step={step}] robot_view {rid} key mismatch: only_orig={only1} only_vec={only2}')
            return True

    for k in sorted(keys1):
        v1 = rv1[k]
        v2 = rv2[k]
        if isinstance(v1, np.ndarray):
            if not np.allclose(v1, v2, atol=1e-6):
                diff = np.abs(v1 - v2)
                idx_max = int(np.argmax(diff))
                print(f'[step={step}] robot_view {rid}.{k} mismatch: max_diff={float(diff[idx_max]):.2e} at idx={idx_max}')
                print(f'  orig: {v1}')
                print(f'  vec : {v2}')
                return True
        elif isinstance(v1, dict):
            for kk in sorted(v1.keys()):
                a1 = np.asarray(v1[kk], dtype=np.float64)
                a2 = np.asarray(v2.get(kk, a1), dtype=np.float64)
                if not np.allclose(a1, a2, atol=1e-6):
                    print(f'[step={step}] robot_view {rid}.{k}.{kk} mismatch')
                    print(f'  orig: {a1}')
                    print(f'  vec : {a2}')
                    return True
        elif isinstance(v1, (int, float)):
            if abs(v1 - v2) > 1e-10:
                print(f'[step={step}] robot_view {rid}.{k} mismatch: orig={v1} vec={v2}')
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--env-blueprint', type=str,
                        default='envs/humanoid21/blueprint.yaml')
    parser.add_argument('--policy-a-blueprint', type=str,
                        default='/data1/mono/things/combatbench_platform/submisiontool/uploads/submission_11/extracted/policy_blueprint.yaml')
    parser.add_argument('--policy-b-blueprint', type=str,
                        default='/data1/mono/things/combatbench_platform/submisiontool/uploads/submission_12/extracted/policy_blueprint.yaml')
    parser.add_argument('--verbose', action='store_true',
                        help='Print every step even if no mismatch')
    args = parser.parse_args()

    # Load blueprint
    bp = EnvBlueprint.load(args.env_blueprint)

    # Load policies
    policy_a = PolicyBlueprint.load(args.policy_a_blueprint).build()
    policy_b = PolicyBlueprint.load(args.policy_b_blueprint).build()

    # --- Side A: original simulator + original plugins ---
    from envs.humanoid21.simulator import MujocoCombatSimulator as SimOrig
    from envs.humanoid21.plugins import CombatScoringPlugin
    from envs.humanoid21.observer_plugins import CombatScoringObserver

    sim_orig = SimOrig(**bp.simulator.config)
    rt_orig = EnvRuntime(
        simulator=sim_orig,
        plugins=[_instantiate(spec) for spec in bp.plugins],
        observer_plugins={name: _instantiate(spec) for name, spec in bp.observer_plugins.items()},
        phy_steps_per_action=bp.phy_steps_per_action,
        max_steps=bp.max_steps,
        strict=bp.strict,
    )

    # --- Side B: vec simulator + vec plugins ---
    from envs.humanoid21.simulator_vec import MujocoCombatSimulator as SimVec
    from envs.humanoid21.plugins_vec import CombatScoringPlugin as CombatScoringPluginVec
    from envs.humanoid21.observer_plugins_vec import CombatScoringObserver as CombatScoringObserverVec

    # Build vec plugins with same configs as original
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

    # Create episode runners
    runner_orig = EpisodeRunner(runtime=rt_orig, policy_a=policy_a, policy_b=policy_b)
    runner_vec = EpisodeRunner(runtime=rt_vec, policy_a=policy_a, policy_b=policy_b)

    # Derive seeds (identical for both since base_seed is the same)
    base_seed = _resolve_seed(args.seed)
    seeds_orig = runner_orig._derive_seeds(base_seed)
    seeds_vec = runner_vec._derive_seeds(base_seed)

    # Verify seeds match
    assert seeds_orig.runtime == seeds_vec.runtime, f"Runtime seeds differ: {seeds_orig.runtime} vs {seeds_vec.runtime}"
    assert seeds_orig.policies == seeds_vec.policies, f"Policy seeds differ"
    print(f"Base seed: {base_seed}")
    print(f"Runtime seed: {seeds_orig.runtime}")
    print(f"Policy seeds: {seeds_orig.policies}")

    # Reset both
    runner_orig._reset_all(seeds_orig)
    runner_vec._reset_all(seeds_vec)

    # Get initial observations
    obs_a_orig, obs_b_orig = rt_orig.get_observation()
    obs_a_vec, obs_b_vec = rt_vec.get_observation()

    mismatches = 0

    for step in range(600):
        # 1. Compare observations BEFORE policy acts
        if not np.allclose(obs_a_orig, obs_a_vec, atol=1e-6):
            diff = np.abs(obs_a_orig - obs_a_vec)
            idx = np.where(diff > 1e-6)[0]
            print(f'\n=== Observation A mismatch at step {step} ===')
            print(f'  diff indices: {idx}')
            print(f'  orig: {obs_a_orig[idx]}')
            print(f'  vec : {obs_a_vec[idx]}')
            mismatches += 1
            if mismatches > 10:
                return

        if not np.allclose(obs_b_orig, obs_b_vec, atol=1e-6):
            diff = np.abs(obs_b_orig - obs_b_vec)
            idx = np.where(diff > 1e-6)[0]
            print(f'\n=== Observation B mismatch at step {step} ===')
            print(f'  diff indices: {idx}')
            print(f'  orig: {obs_b_orig[idx]}')
            print(f'  vec : {obs_b_vec[idx]}')
            mismatches += 1
            if mismatches > 10:
                return

        # 2. Get actions from policies (using orig observation for both)
        action_a, _ = policy_a.act(obs_a_orig, want_extra=False)
        action_b, _ = policy_b.act(obs_b_orig, want_extra=False)

        # 3. Feed actions to both simulators
        rt_orig.step(action_a, action_b)
        rt_vec.step(action_a, action_b)

        # 4. Compare derived_state
        ds_orig = sim_orig.get_derived_state()
        ds_vec = sim_vec.get_derived_state(['torso_distance', 'contacts'])

        # Compare torso_distance
        if not np.allclose(ds_orig['torso_distance'], ds_vec['torso_distance'], atol=1e-6):
            print(f'\n=== torso_distance mismatch at step {step} ===')
            print(f'  orig: {ds_orig["torso_distance"]}')
            print(f'  vec : {ds_vec["torso_distance"]}')
            mismatches += 1
            if mismatches > 10:
                return

        # Per-robot views (feet_forces, observation, etc.) are no longer in
        # vec derived_state — compare observations directly instead.
        obs_a_o, obs_b_o = sim_orig.get_observation().values()
        obs_a_v, obs_b_v = sim_vec.get_observation().values()
        if not np.allclose(obs_a_o, obs_a_v, atol=1e-6):
            diff = np.abs(obs_a_o - obs_a_v)
            idx = np.where(diff > 1e-6)[0]
            print(f'\n=== Observation A mismatch at step {step} ===')
            print(f'  diff indices: {idx}')
            mismatches += 1
            if mismatches > 10:
                return
        if not np.allclose(obs_b_o, obs_b_v, atol=1e-6):
            diff = np.abs(obs_b_o - obs_b_v)
            idx = np.where(diff > 1e-6)[0]
            print(f'\n=== Observation B mismatch at step {step} ===')
            print(f'  diff indices: {idx}')
            mismatches += 1
            if mismatches > 10:
                return

        # 5. Compare health metrics
        ha_orig = rt_orig.ctx.metrics.get('health_a', 0.0)
        ha_vec = rt_vec.ctx.metrics.get('health_a', 0.0)
        hb_orig = rt_orig.ctx.metrics.get('health_b', 0.0)
        hb_vec = rt_vec.ctx.metrics.get('health_b', 0.0)

        if abs(ha_orig - ha_vec) > 1e-5:
            print(f'\n=== health_a mismatch at step {step} ===')
            print(f'  orig: {ha_orig}')
            print(f'  vec : {ha_vec}')
            mismatches += 1
            if mismatches > 10:
                return

        if abs(hb_orig - hb_vec) > 1e-5:
            print(f'\n=== health_b mismatch at step {step} ===')
            print(f'  orig: {hb_orig}')
            print(f'  vec : {hb_vec}')
            mismatches += 1
            if mismatches > 10:
                return

        # 6. Compare damage metrics
        da_orig = rt_orig.ctx.metrics.get('damage_taken_a', 0.0)
        da_vec = rt_vec.ctx.metrics.get('damage_taken_a', 0.0)
        db_orig = rt_orig.ctx.metrics.get('damage_taken_b', 0.0)
        db_vec = rt_vec.ctx.metrics.get('damage_taken_b', 0.0)

        if abs(da_orig - da_vec) > 1e-5:
            print(f'\n=== damage_taken_a mismatch at step {step} ===')
            print(f'  orig: {da_orig}')
            print(f'  vec : {da_vec}')
            mismatches += 1
            if mismatches > 10:
                return

        if abs(db_orig - db_vec) > 1e-5:
            print(f'\n=== damage_taken_b mismatch at step {step} ===')
            print(f'  orig: {db_orig}')
            print(f'  vec : {db_vec}')
            mismatches += 1
            if mismatches > 10:
                return

        # 7. Check termination
        term_orig, trunc_orig = rt_orig.get_termination_flags()
        term_vec, trunc_vec = rt_vec.get_termination_flags()
        if term_orig != term_vec or trunc_orig != trunc_vec:
            print(f'\n=== Termination mismatch at step {step} ===')
            print(f'  orig: term={term_orig} trunc={trunc_orig}')
            print(f'  vec : term={term_vec} trunc={trunc_vec}')
            mismatches += 1
            if mismatches > 10:
                return

        if term_orig or trunc_orig:
            print(f'\nBoth terminated at step {step}')
            break

        # Get next observations
        obs_a_orig, obs_b_orig = rt_orig.get_observation()
        obs_a_vec, obs_b_vec = rt_vec.get_observation()

        if args.verbose and step % 50 == 0:
            print(f'step {step}: health_a={ha_orig:.4f}/{ha_vec:.4f} health_b={hb_orig:.4f}/{hb_vec:.4f} '
                  f'dmg_a={da_orig:.4f}/{da_vec:.4f} dmg_b={db_orig:.4f}/{db_vec:.4f}')

    # Final results
    print(f'\n=== Final Results ===')
    print(f'orig: health_a={rt_orig.ctx.metrics.get("health_a", 0.0)} health_b={rt_orig.ctx.metrics.get("health_b", 0.0)}')
    print(f'vec : health_a={rt_vec.ctx.metrics.get("health_a", 0.0)} health_b={rt_vec.ctx.metrics.get("health_b", 0.0)}')
    print(f'orig: dmg_a={rt_orig.ctx.metrics.get("damage_taken_a", 0.0)} dmg_b={rt_orig.ctx.metrics.get("damage_taken_b", 0.0)}')
    print(f'vec : dmg_a={rt_vec.ctx.metrics.get("damage_taken_a", 0.0)} dmg_b={rt_vec.ctx.metrics.get("damage_taken_b", 0.0)}')

    match = (abs(rt_orig.ctx.metrics.get("health_a", 0.0) - rt_vec.ctx.metrics.get("health_a", 0.0)) < 1e-5
             and abs(rt_orig.ctx.metrics.get("health_b", 0.0) - rt_vec.ctx.metrics.get("health_b", 0.0)) < 1e-5
             and abs(rt_orig.ctx.metrics.get("damage_taken_a", 0.0) - rt_vec.ctx.metrics.get("damage_taken_a", 0.0)) < 1e-5
             and abs(rt_orig.ctx.metrics.get("damage_taken_b", 0.0) - rt_vec.ctx.metrics.get("damage_taken_b", 0.0)) < 1e-5)
    print(f'MISMATCHES: {mismatches}')
    print(f'MATCH: {match and mismatches == 0}')


if __name__ == '__main__':
    main()
