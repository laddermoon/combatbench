"""deepmimic28 确定性评估：RSI 从 t=0 起步，跑整段 episode，出指标+视频。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint
from envs.framework.common_plugins import VideoRecorderPlugin

_PKG = Path(__file__).resolve().parent


def _load_policy(policy_arg: str):
    p = Path(policy_arg).resolve()
    if (p / 'policy').is_dir():              # run 目录 → best-of-run
        p = p / 'policy'
    import json as _json
    manifest = _json.loads((p / 'MANIFEST.json').read_text())
    cls = manifest.get('exported_class', 'ExportedGaussianMLPPolicy')
    bp = PolicyBlueprint.from_dict({
        'version': 1,
        'cls': f'file:{p}/policy.py:{cls}',
        'config': {},
    })
    return bp.build()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True, help='policy_exports/uNNNNN 或 policy/ 目录')
    ap.add_argument('--motion', default='humanoid_spinkick')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--video', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--start-time', type=float, default=0.0)
    args = ap.parse_args()

    env_bp = ParameterizedEnvBlueprint.load(_PKG / 'envs' / 'track_env.yaml').materialize(
        motion=args.motion, rand_reset=False, episode_steps=args.steps)

    pol = _load_policy(args.policy)
    zero = PolicyBlueprint.load(_PKG / 'blueprints' / 'zero.yaml').build()

    rt = env_bp.build()
    if args.video:
        vp = Path(args.video)
        vp.parent.mkdir(parents=True, exist_ok=True)
        rt.attach_plugin(VideoRecorderPlugin(fps=20, output_path=str(vp)))

    rt.reset(seed=0, options={'start_time': args.start_time})

    rows = []
    terminated_reason = None
    obs_a, obs_b = rt.get_observation()
    while not rt.is_episode_over():
        aa, _ = pol.act(obs_a)
        ab, _ = zero.act(obs_b)
        rt.step(aa, ab)
        out = rt.get_observer_output('track_a') or {}
        rows.append({k: float(v) for k, v in out.items()
                     if isinstance(v, (int, float, np.floating))})
        obs_a, obs_b = rt.get_observation()
    terminated_reason = rt.get_agent_termination().get('robot_a')
    rt.close()

    n = len(rows)
    metrics = {
        'steps_survived': n,
        'survive_ratio': n / args.steps,
        'terminated': terminated_reason,
        'completed': terminated_reason in (None, 'timeout'),
        'reward_mean': float(np.mean([r['reward'] for r in rows])) if rows else 0.0,
        'root_dev_mean': float(np.mean([r['root_dev'] for r in rows])) if rows else 0.0,
        'root_dev_max': float(np.max([r['root_dev'] for r in rows])) if rows else 0.0,
        'key_err_mean': float(np.mean([r['key_err'] for r in rows])) if rows else 0.0,
        'pose_err_mean': float(np.mean([r['pose_err'] for r in rows])) if rows else 0.0,
    }
    txt = json.dumps(metrics, indent=2)
    print(txt)
    if args.json:
        Path(args.json).write_text(txt)


if __name__ == '__main__':
    main()
