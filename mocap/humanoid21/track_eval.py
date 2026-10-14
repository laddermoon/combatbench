"""训练后跟踪策略的确定性评估（M2 L2 判分用）。

与 track_replay.py（open-loop 基线）对应：这里用训练好的策略做
closed-loop 回放 —— RSI 固定从帧 0 起，全程不覆写状态，直到
track_clip_end 或 track_fall。

用法:
    PYTHONPATH=. python3 mocap/humanoid21/track_eval.py \
        --policy baseline/runs/<run>/policy_exports/uNNNNN \
        --clip cmu_07_01 --video

输出 /data1/dev/combatbench_mocap/track_eval/<clip>_<policy>.{json,mp4}
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint
from envs.framework.common_plugins import VideoRecorderPlugin


def eval_policy(policy_dir: str, clip_id: str, video: bool = False,
                seed: int = 0, max_steps: int = 2000):
    policy_dir = Path(policy_dir).resolve()
    bp = PolicyBlueprint.from_dict({
        'version': 1,
        'cls': f'file:{policy_dir}/policy.py:ExportedTruncNormPolicy',
        'config': {},
    })
    pol = bp.build()
    standing = PolicyBlueprint.load(
        Path(__file__).resolve().parents[2]
        / 'policy' / 'blueprints' / 'humanoid21' / 'standing.yaml').build()

    env_pb = ParameterizedEnvBlueprint.load(
        Path(__file__).resolve().parents[2]
        / 'baseline' / 'humanoid21' / 'blueprints' / 'track_ref_env.yaml')
    env_bp = env_pb.materialize(
        clip_id=clip_id, rsi='first', max_steps=max_steps)
    rt = env_bp.build()
    tag = f'{policy_dir.parent.parent.name}_{policy_dir.name}'
    if video:
        out_dir = Path('/data1/dev/combatbench_mocap/track_eval')
        out_dir.mkdir(parents=True, exist_ok=True)
        rt.attach_plugin(VideoRecorderPlugin(
            fps=20,
            output_path=str(out_dir / f'{clip_id}_{tag}_track_eval.mp4')))

    rt.reset(seed=seed, options={'initial_distance': 6.0})
    rec = {'reward': [], 'kp_err_m': [], 'kp_err_rel_m': [],
           'root_dxy_m': [], 'root_dyaw_deg': [],
           'q_rmse_deg': [], 'phase': []}
    obs_a, obs_b = rt.get_observation()
    while not rt.is_episode_over():
        aa, _ = pol.act(obs_a)
        ab, _ = standing.act(obs_b)
        rt.step(aa, ab)
        out = rt.get_observer_output('track_a')
        for k in rec:
            rec[k].append(float(out[k]))
        obs_a, obs_b = rt.get_observation()

    terms = rt.get_agent_termination()
    n = len(rec['reward'])
    metrics = {
        'clip_id': clip_id,
        'policy_dir': str(policy_dir),
        'steps': n,
        'terminated_reason': terms.get('robot_a'),
        'phase_reached': rec['phase'][-1] if n else 0.0,
        'completed': terms.get('robot_a') == 'track_clip_end',
        'reward_mean': float(np.mean(rec['reward'])) if n else 0.0,
        'kp_err_m_mean': float(np.mean(rec['kp_err_m'])) if n else 0.0,
        'kp_err_m_max': float(np.max(rec['kp_err_m'])) if n else 0.0,
        'kp_err_rel_m_mean': float(np.mean(rec['kp_err_rel_m'])) if n else 0.0,
        'root_dxy_m_mean': float(np.mean(rec['root_dxy_m'])) if n else 0.0,
        'root_dxy_m_max': float(np.max(rec['root_dxy_m'])) if n else 0.0,
        'root_dyaw_deg_mean': float(np.mean(np.abs(rec['root_dyaw_deg']))) if n else 0.0,
        'root_dyaw_deg_max': float(np.max(np.abs(rec['root_dyaw_deg']))) if n else 0.0,
        'q_rmse_deg_mean': float(np.mean(rec['q_rmse_deg'])) if n else 0.0,
        'q_rmse_deg_max': float(np.max(rec['q_rmse_deg'])) if n else 0.0,
    }
    out_dir = Path('/data1/dev/combatbench_mocap/track_eval')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{clip_id}_{tag}_track_eval.json').write_text(
        json.dumps(metrics, indent=1))
    np.savez(out_dir / f'{clip_id}_{tag}_track_eval_trace.npz',
             **{k: np.asarray(v) for k, v in rec.items()})
    rt.close()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--clip', default='cmu_07_01')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(eval_policy(a.policy, a.clip, a.video, a.seed), indent=1))


if __name__ == '__main__':
    main()
