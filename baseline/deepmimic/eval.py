"""DeepMimic 跟踪策略的确定性评估 + 视频（验收口径，M2 同款流程）。

RSI 固定从首个合法帧（rsi_frames[0]）起，全程不覆写状态，直到
track_clip_end / track_fall / track_drift / max_steps。

用法（repo 根 = things/combatbench）:
    PYTHONPATH=. python3 baseline/deepmimic/eval.py \
        --policy baseline/runs/dm_<clip>_*/policy_exports/uNNNNN \
        --clip cmu_13_17 --video /tmp/eval.mp4 --json /tmp/metrics.json

``--policy`` 也接受训练 run 目录（自动取 policy/ best-of-run 导出）。
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')       # 本机 EGL 坏了
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')

import numpy as np

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parents[1]
sys.path.insert(0, str(_REPO))

from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint
from envs.framework.common_plugins import VideoRecorderPlugin

from baseline.deepmimic.ref import CoreRefView

ENV_YAML = _PKG / 'envs' / 'track_env.yaml'
STANDING_YAML = _REPO / 'policy' / 'blueprints' / 'humanoid21' / 'standing.yaml'


def _load_policy(policy_arg: str) -> object:
    """加载导出策略：export 目录 / run 目录（取 policy/ 子目录）。"""
    p = Path(policy_arg).resolve()
    if (p / 'policy').is_dir():              # run 目录 → best-of-run
        p = p / 'policy'
    manifest = json.loads((p / 'MANIFEST.json').read_text())
    cls = manifest.get('exported_class', 'ExportedTruncNormPolicy')
    bp = PolicyBlueprint.from_dict({
        'version': 1,
        'cls': f'file:{p}/policy.py:{cls}',
        'config': {},
    })
    return bp.build(), p


def eval_policy(policy_dir: str, clip_id: str,
                video_path: str = None, json_path: str = None,
                seed: int = 0, fall_mode: str = 'height',
                drift_dxy=None, drift_dyaw_deg=None,
                start_frame: int = None, max_steps: int = None,
                actuation_profile: str = 'simchar') -> dict:
    pol, pol_dir = _load_policy(policy_dir)
    standing = PolicyBlueprint.load(STANDING_YAML).build()
    ref = CoreRefView(clip_id)

    ms = int(max_steps or ref.n_ctrl)
    kw = dict(clip_id=clip_id, rsi='first', max_steps=ms,
              fall_mode=fall_mode, drift_dxy=drift_dxy,
              drift_dyaw_deg=drift_dyaw_deg,
              actuation_profile=actuation_profile)
    env_bp = ParameterizedEnvBlueprint.load(ENV_YAML).materialize(**kw)
    rt = env_bp.build()

    out_video = Path(video_path) if video_path else None
    if out_video:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        rt.attach_plugin(VideoRecorderPlugin(
            fps=20, output_path=str(out_video)))

    opts = {'initial_distance': 6.0}
    if start_frame is not None:
        opts['start_frame'] = int(start_frame)
    rt.reset(seed=seed, options=opts)

    keys = ('reward', 'kp_err_m', 'kp_err_rel_m', 'root_dxy_m',
            'root_dyaw_deg', 'q_rmse_deg', 'phase')
    rec = {k: [] for k in keys}
    acts = []
    jvels = []
    obs_a, obs_b = rt.get_observation()
    while not rt.is_episode_over():
        aa, _ = pol.act(obs_a)
        ab, _ = standing.act(obs_b)
        rt.step(aa, ab)
        acts.append(np.asarray(aa, dtype=np.float64).copy())
        cs = rt.simulator.get_core_state()
        jv = cs.get('robot_a', cs).get('joint_vel_norm') if isinstance(cs, dict) else None
        if jv is not None:
            jvels.append(np.asarray(jv, dtype=np.float64).copy())
        out = rt.get_observer_output('track_a')
        for k in keys:
            rec[k].append(float(out[k]))
        obs_a, obs_b = rt.get_observation()

    terms = rt.get_agent_termination()
    n = len(rec['reward'])

    def _m(k):
        return float(np.mean(rec[k])) if n else 0.0

    metrics = {
        'clip_id': ref.clip_id,
        'policy_dir': str(pol_dir),
        'seed': seed,
        'start_frame': int(start_frame) if start_frame is not None
                       else int(ref.rsi_frames[0]),
        'steps': n,
        'terminated_reason': terms.get('robot_a'),
        'completed': terms.get('robot_a') == 'track_clip_end',
        'phase_reached': rec['phase'][-1] if n else 0.0,
        'reward_mean': _m('reward'),
        'kp_err_m_mean': _m('kp_err_m'),
        'kp_err_m_max': float(np.max(rec['kp_err_m'])) if n else 0.0,
        'kp_err_rel_m_mean': _m('kp_err_rel_m'),
        'root_dxy_m_mean': _m('root_dxy_m'),
        'root_dxy_m_max': float(np.max(rec['root_dxy_m'])) if n else 0.0,
        'root_dyaw_deg_mean': float(np.mean(np.abs(rec['root_dyaw_deg'])))
        if n else 0.0,
        'root_dyaw_deg_max': float(np.max(np.abs(rec['root_dyaw_deg'])))
        if n else 0.0,
        'q_rmse_deg_mean': _m('q_rmse_deg'),
        'q_rmse_deg_max': float(np.max(rec['q_rmse_deg'])) if n else 0.0,
    }
    # 抖动量化：相邻控制步 action 变化率（|Δa| 均值/峰值）与
    # 符号翻转率（每关节平均每步符号改变比例）。|Δa|~0=静止，
    # ~0.1-0.3=正常跟踪，>0.5 或翻转率>30% 即高频抖动。
    if len(acts) > 1:
        A = np.asarray(acts)
        da = np.abs(np.diff(A, axis=0))
        flips = (np.diff(np.sign(A), axis=0) != 0).mean()
        metrics['action_rate_mean'] = float(da.mean())
        metrics['action_rate_max'] = float(da.max())
        metrics['action_flip_rate'] = float(flips)
    # 关节空间抖动：真实关节速度（归一化）的符号翻转率与速度量级。
    # 动作空间 flip 相同的条件下，simchar alpha>1 会把同样的 action
    # 抖动能放大成更大的关节目标抖动，此指标反映视觉真实抖动。
    if len(jvels) > 2:
        JV = np.asarray(jvels)
        jflips = (np.diff(np.sign(JV), axis=0) != 0).mean()
        metrics['jvel_abs_mean'] = float(np.abs(JV).mean())
        metrics['jvel_flip_rate'] = float(jflips)
    rt.close()

    if json_path:
        jp = Path(json_path)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
        np.savez(jp.with_suffix('.trace.npz'),
                 **{k: np.asarray(v) for k, v in rec.items()})
    return metrics


def main():
    ap = argparse.ArgumentParser(description='DeepMimic 确定性整段评估')
    ap.add_argument('--policy', required=True,
                    help='policy_exports/uNNNNN 或 run 目录')
    ap.add_argument('--clip', required=True)
    ap.add_argument('--video', default=None, help='输出 mp4 路径')
    ap.add_argument('--json', default=None, help='输出 metrics json 路径')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--start-frame', type=int, default=None)
    ap.add_argument('--fall-mode', default='height',
                    choices=('height', 'contact', 'none'))
    ap.add_argument('--drift-dxy', type=float, default=None)
    ap.add_argument('--drift-dyaw-deg', type=float, default=None)
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--actuation-profile', default='simchar',
                    help='执行器 profile（须与训练一致）')
    a = ap.parse_args()
    metrics = eval_policy(
        a.policy, a.clip, video_path=a.video, json_path=a.json,
        seed=a.seed, fall_mode=a.fall_mode, drift_dxy=a.drift_dxy,
        drift_dyaw_deg=a.drift_dyaw_deg, start_frame=a.start_frame,
        max_steps=a.max_steps, actuation_profile=a.actuation_profile)
    print(json.dumps(metrics, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
