"""上游 MimicKit 预训练策略在我们仿真器中的验证评估（终极语义链验证）。

若上游策略能在我方 sim 完成 spinkick，则 obs/action/reward/动力学语义全部对齐；
失败则暴露语义 bug。用法：
  PYTHONPATH=. python3 baseline/deepmimic28/eval_pretrained.py [--video out.mp4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint
from envs.framework.policy import PolicyBlueprint
from envs.framework.common_plugins import VideoRecorderPlugin

_PKG = Path(__file__).resolve().parent
CKPT = '/data1/mono/things/combatbench-research/learn/mimickit_data_pack/models/deepmimic_humanoid_spinkick_model.pt'


class UpstreamPolicy:
    """上游 actor 精确重建：464→1024→512→28(ReLU)，obs 归一化 clip10。"""

    def __init__(self, ckpt_path: str = CKPT):
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.obs_mean = sd['_obs_norm._mean'].numpy()
        self.obs_std = sd['_obs_norm._std'].numpy()
        w0 = sd['_model._actor_layers.0.weight'].numpy()
        b0 = sd['_model._actor_layers.0.bias'].numpy()
        w2 = sd['_model._actor_layers.2.weight'].numpy()
        b2 = sd['_model._actor_layers.2.bias'].numpy()
        wm = sd['_model._action_dist._mean_net.weight'].numpy()
        bm = sd['_model._action_dist._mean_net.bias'].numpy()
        self._logstd = sd['_model._action_dist._logstd_net'].numpy()
        self.w0, self.b0, self.w2, self.b2, self.wm, self.bm = w0, b0, w2, b2, wm, bm

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.obs_mean) / self.obs_std
        x = np.clip(x, -10.0, 10.0)
        h = np.maximum(x @ self.w0.T + self.b0, 0.0)     # ReLU
        h = np.maximum(h @ self.w2.T + self.b2, 0.0)     # ReLU
        return (h @ self.wm.T + self.bm).astype(np.float32)  # 归一化动作(mode=mean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default=None)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--start-time', type=float, default=0.0)
    ap.add_argument('--stochastic', action='store_true')
    args = ap.parse_args()

    pol = UpstreamPolicy()
    print(f'logstd 范围: [{pol._logstd.min():.3f}, {pol._logstd.max():.3f}] '
          f'(FIXED 0.05→ln={np.log(0.05):.3f})')

    env_bp = ParameterizedEnvBlueprint.load(_PKG / 'envs' / 'track_env.yaml').materialize(
        motion='humanoid_spinkick', rand_reset=False, episode_steps=args.steps)
    zero = PolicyBlueprint.load(_PKG / 'blueprints' / 'zero.yaml').build()
    rt = env_bp.build()
    if args.video:
        vp = Path(args.video)
        vp.parent.mkdir(parents=True, exist_ok=True)
        rt.attach_plugin(VideoRecorderPlugin(fps=20, output_path=str(vp)))
    rt.reset(seed=0, options={'start_time': args.start_time})

    rows = []
    rng = np.random.default_rng(0)
    obs_a, obs_b = rt.get_observation()
    while not rt.is_episode_over():
        a = pol.act(np.asarray(obs_a, dtype=np.float32))
        if args.stochastic:
            a = a + rng.normal(0, 1, size=a.shape).astype(np.float32) * np.exp(pol._logstd)
        ab, _ = zero.act(obs_b)
        rt.step(a, ab)
        out = rt.get_observer_output('track_a') or {}
        rows.append({k: float(v) for k, v in out.items() if np.isscalar(v)})
        obs_a, obs_b = rt.get_observation()
    reason = rt.get_agent_termination().get('robot_a')
    rt.close()

    n = len(rows)
    print(f'\n生存: {n}/{args.steps} 步  终止原因: {reason}')
    if rows:
        print(f'reward/step: {np.mean([r["reward"] for r in rows]):.4f}  (上游收敛 ~0.9)')
        print(f'root_dev mean/max: {np.mean([r["root_dev"] for r in rows]):.4f} / '
              f'{np.max([r["root_dev"] for r in rows]):.4f}  (上游收敛 root_pos_err ~0.044)')
        print(f'pose_err mean: {np.mean([r["pose_err"] for r in rows]):.4f}  (上游收敛 ~0.13)')
        print(f'key_err mean: {np.mean([r["key_err"] for r in rows]):.4f}')


if __name__ == '__main__':
    main()
