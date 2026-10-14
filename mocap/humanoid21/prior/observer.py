"""AMP 风格奖励 observer（M3 在线接口）。

每个控制步提取与离线 features.py 完全同语义的 67 维帧特征，
维护 4 帧队列，按 150ms 窗 (f_{k-3}, f_k) 送入判别器得风格分。

输入坐标不变性：与离线一致（yaw 对齐系 + 相对骨盆关键点），
与位置/朝向无关——风格分只描述"怎么动"。

输出：
  reward   r_style = -ln(1 - D + eps)，截断 [0, 5]（风格分）
  D        判别器原始输出 ∈ [0,1]（1=像参考库）
  disc_version / feat_gap / n 诊断

用法：blueprint 中 observer_plugins 注册本类，disc_path 指向
/data1/dev/combatbench_mocap/prior/disc_vN.pt。
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from envs.framework import BaseObserverPlugin, ReadOnlySimContext
from mocap.humanoid21.prior.features import extract, TRANS_GAP, AMP_KP_NAMES
from mocap.humanoid21.prior.train_disc import Discriminator
from mocap.humanoid21.robot_kin import CONTROLLED_JOINTS, joint_limits

_MID = np.array([0.5 * (joint_limits(n)[0] + joint_limits(n)[1])
                 for n in CONTROLLED_JOINTS])
_HALF = np.array([0.5 * (joint_limits(n)[1] - joint_limits(n)[0])
                  for n in CONTROLLED_JOINTS])


class AmpStyleObserver(BaseObserverPlugin):
    def __init__(self, agent_id: str = 'robot_a',
                 disc_path: str = '/data1/dev/combatbench_mocap/prior/disc_v4.pt',
                 reward_clip: float = 5.0) -> None:
        self.agent_id = str(agent_id)
        self.disc_path = str(disc_path)
        self.reward_clip = float(reward_clip)
        ck = torch.load(self.disc_path, map_location='cpu',
                        weights_only=False)
        self._disc = Discriminator()
        self._disc.load_state_dict(ck['model'])
        self._disc.eval()
        self._mu = np.asarray(ck['mu'], dtype=np.float32)
        self._sd = np.asarray(ck['sd'], dtype=np.float32)
        self._version = str(ck.get('version', '?'))
        self._hist: deque = deque(maxlen=TRANS_GAP + 1)
        self._output: Dict[str, float] = {'reward': 0.0, 'D': 0.0}

    # ---------------------------------------------------------------
    def _suffix(self) -> str:
        return '_a' if self.agent_id.endswith('_a') else '_b'

    def _frame_features(self, ctx: ReadOnlySimContext) -> np.ndarray:
        core = ctx.accessor.get_core_state()[self.agent_id]
        ds = ctx.accessor.get_derived_state([self.agent_id])[self.agent_id]
        sfx = self._suffix()

        q_raw = np.asarray(core['joint_pos_norm']) * _HALF + _MID
        qv_raw = np.asarray(core['joint_vel_norm']) * _HALF
        q_self = np.asarray(core['root_rot'], dtype=np.float64)
        Rr = R.from_quat([q_self[1], q_self[2], q_self[3], q_self[0]])
        lin_w = Rr.as_matrix() @ np.asarray(core['root_vel_local'])
        ang_w = Rr.as_matrix() @ np.asarray(core['root_angular_vel_local'])

        bx = ds['body_xpos']
        names = ['pelvis'] + list(AMP_KP_NAMES)
        kp = np.stack([np.asarray(bx[f'{n}{sfx}']) for n in names])
        return extract(q_raw, qv_raw, np.asarray(core['root_pos']),
                       q_self, lin_w, ang_w, kp, names)

    # ---------------------------------------------------------------
    def _compute(self, ctx: ReadOnlySimContext) -> Dict[str, float]:
        self._hist.append(self._frame_features(ctx))
        if len(self._hist) <= TRANS_GAP:
            return {'reward': 0.0, 'D': 0.0,
                    'disc_version': 0.0, 'warmup': 1.0}
        tr = np.concatenate([self._hist[0], self._hist[-1]])
        x = torch.tensor((tr - self._mu) / self._sd,
                         dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            d = float(self._disc(x).item())
        r = min(-np.log(max(1.0 - d, 1e-8)), self.reward_clip)
        return {'reward': r, 'D': d, 'warmup': 0.0,
                'disc_version': float(self._version.lstrip('v') or 0)}

    def on_pre_episode(self, ctx: ReadOnlySimContext) -> None:
        self._hist.clear()
        self._hist.append(self._frame_features(ctx))
        self._output = {'reward': 0.0, 'D': 0.0, 'warmup': 1.0}

    def on_post_action_step(self, ctx: ReadOnlySimContext) -> None:
        self._output = self._compute(ctx)

    def get_output(self) -> Dict[str, float]:
        return dict(self._output)

    def to_blueprint(self) -> Dict[str, Any]:
        return {'agent_id': self.agent_id, 'disc_path': self.disc_path,
                'reward_clip': self.reward_clip}

    @classmethod
    def from_blueprint(cls, config: Dict[str, Any]) -> 'AmpStyleObserver':
        return cls(**config)
