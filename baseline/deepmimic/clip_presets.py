"""按 clip meta 推导环境参数默认值。

规则（对应 DESIGN.md §5-D6 与 §4.5）：

- ``max_steps``：= clip 控制帧数（episode 能放完整个片段；所有终止都由
  track_* 语义触发，timeout 不可达）
- ``fall_mode``：labels/clip_id 含 getup/fall/lie/crawl 语义时 height 判据
  （head<pelvis）会误伤合法低姿态 → 建议 'contact'；其余 'height'
- drift：默认沿用 M2 v6 一阶段值 0.8m/75°（对位移类 clip 同样成立——
  dxy 度量的是与参考的偏差而非位移量；如某 clip 训练中被误伤再调）

实验侧把这些值当默认值；显式 ``--set`` 永远覆盖。
"""
from __future__ import annotations

from typing import Any, Dict

from .ref import CoreRefView

_GROUND_LABELS = {'getup', 'fall', 'recovery'}
_GROUND_SUBSTR = ('lie', 'crawl', 'getup', 'fall')


def suggest_env_params(ref: CoreRefView) -> Dict[str, Any]:
    labels = set(ref.meta.get('labels', []))
    cid = ref.clip_id.lower()
    groundish = bool(labels & _GROUND_LABELS) or any(
        s in cid for s in _GROUND_SUBSTR)
    return {
        'max_steps': int(ref.n_ctrl),
        'fall_mode': 'contact' if groundish else 'height',
        'drift_dxy': 0.8,
        'drift_dyaw_deg': 75.0,
    }
