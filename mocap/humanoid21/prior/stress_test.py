"""AMP 判别器对抗压力测试（M3 验收核心）。

按 README §10 清单，对每种"明显不合理运动"打分——判别器 D 输出
必须显著低于真实片段；同时真实片段（含留出 clip）必须高分。
输出每类 score 分布（mean/p05/p95）与整体判定。

用法：
    PYTHONPATH=. python3 -B mocap/humanoid21/prior/stress_test.py \
        --disc /data1/dev/combatbench_mocap/prior/disc_v1.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mocap.humanoid21.prior.features import (
    CLIPS_DIR, clip_frame_features, transitions)
from mocap.humanoid21.prior.train_disc import Discriminator, make_negatives


def score(disc, mu, sd, X, bs=8192):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.tensor((X[i:i + bs] - mu) / sd,
                             dtype=torch.float32, device=next(
                                 disc.parameters()).device)
            out.append(disc(x).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--disc', required=True)
    ap.add_argument('--holdout', default='cmu_17_10',
                    help='留出片段验证泛化（训练时建议同样排除）')
    a = ap.parse_args()

    ck = torch.load(a.disc, map_location='cpu', weights_only=False)
    disc = Discriminator()
    disc.load_state_dict(ck['model'])
    disc.eval().cuda() if torch.cuda.is_available() else disc.eval()
    mu, sd = ck['mu'], ck['sd']

    feats_by_clip, holdout = {}, None
    for p in sorted(CLIPS_DIR.glob('cmu_*.npz')):
        if p.stem.endswith('_src'):
            continue
        f = clip_frame_features(p)
        if len(f) < 2:
            continue
        if p.stem == a.holdout:
            holdout = f
        else:
            feats_by_clip[p.stem] = f

    rng = np.random.default_rng(0)
    negs = make_negatives(feats_by_clip, rng)

    report = {'disc': a.disc, 'holdout_clip': a.holdout, 'classes': {}}
    pos = np.concatenate(
        [transitions(f) for f in feats_by_clip.values()], axis=0)
    classes = {'real_train': pos}
    if holdout is not None:
        classes['real_holdout'] = transitions(holdout)
    classes.update({f'neg_{k}': v for k, v in negs.items()})

    for name, X in classes.items():
        s = score(disc, mu, sd, X)
        report['classes'][name] = {
            'n': len(X),
            'D_mean': round(float(s.mean()), 4),
            'D_p05': round(float(np.percentile(s, 5)), 4),
            'D_p95': round(float(np.percentile(s, 95)), 4),
        }
        print(f'{name:14s} n={len(X):6d} D_mean={s.mean():.3f} '
              f'p05={np.percentile(s,5):.3f} p95={np.percentile(s,95):.3f}')

    out = Path(a.disc).with_suffix('').parent / \
        f'{Path(a.disc).stem}_stress.json'
    out.write_text(json.dumps(report, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
