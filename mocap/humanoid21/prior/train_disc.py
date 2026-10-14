"""AMP 风格判别器训练（M3 原型）。

正例：17 个 L1+ 片段的 20Hz 状态转换（mocap/humanoid21/prior/features.py）。
负例：对正例做对抗性破坏合成——按 README §10 清单：
  frozen    冻结：f_{t+1}=f_t（静止）
  repeat    单帧重复：过渡序列只含同一转换
  shuffle   时间打乱：帧序重排（转换断裂）
  reversed  倒放：时间反向
  jitter    抖动：关节角/速度加高频噪声（对"打拳蹭步"类颤动敏感）
  speed     不合理变速：速度/速度差通道 ×k
  random    限位内平滑随机游走（物理可摆出的乱动）

判别器：MLP 134→256→256→1，BCE + 实样本梯度惩罚（AMP 风格，平滑决策面）。
输出 prior/disc_vN.pt + meta.json（版本、特征语义、归一化统计、负例配比）。

用法：
    PYTHONPATH=. python3 -B mocap/humanoid21/prior/train_disc.py \
        --version v1 [--epochs 30] [--out /data1/dev/combatbench_mocap/prior]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mocap.humanoid21.prior.features import (
    TRANS_DIM, clip_frame_features, library_transitions, transitions,
    CLIPS_DIR)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_negatives(feats_by_clip: dict, rng: np.random.Generator) -> dict:
    """每种破坏类型返回 (N,134) 负例转换。feats_by_clip: {cid:(T,67)}"""
    negs = {}
    all_tr = np.concatenate([transitions(f) for f in feats_by_clip.values()])
    # 动态子集：增量幅度 top 50% 的转换。静止段转换增量≈0，在库内
    # 属真实分布——"冻结"/"倒放"负例只有取自动态段才是真的
    # 离分布（动态中突然定住/时间反演），否则判别器学到的是
    # "静止=假"这条与 README §10 冲突的错误规则。
    delta = np.linalg.norm(all_tr[:, 67:] - all_tr[:, :67], axis=1)
    dyn = delta > np.median(delta)
    tr_dyn = all_tr[dyn]

    # frozen：动态段冻结 f_{t+1}=f_t（运动中途定住——离分布）
    negs['frozen'] = np.concatenate([tr_dyn[:, :67], tr_dyn[:, :67]], axis=1)

    # repeat：单段序列随机抽一段重复填充（过渡=同一转换的循环）
    reps = []
    for f in feats_by_clip.values():
        tr = transitions(f)
        if len(tr) < 4:
            continue
        k = rng.integers(0, len(tr) - 4)
        seg = np.repeat(tr[k:k + 1], min(8, len(tr)), axis=0)
        reps.append(seg)
    negs['repeat'] = np.concatenate(reps)

    # shuffle：随机打乱两个半帧的组合（过渡缺失/瞬移式跳变）
    idx = rng.permutation(len(all_tr))
    negs['shuffle'] = np.concatenate(
        [all_tr[:, :67], all_tr[idx][:, 67:]], axis=1)

    # reversed：动态段时间反演（出拳伸展↔收回交换等非对称动态）
    negs['reversed'] = np.concatenate(
        [tr_dyn[:, 67:], tr_dyn[:, :67]], axis=1)

    # jitter：对关节角/速度通道加高频噪声
    jit = all_tr.copy()
    noise = rng.normal(0, 0.35, size=(len(jit), 84)).astype(np.float32)
    jit[:, np.r_[0:21, 21:42, 67:88, 88:109]] += noise
    negs['jitter'] = jit

    # speed：速度类通道放大 ×2.5（不合理的瞬时速度跳变）
    sp = all_tr.copy()
    vel_idx = np.r_[21:42, 43:49, 88:109, 110:116]
    sp[:, vel_idx] *= 2.5
    negs['speed'] = sp

    # random：限位内平滑随机游走序列
    rw = []
    n_rw = 4000
    steps = 30
    q = rng.uniform(-0.8, 0.8, size=(n_rw, 21)).astype(np.float32)
    for _ in range(steps):
        dq = rng.normal(0, 0.05, size=(n_rw, 21)).astype(np.float32)
        q_next = np.clip(q + dq, -1.0, 1.0)
        f_t = np.zeros((n_rw, 67), np.float32)
        f_n = np.zeros((n_rw, 67), np.float32)
        f_t[:, 0:21] = q
        f_n[:, 0:21] = q_next
        f_t[:, 21:42] = (dq / 0.05) * 0.1     # 近似角速度通道
        f_n[:, 21:42] = (dq / 0.05) * 0.1
        f_t[:, 42] = f_n[:, 42] = 1.0          # 根高占位
        rw.append(np.concatenate([f_t, f_n], axis=1))
        q = q_next
    negs['random'] = np.concatenate(rw)

    return negs


class Discriminator(nn.Module):
    def __init__(self, dim: int = TRANS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='v1')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--gp', type=float, default=5.0,
                    help='gradient penalty weight on real samples')
    ap.add_argument('--out', default='/data1/dev/combatbench_mocap/prior')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    feats_by_clip = {}
    for p in sorted(CLIPS_DIR.glob('cmu_*.npz')):
        if p.stem.endswith('_src'):
            continue
        f = clip_frame_features(p)
        if len(f) >= 2:
            feats_by_clip[p.stem] = f
    pos = np.concatenate(
        [transitions(f) for f in feats_by_clip.values()], axis=0)
    print(f'[data] positives {pos.shape} from {len(feats_by_clip)} clips')

    negs = make_negatives(feats_by_clip, rng)
    for k, v in negs.items():
        print(f'[data] neg {k}: {v.shape}')
    # 每类等量采样（random 负例有 120k，全局采样会淹没难例类）
    per_cls = max(len(pos) // len(negs), 1)
    neg_sub = np.concatenate([
        v[rng.choice(len(v), size=min(per_cls, len(v)), replace=False)]
        for v in negs.values()], axis=0)

    mu, sd = pos.mean(0), pos.std(0) + 1e-6
    Xp = torch.tensor((pos - mu) / sd, dtype=torch.float32)
    Xn = torch.tensor((neg_sub - mu) / sd, dtype=torch.float32)

    disc = Discriminator().to(DEVICE)
    opt = torch.optim.Adam(disc.parameters(), lr=a.lr)
    bce = nn.BCELoss()
    bs = 4096
    t0 = time.time()
    for ep in range(a.epochs):
        perm_p = torch.randperm(len(Xp))
        perm_n = torch.randperm(len(Xn))
        ep_loss, n_correct, n_tot = 0.0, 0, 0
        for i in range(0, len(Xp), bs):
            bp = Xp[perm_p[i:i + bs]].to(DEVICE)
            bn = Xn[perm_n[i:i + bs]].to(DEVICE)
            bp.requires_grad_(True)
            out_p, out_n = disc(bp), disc(bn)
            loss = (bce(out_p, torch.ones_like(out_p))
                    + bce(out_n, torch.zeros_like(out_n)))
            if a.gp > 0:
                g = torch.autograd.grad(out_p.sum(), bp,
                                        create_graph=True)[0]
                loss = loss + a.gp * (g.norm(dim=1) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss)
            n_correct += int((out_p > .5).sum() + (out_n < .5).sum())
            n_tot += len(bp) + len(bn)
        if ep % 5 == 4 or ep == 0:
            print(f'[ep {ep+1}] loss={ep_loss:.2f} acc={n_correct/n_tot:.3f}')

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'model': disc.state_dict(), 'mu': mu, 'sd': sd,
        'feat_dim': 67, 'trans_dim': TRANS_DIM, 'version': a.version,
    }
    torch.save(ckpt, out_dir / f'disc_{a.version}.pt')
    meta = {
        'version': a.version,
        'feature': 'features.py extract() 67d/frame, transition=2 frames @20Hz',
        'positives': f'{len(pos)} transitions from {len(feats_by_clip)} clips',
        'negatives': {k: len(v) for k, v in negs.items()},
        'train_seconds': round(time.time() - t0, 1),
        'final_acc': round(n_correct / n_tot, 4),
        'device': DEVICE,
    }
    (out_dir / f'disc_{a.version}.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
