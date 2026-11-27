# SuperMimic 官方实现复现 — 部署详细日志

按时间顺序记录在本机（instance-1f1igpaq, 8×4090 24GB）复现官方 SuperMimic 训练管线的每一步。
仓库位置：`/tmp/supermimic_mjlab`（官方仓库 clone + 本地 patch）。

## [环境] uv + venv

**Why:** 官方用 uv/lockfile 管理环境，Python 3.11 + mjlab + mujoco-warp。

**命令:**
```bash
pip install uv
cd /tmp/supermimic_mjlab && uv sync
```

**结果:** lockfile 默认 torch 2.14.1+cu130 安装失败——本机驱动 CUDA 12.4 太旧。
**解决:** venv 内降级 `torch==2.7.1`（cu126 wheel），验证 `torch.cuda.is_available()=True`。

## [数据] LAFAN1 G1 retarget

**Why:** 官方训练用 40 条 G1-retarget 的 LAFAN1 clip（~2.45h），仓库不含数据。

**尝试:**
- `unitreerobotics/LAFAN1_Retargeting_Dataset`（HF 官方）→ 401，gated（需账号+接受 CC BY-NC-ND 4.0）
- `lvhaidong/LAFAN1_Retargeting_Dataset`（HF 镜像）→ 可用，`g1/*.csv` 40 文件 ~90MB，走代理 `http://192.168.16.76:18000`
- `AI-ModelScope/LAFAN1_Retargeting_Dataset`（ModelScope）→ 备用，国内直连可用

**命令:** snapshot_download g1/*.csv → `data/raw/lafan1/` → `scripts/data/convert_lafan_to_npz.py` → `src/assets/motions/g1/lafan1_npz/`（40 NPZ，2.451h）+ manifest.json。

**结果:** 与官方数据规模一致（40 clips / 2.45h / 30Hz CSV / 36 列 schema）。

## [冒烟] 60 轮小规模（64 envs 默认）

**命令:**
```bash
.venv/bin/python scripts/train_supermimic.py --device cuda:0 --iterations 60 --output outputs/smoke
```

**结果（两次失败 + 修复）:**
1. torch 2.7 dynamo 无法 trace `dict | dict`（官方开发环境 torch 2.14 支持）→ patch `src/tasks/supermimic/mdp/rewards.py:72` `terms | metrics` → `{**terms, **metrics}`
2. 同模式 crash → patch `src/tasks/supermimic/mdp/velocity.py:65` 同样改法

第三次（outputs/smoke2）跑通：0.34 s/iter，475 transitions/s，峰值 1.89 GiB。
WM loss 2.85→0.67，imagined return 0.16→0.50。eval 0.002 return（60 轮正常）。

## [长跑] 512 envs × 15000 轮

**命令:**
```bash
nohup .venv/bin/python scripts/train_supermimic.py --device cuda:0 \
  --num-envs 512 --wm-batch-size 1024 --policy-batch-size 512 \
  --iterations 15000 --eval-interval 500 --checkpoint-interval 500 \
  --output outputs/run512_15k > outputs/run512_15k.log 2>&1 &
```

**结果:** ~0.4-0.6 s/iter，2-3k transitions/s，GPU 峰值 ~5.7 GiB，ETA ~1.5-2.5h。
iter 70 时 imagined return 0.55 上升中；critic loss 大且振荡（1k-12k）需后续观察。
