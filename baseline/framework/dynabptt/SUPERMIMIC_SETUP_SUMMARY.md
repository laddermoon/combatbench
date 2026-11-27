# SuperMimic 官方复现 Summary

在本机（8×4090 24GB, CUDA 12.4 驱动）复现官方 SuperMimic（mjlab 版）训练。

## The Correct Path

1. **建环境**
   ```bash
   pip install uv
   cd /tmp/supermimic_mjlab && uv sync
   .venv/bin/pip install "torch==2.7.1"   # 降级，见 Pitfall 1
   ```

2. **下载 LAFAN1 G1 数据**（HF 镜像，走代理）
   ```bash
   export https_proxy=http://192.168.16.76:18000
   huggingface-cli download lvhaidong/LAFAN1_Retargeting_Dataset \
     --repo-type dataset --include "g1/*.csv" --local-dir data/raw/lafan1
   ```

3. **转 NPZ**
   ```bash
   .venv/bin/python scripts/data/convert_lafan_to_npz.py \
     --src data/raw/lafan1/g1 --dst src/assets/motions/g1/lafan1_npz
   ```
   预期：40 clips，2.451h。

4. **训练**（24GB 缩规模配置）
   ```bash
   .venv/bin/python scripts/train_supermimic.py --device cuda:0 \
     --num-envs 512 --wm-batch-size 1024 --policy-batch-size 512 \
     --iterations 15000 --eval-interval 500 --checkpoint-interval 500 \
     --output outputs/run512_15k
   ```
   预期：~0.5 s/iter，~5.7 GiB 峰值，ETA ~2h。

## Pitfalls & Solutions

### torch 版本与驱动
- **症状:** lockfile 的 torch 2.14.1+cu130 安装/加载失败
- **根因:** 驱动 CUDA 12.4，cu130 要求更高
- **解决:** venv 内 `torch==2.7.1`（cu126）

### torch.compile 追踪 `dict | dict` 失败
- **症状:** dynamo `Unsupported: dict union/merge`，崩溃在 rewards/velocity 计算
- **根因:** 官方开发用 torch 2.14，dynamo 支持 `|` dict merge；2.7 不支持
- **解决:** `src/tasks/supermimic/mdp/rewards.py:72` 与 `velocity.py:65` 把 `a | b` 改为 `{**a, **b}`。若再遇到同类错，grep `| metrics` / `| {` 或加 `--no-compile` 绕过

### 官方 HF 数据集 gated
- **症状:** `unitreerobotics/LAFAN1_Retargeting_Dataset` 401
- **解决:** 用镜像 `lvhaidong/...`（HF）或 `AI-ModelScope/...`（国内直连）
- **注意:** 数据协议 CC BY-NC-ND 4.0，非商用、禁再分发

### 显存余量大
- 512 envs / policy-batch 512 只用 ~5.7 GiB；24GB 理论上可再放大数倍，主要敏感参数是 `--policy-batch-size`（32 步想象反传图）。下次可试 1024。
