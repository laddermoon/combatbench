# DeepMimic 单 clip 运动跟踪

在 `data/core` 的任意一条 CoreClip 上训练 PPO 模仿策略（DeepMimic 式：
RSI + 加权模仿奖励 + 早停）。设计与论文差异见 `DESIGN.md` §5。

## 训练

```bash
cd /data1/mono/things/combatbench

# 冒烟（先验证管线）
./baseline/deepmimic/train.sh cmu_13_17 --smoke

# 正式训练（后台，日志在 run_dir/train.log）
./baseline/deepmimic/train.sh cmu_13_17
./baseline/deepmimic/train.sh accad_m2ma_punches_e1_jab_left

# 覆盖默认参数（透传给 --set）
./baseline/deepmimic/train.sh cmu_13_17 --set drift_dxy=0.6 \
    --set drift_dyaw_deg=50

# 前台调试
./baseline/deepmimic/train.sh cmu_13_17 --fg --smoke
```

输出到 `baseline/runs/dm_<clip>_<ts>/`：`checkpoints/`（周期）、
`policy/`（best-of-run）、`policy_exports/uNNNNN/`（每 update 导出）、
`videos/uNNNNN.mp4`（eval 视频）。

默认值：实验自动取 `max_steps=clip.n_ctrl`；getup/fall/lie/crawl 类 clip
自动 `fall_mode=contact`；drift 默认 0.8m/75°，`--set drift_dxy=None`
关闭。

## 评估视频

```bash
# 确定性 rsi=first 整段回放 + 指标（M2 验收口径）
PYTHONPATH=. python3 baseline/deepmimic/eval.py \
    --policy baseline/runs/dm_cmu_13_17_XXXXXX/policy_exports/u01900 \
    --clip cmu_13_17 \
    --video /tmp/cmu_13_17_eval.mp4 \
    --json /tmp/cmu_13_17_eval.json
```

- `--policy` 接受 `policy_exports/uNNNNN` 或整个 run 目录（取 `policy/`）
- 指标：completed / phase / reward_mean / kp_err / dxy / dyaw / q_rmse
- `--fall-mode contact` / `--drift-dxy` 等可改评估终止条件
- `--start-frame K` 从指定控制帧起回放

## 选片段

```bash
# 查看 clip 统计（时长/标签/质量/净位移）
PYTHONPATH=. python3 -c "
from baseline.deepmimic.ref import CoreRefView
import sys, json
print(json.dumps(CoreRefView(sys.argv[1]).clip_stats(), indent=1, ensure_ascii=False))
" cmu_13_17
```

`data/core/README.md` 有标签/质量分布；建议先跑 `strike_*` /
`stance_*`（净位移小、M2 已验证可收敛），getup/lie/crawl 类需
`fall_mode=contact|none`。

## 实验入口

`baseline/experiments_ppo/exp_deepmimic.py`（注册名 `deepmimic`）——
也可不用脚本直接：

```bash
PYTHONPATH=. python3 baseline/framework/train.py --experiment deepmimic \
    --set clip_id=cmu_13_17 --background
```
