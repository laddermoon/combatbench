#!/usr/bin/env bash
# V1 技能训练启动脚本（DECISION_LOG D2/D3）。
#
# 用法：
#   ./train_v1.sh smoke            # 冒烟：2 update × 8 ep，随机初始化
#   ./train_v1.sh train            # 正式训练（前台，随机初始化）
#   ./train_v1.sh warm             # warm-start：s42i_b u01680 续训
#   ./train_v1.sh warm_smoke       # warm-start 冒烟
#
# warm-start 用 --resume-from + --reset-update：加载 s42i_b 权重、
# 从 update 0 起算（66 维输入一致，cmd[0:6] 放技能码、cmd[6:9]=0）。

set -euo pipefail
cd "$(dirname "$0")/../../.."          # → things/combatbench
export PYTHONPATH="$PWD"

CKPT="baseline/runs/hier_s2_balance_s42i_b/checkpoints/checkpoint_u01680.pt"
MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    python3 -B baseline/framework/train.py --experiment v1_skill --algo ppo --smoke
    ;;
  train)
    python3 -B baseline/framework/train.py --experiment v1_skill --algo ppo --background
    ;;
  warm)
    python3 -B baseline/framework/train.py --experiment v1_skill --algo ppo --background \
        --resume-from "$CKPT" --reset-update
    ;;
  warm_smoke)
    python3 -B baseline/framework/train.py --experiment v1_skill --algo ppo --smoke \
        --resume-from "$CKPT" --reset-update
    ;;
  *)
    echo "usage: $0 {smoke|train|warm|warm_smoke}" >&2; exit 2
    ;;
esac
