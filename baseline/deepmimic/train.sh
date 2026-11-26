#!/usr/bin/env bash
# DeepMimic 单 clip 训练启动脚本。
#
# 用法：
#   ./train.sh <clip_id> [选项] [--set K=V ...] [train.py 其他参数]
#
#   --smoke   冒烟（2 updates / 8 episodes）
#   --fg      前台跑（默认 --background）
#   其余参数原样传给 baseline/framework/train.py
#
# 例：
#   ./baseline/deepmimic/train.sh cmu_13_17 --smoke
#   ./baseline/deepmimic/train.sh accad_m2ma_punches_e1_jab_left
#   ./baseline/deepmimic/train.sh cmu_07_01 --set drift_dxy=1.2
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # things/combatbench
CLIP="${1:?usage: train.sh <clip_id> [--smoke] [--fg] [extra args]}"
shift

FG=0; SMOKE=0; PROFILE="legacy"; EXTRA=()
for a in "$@"; do
  case "$a" in
    --fg)     FG=1 ;;
    --smoke)  SMOKE=1 ;;
    --simchar) PROFILE="simchar" ;;
    *)        EXTRA+=("$a") ;;
  esac
done

# 查 clip 信息（n_ctrl 做 max_steps 提示；实验侧 max_steps<=0 也会自动，
# 这里显式设置只是为了写进 run 名便于辨认）。
cd "$ROOT"
N_CTRL=$(PYTHONPATH="$ROOT" python3 - "$CLIP" <<'PY'
import sys
from baseline.deepmimic.ref import CoreRefView
print(CoreRefView(sys.argv[1]).n_ctrl)
PY
)

TS=$(date +%m%d_%H%M)
TAG=$([[ "$PROFILE" == "simchar" ]] && echo "_simchar" || echo "")
RUN_NAME="dm_${CLIP}${TAG}_${TS}"

ARGS=(--experiment deepmimic
      --set "clip_id=${CLIP}"
      --set "max_steps=${N_CTRL}"
      --set "actuation_profile=${PROFILE}"
      --run-name "${RUN_NAME}")
[[ $SMOKE == 1 ]] && ARGS+=(--smoke)
[[ $FG == 0 ]] && ARGS+=(--background)

echo "[train.sh] clip=${CLIP} n_ctrl=${N_CTRL} profile=${PROFILE} run=${RUN_NAME}"
exec env PYTHONPATH="$ROOT" python3 -B baseline/framework/train.py \
    "${ARGS[@]}" "${EXTRA[@]}"
