#!/usr/bin/env bash
# Launch the curriculum trainer in the background.
#
# Usage: bash baseline/humanoid21/launch_curriculum.sh [RUN_NAME]
#
# - Runs from the combatbench repo root.
# - Uses CUDA_VISIBLE_DEVICES=1 (4090, 24GB free) by default; override
#   via the `CUDA` env var.
# - Loads stage1 best checkpoint as the actor prior.
# - Writes log to baseline/humanoid21/logs/<RUN_NAME>.log
# - Writes PID to baseline/humanoid21/logs/<RUN_NAME>.pid
# - Records the run name in baseline/humanoid21/logs/LATEST_RUN
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_NAME="${1:-curriculum_resumed_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="baseline/humanoid21/logs"
mkdir -p "$LOG_DIR"

CUDA_DEVICE="${CUDA:-1}"
RESUME_FROM="baseline/humanoid21/runs/stage1_20260430_093352/policy/model.pt"

echo "RUN_NAME=$RUN_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
echo "log: $LOG_DIR/${RUN_NAME}.log"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
nohup python3 -u baseline/humanoid21/curriculum.py \
    --resume-from "$RESUME_FROM" \
    --run-name "$RUN_NAME" \
    > "$LOG_DIR/${RUN_NAME}.log" 2>&1 &

PID=$!
echo "$PID" > "$LOG_DIR/${RUN_NAME}.pid"
echo "$RUN_NAME" > "$LOG_DIR/LATEST_RUN"

echo "PID=$PID"
sleep 2
if kill -0 "$PID" 2>/dev/null; then
    echo "Process is alive. Use the monitor to watch progress:"
    echo "  python3 baseline/humanoid21/curriculum_monitor.py"
else
    echo "ERROR: process exited within 2s. Tail of log:"
    tail -20 "$LOG_DIR/${RUN_NAME}.log"
    exit 1
fi
