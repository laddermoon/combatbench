#!/bin/bash
# MimicKit DeepMimic28 训练启动器。
# 用法: ./baseline/deepmimic28/train.sh [multi] [--smoke] [--fg] [额外 --set k=v ...]
#   multi      — 多 critic 变体（默认单 critic）
set -e
cd "$(dirname "$0")/../.."

MULTI=false
SMOKE=""
BG="--background"
EXTRA=()
for a in "$@"; do
  case "$a" in
    multi) MULTI=true ;;
    --smoke) SMOKE="--smoke" ;;
    --fg) BG="" ;;
    *) EXTRA+=("$a") ;;
  esac
done

if $MULTI; then EXTRA+=("--set" "multi_critic=True"); fi

PYTHONPATH=. python3 baseline/framework/train.py \
  --experiment deepmimic28 $SMOKE $BG "${EXTRA[@]}"
