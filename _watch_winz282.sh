#!/bin/bash
# Launch winsorize run from ckpt_u00280: u281 unwinsorized (prefix must
# stay bit-identical to run A), winsorize ±4σ active from u282 onward.
set -u
cd /data1/mono/things/combatbench
A=baseline/runs/verify_resume_A
B=baseline/runs/winsorize4_from282
CKPT=$A/checkpoints/checkpoint_u00280.pt

PYTHONPATH=/data1/mono/things/combatbench CUDA_VISIBLE_DEVICES=1 \
  python3 -B baseline/framework/train.py \
    --experiment standup_floor04 --algo ppo \
    --run-name winsorize4_from282 \
    --resume-from $CKPT \
    --adv-winsorize-sigma 4.0 --adv-winsorize-from-update 282 \
    --background --no-snapshot
sleep 2

# Sentinel #1: consumed at u281's poll → dumps/u00281 (prefix check)
mkdir -p $B
cat > $B/dump_request.json <<'EOJ'
{"hypothesis": "winsorize-gated-off u281 — must be bit-identical to run A's u281; verifies the from_update gate preserves the prefix"}
EOJ
while [ ! -f $B/dumps/u00281/request.json ]; do sleep 0.5; done
cat > $B/dump_request.json <<'EOJ'
{"hypothesis": "winsorized u282 — expect no grad spike at mb28 (cap ±4σ kills the -13σ whip frames), all 4 epochs run to ~200 actor steps, KL stays under window threshold"}
EOJ
echo "[watch] sentinels placed; u282 dump armed"
