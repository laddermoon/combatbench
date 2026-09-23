#!/bin/bash
# Dual-clip arm: resume from ckpt_u00280 → u281 standard surrogate
# (identical prefix), dual_clip_c=3 floor from u282 onward.
set -u
cd /data1/mono/things/combatbench
A=baseline/runs/verify_resume_A
B=baseline/runs/dualclip_from282
CKPT=$A/checkpoints/checkpoint_u00280.pt

PYTHONPATH=/data1/mono/things/combatbench CUDA_VISIBLE_DEVICES=3 \
  python3 -B baseline/framework/train.py \
    --experiment standup_floor04 --algo ppo \
    --run-name dualclip_from282 \
    --resume-from $CKPT \
    --dual-clip-c 3.0 --dual-clip-from-update 282 \
    --background --no-snapshot
sleep 2

mkdir -p $B
cat > $B/dump_request.json <<'EOJ'
{"hypothesis": "u281 prefix check — dual-clip gated at 282, this update must be bit-identical to runs A / winsorize-B / gauss_rank-C"}
EOJ
while [ ! -f $B/dumps/u00281/request.json ]; do sleep 0.5; done
cat > $B/dump_request.json <<'EOJ'
{"hypothesis": "dual-clip u282 — mb28 whipsaw frame (adv<0, ratio=117) should sit on the c·adv floor with zero gradient; no grad spike, KL climb may still occur since direction is otherwise unchanged"}
EOJ
echo "[watch] dual-clip run launched, sentinels placed"
