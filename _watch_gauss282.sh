#!/bin/bash
# gauss_rank arm: resume from ckpt_u00280 → u281 zscore (identical
# prefix), switch to gauss_rank from u282 onward.
set -u
cd /data1/mono/things/combatbench
A=baseline/runs/verify_resume_A
B=baseline/runs/gaussrank_from282
CKPT=$A/checkpoints/checkpoint_u00280.pt

PYTHONPATH=/data1/mono/things/combatbench CUDA_VISIBLE_DEVICES=2 \
  python3 -B baseline/framework/train.py \
    --experiment standup_floor04 --algo ppo \
    --run-name gaussrank_from282 \
    --resume-from $CKPT \
    --adv-norm gauss_rank --adv-norm-from-update 282 \
    --background --no-snapshot
sleep 2

mkdir -p $B
cat > $B/dump_request.json <<'EOJ'
{"hypothesis": "u281 prefix check — adv_norm switch gated at 282, this update must be bit-identical to runs A and winsorize-B"}
EOJ
while [ ! -f $B/dumps/u00281/request.json ]; do sleep 0.5; done
cat > $B/dump_request.json <<'EOJ'
{"hypothesis": "gauss_rank u282 — bounded order-statistic advs should prevent the mb28 grad spike and let all 4 epochs run; compare timeline vs winsorize arm and original"}
EOJ
echo "[watch] gauss_rank run launched, sentinels placed"
