#!/bin/bash
# Watcher: wait for verify_resume_A's checkpoint_u00280, launch a resumed
# run B (GPU2), and trigger dumps at u281 (baseline) + u282 (incident).
set -u
cd /data1/mono/things/combatbench
A=baseline/runs/verify_resume_A
B=baseline/runs/verify_u282_scene
CKPT=$A/checkpoints/checkpoint_u00280.pt

echo "[watch] waiting for $CKPT"
while [ ! -f "$CKPT" ]; do sleep 5; done
# Wait until A has logged u282 too (its own copy of the incident)
while ! grep -q '"update": 282' $A/train.log; do sleep 5; done
echo "[watch] ckpt ready and A passed u282; launching B"

PYTHONPATH=/data1/mono/things/combatbench CUDA_VISIBLE_DEVICES=2 \
  python3 -B baseline/framework/train.py \
    --experiment standup_floor04 --algo ppo \
    --run-name verify_u282_scene \
    --resume-from $CKPT --background --no-snapshot
sleep 2

# Sentinel #1: consumed at u281's poll → dumps/u00281
mkdir -p $B
cat > $B/dump_request.json <<'EOF'
{"hypothesis": "u281 baseline — normal update immediately before the u282 early stop; reference for per-minibatch KL/ratio/Δθ comparison"}
EOF
echo "[watch] sentinel for u281 written"

# u281's poll moves the sentinel to dumps/u00281/request.json at update start
while [ ! -f $B/dumps/u00281/request.json ]; do sleep 0.5; done
cat > $B/dump_request.json <<'EOF'
{"hypothesis": "u282 first early stop — reconstruct minibatch-level causal chain: ADV composition → ratio extremes → Δθ → KL climb → stop at ~mb30"}
EOF
echo "[watch] sentinel for u282 written"

# Wait for B's u282 to finish (stats + dump written)
while ! grep -q '"update": 282' $B/train.log; do sleep 1; done
sleep 10
echo "[watch] B u282 done"
ls $B/dumps/
grep "early_stop" $B/train.log | head -3
