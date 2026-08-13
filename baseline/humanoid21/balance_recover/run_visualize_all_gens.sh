#!/bin/bash
set -e

REPO=/data1/mono/things/combatbench
RUN_ROOT=$REPO/baseline/humanoid21/balance_recover/run_recovery_v2
OUT_BASE=$RUN_ROOT/sample_images

POLICIES=(
  "baseline/runs/fixaw_survonly_crossphi2_s42/policy_exports/u00460/policy_blueprint.yaml"
  "baseline/runs/recovery_v2_gen0/policy_exports/u00495/policy_blueprint.yaml"
  "baseline/runs/recovery_v2_gen1/policy_exports/u00315/policy_blueprint.yaml"
  "baseline/runs/recovery_v2_gen2/policy_exports/u00335/policy_blueprint.yaml"
  "baseline/runs/recovery_v2_gen3/policy_exports/u00215/policy_blueprint.yaml"
  "baseline/runs/recovery_v2_gen4/policy_exports/u00205/policy_blueprint.yaml"
  "baseline/runs/recovery_v2_gen5/policy_exports/u00305/policy_blueprint.yaml"
  "baseline/runs/recovery_v2_gen6/policy_exports/u00185/policy_blueprint.yaml"
  "baseline/runs/recovery_v2_gen7/policy_exports/u00130/policy_blueprint.yaml"
  "baseline/runs/recovery_v2_gen8/policy_exports/u00215/policy_blueprint.yaml"
)

cd $REPO

for i in $(seq 0 9); do
  NPZ="$RUN_ROOT/sample_weights_gen${i}.npz"
  POLICY="${POLICIES[$i]}"
  OUT_DIR="$OUT_BASE/gen${i}"

  echo "=== Gen $i ==="
  PYTHONPATH=$REPO python3 baseline/humanoid21/balance_recover/visualize_samples.py \
    --npz "$NPZ" \
    --policy "$POLICY" \
    --num-samples 200 \
    --output-dir "$OUT_DIR" \
    --seed $((42 + i)) 2>&1 | tail -3
  echo ""
done

echo "Done! All images in $OUT_BASE/"
