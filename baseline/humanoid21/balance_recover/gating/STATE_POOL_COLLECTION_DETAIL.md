# 状态池收集

## 复现命令

```bash
cd /data1/mono/things/combatbench

# Path A (impulse perturbation)
PYTHONPATH=. python3 baseline/humanoid21/balance_recover/gating/collect_state_pool.py \
    --config baseline/humanoid21/balance_recover/gating/collect_path_a.yaml \
    --output baseline/humanoid21/balance_recover/gating/state_pool_a.npz \
    --workers 96

# Path B (initial state perturbation)
PYTHONPATH=. python3 baseline/humanoid21/balance_recover/gating/collect_state_pool.py \
    --config baseline/humanoid21/balance_recover/gating/collect_path_b.yaml \
    --output baseline/humanoid21/balance_recover/gating/state_pool_b.npz \
    --workers 96
```

输出：`state_pool_a.npz` + `state_pool_b.npz`（各 60k/40k states，~3 min/path）

## 可视化

```bash
cd /data1/mono/things/combatbench
PYTHONPATH=. python3 baseline/humanoid21/balance_recover/gating/render_state_images.py \
    --input baseline/humanoid21/balance_recover/gating/state_pool_a.npz \
    --output-dir baseline/humanoid21/balance_recover/gating/state_pool_a_images \
    --max-images 100
```

```bash
cd /data1/mono/things/combatbench
PYTHONPATH=. python3 baseline/humanoid21/balance_recover/gating/render_state_images.py \
    --input baseline/humanoid21/balance_recover/gating/state_pool_b.npz \
    --output-dir baseline/humanoid21/balance_recover/gating/state_pool_b_images \
    --max-images 100
```

## 状态标注

```bash
cd /data1/mono/things/combatbench
PYTHONPATH=. python3 baseline/humanoid21/balance_recover/gating/label_state_pool.py \
    --input baseline/humanoid21/balance_recover/gating/state_pool_a.npz \
    --output baseline/humanoid21/balance_recover/gating/labeled_state_pool_a.npz \
    --policy baseline/runs/recovery_v5_gen9/policy_exports/u00635/policy_blueprint.yaml \
    --workers 96 \
    --min-torso-height 0.5
```

Results:
  Safe (label=1):   37459 (37.5%)
  Unsafe (label=0): 62541 (62.5%)


Labeling time: 200.9s (0.003s/state)

Results:
  Safe (label=1):   36131 (57.2%)
  Unsafe (label=0): 27083 (42.8%)

```bash
cd /data1/mono/things/combatbench
PYTHONPATH=. python3 baseline/humanoid21/balance_recover/gating/label_state_pool.py \
    --input baseline/humanoid21/balance_recover/gating/state_pool_b.npz \
    --output baseline/humanoid21/balance_recover/gating/labeled_state_pool_b.npz \
    --policy baseline/runs/recovery_v5_gen9/policy_exports/u00635/policy_blueprint.yaml \
    --workers 96
```
