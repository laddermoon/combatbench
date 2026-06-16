# Train Fight Operations Detail Log

## 2026-06-16 11:00 Action: Start Fight Curriculum Training

**Why:** Start stage 3 Combat (Fight) curriculum training on `instance-1f1igpaq` starting from the latest pre-trained chaser checkpoint (`checkpoint_u10294.pt` under `curriculum_follow_20260615_211441`).

**Command:**
```bash
PYTHONPATH=. nohup python3 -m baseline.humanoid21.curriculum.train --v2 --experiment fight --resume-from /data1/mono/things/combatbench/baseline/humanoid21/runs/curriculum_follow_20260615_211441/checkpoints/checkpoint_u10294.pt &> fight.log &
```

**Next step:** Run the command and monitor `fight.log` to ensure successful initialization and start of rollout episodes.

## 2026-06-16 11:10 Action: Implement and Run Custom Combat Monitoring Script

**Why:** Create a robust, combat-specialized log analyzer `analyze_fight_logs.py` to parse 3-way fallback state ratios, damage dealt, and gating switches.

**Command:**
```bash
# Create and make script executable
chmod +x baseline/humanoid21/curriculum/analyze_fight_logs.py

# Run analyzer
python3 baseline/humanoid21/curriculum/analyze_fight_logs.py fight.log
```

**Result:**
The script executed successfully, parsing 23 updates and providing the following core stats:
* Curriculum Level 0, Opponent Speed 0.0 m/s
* Fight/Follow/Recover Ratios: fight=0.402, follow=0.188, recover=0.410 (proving three-way switching is active)
* Net Dealt Damage: +0.44 (demonstrating attack reward calculation is functional)
* Verified 7 reward components with healthy Explained Variances (e.g. EV of r_fall = +0.439)

