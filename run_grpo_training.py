#!/usr/bin/env python3
"""
GRPO 训练启动脚本
"""
import sys
sys.path.insert(0, '/data1/mono/things/combatbench')

from baseline.humanoid21_nonfall.train_grpo import main

if __name__ == "__main__":
    main()
