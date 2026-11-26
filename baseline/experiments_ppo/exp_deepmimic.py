"""DeepMimic 单 clip 运动跟踪实验 — 注册壳。

实现在 ``baseline/deepmimic/``（registry 只扫本目录顶层 exp_*.py，
故实验类本体放包里、此处仅转接）。

用法：
    PYTHONPATH=. python3 baseline/framework/train.py \
        --experiment deepmimic --set clip_id=cmu_13_17 --smoke
或直接 ``baseline/deepmimic/train.sh <clip_id>``。
"""
from baseline.deepmimic.experiment import DeepMimicTrack

EXPERIMENT_CLASS = DeepMimicTrack
