"""V2 线 Stage 3 — 随机臂脚本接管下的推力平衡课程（79 维）。

在 S2 收敛模型（level 11、抗 100N）基础上重训同一 12 级课程：
站稳后（h>1.10×20 步）flag=1，env 内 act[15:21] 被覆写为随机
脚本目标（每臂 U(2,20) 步重采样、U(-1,1) norm、双机双臂独立
rng）——既是平衡扰动，也撤掉双臂这个抗扰手段；起身/摔倒期间
flag=0 策略自控双臂。obs[57]=flag、obs[58:64]=arm_tgt 由
V2Simulator 同源写入（承诺=实际施加）。

通道/课程/超参全部继承 v2_s2_balance 不变；轨迹存策略原始
action（覆写前）——变换 MDP 上的标准 PPO，臂维成死动作维，
flag=1 期间只有有界漂移（已知风险，必要时加 imitation 项）。

启动（warm-start S2 最强 ckpt，同 --reset-update）：

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 -B \
      baseline/framework/train.py --experiment v2_s3_balance_arm \
      --algo ppo \
      --resume-from baseline/runs/v2_s2_balance/checkpoints/checkpoint_uXXXXX.pt \
      --reset-update --run-name v2_s3_balance_arm --background

配套文件（baseline/humanoid21/v1/）：
  arm_script.py · v2_simulator.py · v2_balance_arm_env.yaml
"""
from __future__ import annotations

from pathlib import Path

from envs.framework.parameterized_blueprint import ParameterizedEnvBlueprint

from .exp_v2_s2_balance import V2S2Balance

_V1 = Path(__file__).resolve().parent.parent / "humanoid21" / "v1"


class V2S3BalanceArm(V2S2Balance):
    name = "v2_s3_balance_arm"

    # 关闭实验级早停：level 6+ 长推平台期 quality 可长期不涨，
    # 200-eval 无改善判停过早（u1450 误停）。仍保留 best 追踪/保存。
    _no_improvement_limit = 10 ** 9

    def _env_pb(self):
        return ParameterizedEnvBlueprint.load(
            _V1 / "v2_balance_arm_env.yaml")


EXPERIMENT_CLASS = V2S3BalanceArm


