"""GaussianMLPPolicy — MimicKit 式无界对角高斯策略（弧度原生动作空间）。

设计来源：MimicKit ``DistributionGaussianDiag``（DeepMimic/AMP 谱系）：
  - mean = 线性输出层，无 tanh squash，无界（弧度空间直接输出）
  - 输出层小初始化（init_output_scale=0.01，零 bias）→ 初始 mean ≈ 0 rad
  - per-dim 对角 σ，可学习（CONSTANT 型），log_std 参数化
  - 无截断、无裁剪：软约束由环境侧 action bounds 裁剪承担

与框架 TruncatedNormal 家族共享的约定：
  - explore_factor → σ 乘法缩放（scale = exp(ei·ln3)）
  - uncertainty U ∈ [0,1]，action-independent：U = mean(min(σ/σ_REF, 1))，
    σ_REF = 1 rad（σ 达到 1 rad 视为"最大不确定"）
  - 不支持 reference-delta（SUPPORTS_REFERENCE_DELTA = False）
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from baseline.framework.ppo.experiment import ActorEval, TrainablePolicy
from envs.framework.policy import Policy, PolicyBlueprint
from baseline.framework.ppo.sampling_context import SamplingContext

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_EXPLORE_K = math.log(3.0)
_LOG_STD_MIN, _LOG_STD_MAX = -5.0, 2.0
_SIGMA_REF = 1.0  # rad，uncertainty 归一化参考

# 尝试复用 truncnorm 的 export codegen 机制；不存在则退化为简单拷贝
try:
    from baseline.framework.ppo.policies.truncated_normal_mlp import _build_export_policy_code
except Exception:  # pragma: no cover
    def _build_export_policy_code(template_name: str) -> str:
        template_path = Path(__file__).parent / template_name
        return template_path.read_text(encoding="utf-8")


class GaussianMLPPolicy(nn.Module, TrainablePolicy, Policy):
    """无界对角高斯 MLP 策略（弧度原生）。"""

    SUPPORTS_REFERENCE_DELTA = False

    _POLICY_CLASS = "GaussianMLPPolicy"
    _EXPORTED_CLASS = "ExportedGaussianMLPPolicy"
    _EXPORT_TEMPLATE = "_export_template_gaussian.py"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device | str = "cpu",
        init_log_std: float = math.log(0.1),
        init_output_scale: float = 0.01,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.init_log_std = float(init_log_std)
        self.init_output_scale = float(init_output_scale)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # MimicKit 式小初始化：初始 mean ≈ 0 rad
        out = self.net[-1]
        nn.init.uniform_(out.weight, -init_output_scale, init_output_scale)
        nn.init.zeros_(out.bias)

        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(init_log_std), dtype=torch.float32)
        )

        self.to(torch.device(device))
        self._gen = torch.Generator(device=self.device)
        self._last_seed: Optional[int] = None

    # ---------------- RNG ----------------
    def _ensure_gen(self, device: torch.device) -> torch.Generator:
        if self._gen.device != device:
            last = self._last_seed
            self._gen = torch.Generator(device=device)
            if last is not None:
                self._gen.manual_seed(last)
        return self._gen

    def reset(self, seed: Optional[int] = None) -> None:
        self._last_seed = None if seed is None else int(seed)
        if seed is not None:
            self._ensure_gen(self.device).manual_seed(int(seed))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ---------------- distribution ----------------
    def effective_log_std(self) -> torch.Tensor:
        return torch.clamp(self.log_std, _LOG_STD_MIN, _LOG_STD_MAX)

    def policy_sigma(self) -> torch.Tensor:
        return self.effective_log_std().exp()

    def _explore_scale(self, explore_factor: Any = 0.0) -> Any:
        if isinstance(explore_factor, torch.Tensor):
            return torch.exp(explore_factor * _EXPLORE_K)
        return math.exp(float(explore_factor) * _EXPLORE_K)

    def effective_sigma(self, policy_sigma: torch.Tensor, explore_factor: Any = 0.0) -> torch.Tensor:
        scale = self._explore_scale(explore_factor)
        if isinstance(scale, torch.Tensor):
            return policy_sigma * scale.unsqueeze(-1)
        return policy_sigma * scale

    def forward(self, obs: torch.Tensor, *, ctx: Optional[SamplingContext] = None):
        mean = self.net(obs)
        ef = ctx.explore_factor if ctx is not None else 0.0
        sigma = self.effective_sigma(self.policy_sigma(), ef)
        return mean, sigma.expand_as(mean) if sigma.dim() < 2 else sigma

    # ---------------- sampling ----------------
    def sample_action(self, obs: torch.Tensor, *, ctx: Optional[SamplingContext] = None,
                      u: Optional[torch.Tensor] = None):
        mean, sigma = self.forward(obs, ctx=ctx)
        if u is None:
            eps = torch.randn(mean.shape, generator=self._ensure_gen(mean.device),
                              device=mean.device, dtype=mean.dtype)
        else:
            eps = torch.as_tensor(u, device=mean.device, dtype=mean.dtype)
        action = mean + sigma * eps
        z = (action - mean) / sigma
        log_prob = (-0.5 * z * z - torch.log(sigma) - 0.5 * math.log(2 * math.pi)).sum(dim=-1)
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return mean

    # ---------------- training-side ----------------
    def evaluate_actions(self, obs, actions, ctx=None, *, want_stats=False) -> ActorEval:
        mean = self.net(obs)
        policy_sigma = self.policy_sigma()
        ef = ctx.explore_factor if ctx is not None else 0.0
        eff_sigma = self.effective_sigma(policy_sigma, ef)

        z = (actions - mean) / eff_sigma
        log_prob = (-0.5 * z * z - torch.log(eff_sigma)
                    - 0.5 * math.log(2 * math.pi)).sum(dim=-1)

        # U: action-independent；σ 为常量参数，故对 batch 亦为常量（契约允许）
        U_per_dim = (policy_sigma / _SIGMA_REF).clamp(max=1.0)
        uncertainty = U_per_dim.mean(dim=-1).expand(log_prob.shape).clamp(0.0, 1.0)

        stats = None
        if want_stats:
            with torch.no_grad():
                stats = {
                    "uncertainty": float(uncertainty.mean().item()),
                    "std_mean": float(policy_sigma.mean().item()),
                    "eff_std_mean": float(eff_sigma.mean().item()),
                    "std_min": float(policy_sigma.min().item()),
                    "std_max": float(policy_sigma.max().item()),
                    "mean_abs": float(mean.abs().mean().item()),
                }
        return ActorEval(log_prob=log_prob, uncertainty=uncertainty, stats=stats, mean=mean)

    # ---------------- Policy contract ----------------
    def act(self, observation: Any, *, want_extra: bool = False):
        obs_t = torch.as_tensor(np.asarray(observation, dtype=np.float32),
                                dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.deterministic_action(obs_t)
        return action.squeeze(0).cpu().numpy().astype(np.float32), None

    def sample(self, observation: Any, *, ctx: Optional[SamplingContext] = None,
               want_extra: bool = False):
        obs_t = torch.as_tensor(np.asarray(observation, dtype=np.float32),
                                dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.sample_action(obs_t, ctx=ctx)
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        if not want_extra or log_prob is None:
            return action_np, None
        return action_np, {"log_prob": float(log_prob.item())}

    # ---------------- export ----------------
    def to_blueprint(self, dest_path: Optional[str] = None) -> "PolicyBlueprint":
        if dest_path is None:
            dest_path = tempfile.mkdtemp(prefix="policy_export_")
        policy_dir = Path(dest_path)
        policy_dir.mkdir(parents=True, exist_ok=True)

        state_dict = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        payload = {
            "format_version": 1,
            "policy_class": self._POLICY_CLASS,
            "arch": {"obs_dim": self.obs_dim, "action_dim": self.action_dim,
                     "hidden_dim": self.hidden_dim},
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "state_dict": state_dict,
            "state_dict_keys": sorted(state_dict.keys()),
            **self._export_extra(),
        }
        torch.save(payload, policy_dir / "model.pt")

        policy_code = _build_export_policy_code(self._EXPORT_TEMPLATE)
        (policy_dir / "policy.py").write_text(policy_code, encoding="utf-8")

        manifest = {
            "format_version": 1,
            "policy_class": self._POLICY_CLASS,
            "arch": payload["arch"],
            "files": ["model.pt", "policy.py", "MANIFEST.json"],
            "exported_class": self._EXPORTED_CLASS,
            **self._export_extra(),
        }
        (policy_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return PolicyBlueprint(cls=f"file:{policy_dir / 'policy.py'}:{self._EXPORTED_CLASS}")

    def _export_extra(self) -> Dict[str, Any]:
        return {
            "distribution_kind": "diagonal_gaussian_v1",
            "std_source": "shared",
            "std_parameterization": "log_std_v1",
            "uncertainty_kind": "sigma_ref_ratio_v1",
            "exploration_kind": "log_std_multiplicative_v1",
        }
