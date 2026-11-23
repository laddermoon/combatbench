"""Self-contained exported policy template for
StateMixtureBoundedStdTruncatedNormalPolicy.

This file is the source of the ``policy.py`` that ``to_blueprint``
writes into every export directory.  It is **self-contained** — it does
NOT import from ``baseline.*`` or any other repo module.  The only
dependencies are ``torch``, ``numpy``, ``math``, and the Python standard
library.  This means:

- Exported policies work without the repo on ``sys.path``.
- Internal refactoring (renaming modules, moving files) does not break
  historical artifacts.
- The export is suitable for benchmark/competition submission where the
  user may not have the repo at all.

The inference logic (K-component truncated-normal mixture, shared
component sampling, erf-space inverse-CDF, mixture log_prob, bounded
sigmoid σ map with v+αe explore) is inlined from
``state_mixture_bounded_std_truncated_normal_mlp.py``.  A parity test
in ``test_state_mixture_bounded_std_truncated_normal.py`` verifies
that the training-side policy and this exported version produce
identical outputs on the same inputs.

This file is lint-able, type-checkable, and testable on its own.  Do not
add imports from ``baseline.*`` or ``envs.*`` — that would re-introduce
the P0-6 bug.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Environment protocol (minimal inline stubs — no repo import needed)
# ---------------------------------------------------------------------------


class Policy:
    """Minimal Policy ABC stub for the exported policy."""

    def act(self, observation: Any, *, want_extra: bool = False) -> Tuple[np.ndarray, Any]:
        raise NotImplementedError


class SamplingContext:
    """Self-contained sampling context — mirrors the framework's
    ``baseline.framework.ppo.sampling_context.SamplingContext`` so this
    file needs no repo import.  Fields are per-frame values (scalars or
    arrays); the exported policies consume ``explore_factor`` plus the
    ``reference_action``/``delta_factor``/``delta`` fields.

    A plain class (not a dataclass): this file is exec-loaded without a
    real module entry, so ``from __future__ import annotations`` +
    ``@dataclass`` would fail to resolve string annotations.
    """

    __slots__ = (
        "explore_factor", "reference_action", "delta_factor",
        "delta",
    )

    def __init__(self, explore_factor=None, reference_action=None,
                 delta_factor=None,
                 delta=None):
        self.explore_factor = explore_factor
        self.reference_action = reference_action
        self.delta_factor = delta_factor
        self.delta = delta

    def has_delta(self) -> bool:
        """True iff the reference-delta scale is active — mirrors the
        upstream ``SamplingContext.has_delta``."""
        if self.reference_action is None and self.delta is None:
            return False
        c = self.delta_factor
        if c is None:
            return False
        if hasattr(c, "any"):  # ndarray / torch.Tensor
            return bool((c != 0).any())
        return c != 0



def _ctx_bcast(x: Any, target: torch.Tensor) -> Any:
    """ctx scalar field → float, or a (B,) tensor → broadcastable."""
    if not torch.is_tensor(x):
        return float(x)
    while x.ndim < target.ndim:
        x = x.unsqueeze(-1)
    return x


def _ctx_nonzero(x):
    """True iff a ctx scalar field is nonzero (scalar or any-element)."""
    if x is None:
        return False
    return bool((x != 0).any()) if hasattr(x, "any") else bool(x != 0)


def _delta_of(mean, ctx):
    """The Δ payload the σ floor consumes this call, or ``None``.

    An explicit ``ctx.delta`` payload wins over the fresh
    ``mean − a_ref`` path — ``ctx.delta`` is the action-level Δ
    (``det_action − a_ref``) supplied by the sampling layer / replay,
    never a function of current θ.  Payload presence is the contract;
    ``delta_max_sigma`` decides what a missing payload means.
    """
    if ctx is None:
        return None
    def _bcast(v):
        t = torch.as_tensor(v, dtype=mean.dtype, device=mean.device)
        while t.ndim < mean.ndim:
            t = t.unsqueeze(-2)
        return t

    frozen = getattr(ctx, "delta", None)
    if frozen is not None:
        return _bcast(frozen)
    ref = getattr(ctx, "reference_action", None)
    if ref is None:
        return None
    return mean - _bcast(ref)


def delta_max_sigma(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    ctx: Optional["SamplingContext"],
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
) -> torch.Tensor:
    """Element-wise σ floor from the reference delta —
    inlined copy of ``truncated_normal_mlp.delta_max_sigma``.

        σ_eff = max(σ, c·|Δ|)

    ``mean`` is detached in the Δ term (σ_eff exogenous w.r.t. m_θ —
    see the source docstring).  The max keeps σ_eff ≥ σ in every dim:
    the delta can only widen exploration, never narrow it."""
    # Attribute-based activation check — NOT ctx.has_delta(): ``ctx`` may
    # be an instance of an older SamplingContext class held by a spawned
    # rollout worker; its fields exist but methods may not.
    c = getattr(ctx, "delta_factor", None) if ctx is not None else None
    if not _ctx_nonzero(c):
        return sigma
    delta = _delta_of(mean, ctx)
    if delta is None:
        # c ≠ 0 but no Δ source — the rollout-side record lost its
        # payload (stale buffer or pipeline bug); fail loud.
        raise ValueError(
            "delta ctx carries delta_factor != 0 but neither `delta` "
            "nor `reference_action` — the rollout-side sctx__ "
            "payload record is missing (stale buffer or pipeline "
            "bug)"
        )
    out = torch.maximum(
        sigma, (_ctx_bcast(c, sigma) * delta.detach()).abs(),
    )
    if sigma_min is not None:
        out = out.clamp(sigma_min, sigma_max)
    return out


class StochasticPolicy:
    """Minimal StochasticPolicy stub for the exported policy."""

    def sample(
        self,
        observation: Any,
        *,
        ctx: Optional["SamplingContext"] = None,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None) -> None:
        pass


# ---------------------------------------------------------------------------
# Truncated-normal mixture math (inlined from the mixture policies)
# ---------------------------------------------------------------------------

_SQRT_2 = math.sqrt(2.0)
_LOG_2PI_HALF = 0.5 * math.log(2.0 * math.pi)

_ACTION_LOW = -1.0
_ACTION_HIGH = 1.0

_ERFINV_EPS = 1e-7
_ACTION_EPS = 1e-6

_EI_TOLERANCE = 1e-6            # float32 slack on the [-1, 1] ei range


# ---------------------------------------------------------------------------
# Self-contained inference network
# ---------------------------------------------------------------------------


class _StateMixtureBoundedStdInferenceNet(nn.Module):
    """Inference-only K-component mixture with state-dependent bounded σ.

    Self-contained copy of the inference subset of
    ``StateMixtureBoundedStdTruncatedNormalPolicy`` — excludes
    ``evaluate_actions`` (training-only) and ``to_blueprint``
    (export-only).  State-dict keys are identical to the training side
    (``trunk.*``, ``head.*``).

    Head layout: ``logits (K) | raw_mean (K·D) | raw_v (K·D)``,
    component-major; σ is the bounded map of the per-state ``v`` block.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_components: int,
        sigma_min: float,
        sigma_max: float,
        explore_alpha: float,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_components = int(num_components)

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        K, D = self.num_components, self.action_dim
        self.head = nn.Linear(hidden_dim, K + 2 * K * D)

        self._r_min = math.log(float(sigma_min))
        self._delta_r = math.log(float(sigma_max)) - self._r_min
        self._sigma_min = float(sigma_min)
        self._sigma_max = float(sigma_max)
        self._explore_alpha = float(explore_alpha)
        self._gen = torch.Generator()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reseed this net's private RNG (per-episode reproducibility)."""
        if seed is not None:
            self._gen.manual_seed(int(seed))

    def _bounded_sigma(self, v_e: torch.Tensor) -> torch.Tensor:
        """σ = exp(r_min + Δr · sigmoid(v_e))  ∈ (σ_min, σ_max)."""
        return torch.exp(
            self._r_min + self._delta_r * torch.sigmoid(v_e)
        )

    @staticmethod
    def _check_ei(explore_factor: Any) -> None:
        """explore_factor must stay within [-1, 1] — no silent clamp."""
        if (
            isinstance(explore_factor, torch.Tensor)
            and explore_factor.is_cuda
            and torch.cuda.is_current_stream_capturing()
        ):
            # GPU→CPU validation syncs are illegal while a CUDA graph is
            # being captured; eager calls keep the full check.
            return
        if isinstance(explore_factor, torch.Tensor):
            if not bool(torch.isfinite(explore_factor).all()):
                raise ValueError("explore_factor contains non-finite values")
            bad = (explore_factor < -1.0 - _EI_TOLERANCE) | (
                explore_factor > 1.0 + _EI_TOLERANCE
            )
            if bool(bad.any()):
                raise ValueError(
                    f"explore_factor out of [-1, 1]: "
                    f"min={float(explore_factor.min())}, "
                    f"max={float(explore_factor.max())}"
                )
        else:
            e = float(explore_factor)
            if not math.isfinite(e) or abs(e) > 1.0 + _EI_TOLERANCE:
                raise ValueError(
                    f"explore_factor out of [-1, 1]: {explore_factor}"
                )

    def _head_forward(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """→ (log_pi, mean, v) — v is the head's raw σ-control block."""
        K, D = self.num_components, self.action_dim
        out = self.head(self.trunk(obs))
        logits = out[..., :K]
        raw_mean = out[..., K:K + K * D].reshape(-1, K, D)
        v = out[..., K + K * D:].reshape(-1, K, D)
        log_pi = torch.log_softmax(logits, dim=-1)
        mean = torch.tanh(raw_mean)
        return log_pi, mean, v

    def _effective_sigma(
        self, obs_v: torch.Tensor, explore_factor: Any = 0.0,
    ) -> torch.Tensor:
        """σ(v + αe) — additive shift on the raw sigmoid input."""
        v = obs_v
        if isinstance(explore_factor, torch.Tensor):
            v_e = v + self._explore_alpha * explore_factor.view(-1, 1, 1)
        else:
            v_e = v + self._explore_alpha * float(explore_factor)
        return self._bounded_sigma(v_e)

    @staticmethod
    def _log_trunc_Z(
        mean: torch.Tensor, sigma: torch.Tensor,
    ) -> torch.Tensor:
        erf_hi = torch.erf((_ACTION_HIGH - mean) / (_SQRT_2 * sigma))
        erf_lo = torch.erf((_ACTION_HIGH + mean) / (_SQRT_2 * sigma))
        Z = 0.5 * (erf_hi + erf_lo)
        Z = torch.clamp_min(Z, torch.finfo(Z.dtype).tiny)
        return torch.log(Z)

    def _mixture_log_prob(
        self,
        actions: torch.Tensor,
        mean: torch.Tensor,
        sigma: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        a = actions.unsqueeze(1)
        log_Z = self._log_trunc_Z(mean, sigma)
        z = (a - mean) / sigma
        comp_lp = (
            -0.5 * z * z - torch.log(sigma) - _LOG_2PI_HALF - log_Z
        ).sum(dim=-1)
        return torch.logsumexp(comp_lp + log_pi, dim=-1)

    def sample_action(
        self, obs: torch.Tensor, *, ctx: Optional["SamplingContext"] = None,
        uniform: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ``uniform`` (B, D+1): column 0 selects the mixture component via
        # inverse-CDF on the cumulated weights (replacing multinomial);
        # columns 1: are the per-dim action draws (replacing the internal
        # RNG).  The batched inference server injects per-request noise
        # here so results are independent of batch composition.
        ef = ctx.explore_factor if ctx is not None else 0.0
        self._check_ei(ef)
        log_pi, mean, v = self._head_forward(obs)
        sigma = self._effective_sigma(v, ef)
        sigma = delta_max_sigma(
            mean, sigma, ctx, self._sigma_min, self._sigma_max,
        )
        B, K, D = mean.shape

        if K == 1:
            idx = torch.zeros(B, dtype=torch.long, device=mean.device)
        elif uniform is not None:
            cum = torch.cumsum(log_pi.exp(), dim=-1)
            idx = (
                torch.searchsorted(cum, uniform[:, 0].unsqueeze(-1).contiguous())
                .squeeze(-1)
                .clamp_(max=K - 1)
            )
        else:
            idx = torch.multinomial(
                log_pi.exp(), 1, generator=self._gen,
            ).squeeze(-1)
        sel = idx.view(-1, 1, 1).expand(-1, 1, D)
        mu_k = mean.gather(1, sel).squeeze(1)
        sg_k = sigma.gather(1, sel).squeeze(1)

        erf_a = torch.erf((_ACTION_LOW - mu_k) / (_SQRT_2 * sg_k))
        erf_b = torch.erf((_ACTION_HIGH - mu_k) / (_SQRT_2 * sg_k))
        u = (
            uniform[:, 1:]
            if uniform is not None
            else torch.rand(
                mu_k.shape,
                generator=self._gen,
                device=mu_k.device,
                dtype=mu_k.dtype,
            )
        )
        q = (1.0 - u) * erf_a + u * erf_b
        q = q.clamp(-1.0 + _ERFINV_EPS, 1.0 - _ERFINV_EPS)
        action = mu_k + _SQRT_2 * sg_k * torch.erfinv(q)
        action = torch.clamp(
            action, _ACTION_LOW + _ACTION_EPS, _ACTION_HIGH - _ACTION_EPS,
        )

        log_prob = self._mixture_log_prob(action, mean, sigma, log_pi)
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        log_pi, mean, _ = self._head_forward(obs)
        idx = log_pi.argmax(dim=-1)
        sel = idx.view(-1, 1, 1).expand(-1, 1, self.action_dim)
        return mean.gather(1, sel).squeeze(1)


# ---------------------------------------------------------------------------
# Exported policy class (loaded by PolicyBlueprint at runtime)
# ---------------------------------------------------------------------------

_EXPORT_FORMAT_VERSION = 1
_DISTRIBUTION_KIND = "bounded_std_mixture_truncated_normal_v1"
_STD_PARAMETERIZATION = "sigmoid_log_std_v1"
_EXPLORATION_KIND = "raw_std_additive_shift_v1"
_STD_SOURCE = "state"


class ExportedStateMixtureBoundedStdTruncNormPolicy(Policy, StochasticPolicy):
    """Runtime-loadable policy backed by a ``model.pt`` checkpoint.

    Implements both ``Policy`` (deterministic ``act()`` → highest-weight
    component mean) and ``StochasticPolicy`` (mixture ``sample()``).

    Loading is strict (``strict=True``) and validates
    ``format_version``, ``policy_class``, the distribution-identity
    metadata fields, the bounded-σ config (``sigma_min``/``sigma_max``/
    ``explore_alpha``), and ``num_components`` before attempting to load.
    """
    # Capability flag read by SamplingPolicy at wrap time — this cell
    # implements the reference-delta σ floor (delta_max_sigma above).
    SUPPORTS_REFERENCE_DELTA = True


    def __init__(self, model_path: Optional[str] = None):
        payload_path = (
            Path(model_path) if model_path is not None
            else Path(__file__).resolve().parent / "model.pt"
        )
        payload = torch.load(payload_path, map_location="cpu")

        fv = payload.get("format_version", 0)
        if fv != _EXPORT_FORMAT_VERSION:
            raise RuntimeError(
                f"Policy export format version mismatch: "
                f"file has {fv}, loader expects {_EXPORT_FORMAT_VERSION}. "
                f"This export was created by a different version of "
                f"the framework. Re-export the policy with the current code."
            )
        pcls = payload.get("policy_class", "unknown")
        if pcls != "StateMixtureBoundedStdTruncatedNormalPolicy":
            raise RuntimeError(
                f"Policy class mismatch: file says {pcls!r}, "
                f"loader expects 'StateMixtureBoundedStdTruncatedNormalPolicy'."
            )
        for key, expected in (
            ("distribution_kind", _DISTRIBUTION_KIND),
            ("std_parameterization", _STD_PARAMETERIZATION),
            ("exploration_kind", _EXPLORATION_KIND),
            ("std_source", _STD_SOURCE),
        ):
            got = payload.get(key, "missing")
            if got != expected:
                raise RuntimeError(
                    f"{key} mismatch: file says {got!r}, "
                    f"loader expects {expected!r}."
                )
        for key in ("sigma_min", "sigma_max", "explore_alpha"):
            val = payload.get(key)
            if val is None or not math.isfinite(float(val)):
                raise RuntimeError(
                    f"missing or non-finite {key} in export payload: {val!r}"
                )
        sigma_min = float(payload["sigma_min"])
        sigma_max = float(payload["sigma_max"])
        explore_alpha = float(payload["explore_alpha"])
        if not (0.0 < sigma_min < sigma_max):
            raise RuntimeError(
                f"invalid σ bounds: sigma_min={sigma_min} "
                f"sigma_max={sigma_max}"
            )
        if explore_alpha <= 0.0:
            raise RuntimeError(
                f"invalid explore_alpha: {explore_alpha}"
            )
        arch = payload.get("arch", {})
        obs_dim = int(arch.get("obs_dim", payload.get("obs_dim", 0)))
        action_dim = int(arch.get("action_dim", payload.get("action_dim", 0)))
        hidden_dim = int(arch.get("hidden_dim", payload.get("hidden_dim", 0)))
        num_components = int(
            arch.get("num_components", payload.get("num_components", 0))
        )
        if num_components < 1:
            raise RuntimeError(
                f"Invalid num_components in payload: {num_components}"
            )

        self._policy = _StateMixtureBoundedStdInferenceNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_components=num_components,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            explore_alpha=explore_alpha,
        )
        self._policy.load_state_dict(payload["state_dict"], strict=True)
        self._policy.eval()

    def act(
        self,
        observation: Any,
        *,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, None]:
        """Deterministic action — highest-weight component mean."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self._policy.deterministic_action(obs_tensor)
        return action.squeeze(0).cpu().numpy().astype(np.float32), None

    def sample(
        self,
        observation: Any,
        *,
        ctx: Optional["SamplingContext"] = None,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Stochastic action — mixture sample with explore_factor."""
        obs_array = np.asarray(observation, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self._policy.sample_action(
                obs_tensor, ctx=ctx,
            )
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        if not want_extra or log_prob is None:
            return action_np, None
        return action_np, {"log_prob": float(log_prob.item())}

    def reset(self, seed: Optional[int] = None) -> None:
        """Reseed the inference net's private RNG for reproducible
        rollouts (per-policy stream, independent of other agents)."""
        self._policy.reset(seed)
        return None


# ============================================ attack planner ==
_DEFAULT_PARAMS: Dict[str, Any] = {
    "strike_range": 0.62,    # 根间距 ≤ 此值且对准 → 出拳
    "approach_range": 0.05,  # 逼近目标深度（拳程~0.5，根距需
                           #   ≤~0.4 拳才能压进躯干半径~0.2）
                           #   必须 < strike_range，否则出现
                           #   (strike_range, strike_range+standoff_step)
                           #   既不迈步也不出拳的死锁区）
    "flank_off": 0.55,       # 侧翼航点偏置 m——实测冻结/站立对手
                           #   双臂前置护胸，正面拳 100% 落在前臂
                           #   上不记分；躯干侧面无臂遮挡。
                           #   逼近目标 = 对手位 + perp(其朝向)·off
    "flank_commit": 0.90,    # 根距 > 此值时走侧翼航点；
                           #   ≤ 后转为直接对准躯干（终段 homing）
    "front_thresh": 0.4,     # frontness=-cos(face,opp_rel) < 此值
                           #   才算在正面锥外（侧/背）允许出拳；
                           #   正面被顶住太久则由 flank_patience 放行
    "flank_patience": 60,    # 帧：近身仍正面 → 原地等后骨盆拳放行
    "avoid_ang": 0.7,        # rad：直线穿身时航向外旋量（绕弧切向）
    "flank_enable": False,   # 绕行总开关——实测基座步令做不到
                           #   连续弧线走位（悬停/漂移/摔），v1 关：
                           #   正面逼近 + 骨盆拳；绕行留作 v2
    "strike_yaw_close": 0.30,# 近身放行 yaw 容差（骨盆目标~0.25m 半径，
                           #   0.3rad@0.55m 偏差 ~0.17m 可容）
    "strike_yaw": 0.15,      # rad，|bearing|≤此值才算对准——实测
                             #   θ~0.3 出拳从体侧擦过（躯干半径~0.15）
    "abort_yaw": 0.40,       # STRIKE 中偏航超此 → 中止重对准
    "step_cap": 0.40,        # 单步残差位移上限 m
    "dyaw_cap": 0.45,        # 单步转角残差上限 rad
    "standoff_step": 0.12,   # 残余步距 < 此值不迈步（防原地碎步）
    "max_age": 15,           # 步令重规划间隔兜底帧
    "punch_ext": (0.5, 0.5, -1.0),    # 骨盆拳（z-.21 fwd .55）——计分件
                                    #   含 pelvis/waist（全归 torso），
                                    #   护臂 X 交叉只封胸口以上，
                                    #   下腹 z~0.9m 是空门
    "punch_high": (-0.5, 1.0, -1.0),  # 高位拳（z+.29 fwd .56）只对
                                    #   蹲姿/弯腰对手（头低到拳程内）
    "punch_k": 14,           # 伸展段帧数——sweep 显示 8-12f 才到
                           #   最大伸展，8 会切短有效拳程
    "recover_k": 8,          # 收回段帧数（之后回 APPROACH）
    "high_z": 0.05,          # 对手头 ego-z < 此值（蹲/弯腰）才选
                             #   高位拳——站立对手头在 1.6m 拳程外，
                             #   且 X 护臂封胸口；骨盆拳走空门
    "guard_z": 0.45,         # upright < 此 → GUARD
    "guard_recover": 0.60,   # upright > 此持续 guard_frames → 解除
    "guard_frames": 10,
    "contact_hi": 0.30,      # 落地阈值（体重倍数）
    "contact_lo": 0.15,      # 离地阈值
    "contact_m1_idx": 52,    # 实测 foot<0 ↔ obs[52]
    "contact_p1_idx": 53,    # 实测 foot>0 ↔ obs[53]
    "decay_k": 12.0,         # 残差线性衰减常数（帧）——扫出最优
    "lunge_step": 0.12,      # STRIKE 期突进——0.25 实测弯腰缩拳程；
                           #   0.12 站直小步压最后 0.15m
}


class AttackPlanner:
    """obs96 → cmd22。每帧调用，内部自持状态机。"""

    def __init__(self, params: Optional[Dict[str, Any]] = None,
                 guard_tgt=(0.172, 0.172, 0.333)) -> None:
        self.p = dict(_DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        self.guard_tgt = np.asarray(guard_tgt, float)
        self.reset()

    # ------------------------------------------------------------ 状态 --
    def reset(self) -> None:
        self.state = "APPROACH"
        self._foot = -1              # 下一条步令的动脚（-1 左 / +1 右）
        self._age = 10 ** 9          # 当前步令令龄（强制首帧发令）
        self._armed = False          # 落地门控武装（离地→落地语义）
        self._strike_t = 0
        self._guard_ok = 0
        self._flank_wait = 0
        self._punch = np.asarray(self.p["punch_ext"], float)

    # ------------------------------------------------------------ 主入口 --
    def update(self, obs: np.ndarray) -> np.ndarray:
        p = self.p
        obs = np.asarray(obs, float)
        opp = obs[57:60]                       # 对手根：ego 系
        d = float(np.hypot(opp[0], opp[1]))
        th = float(math.atan2(opp[1], opp[0]))
        head_z = float(obs[68])                # 对手头 ego-z
        up = -float(obs[44])                   # uprightness（-g_z）
        c_m1 = float(obs[p["contact_m1_idx"]])
        c_p1 = float(obs[p["contact_p1_idx"]])

        if self.state != "GUARD" and up < p["guard_z"]:
            self.state, self._guard_ok = "GUARD", 0
        if self.state == "GUARD":
            if up > p["guard_recover"]:
                self._guard_ok += 1
                if self._guard_ok >= p["guard_frames"]:
                    self.state = "APPROACH"
                    self._age = 10 ** 9        # 立即重发步令
            else:
                self._guard_ok = 0
            return self._cmd(step=None, arm=None)   # 全零：基座自救

        self._age += 1
        if self.state == "STRIKE":
            self._strike_t += 1
            if (self._strike_t >= p["punch_k"] + p["recover_k"]
                    or abs(th) > p["abort_yaw"]):
                self.state, self._age = "APPROACH", 10 ** 9
                return self._cmd(step=None, arm=None)
            tgt = (self._punch if self._strike_t <= p["punch_k"]
                   else self.guard_tgt)
            # 突进拳：伸展段同时给小步令压上最后 0.2m
            # （实测站立拳 ~0.6m 根距仍差 0.25m 接触）
            lunge = np.array([self._foot,
                              p["lunge_step"] * math.cos(th),
                              p["lunge_step"] * math.sin(th),
                              0.3 * th, 0, 0])
            return self._cmd(step=(lunge, np.zeros(6)),
                             arm=(tgt, None))

        # APPROACH
        # frontness：我在对手正面锥内的程度（+1=正对其面朝方向）。
        # 实测：冻结/站立对手双臂前置护胸，正面拳 100% 落在
        # 前臂上不记分 → 须绕到正面锥外打躯干侧面。
        face = obs[63:65]
        fn = float(np.hypot(face[0], face[1]))
        frontness = (-float(face[0] * opp[0] + face[1] * opp[1])
                     / (fn * d + 1e-9)) if fn > 1e-6 and d > 1e-6 else 1.0
        in_front = frontness >= p["front_thresh"]

        aligned = abs(th) <= p["strike_yaw"]
        aligned_close = abs(th) <= p["strike_yaw_close"]
        if d <= p["strike_range"]:
            if aligned or (aligned_close
                           and self._flank_wait >= p["flank_patience"]):
                self.state = "STRIKE"          # 对准即打（骨盆拳钻臂下）
                self._strike_t = 0
                self._punch = (np.asarray(p["punch_high"], float)
                               if head_z < p["high_z"]
                               else np.asarray(p["punch_ext"], float))
                return self._cmd(step=None,
                                 arm=(self._punch, None))
            if aligned_close:
                self._flank_wait += 1          # 贴近但未完全对准：等
                return self._cmd(step=None, arm=None)
            self._flank_wait = 0

        # 航点：正面锥内且绕行开 → 绕后点（experimental，v1 默认关）；
        # 否则 → 躯干中心。Δψ 恒朝中心。
        if p["flank_enable"] and in_front and fn > 1e-6:
            th_f = math.atan2(face[1], face[0])
            r_orb = p["approach_range"] + 0.15
            wp_l = opp[:2] + r_orb * np.array(
                [math.cos(th_f + 2.1), math.sin(th_f + 2.1)])
            wp_r = opp[:2] + r_orb * np.array(
                [math.cos(th_f - 2.1), math.sin(th_f - 2.1)])
            wp = (wp_l if abs(math.atan2(wp_l[1], wp_l[0]))
                  <= abs(math.atan2(wp_r[1], wp_r[0])) else wp_r)
            aim_d = float(np.hypot(wp[0], wp[1]))
            aim_th = float(math.atan2(wp[1], wp[0]))
            # 切向避障：直线到航点会穿过对手（|侧向距|<r_clear 且
            # 在前方）→ 把航向旋到障碍圆切线，形成自然绕弧
            un = wp / (aim_d + 1e-9)
            cross = float(un[0] * opp[1] - un[1] * opp[0])
            if un[0] * opp[0] + un[1] * opp[1] > 0 \
                    and abs(cross) < 0.45:
                aim_th += -math.copysign(p["avoid_ang"], cross)
        else:
            aim_d, aim_th = d, th

        # 重规划门控：首令 / 动脚"离地→落地" / 超时
        c_mv = c_m1 if self._foot < 0 else c_p1
        if c_mv < p["contact_lo"]:
            self._armed = True
        landed = self._armed and c_mv >= p["contact_hi"]
        if self._age > p["max_age"] or (self._age > 2 and landed):
            self._foot = -self._foot
            self._age = 0
            self._armed = c_mv < p["contact_lo"]

        step = self._step_res(aim_d, aim_th, th, self._foot, self._age)
        nxt = self._step_res(aim_d, aim_th, th, -self._foot, 0)
        return self._cmd(step=(step, nxt), arm=None)

    # ------------------------------------------------------------ 构造 ----
    def _step_res(self, d: float, th_move: float, th_face: float,
                  foot: int, age: int) -> np.ndarray:
        """逼近步残差：min(d−range,cap)·decay(age) 沿 th_move 方向。

        幅度对目标距离锚定，剖面按令龄线性衰减，位移方向对
        航点 bearing 跟踪；Δψ 对 th_face（躯干中心）——绕侧翼时
        位移朝航点、身体转向对手，形成 circle-strafe。
        """
        p = self.p
        k = max(0.0, 1.0 - age / p["decay_k"])
        L = max(0.0, min(d - p["approach_range"],
                         p["step_cap"])) * k
        dyaw = float(np.clip(th_face, -p["dyaw_cap"], p["dyaw_cap"])) * k
        if L < p["standoff_step"] * k:
            return np.array([foot, 0.0, 0.0, dyaw, 0, 0])
        return np.array([foot,
                         L * math.cos(th_move), L * math.sin(th_move),
                         dyaw, 0, 0])

    def _cmd(self, step: Optional[Tuple[np.ndarray, np.ndarray]],
             arm: Optional[Tuple[np.ndarray, Optional[np.ndarray]]]
             ) -> np.ndarray:
        c = np.zeros(22)
        if arm is not None:
            tr, tl = arm
            if tr is not None:
                c[0] = 1.0
                c[1:4] = tr
            if tl is not None:
                c[4] = 1.0
                c[5:8] = tl
        if step is not None:
            c[10:16] = step[0]
            c[16:22] = step[1]
        return c



# ====================================================== attack wrapper ==


class ExportedAttackPolicy(Policy):
    """obs96 → AttackPlanner → cmd22 → obs79 → 基座 → action21。

    部署侧组合策略：高层脚本状态机（内联）+ 已训练 steparm 基座。
    obs96 = 原始 96 维观测；cmd22 段格式见 AttackPlanner 注释。
    """

    def __init__(self, model_path=None):
        self._base = ExportedStateMixtureBoundedStdTruncNormPolicy(
            model_path)
        self._planner = AttackPlanner()
        self._last_cmd = np.zeros(22)

    def reset(self, seed=None):
        self._planner.reset()
        if hasattr(self._base, "reset"):
            self._base.reset(seed)

    def act(self, observation, *, want_extra=False):
        obs = np.asarray(observation, dtype=np.float32)
        cmd = self._planner.update(obs)
        self._last_cmd = cmd
        act, extra = self._base.act(
            np.concatenate([obs[:57], cmd]).astype(np.float32),
            want_extra=want_extra)
        if want_extra:
            return act, {"cmd22": cmd,
                         "planner": self._planner.state,
                         "base_extra": extra}
        return act, None
