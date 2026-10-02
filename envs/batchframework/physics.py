"""PhysicsBackend — 设备后端物理契约（E1-W1，对应 discuss.md D4）。

设计要点（含 W0 探针实测结论）：

- **advance 是全行推进**：warp 1.12.1 无 per-world masked-step 原语；
  ENDED 行冻结由 runtime 用 ``capture``/``restore`` 组合实现
  （每 action step 边界一次 write-back，窗内漂移有界不累积）。
- **视图是借用，不是快照**：``views()`` 返回后端活跃存储的零拷贝
  视图；任何 ``advance``/``initialize``/``apply_patch``/``restore``
  都可能使其失效。跨操作保留须自行 clone。
- **派生字段时效**：``advance`` 后 post_integrate 视图（xpos/cvel/
  contact）有效（mjw.step 内部已刷新，W0-P3 实测）；``apply_patch``
  的刷新由 ``refresh`` 参数显式控制。
- **快照是近似恢复**（W0-P2 实测）：capture/restore 后重推进与不间断
  轨迹漂移 ~1e-7/10 步——warp 本身同输入运行间就不逐位确定，契约
  不承诺逐位续跑。
- 本模块不 import warp/mjx/humanoid21/baseline——核心契约层零后端依赖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, MutableMapping, Optional, Protocol

import torch


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------
class BackendError(Exception):
    """后端执行失败（CUDA 断言、非法调用时序等）。"""


class CapacityError(BackendError):
    """容量溢出（contact/efc/record buffer）。必须是显式失败。"""


class ContractError(BackendError):
    """违反调用契约（shape/dtype 不符、未初始化访问、视图过期使用）。"""


# ---------------------------------------------------------------------------
# 描述符
# ---------------------------------------------------------------------------
class SamplePhase(Enum):
    """视图字段的求值时刻。"""
    INTEGRATION = "integration"    # 积分状态（qpos/qvel/ctrl/warmstart）
    POST_INTEGRATE = "post_integrate"  # advance 后有效（xpos/cvel/contact）
    INPUT = "input"               # 消费型输入（xfrc_pending/sched/act_target）
    META = "meta"                 # 静态描述（查找表等，非张量亦允许）


class RefreshPolicy(Enum):
    """apply_patch 后派生字段的刷新策略。"""
    FORWARD = "forward"   # 立即重建运动学/接触（等价 mj_forward）
    NONE = "none"         # 不刷新——derived 视图保持陈旧直到下次 advance/forward


class SnapshotLevel(Enum):
    SEMANTIC = "semantic"        # 可移植最小集（qpos/qvel），跨后端对照用
    INTEGRATION = "integration"  # 后端完整推进状态（近似恢复，非逐位）


@dataclass(frozen=True)
class FieldSpec:
    """单个物理视图字段的契约描述。"""
    name: str
    phase: SamplePhase
    writable: bool = False      # 允许 apply_patch 直写
    doc: str = ""


@dataclass(frozen=True)
class CapacitySpec:
    """容量声明。``per_world=True`` 表示值是每 world 的语义。"""
    name: str
    value: int
    per_world: bool = True


@dataclass(frozen=True)
class BackendDescriptor:
    """后端静态描述——装配期校验用。"""
    backend: str                 # "warp" / "fake" / ...
    backend_version: str
    device: str                  # torch device 字符串
    batch_size: int
    nq: int
    nv: int
    nbody: int
    nu: int
    fields: Mapping[str, FieldSpec] = field(default_factory=dict)
    capacities: Mapping[str, CapacitySpec] = field(default_factory=dict)
    # 能力 flags：{"fused_control", "park_rows", "snapshot:integration", ...}
    capabilities: frozenset = frozenset()


# ---------------------------------------------------------------------------
# 控制程序（每子步由 backend 调用，binding 提供实现）
# ---------------------------------------------------------------------------
class ControlProgram(Protocol):
    """action→ctrl 的状态反馈控制（如 PD）。

    backend 在每个物理子步前调用 ``apply(views)``——实现用 torch 就地写
    views["ctrl"]（warp 后端亦可 launch wp.kernel；实现细节归实现）。
    """

    def apply(self, views: MutableMapping[str, torch.Tensor]) -> None:
        """根据当前物理视图计算 ctrl（就地写 views 中的控制字段）。"""
        ...


# ---------------------------------------------------------------------------
# 后端协议
# ---------------------------------------------------------------------------
class PhysicsBackend(Protocol):
    """批量物理后端契约。

    生命周期：构造 → ``views()``（initialize 前可拿，内容未定义）
    → ``initialize`` → ``advance``/``apply_patch``/``capture`` 循环
    → ``close``。

    所有张量参数为设备端 ``torch.Tensor``，第一维是 batch；``mask``
    为 ``(B,) bool``——None 表示全行。
    """

    # --- 描述与视图 ---
    def describe(self) -> BackendDescriptor:
        ...

    def views(self) -> MutableMapping[str, torch.Tensor]:
        """活跃物理视图的借用字典。键由 BackendDescriptor.fields 声明。"""
        ...

    # --- 生命周期 ---
    def initialize(self, qpos: torch.Tensor, *,
                   qvel: Optional[torch.Tensor] = None,
                   mask: Optional[torch.Tensor] = None) -> None:
        """对 mask 行写入全新积分状态并刷新派生字段。

        语义是"全新求解"：选中行的 warmstart/ctrl/外力清零后 forward。
        ``qpos`` 为 ``(M, nq)`` 或全量 ``(B, nq)``（mask=None 时）。
        """
        ...

    def apply_patch(self, mask: torch.Tensor,
                    fields: Mapping[str, torch.Tensor], *,
                    refresh: RefreshPolicy = RefreshPolicy.FORWARD) -> None:
        """对 mask 行写入具名字段（如 set_integration_state 的行级版本）。

        fields 的键必须在 descriptor 中声明为 writable。
        """
        ...

    def advance(self, n_substeps: int,
                control: Optional[ControlProgram] = None, *,
                pre_step=None, post_step=None) -> None:
        """全行推进 n_substeps 个物理子步。

        ``control`` 非 None 时每个子步前调用 ``control.apply(views)``。
        ``pre_step(i)``/``post_step(i)``（可选）在每个物理子步的积分
        前/后回调——用于 runtime 的逐子步插件 hook（必须是设备侧
        torch 操作，禁止 host 同步）。消费型输入按契约处理：pending
        wrench 仅首个子步、schedule 逐子步消费后清除。
        """
        ...

    # --- 快照 ---
    def capture(self, mask: torch.Tensor,
                level: SnapshotLevel = SnapshotLevel.INTEGRATION
                ) -> Dict[str, torch.Tensor]:
        """对 mask 行捕获快照，返回独立拥有的张量拷贝（非借用视图）。"""
        ...

    def restore(self, mask: torch.Tensor,
                snapshot: Mapping[str, torch.Tensor]) -> None:
        """把 capture 产物写回 mask 行（含必要的派生刷新）。"""
        ...

    # --- 状态与收尾 ---
    def status(self) -> Dict[str, Any]:
        """容量/健康状态（nacon/nefc 用量、错误标志）。低频调用。"""
        ...

    def close(self) -> None:
        """释放后端资源。可重复调用。"""
        ...


# ---------------------------------------------------------------------------
# 便捷检查
# ---------------------------------------------------------------------------
def check_descriptor(d: BackendDescriptor) -> None:
    """装配期自检：必需字段与形状一致性。"""
    if d.batch_size <= 0:
        raise ContractError("batch_size must be > 0")
    for req in ("qpos", "qvel"):
        if req not in d.fields:
            raise ContractError(f"descriptor missing required field {req!r}")
