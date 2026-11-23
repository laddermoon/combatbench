"""BUILD_ATTACK_EXPORT：组装自包含可部署攻击 policy 包。

输出目录（默认 attack_policy_v1/）：
    policy.py    = 基座导出 policy.py 全文（torch 推理，自包含）
                 + 内联 AttackPlanner（纯 numpy/math）
                 + ExportedAttackPolicy 组合类
    model.pt     = 从基座导出目录复制
    MANIFEST.json

Loader 约定（与 step_accept._load_policy / PolicyBlueprint 一致）：
扫模块内首个 Exported* 类并实例化——'ExportedAttackPolicy' 字典序
先于 'ExportedStateMixture...'，自动成为入口。

用法：
    python3 build_attack_export.py \
        --base baseline/runs/v2_steparm_r1/policy_exports/u04735 \
        --out  baseline/humanoid21/v1/attack_policy_v1
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

_V1 = Path(__file__).resolve().parent


def _planner_source() -> str:
    """取 attack_planner.py 中纯算法部分（参数表 + AttackPlanner 类），
    剥掉 import/Policy-stub/wrapper/对立方 helper——打包文件里
    Policy 类已由基座导出代码定义。"""
    src = (_V1 / "attack_planner.py").read_text(encoding="utf-8")
    start = src.index("_DEFAULT_PARAMS")
    end = src.index("# ============================================================ wrappers ==")
    body = src[start:end]
    # 移除 AttackPlanner 头部对 Policy 的 try-import 已不在截取范围内
    return body


_WRAPPER = '''
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
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="基座导出目录（含 policy.py + model.pt）")
    ap.add_argument("--out", default=str(_V1 / "attack_policy_v1"))
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_src = (base / "policy.py").read_text(encoding="utf-8")
    planner_src = _planner_source()
    pkg = (base_src
           + "\n\n# ============================================ attack planner ==\n"
           + planner_src + _WRAPPER)
    (out / "policy.py").write_text(pkg, encoding="utf-8")
    shutil.copy2(base / "model.pt", out / "model.pt")

    manifest = {
        "format_version": 1,
        "policy_class": "ExportedAttackPolicy",
        "kind": "scripted_planner_over_trained_base",
        "base_export": str(base),
        "obs_in": 96, "obs_to_base": 79, "cmd_dim": 22,
        "action_dim": 21,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": ["model.pt", "policy.py", "MANIFEST.json"],
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False))

    # 自检：包内不得有 baseline./envs. import
    bad = re.findall(r"(?:from|import)\s+(baseline|envs)\.", pkg)
    if bad:
        raise RuntimeError(f"打包文件含 repo import: {set(bad)}")
    print(f"[built] {out}  ({len(pkg)} bytes policy.py)")


if __name__ == "__main__":
    main()
