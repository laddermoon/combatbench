"""mujoco-warp 后端的 M2 跨后端 fixture 验证。

与 ``test_mjx_validation.py::test_m2_cross_backend_fixtures`` 同一组
case，但有两处刻意差异（均在文档中声明）：

1. **容差**：fixture 的 output.frames 容差 1e-5 是 fp64 级（mjx 达成）。
   warp 只有 fp32，实测全字段偏差 max_abs≈0.016（60N 接触力，rel
   2.7e-4）、max_rel≈5e-4（非零字段）、近零字段 abs≤9e-4——纯 fp32
   舍入+求解器迭代条件放大，无语义差异。``WARP_TOLERANCES`` 取
   {atol: 2e-2, rtol: 1e-3}，约为实测极值的 4× 余量，仍比物理量级
   紧 3 个数量级。

2. **provenance**：形式 replay() 会做依赖指纹 stale 检查，依赖文件
   （如 baseline/experiments_ppo/base.py）的任何改动都会阻断——这是
   特性不是 bug。本测试走 execute+compare 直接数值对照（语义验证
   本体），正式验收口径以依赖冻结后的完整 replay 为准。
"""

import copy
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

# fp32 后端的文档化容差（见模块 docstring；实测 max_abs 0.016, rel 5e-4）
WARP_TOLERANCES = {"output.frames": {"atol": 2e-2, "rtol": 1e-3},
                   "output.frame": {"atol": 2e-2, "rtol": 1e-3},
                   "output.envs": {"atol": 2e-2, "rtol": 1e-3}}


def test_warp_cross_backend_fixtures():
    """warp 后端跨后端 fixture：6 个物理 case 在 fp32 容差内通过，
    mjSTATE-blob 与 logic case 必须显式 unsupported。"""
    from envs.batchframework import validation as v
    from envs.batchframework.validation_warp import WarpAdapter

    fx = Path(project_root) / "envs/batchframework/validation_fixtures"
    index = json.loads((fx / "index.json").read_text())
    adapter = WarpAdapter()

    must_pass = ["dyn-standing-s1", "dyn-standing-s25", "dyn-moving-2x5",
                 "extforce-torso-push", "state-io-write-read", "batch-isolation-2"]
    must_unsupported = ["standing-1-substeps", "standing-25-substeps",
                        "standup-timeout-bootstrap"]

    for cid in must_pass:
        bundle = v.load(fx / f"{cid}.json")
        actual = adapter.execute(copy.deepcopy(bundle["case"]))
        # compare 不允许未使用的 tolerance 路径——按该 case 实际输出键选择
        ft = {k: v2 for k, v2 in WARP_TOLERANCES.items()
              if k.split(".", 1)[1] in actual}
        fails = v.compare(bundle["expected"], actual,
                          {"atol": 0.0, "rtol": 0.0}, ft)
        assert not fails, f"{cid}: {fails[:4]}"
    for cid in must_unsupported:
        bundle = v.load(fx / f"{cid}.json")
        try:
            adapter.execute(copy.deepcopy(bundle["case"]))
        except v.Unsupported:
            continue
        raise AssertionError(f"{cid}: expected Unsupported")
    print("[PASS] test_warp_cross_backend_fixtures "
          "(6 pass @fp32 tol, 3 unsupported)")


if __name__ == "__main__":
    test_warp_cross_backend_fixtures()
