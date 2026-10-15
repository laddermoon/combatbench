"""humanoid21 动作库标准查询接口（M5-W4）。

对 ``/data1/dev/combatbench_mocap`` 的 index.json + clips/ 提供只读访问：

    from mocap.humanoid21.library import MotionLibrary
    lib = MotionLibrary()
    lib.list_clips(labels=["kick"])          # 按标签检索
    lib.list_clips(kind="original")          # 原始片段（排除镜像/切分/_src）
    npz = lib.load("cmu_135_07")             # 加载 npz（lazy dict-like）
    meta = lib.info("cmu_135_07_m")          # index 条目

约定（见 DATASPEC.md）：
- 片段 id：cmu_<subj>_<trial>[_segNN][_m]；_src 为源骨架参考数据。
- quality.level: L0 运动学导入 / L1 通过运动学检查 / L2 物理跟踪验证。
- 派生片段的 identity.derivation ∈ {segment, mirror_sagittal_x}。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

LIBRARY = Path("/data1/dev/combatbench_mocap")


class MotionLibrary:
    def __init__(self, root: Path = LIBRARY):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text())
        self.clips_dir = self.root / "clips"

    # ---------------------------------------------------------------
    def _kind(self, cid: str) -> str:
        if cid.endswith("_src"):
            return "source"
        if "_seg" in cid:
            return "segment"
        if cid.endswith("_m"):
            return "mirror"
        return "original"

    def list_clips(self, labels: Optional[List[str]] = None,
                   kind: Optional[str] = None,
                   level: Optional[str] = None) -> List[str]:
        """按标签/类型/验收等级过滤，返回排序后的 clip_id 列表。"""
        out = []
        for cid, e in self.index["clips"].items():
            k = self._kind(cid)
            if kind == "robot":
                if k == "source":
                    continue
            elif kind is not None and k != kind:
                continue
            if labels is not None:
                have = set(e.get("classification", {}).get("labels", []))
                if not set(labels) <= have:
                    continue
            if level is not None and \
                    e.get("quality", {}).get("level") != level:
                continue
            if e.get("quality", {}).get("rejected"):
                continue
            out.append(cid)
        return sorted(out)

    def load(self, clip_id: str) -> Dict[str, np.ndarray]:
        """加载 clip npz 为 dict（np.load lazy → 此处读全量）。"""
        p = self.clips_dir / f"{clip_id}.npz"
        if not p.exists():
            raise KeyError(f"{clip_id} not in {self.clips_dir}")
        z = np.load(p)
        return {k: z[k] for k in z.files}

    def info(self, clip_id: str) -> dict:
        return self.index["clips"][clip_id]

    def summary(self) -> dict:
        kinds: Dict[str, int] = {}
        labels: Dict[str, int] = {}
        for cid, e in self.index["clips"].items():
            kinds[self._kind(cid)] = kinds.get(self._kind(cid), 0) + 1
            for lb in e.get("classification", {}).get("labels", []):
                labels[lb] = labels.get(lb, 0) + 1
        return {"total": len(self.index["clips"]), "by_kind": kinds,
                "by_label": dict(sorted(labels.items()))}


if __name__ == "__main__":
    import sys
    lib = MotionLibrary()
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        s = lib.summary()
        print(json.dumps(s, indent=1, ensure_ascii=False))
    elif len(sys.argv) > 1:
        print("\n".join(lib.list_clips(labels=sys.argv[1].split(","))))
    else:
        print("\n".join(lib.list_clips()))
