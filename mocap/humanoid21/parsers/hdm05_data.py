"""HDM05 预处理 .data 格式解析器（fabiocarrara/mocap release 版）。

每条记录 = 一个带标签的片段：
    #objectKey messif.objects.keys.AbstractObjectKey {take}_{cls}_{start}_{n}
    {n};mcdr.objects.ObjectMocapPose
    x1,y1,z1; x2,y2,z2; ... x31,y31,z31        <- 每行一帧，31 个关节坐标
    ...

关键点：
  - pt0 恒为 (0,0,0) = root/pelvis 锚点（normPOS：根相对 + 偏航归一化，
    原地化数据——世界位移与朝向已被剥离）
  - pt1..30 按 ASF :bonedata 顺序（见 JOINT_NAMES）
  - 坐标系：y 上、+x 左、z 矢状；单位未定（经标定，骨盆到头约 30 单位 ≈ 1.05m）
  - 120 Hz（HDM05 标称）

类标签：key 第二段是类别 id，映射见 hdm05_action_descriptions.txt
（id;name）。130 细类可由 HDM05-category_mapper-130vs65.csv 合并到 65 类。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np

# pt0 = root，pt1..30 按 ASF :bonedata 顺序
JOINT_NAMES = [
    "root",
    "lhipjoint", "lfemur", "ltibia", "lfoot", "ltoes",
    "rhipjoint", "rfemur", "rtibia", "rfoot", "rtoes",
    "lowerback", "upperback", "thorax", "lowerneck", "upperneck", "head",
    "lclavicle", "lhumerus", "lradius", "lwrist", "lhand", "lfingers",
    "lthumb",
    "rclavicle", "rhumerus", "rradius", "rwrist", "rhand", "rfingers",
    "rthumb",
]

_REC = re.compile(
    r"#objectKey \S+ (\d+)_(\d+)_(\d+)_(\d+)\n"
    r"\d+;mcdr\.objects\.ObjectMocapPose\n")


@dataclass
class HdmSegment:
    key: str
    take: int
    class_id: int
    start_frame: int
    n_frames: int
    points: np.ndarray          # (T, 31, 3)
    label: str = ""

    def joints(self) -> Dict[str, np.ndarray]:
        """bone_name -> (T,3) 位置。"""
        return {n: self.points[:, i] for i, n in enumerate(JOINT_NAMES)}


def load_action_names(path: str | Path) -> Dict[int, str]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if ";" in line:
            i, name = line.split(";", 1)
            out[int(i.strip())] = name.strip()
    return out


def iter_segments(path: str | Path,
                  action_names: Optional[Dict[int, str]] = None,
                  max_records: Optional[int] = None) -> Iterator[HdmSegment]:
    """流式解析 .data 文件，逐条 yield 带标签片段（不占全内存）。"""
    path = Path(path)
    count = 0
    with open(path) as f:
        key = take = cls = start = n = None
        frames = []
        for line in f:
            if line.startswith("#objectKey"):
                if key is not None:
                    yield HdmSegment(
                        key=key, take=take, class_id=cls,
                        start_frame=start, n_frames=len(frames),
                        points=np.asarray(frames, dtype=np.float64).reshape(-1, 31, 3),
                        label=(action_names or {}).get(cls, ""))
                    count += 1
                    if max_records and count >= max_records:
                        return
                m = _REC.match(line + f.readline())
                if m is None:
                    raise ValueError(f"bad record header near {line!r}")
                take, cls, start, n = map(int, m.groups())
                key = f"{take}_{cls}_{start}_{n}"
                frames = []
            elif line.strip():
                frames.append([float(x) for p in line.split(";")
                               for x in p.split(",")])
    if key is not None and frames:
        yield HdmSegment(
            key=key, take=take, class_id=cls, start_frame=start,
            n_frames=len(frames), points=np.asarray(frames, dtype=np.float64).reshape(-1, 31, 3),
            label=(action_names or {}).get(cls, ""))


def list_classes(path: str | Path) -> Dict[int, int]:
    """统计文件中各类别片段数。"""
    from collections import Counter
    c = Counter()
    for seg in iter_segments(path):
        c[seg.class_id] += 1
    return dict(c)
