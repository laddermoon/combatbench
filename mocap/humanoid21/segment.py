"""M5-W2 片段切分工具：把长 clip 切成带标签的动作段。

三种模式：
  --auto          自动按"低活动度静默期"切分：全部关节 |dq| < thresh
                  且根速度 < thresh 持续 >= idle_min 帧的区间视为间隔，
                  两段静默之间的活跃区为一个 segment。
  --detect KIND   事件探测切分：kick=脚速峰值, punch=手速峰值,
                  jump=双脚离地, block=手高过头, ground=骨盆贴地。
  --segments S-E:label,S-E:label   手工指定帧区间与标签。

输出：``<clip>_seg<NN>.npz``（关节/根/关键点/接触数组切片，时间戳
归零，lead_stable_frame 重算）+ index.json 溯源条目
{derivation: segment, parent_clip, frame_range}。

用途：长片段（如 40s 拳击含多个爆发-间歇循环）切成聚焦动作段，
供任务采样与条件先验用。

用法：
  PYTHONPATH=. python3 mocap/humanoid21/segment.py --clip cmu_13_17 --auto
  PYTHONPATH=. python3 mocap/humanoid21/segment.py --clip cmu_13_17 \
      --segments 600-1000:combo,2400-3000:punch
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np

LIBRARY = Path("/data1/dev/combatbench_mocap")
CLIPS = LIBRARY / "clips"
INDEX = LIBRARY / "index.json"

# 逐帧数组字段切片清单（npz 中第一维=T 的字段都切）
_SLICE_KEYS = ("timestamps", "root_pos", "root_quat", "root_linvel",
               "root_angvel", "joint_angles", "joint_velocities",
               "keypoints", "foot_contacts", "retarget_residuals")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auto_segments(npz, idle_vel: float = 0.15, idle_linvel: float = 0.25,
                  idle_min: int = 48, min_seg: int = 120,
                  merge_gap: int = 48) -> List[Tuple[int, int]]:
    """按低活动度静默期自动切段。返回 [start,end) 帧区间。

    idle_vel: 关节角速度阈值 (rad/s)；idle_linvel: 根线速度阈值 (m/s)；
    idle_min: 持续多少帧(120Hz)的静默才算分隔（48=0.4s）；
    min_seg: 最短片段长度（120=1s）；merge_gap: 静默小于该值直接并入。
    """
    qv = np.asarray(npz["joint_velocities"])
    lv = np.asarray(npz["root_linvel"])
    activity = np.maximum(np.abs(qv).max(axis=1),
                          np.linalg.norm(lv, axis=1) / max(idle_linvel, 1e-6)
                          * idle_vel)
    active = activity >= idle_vel

    # 找活跃段：active 连续区间，合并相隔 < merge_gap 的段
    segs: List[List[int]] = []
    t = 0
    T = len(active)
    while t < T:
        if active[t]:
            s = t
            while t < T and (active[t] or
                             (t - s < T and not active[t] and
                              t + merge_gap < T and
                              np.any(active[t:t + merge_gap]))):
                t += 1
            segs.append([s, t])
        else:
            t += 1

    # 只有间隔 >= idle_min 的静默才做切分点
    out = []
    for s, e in segs:
        if e - s >= min_seg:
            out.append((s, e))
    return out


def _kp(npz, name: str) -> np.ndarray:
    i = list(npz["keypoint_names"]).index(name)
    return np.asarray(npz["keypoints"])[:, i]


def event_signal(npz, kind: str) -> np.ndarray:
    """逐帧事件信号标量。kind: kick|punch|jump|block|ground。"""
    dt = 1.0 / 120.0
    if kind == "kick":
        fl, fr = _kp(npz, "foot_left"), _kp(npz, "foot_right")
        v = np.maximum(np.linalg.norm(np.gradient(fl, dt, axis=0), axis=1),
                       np.linalg.norm(np.gradient(fr, dt, axis=0), axis=1))
        return v
    if kind == "punch":
        hl, hr = _kp(npz, "hand_left"), _kp(npz, "hand_right")
        v = np.maximum(np.linalg.norm(np.gradient(hl, dt, axis=0), axis=1),
                       np.linalg.norm(np.gradient(hr, dt, axis=0), axis=1))
        return v
    if kind == "jump":
        fl, fr = _kp(npz, "foot_left"), _kp(npz, "foot_right")
        z = np.minimum(fl[:, 2], fr[:, 2])
        return z - np.percentile(z, 5)          # 双脚离地高度
    if kind == "block":
        hl, hr = _kp(npz, "hand_left"), _kp(npz, "hand_right")
        head = _kp(npz, "head")
        return np.maximum(hl[:, 2], hr[:, 2]) - head[:, 2] + 0.15
    if kind == "ground":
        return -_kp(npz, "pelvis")[:, 2]        # 骨盆越低信号越大
    raise ValueError(kind)


_EVENT_THRESH = {            # (触发阈值, 回落阈值) — 120Hz 信号单位
    "kick": (3.0, 1.0), "punch": (2.5, 1.0), "jump": (0.10, 0.05),
    "block": (0.10, 0.0), "ground": (-0.55, -0.75),
}


def detect_events(npz, kind: str, pad: int = 12, merge_gap: int = 36,
                  min_len: int = 24) -> List[Tuple[int, int]]:
    """信号超阈即事件核心，向两侧扩展到回落阈值以下，合并近邻。"""
    hi, lo = _EVENT_THRESH[kind]
    sig = event_signal(npz, kind)
    T = len(sig)
    out, t = [], 0
    while t < T:
        if sig[t] >= hi:
            s = t
            while s > 0 and sig[s - 1] > lo:      # 左扩到回落点
                s -= 1
            e = t
            while e < T and sig[e] > lo:           # 右扩到回落点
                e += 1
            while e < T and np.any(sig[e:e + merge_gap] >= hi):  # 合并邻近事件
                e2 = e + int(np.argmax(sig[e:e + merge_gap] >= hi))
                e = e2
                while e < T and sig[e] > lo:
                    e += 1
            s, e = max(0, s - pad), min(T, e + pad)
            if e - s >= min_len:
                out.append((s, e))
            t = e
        else:
            t += 1
    return out


def slice_npz(npz, s: int, e: int) -> dict:
    out = {}
    for k in npz.files:
        arr = npz[k]
        if k in _SLICE_KEYS and arr.shape[0] == npz["timestamps"].shape[0]:
            arr = arr[s:e].copy()
            if k == "timestamps":
                arr = arr - arr[0]
        out[k] = arr
    # 重算 lead_stable_frame（段内相对帧号）
    q = np.asarray(out["joint_angles"])
    if len(q) > 1:
        dq = np.abs(np.diff(q, axis=0)).max(axis=1)
        lead = 0
        for t in range(len(dq)):
            if np.all(dq[t:t + 6] < 0.08):
                lead = t
                break
        out["lead_stable_frame"] = np.int32(lead)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--detect", choices=list(_EVENT_THRESH), default=None,
                    help="事件探测切段：kick=脚速/punch=手速/jump=双脚离地/"
                         "block=举手/ground=骨盆贴地")
    ap.add_argument("--segments", type=str, default=None,
                    help="S-E:label,S-E:label (120Hz 帧号)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    npz = np.load(CLIPS / f"{args.clip}.npz")
    T = len(npz["timestamps"])

    if args.auto:
        ranges = auto_segments(npz)
        labels = [None] * len(ranges)
        print(f"auto: {len(ranges)} segments")
    elif args.detect:
        ranges = detect_events(npz, args.detect)
        labels = [args.detect] * len(ranges)
        print(f"detect({args.detect}): {len(ranges)} events")
        for i, (s, e) in enumerate(ranges):
            print(f"  seg{i:02d}: [{s},{e}) = {e-s}f = {(e-s)/120:.2f}s")
    else:
        ranges, labels = [], []
        for spec in args.segments.split(","):
            se, _, lab = spec.partition(":")
            s, e = se.split("-")
            ranges.append((int(s), int(e)))
            labels.append(lab or None)

    if args.dry_run:
        return

    index = json.loads(INDEX.read_text())
    parent = index["clips"][args.clip]
    n_written = 0
    for i, ((s, e), lab) in enumerate(zip(ranges, labels), 1):
        if s < 0 or e > T or e - s < 24:
            print(f"  skip seg{i:02d} [{s},{e}) out-of-range/too short")
            continue
        seg_id = f"{args.clip}_seg{i:02d}"
        sub = slice_npz(npz, s, e)
        np.savez_compressed(CLIPS / f"{seg_id}.npz", **sub)
        seg_labels = list(parent["classification"]["labels"])
        if lab:
            seg_labels = [lab] + [l for l in seg_labels if l != lab]
        index["clips"][seg_id] = {
            "identity": {"source": "derived", "parent_clip": args.clip,
                         "derivation": "segment",
                         "frame_range": [s, e],
                         "npz_sha256": _sha256(CLIPS / f"{seg_id}.npz"),
                         "version": 1},
            "time": {"src_fps": 120.0, "fps": 120.0,
                     "frames": e - s, "duration_s": round((e - s) / 120, 2)},
            "frames_convention": "mujoco_world_z_up",
            "joint_space": "humanoid21_v2",
            "classification": {"labels": seg_labels,
                               "label_source": "segment_auto"
                               if args.auto else
                               ("event_detect:" + args.detect)
                               if args.detect else "manual",
                               "confidence": "medium"},
            "quality": {"level": "L0", "physics": None,
                        "issues": [], "rejected": False,
                        "reject_reason": None},
            "adaptation": parent["adaptation"],
        }
        n_written += 1
        print(f"  wrote {seg_id} [{s},{e}) label={lab}")
    INDEX.write_text(json.dumps(index, indent=1, ensure_ascii=False))
    print(f"{n_written} segments indexed")


if __name__ == "__main__":
    main()
