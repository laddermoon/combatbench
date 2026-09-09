"""``frame`` — frame-level inspector.

S3: Inspect a single frame or filter frames by condition.

Two modes:
- ``--id ep0003:robot_a:137``: show one frame's full data.
- ``--where "combine.aw_normed.r_left_foot < 0"``: filter and list.

Data comes from snapshot replay .npz files (S2) + episodes.

See ``DEBUG_GUIDE.md`` §3.4 ``frame``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Frame ID parsing
# ---------------------------------------------------------------------------

def parse_frame_id(frame_id: str) -> Tuple[int, str, int]:
    """Parse ``ep0003:robot_a:137`` → (3, "robot_a", 137).

    Raises ValueError on malformed IDs.
    """
    parts = frame_id.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"frame ID must be ep{{N}}:{{agent}}:{{t}}, got {frame_id!r}"
        )
    ep_str, agent, t_str = parts
    if not ep_str.startswith("ep"):
        raise ValueError(f"frame ID episode part must start with 'ep': {ep_str!r}")
    try:
        ep_idx = int(ep_str[2:])
    except ValueError:
        raise ValueError(f"frame ID episode number not parseable: {ep_str!r}")
    try:
        t = int(t_str)
    except ValueError:
        raise ValueError(f"frame ID time step not parseable: {t_str!r}")
    return ep_idx, agent, t


# ---------------------------------------------------------------------------
# Frame index mapping (from episodes + provenance)
# ---------------------------------------------------------------------------

@dataclass
class FrameIndex:
    """Maps flat indices to (episode_index, agent_id, t) and back."""
    # Per-segment metadata.
    seg_episode: List[int] = field(default_factory=list)
    seg_agent: List[str] = field(default_factory=list)
    seg_t_start: List[int] = field(default_factory=list)
    seg_length: List[int] = field(default_factory=list)
    total_frames: int = 0

    def flat_index(self, episode_index: int, agent_id: str, t: int) -> int:
        """Find the flat index for a given (episode, agent, t)."""
        offset = 0
        for i in range(len(self.seg_episode)):
            if (self.seg_episode[i] == episode_index
                    and self.seg_agent[i] == agent_id
                    and self.seg_t_start[i] <= t < self.seg_t_start[i] + self.seg_length[i]):
                return offset + (t - self.seg_t_start[i])
            offset += self.seg_length[i]
        raise KeyError(f"frame ep{episode_index:04d}:{agent_id}:{t} not found")

    def frame_id(self, flat_index: int) -> str:
        """Convert flat index to frame ID string."""
        offset = 0
        for i in range(len(self.seg_episode)):
            if flat_index < offset + self.seg_length[i]:
                t = self.seg_t_start[i] + (flat_index - offset)
                return f"ep{self.seg_episode[i]:04d}:{self.seg_agent[i]}:{t}"
            offset += self.seg_length[i]
        raise IndexError(f"flat index {flat_index} out of range (total={self.total_frames})")


def build_frame_index(snapshot_dir: Path) -> FrameIndex:
    """Build a FrameIndex from the snapshot's episodes."""
    from baseline.framework.rollout import EpisodeCollection

    episodes_path = Path(snapshot_dir) / "episodes"
    if not episodes_path.exists():
        raise FileNotFoundError(f"No episodes/ in snapshot {snapshot_dir}")

    coll = EpisodeCollection.load(episodes_path)
    idx = FrameIndex()
    for ep in coll:
        for agent_id in ep.observations:
            idx.seg_episode.append(ep.episode_index)
            idx.seg_agent.append(agent_id)
            idx.seg_t_start.append(0)
            idx.seg_length.append(ep.num_frames)
            idx.total_frames += ep.num_frames
    return idx


# ---------------------------------------------------------------------------
# Frame data extraction
# ---------------------------------------------------------------------------

@dataclass
class FrameData:
    """All data for one frame."""
    frame_id: str
    flat_index: int
    episode_index: int
    agent_id: str
    t: int
    # Per-stage per-key values (scalars extracted from arrays).
    values: Dict[str, float] = field(default_factory=dict)
    # Observation (from episode).
    observation: Optional[np.ndarray] = None
    # Action (from episode).
    action: Optional[np.ndarray] = None
    # Observer outputs (from episode, if available).
    observer_outputs: Dict[str, Any] = field(default_factory=dict)
    # Debug arrays (experiment-owned, per-frame).
    debug_arrays: Dict[str, float] = field(default_factory=dict)


def _extract_frame_values(
    arrays: Dict[str, np.ndarray],
    flat_index: int,
) -> Dict[str, float]:
    """Extract scalar values for one frame from all arrays."""
    values: Dict[str, float] = {}
    for key, arr in arrays.items():
        arr = np.asarray(arr)
        if arr.ndim == 0:
            # Scalar — same for all frames.
            values[key] = float(arr)
        elif arr.ndim == 1 and len(arr) > flat_index:
            values[key] = float(arr[flat_index])
        elif arr.ndim == 2 and len(arr) > flat_index:
            # Multi-dimensional per-frame — store norm or first element.
            row = arr[flat_index]
            values[f"{key}[0]"] = float(row[0])
            values[f"{key}.norm"] = float(np.linalg.norm(row))
        # Skip higher-dim arrays.
    return values


def inspect_frame(
    snapshot_dir: Path,
    frame_id: str,
) -> FrameData:
    """Inspect a single frame by ID.

    Args:
        snapshot_dir: Snapshot directory (contains episodes/ + replay/).
        frame_id: Frame ID like ``ep0003:robot_a:137``.

    Returns:
        :class:`FrameData` with all available data for that frame.
    """
    snapshot_dir = Path(snapshot_dir)
    ep_idx, agent_id, t = parse_frame_id(frame_id)

    # Build frame index.
    fidx = build_frame_index(snapshot_dir)
    flat = fidx.flat_index(ep_idx, agent_id, t)

    # Load replay arrays.
    from .where import build_frame_arrays
    replay_dir = snapshot_dir / "replay"
    arrays = build_frame_arrays(replay_dir) if replay_dir.exists() else {}

    # Extract per-frame values.
    values = _extract_frame_values(arrays, flat)

    # Load episode for observation/action/observer.
    from baseline.framework.rollout import EpisodeCollection
    coll = EpisodeCollection.load(snapshot_dir / "episodes")
    observation = None
    action = None
    observer_outputs: Dict[str, Any] = {}
    for ep in coll:
        if ep.episode_index == ep_idx and agent_id in ep.observations:
            if t < ep.num_frames:
                observation = ep.observations[agent_id][t]
                action = ep.actions[agent_id][t]
                # Observer outputs may be per-frame or per-episode.
                for obs_name, obs_data in ep.observer_outputs.items():
                    if isinstance(obs_data, dict) and agent_id in obs_data:
                        agent_obs = obs_data[agent_id]
                        if hasattr(agent_obs, '__getitem__') and t < len(agent_obs):
                            observer_outputs[obs_name] = agent_obs[t]
                        else:
                            observer_outputs[obs_name] = agent_obs
            break

    # Separate debug arrays from stage arrays.
    debug_values = {k: v for k, v in values.items() if k.startswith("debug.")}

    return FrameData(
        frame_id=frame_id,
        flat_index=flat,
        episode_index=ep_idx,
        agent_id=agent_id,
        t=t,
        values=values,
        observation=observation,
        action=action,
        observer_outputs=observer_outputs,
        debug_arrays=debug_values,
    )


def filter_frames(
    snapshot_dir: Path,
    where_expr: str,
    limit: int = 20,
) -> List[FrameData]:
    """Filter frames by a --where expression and return matches.

    Args:
        snapshot_dir: Snapshot directory.
        where_expr: ``--where`` expression (e.g. ``combine.aw_normed.r_left_foot < 0``).
        limit: Maximum number of matches to return.

    Returns:
        List of :class:`FrameData` for matching frames.
    """
    snapshot_dir = Path(snapshot_dir)
    from .where import parse, build_frame_arrays

    replay_dir = snapshot_dir / "replay"
    arrays = build_frame_arrays(replay_dir) if replay_dir.exists() else {}

    clause = parse(where_expr)
    mask = clause.match(arrays)
    matching_indices = np.where(mask)[0]

    fidx = build_frame_index(snapshot_dir)

    results: List[FrameData] = []
    for flat in matching_indices[:limit]:
        fid = fidx.frame_id(int(flat))
        ep_idx, agent_id, t = parse_frame_id(fid)
        values = _extract_frame_values(arrays, int(flat))
        results.append(FrameData(
            frame_id=fid,
            flat_index=int(flat),
            episode_index=ep_idx,
            agent_id=agent_id,
            t=t,
            values=values,
        ))
    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frame(data: FrameData) -> str:
    """Render a single FrameData as a human-readable string."""
    lines: List[str] = []
    lines.append(f"帧 {data.frame_id}        "
                 f"（episode {data.episode_index}，{data.agent_id}，第 {data.t} 步）")
    lines.append("")

    # Observation.
    if data.observation is not None:
        obs = np.asarray(data.observation)
        lines.append(f"观测        shape={obs.shape} norm={np.linalg.norm(obs):.3f}")
    lines.append("")

    # Observer outputs.
    if data.observer_outputs:
        lines.append("observer")
        for name, val in data.observer_outputs.items():
            val = np.asarray(val).flatten() if hasattr(val, '__len__') else np.array([val])
            if len(val) <= 8:
                parts = [f"{v:.4f}" for v in val[:8]]
                lines.append(f"    {name:<20s} {', '.join(parts)}")
            else:
                lines.append(f"    {name:<20s} shape={val.shape}")
        lines.append("")

    # Debug arrays (experiment-owned).
    if data.debug_arrays:
        lines.append("实验中间量")
        for key, val in sorted(data.debug_arrays.items()):
            lines.append(f"    {key:<30s} {val:.6f}")
        lines.append("")

    # Per-channel data (from combine stage).
    lines.append("通道        reward      aw(原始)  aw(归一化)  V(s)     adv     贡献")
    # Group values by channel.
    channels: Dict[str, Dict[str, float]] = {}
    for key, val in data.values.items():
        # Keys like "combine.aw_frame.r_left_foot", "combine.aw_normed.r_left_foot"
        # "gae.advantages.r_left_foot", "gae.returns.r_left_foot"
        # "gae.values.r_left_foot"
        parts = key.split(".")
        if len(parts) >= 3:
            stage, name, ch = parts[0], parts[1], ".".join(parts[2:])
            channels.setdefault(ch, {})[f"{stage}.{name}"] = val

    for ch in sorted(channels.keys()):
        ch_data = channels[ch]
        reward = ch_data.get("gae.returns", 0.0)
        aw_raw = ch_data.get("combine.aw_frame", 0.0)
        aw_normed = ch_data.get("combine.aw_normed", 0.0)
        v = ch_data.get("gae.values", 0.0)
        adv = ch_data.get("gae.advantages", 0.0)
        contribution = ch_data.get("combine.contribution", 0.0)
        lines.append(
            f"{ch:<12s} {reward:>+10.4f} {aw_raw:>+8.3f} {aw_normed:>+10.3f} "
            f"{v:>+7.3f} {adv:>+7.4f} {contribution:>+7.4f}"
        )

    # combined_adv.
    combined = data.values.get("combine.combined_adv", None)
    if combined is not None:
        lines.append("")
        lines.append(f"combined_adv  {combined:+.4f}")

    return "\n".join(lines)


def render_frame_list(frames: List[FrameData]) -> str:
    """Render a list of frames (from --where) as a compact table."""
    if not frames:
        return "无匹配帧。"

    lines: List[str] = []
    lines.append(f"找到 {len(frames)} 个匹配帧：")
    lines.append("")
    lines.append(f"{'frame_id':<24s} {'reward':>10s} {'aw_normed':>10s} {'adv':>10s} {'contrib':>10s}")

    for f in frames:
        # Extract a representative channel's values.
        reward = f.values.get("gae.returns.r_potential", 0.0)
        aw = f.values.get("combine.aw_normed.r_potential", 0.0)
        adv = f.values.get("gae.advantages.r_potential", 0.0)
        contrib = f.values.get("combine.contribution.r_potential", 0.0)
        lines.append(
            f"{f.frame_id:<24s} {reward:>+10.4f} {aw:>+10.3f} {adv:>+10.4f} {contrib:>+10.4f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def frame(
    snapshot_dir: Path,
    *,
    frame_id: Optional[str] = None,
    where: Optional[str] = None,
    limit: int = 20,
    render: bool = False,
) -> str:
    """Run the frame inspector.

    Args:
        snapshot_dir: Snapshot directory.
        frame_id: If given, inspect this single frame.
        where: If given, filter frames by this expression.
        limit: Max frames to return from --where.
        render: If True, render the frame (TODO: not yet implemented).

    Returns:
        Human-readable report string.
    """
    snapshot_dir = Path(snapshot_dir)

    if frame_id is not None:
        data = inspect_frame(snapshot_dir, frame_id)
        return render_frame(data)

    if where is not None:
        frames = filter_frames(snapshot_dir, where, limit=limit)
        return render_frame_list(frames)

    raise ValueError("either --id or --where must be given")


__all__ = [
    "FrameData",
    "FrameIndex",
    "parse_frame_id",
    "build_frame_index",
    "inspect_frame",
    "filter_frames",
    "render_frame",
    "render_frame_list",
    "frame",
]
