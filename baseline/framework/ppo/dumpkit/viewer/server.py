"""HTTP server for the debug viewer.

Serves a single-page frontend plus JSON API endpoints that read dump
NPZ files lazily and cache them in memory.

Usage::

    PYTHONPATH=. python3 baseline/framework/ppo/debug.py viewer <dump_dir>
"""
from __future__ import annotations

import json
import socketserver
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_BUNDLED_HTML = _HERE / "index.html"


# ---------------------------------------------------------------------------
# DumpData — lazy NPZ loader with caching
# ---------------------------------------------------------------------------

class DumpData:
    """Lazily loads and caches all NPZ/JSON files in a dump directory.

    All arrays are kept in memory after first load.  For a typical dump
    (~200 MB total) this is acceptable for a debug tool.
    """

    def __init__(self, dump_dir: Path):
        self.dump_dir = Path(dump_dir).resolve()
        if not self.dump_dir.is_dir():
            raise NotADirectoryError(f"dump dir not found: {self.dump_dir}")

        self._cache: Dict[str, Any] = {}

    # -- generic lazy loader ------------------------------------------------

    def _load_npz(self, name: str) -> Optional[Dict[str, np.ndarray]]:
        if name in self._cache:
            return self._cache[name]
        path = self.dump_dir / f"{name}.npz"
        if not path.exists():
            self._cache[name] = None
            return None
        data = dict(np.load(path, allow_pickle=True))
        self._cache[name] = data
        return data

    def _load_json(self, name: str) -> Optional[Any]:
        if name in self._cache:
            return self._cache[name]
        path = self.dump_dir / f"{name}.json"
        if not path.exists():
            self._cache[name] = None
            return None
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        self._cache[name] = obj
        return obj

    # -- accessors ----------------------------------------------------------

    @property
    def manifest(self) -> Dict[str, Any]:
        m = self._load_json("manifest")
        if m is None:
            raise FileNotFoundError(f"manifest.json not found in {self.dump_dir}")
        return m

    @property
    def traj_map(self) -> List[Dict[str, Any]]:
        tm = self._load_json("traj_map")
        if tm is not None:
            return tm
        # Fallback: build from frame_ids in trajectories.npz
        traj_npz = self.trajectories_npz
        if traj_npz is None:
            return []
        frame_ids = traj_npz.get("frame_id")
        ep_lengths = traj_npz.get("ep_lengths", np.array([]))
        if frame_ids is None or len(ep_lengths) == 0:
            return []
        return _build_traj_map_from_frame_ids(frame_ids, ep_lengths)

    @property
    def episodes_npz(self) -> Optional[Dict[str, np.ndarray]]:
        return self._load_npz("episodes")

    @property
    def trajectories_npz(self) -> Optional[Dict[str, np.ndarray]]:
        return self._load_npz("trajectories")

    @property
    def buffer_npz(self) -> Optional[Dict[str, np.ndarray]]:
        return self._load_npz("buffer")

    @property
    def gae_npz(self) -> Optional[Dict[str, np.ndarray]]:
        return self._load_npz("gae")

    @property
    def combine_npz(self) -> Optional[Dict[str, np.ndarray]]:
        return self._load_npz("combine")

    @property
    def update_npz(self) -> Optional[Dict[str, np.ndarray]]:
        return self._load_npz("update")

    @property
    def timeline_npz(self) -> Optional[Dict[str, np.ndarray]]:
        return self._load_npz("timeline")

    @property
    def epoch_frames_npz(self) -> Optional[Dict[str, np.ndarray]]:
        return self._load_npz("epoch_frames")

    # -- derived helpers ----------------------------------------------------

    @property
    def channel_names(self) -> List[str]:
        traj = self.trajectories_npz
        if traj is None:
            return []
        cn = traj.get("channel_names")
        if cn is None:
            return []
        return [str(x) for x in cn.tolist()]

    @property
    def observer_keys(self) -> List[str]:
        """Return observer output keys like ['standing_balance_a', 'height_phi_a', ...]."""
        ep = self.episodes_npz
        if ep is None:
            return []
        keys = set()
        for k in ep:
            if k.startswith("observer_outputs."):
                # observer_outputs.{key}.{field} or observer_outputs.{key}
                rest = k[len("observer_outputs."):]
                parts = rest.split(".")
                keys.add(parts[0])
        return sorted(keys)

    @property
    def agent_ids(self) -> List[str]:
        """Return agent IDs from episodes.npz (e.g. ['robot_a', 'robot_b'])."""
        ep = self.episodes_npz
        if ep is None:
            return []
        ids = set()
        for k in ep:
            if k.startswith("obs."):
                ids.add(k[len("obs."):])
        return sorted(ids)

    @property
    def has_images(self) -> bool:
        record_dir = self.dump_dir / "record"
        return record_dir.is_dir() and (record_dir / "index.json").exists()

    @property
    def seg_offsets(self) -> np.ndarray:
        """Cumulative offsets for trajectory slicing in flattened arrays."""
        traj = self.trajectories_npz
        if traj is None:
            return np.array([0])
        ep_lengths = traj["ep_lengths"]
        offsets = np.zeros(len(ep_lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(ep_lengths)
        return offsets

    @property
    def episode_frame_offsets(self) -> np.ndarray:
        """Cumulative offsets for episode slicing in flattened arrays."""
        ep = self.episodes_npz
        if ep is None:
            return np.array([0])
        return ep["episode_frame_offsets"]

    # -- image path ---------------------------------------------------------

    def image_path(self, episode_pos: int, frame: int) -> Optional[Path]:
        """Return the PNG path for a given episode and frame.

        Recorded step N+1 corresponds to dump frame N (step_00000 is
        the initial state before any action).
        """
        record_dir = self.dump_dir / "record"
        ep_dir = record_dir / f"episode_{episode_pos:05d}"
        if not ep_dir.is_dir():
            return None
        # Recorded step = frame + 1
        png = ep_dir / f"step_{frame + 1:05d}.png"
        if png.exists():
            return png
        return None


def _build_traj_map_from_frame_ids(
    frame_ids: np.ndarray,
    ep_lengths: np.ndarray,
) -> List[Dict[str, Any]]:
    """Build traj_map from frame_id array when traj_map.json is missing."""
    n_trajs = len(ep_lengths)
    # We don't have episode info, so build a minimal map.
    ep_map: Dict[int, List[Dict[str, Any]]] = {}
    offset = 0
    for traj_idx in range(n_trajs):
        T = int(ep_lengths[traj_idx])
        if T == 0:
            continue
        fid = str(frame_ids[offset])
        if fid.startswith("flat:"):
            offset += T
            continue
        parts = fid.split(":")
        ep_pos = int(parts[0][2:])
        agent_id = parts[1]
        t_start = int(parts[2])
        ep_map.setdefault(ep_pos, []).append({
            "traj_idx": traj_idx,
            "agent_id": agent_id,
            "t_start": t_start,
            "length": T,
        })
        offset += T
    max_ep = max(ep_map.keys()) if ep_map else 0
    result = []
    for i in range(max_ep + 1):
        result.append({
            "list_pos": i,
            "seed": None,
            "num_frames": None,
            "trajectories": ep_map.get(i, []),
        })
    return result


# ---------------------------------------------------------------------------
# API handlers
# ---------------------------------------------------------------------------

def _arr_to_list(arr: np.ndarray) -> Any:
    """Convert numpy array to JSON-serializable list."""
    if arr.dtype == object:
        return [str(x) for x in arr]
    return arr.tolist()


def _dict_item(arr: np.ndarray) -> Any:
    """Extract dict from object-dtype array (size 1)."""
    if arr.dtype == object and arr.size == 1:
        return arr.item()
    return arr


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

class ViewerAPI:
    """Handles all /api/* endpoints.  Each method returns (status, json_body)."""

    def __init__(self, data: DumpData):
        self.data = data

    def handle(self, path: str) -> Tuple[int, Any]:
        """Route an /api/... path and return (HTTP status, response body)."""
        parts = path.strip("/").split("/")
        # parts[0] == "api"
        if len(parts) < 2:
            return 404, {"error": "unknown endpoint"}

        endpoint = parts[1]

        try:
            if endpoint == "manifest":
                return 200, self._manifest()
            elif endpoint == "episode_list":
                return 200, self._episode_list()
            elif endpoint == "traj_map":
                return 200, self.data.traj_map
            elif endpoint == "episode" and len(parts) >= 4 and parts[3] == "frame":
                return self._episode_frame(int(parts[2]), int(parts[4]))
            elif endpoint == "image" and len(parts) >= 4:
                return self._image(int(parts[2]), int(parts[3]))
            elif endpoint == "trajectory" and len(parts) >= 3:
                traj_idx = int(parts[2])
                if len(parts) >= 5 and parts[3] == "frame":
                    return self._traj_frame(traj_idx, int(parts[4]))
                elif len(parts) >= 6 and parts[3] == "epoch" and parts[5] == "overview":
                    return self._traj_epoch_overview(traj_idx, int(parts[4]))
                elif len(parts) >= 7 and parts[3] == "epoch" and parts[5] == "frame":
                    return self._traj_epoch_frame(traj_idx, int(parts[4]), int(parts[6]))
                elif len(parts) >= 4 and parts[3] == "epoch_compare":
                    return self._traj_epoch_compare(traj_idx)
                elif len(parts) >= 3:
                    return self._traj_overview(traj_idx)
            elif endpoint == "timeline" and len(parts) >= 3:
                if parts[2] == "overview":
                    return 200, self._timeline_overview()
                elif parts[2] == "step" and len(parts) >= 4:
                    return self._timeline_step(int(parts[3]))
            return 404, {"error": f"unknown endpoint: {endpoint}"}
        except (FileNotFoundError, ValueError, IndexError, KeyError) as e:
            return 404, {"error": str(e)}

    # -- /api/manifest ------------------------------------------------------

    def _manifest(self) -> Dict[str, Any]:
        m = dict(self.data.manifest)
        m["channel_names"] = self.data.channel_names
        m["observer_keys"] = self.data.observer_keys
        m["agent_ids"] = self.data.agent_ids
        m["has_images"] = self.data.has_images
        m["has_timeline"] = self.data.timeline_npz is not None
        m["has_epoch_frames"] = self.data.epoch_frames_npz is not None
        # clip_eps and target_kl from timeline
        tl = self.data.timeline_npz
        if tl is not None:
            m["clip_eps"] = float(tl["clip_eps"]) if "clip_eps" in tl else 0.2
            m["target_kl"] = float(tl["target_kl"]) if "target_kl" in tl else float("nan")
            m["n_epochs"] = int(tl["n_epochs"]) if "n_epochs" in tl else 0
            m["n_steps"] = int(tl["n_steps"]) if "n_steps" in tl else 0
            m["early_stop_step"] = int(tl["early_stop_step"]) if "early_stop_step" in tl else -1
        else:
            m["clip_eps"] = 0.2
            m["target_kl"] = float("nan")
            m["n_epochs"] = 0
            m["n_steps"] = 0
            m["early_stop_step"] = -1
        return m

    # -- /api/episode_list --------------------------------------------------

    def _episode_list(self) -> List[Dict[str, Any]]:
        tm = self.data.traj_map
        result = []
        for ep in tm:
            result.append({
                "list_pos": ep["list_pos"],
                "seed": ep.get("seed"),
                "num_frames": ep.get("num_frames"),
                "n_trajectories": len(ep.get("trajectories", [])),
            })
        return result

    # -- /api/episode/<pos>/frame/<f> --------------------------------------

    def _episode_frame(self, ep_pos: int, frame: int) -> Tuple[int, Dict[str, Any]]:
        ep_npz = self.data.episodes_npz
        if ep_npz is None:
            return 404, {"error": "episodes.npz not found"}

        offsets = self.data.episode_frame_offsets
        if ep_pos < 0 or ep_pos + 1 >= len(offsets):
            return 404, {"error": f"episode {ep_pos} out of range"}
        start = int(offsets[ep_pos])
        end = int(offsets[ep_pos + 1])
        if frame < 0 or frame >= end - start:
            return 404, {"error": f"frame {frame} out of range for episode {ep_pos}"}

        idx = start + frame
        result: Dict[str, Any] = {
            "episode_pos": ep_pos,
            "frame": frame,
            "has_image": self.data.image_path(ep_pos, frame) is not None,
        }

        # Per-agent obs/actions
        agents = self.data.agent_ids
        for aid in agents:
            obs_key = f"obs.{aid}"
            act_key = f"actions.{aid}"
            if obs_key in ep_npz:
                result[f"obs_{aid}"] = _arr_to_list(ep_npz[obs_key][idx])
            if act_key in ep_npz:
                result[f"actions_{aid}"] = _arr_to_list(ep_npz[act_key][idx])
            ef_key = f"explore_factors.{aid}"
            if ef_key in ep_npz:
                result[f"explore_factor_{aid}"] = float(ep_npz[ef_key][idx])

        # Observer outputs for this frame
        observer_data: Dict[str, Any] = {}
        for k in ep_npz:
            if not k.startswith("observer_outputs."):
                continue
            rest = k[len("observer_outputs."):]
            # observer_outputs.{key}.{field} — field is per-frame
            parts = rest.split(".", 1)
            obs_key = parts[0]
            if len(parts) == 2:
                field = parts[1]
                val = ep_npz[k][idx]
                if np.isscalar(val) or val.ndim == 0:
                    observer_data[f"{obs_key}.{field}"] = float(val)
                else:
                    observer_data[f"{obs_key}.{field}"] = _arr_to_list(val)
            else:
                # Scalar observer output (not per-frame)
                pass
        result["observer_outputs"] = observer_data

        # Termination info
        term_records = ep_npz.get("_termination_records")
        if term_records is not None and term_records.size > 0:
            try:
                if term_records.dtype == object:
                    raw = str(term_records.item())
                else:
                    raw = str(term_records.item() if term_records.ndim == 0 else term_records[0])
                term_data = json.loads(raw)
                ep_key = f"ep{ep_pos:04d}"
                if ep_key in term_data:
                    result["termination"] = term_data[ep_key]
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        # Associated trajectories with per-frame channel data
        tm = self.data.traj_map
        ep_trajs = tm[ep_pos].get("trajectories", []) if ep_pos < len(tm) else []
        traj_npz = self.data.trajectories_npz
        seg_offsets = self.data.seg_offsets

        enriched_trajs = []
        for t in ep_trajs:
            traj_idx = t["traj_idx"]
            t_start = t["t_start"]
            t_len = t["length"]
            # The trajectory frame corresponding to this episode frame
            traj_frame = frame - t_start
            covered = 0 <= traj_frame < t_len

            entry: Dict[str, Any] = {
                "traj_idx": traj_idx,
                "agent_id": t["agent_id"],
                "t_start": t_start,
                "length": t_len,
                "covered": covered,
                "traj_frame": traj_frame if covered else -1,
            }

            # Per-frame channel data for this trajectory at this frame
            if covered and traj_npz is not None and traj_idx < len(seg_offsets) - 1:
                flat_idx = int(seg_offsets[traj_idx]) + traj_frame
                for ch in self.data.channel_names:
                    rew_key = f"reward.{ch}"
                    aw_key = f"actor_weight.{ch}"
                    if rew_key in traj_npz:
                        entry[f"reward_{ch}"] = float(traj_npz[rew_key][flat_idx])
                    if aw_key in traj_npz:
                        entry[f"actor_weight_{ch}"] = float(traj_npz[aw_key][flat_idx])
                # floor_weight, explore_factor
                if "floor_weight" in traj_npz:
                    entry["floor_weight"] = float(traj_npz["floor_weight"][flat_idx])
                if "explore_factor" in traj_npz:
                    entry["explore_factor"] = float(traj_npz["explore_factor"][flat_idx])
                # Per-channel is_terminated (per-trajectory scalar)
                for ch in self.data.channel_names:
                    it_key = f"is_terminated.{ch}"
                    if it_key in traj_npz:
                        entry[f"is_terminated_{ch}"] = bool(traj_npz[it_key][traj_idx])
                # importance (per-trajectory scalar)
                if "importance" in traj_npz:
                    entry["importance"] = float(traj_npz["importance"][traj_idx])
                # frame_id
                if "frame_id" in traj_npz:
                    entry["frame_id"] = str(traj_npz["frame_id"][flat_idx])

            enriched_trajs.append(entry)

        result["trajectories"] = enriched_trajs

        return 200, result

    # -- /api/image/<pos>/<f> -----------------------------------------------

    def _image(self, ep_pos: int, frame: int) -> Tuple[int, Any]:
        png_path = self.data.image_path(ep_pos, frame)
        if png_path is None or not png_path.exists():
            return 404, {"error": "image not found"}
        return 200, png_path  # signal: serve as binary

    # -- /api/trajectory/<idx>/overview ------------------------------------

    def _find_traj_provenance(self, traj_idx: int) -> Tuple[Optional[int], Optional[int]]:
        """Find episode_pos and t_start for a trajectory by searching traj_map.

        Returns (episode_pos, t_start) or (None, None) if not found.
        """
        tm = self.data.traj_map
        for ep in tm:
            for t in ep.get("trajectories", []):
                if t.get("traj_idx") == traj_idx:
                    return ep.get("list_pos"), t.get("t_start")
        return None, None

    def _traj_overview(self, traj_idx: int) -> Tuple[int, Dict[str, Any]]:
        offsets = self.data.seg_offsets
        if traj_idx < 0 or traj_idx + 1 >= len(offsets):
            return 404, {"error": f"trajectory {traj_idx} out of range"}
        start = int(offsets[traj_idx])
        end = int(offsets[traj_idx + 1])
        T = end - start

        result: Dict[str, Any] = {
            "traj_idx": traj_idx,
            "length": T,
            "channels": self.data.channel_names,
        }

        # Episode provenance (for image lookup)
        ep_pos, t_start = self._find_traj_provenance(traj_idx)
        result["episode_pos"] = ep_pos
        result["t_start"] = t_start

        # Frame IDs
        traj_npz = self.data.trajectories_npz
        if traj_npz is not None and "frame_id" in traj_npz:
            fids = traj_npz["frame_id"][start:end]
            result["frame_ids"] = [str(x) for x in fids]

        # Per-channel reward / actor_weight
        for ch in self.data.channel_names:
            if traj_npz is not None:
                rew_key = f"reward.{ch}"
                aw_key = f"actor_weight.{ch}"
                if rew_key in traj_npz:
                    result[f"reward_{ch}"] = _arr_to_list(traj_npz[rew_key][start:end])
                if aw_key in traj_npz:
                    result[f"actor_weight_{ch}"] = _arr_to_list(traj_npz[aw_key][start:end])

        # GAE: value, advantage, return per channel
        gae = self.data.gae_npz
        if gae is not None:
            for ch in self.data.channel_names:
                for prefix, key in [("values", "values_all"), ("advs", "advs_all"), ("rets", "rets_all")]:
                    arr = gae.get(key)
                    if arr is not None:
                        d = _dict_item(arr)
                        if isinstance(d, dict) and ch in d:
                            result[f"{prefix}_{ch}"] = _arr_to_list(
                                np.asarray(d[ch][start:end], dtype=np.float32)
                            )
            # key_frame_mask per channel (per-frame bool)
            kfm = gae.get("key_frame_mask")
            if kfm is not None:
                d = _dict_item(kfm)
                if isinstance(d, dict):
                    result["key_frame_mask"] = {
                        ch: [bool(x) for x in np.asarray(v[start:end])]
                        for ch, v in d.items()
                    }
            # key_seg_active / key_seg_terminated (per-trajectory bool)
            for seg_key in ("key_seg_active", "key_seg_terminated"):
                seg = gae.get(seg_key)
                if seg is not None:
                    d = _dict_item(seg)
                    if isinstance(d, dict):
                        result[seg_key] = {
                            ch: bool(np.asarray(v)[traj_idx]) if traj_idx < len(v) else False
                            for ch, v in d.items()
                        }
            # bootstrap info
            bv = gae.get("bootstrap_values")
            if bv is not None:
                d = _dict_item(bv)
                if isinstance(d, dict):
                    # bootstrap_values is per-trajectory per-channel
                    # Extract the value for this specific trajectory
                    result["bootstrap_values"] = {
                        ch: float(np.asarray(v)[traj_idx])
                        for ch, v in d.items()
                    }
            bi = gae.get("bootstrap_indices")
            if bi is not None:
                result["bootstrap_indices"] = _arr_to_list(np.asarray(bi))
            # Check if this trajectory is a bootstrap segment
            if bi is not None and traj_idx in np.asarray(bi).tolist():
                result["is_bootstrap_seg"] = True

        # Combine: combined_adv
        combine = self.data.combine_npz
        if combine is not None:
            ca = combine.get("combined_adv")
            if ca is not None:
                result["combined_adv"] = _arr_to_list(ca[start:end])
            # aw_l1_sum (per-frame)
            aw_l1 = combine.get("aw_l1_sum")
            if aw_l1 is not None:
                result["aw_l1_sum"] = _arr_to_list(aw_l1[start:end])
            # confidence / EV per channel
            conf = combine.get("confidences")
            if conf is not None:
                d = _dict_item(conf)
                if isinstance(d, dict):
                    result["confidences"] = {ch: float(v) for ch, v in d.items()}
            ev = combine.get("explained_variances")
            if ev is not None:
                d = _dict_item(ev)
                if isinstance(d, dict):
                    result["explained_variances"] = {ch: float(v) for ch, v in d.items()}
            kaw = combine.get("key_actor_weight_frame")
            if kaw is not None:
                d = _dict_item(kaw)
                if isinstance(d, dict):
                    # key_actor_weight_frame is per-frame per-channel
                    result["actor_weights"] = {
                        ch: _arr_to_list(np.asarray(v[start:end], dtype=np.float32))
                        for ch, v in d.items()
                    }

        # Trajectory-level: floor_weight, explore_factor, importance
        if traj_npz is not None:
            if "floor_weight" in traj_npz:
                result["floor_weight"] = _arr_to_list(traj_npz["floor_weight"][start:end])
            if "explore_factor" in traj_npz:
                result["explore_factor"] = _arr_to_list(traj_npz["explore_factor"][start:end])
            if "importance" in traj_npz:
                result["importance"] = float(traj_npz["importance"][traj_idx])
            # is_terminated per channel (per-trajectory)
            for ch in self.data.channel_names:
                it_key = f"is_terminated.{ch}"
                if it_key in traj_npz:
                    result[f"is_terminated_{ch}"] = bool(traj_npz[it_key][traj_idx])

        # Buffer: old log_prob
        buf = self.data.buffer_npz
        if buf is not None and "log_probs" in buf:
            result["old_log_prob"] = _arr_to_list(buf["log_probs"][start:end])

        return 200, result

    # -- /api/trajectory/<idx>/frame/<f> -----------------------------------

    def _traj_frame(self, traj_idx: int, frame: int) -> Tuple[int, Dict[str, Any]]:
        status, overview = self._traj_overview(traj_idx)
        if status != 200:
            return status, overview
        T = overview["length"]
        if frame < 0 or frame >= T:
            return 404, {"error": f"frame {frame} out of range (len={T})"}

        result: Dict[str, Any] = {"traj_idx": traj_idx, "frame": frame, "length": T}

        # Extract single-frame values from overview
        for ch in self.data.channel_names:
            for prefix in ["reward", "actor_weight", "values", "advs", "rets"]:
                key = f"{prefix}_{ch}"
                if key in overview:
                    result[key] = overview[key][frame]

        if "combined_adv" in overview:
            result["combined_adv"] = overview["combined_adv"][frame]
        if "old_log_prob" in overview:
            result["old_log_prob"] = overview["old_log_prob"][frame]
        if "frame_ids" in overview:
            result["frame_id"] = overview["frame_ids"][frame]

        # Bootstrap info for this trajectory
        if "bootstrap_indices" in overview:
            bi = overview["bootstrap_indices"]
            # Find if this trajectory has a bootstrap entry
            # bootstrap_indices maps seg_idx → flattened_frame_idx
            traj_start = int(self.data.seg_offsets[traj_idx])
            for i, val in enumerate(bi):
                if int(val) == traj_start + frame or int(val) == traj_start:
                    result["is_bootstrap_frame"] = True
                    if "bootstrap_values" in overview:
                        result["bootstrap_value"] = overview["bootstrap_values"]
                    break

        return 200, result

    # -- /api/trajectory/<idx>/epoch/<e>/overview --------------------------

    def _traj_epoch_overview(self, traj_idx: int, epoch: int) -> Tuple[int, Dict[str, Any]]:
        ef = self.data.epoch_frames_npz
        if ef is None:
            return 404, {"error": "epoch_frames.npz not found"}

        offsets = self.data.seg_offsets
        if traj_idx < 0 or traj_idx + 1 >= len(offsets):
            return 404, {"error": f"trajectory {traj_idx} out of range"}
        start = int(offsets[traj_idx])
        end = int(offsets[traj_idx + 1])

        n_epochs = int(ef["n_epochs"]) if "n_epochs" in ef else 0
        if epoch < 0 or epoch >= n_epochs:
            return 404, {"error": f"epoch {epoch} out of range (n_epochs={n_epochs})"}

        result: Dict[str, Any] = {
            "traj_idx": traj_idx,
            "epoch": epoch,
            "length": end - start,
        }

        # Episode provenance (for image lookup)
        ep_pos, t_start = self._find_traj_provenance(traj_idx)
        result["episode_pos"] = ep_pos
        result["t_start"] = t_start

        ratio_key = f"ratio.{epoch}"
        clip_key = f"clip_mask.{epoch}"
        lp_key = f"new_log_prob.{epoch}"

        if ratio_key in ef:
            result["ratio"] = _arr_to_list(ef[ratio_key][start:end])
        if clip_key in ef:
            result["clip_mask"] = [bool(x) for x in ef[clip_key][start:end]]
        if lp_key in ef:
            result["new_log_prob"] = _arr_to_list(ef[lp_key][start:end])

        # Per-channel new_value
        for ch in self.data.channel_names:
            nv_key = f"new_value.{epoch}.{ch}"
            if nv_key in ef:
                result[f"new_value_{ch}"] = _arr_to_list(ef[nv_key][start:end])

        # Cross-reference: old_log_prob from buffer
        buf = self.data.buffer_npz
        if buf is not None and "log_probs" in buf:
            result["old_log_prob"] = _arr_to_list(buf["log_probs"][start:end])

        # Cross-reference: old_value and return from gae
        gae = self.data.gae_npz
        if gae is not None:
            for ch in self.data.channel_names:
                va = gae.get("values_all")
                if va is not None:
                    d = _dict_item(va)
                    if isinstance(d, dict) and ch in d:
                        result[f"old_value_{ch}"] = _arr_to_list(
                            np.asarray(d[ch][start:end], dtype=np.float32)
                        )
                ra = gae.get("rets_all")
                if ra is not None:
                    d = _dict_item(ra)
                    if isinstance(d, dict) and ch in d:
                        result[f"return_{ch}"] = _arr_to_list(
                            np.asarray(d[ch][start:end], dtype=np.float32)
                        )

        # Cross-reference: combined_adv from combine
        combine = self.data.combine_npz
        if combine is not None:
            ca = combine.get("combined_adv")
            if ca is not None:
                result["combined_adv"] = _arr_to_list(ca[start:end])

        # clip_eps from timeline or update
        tl = self.data.timeline_npz
        if tl is not None and "clip_eps" in tl:
            result["clip_eps"] = float(tl["clip_eps"])
        else:
            result["clip_eps"] = 0.2

        # actor_stopped_epoch
        if "actor_stopped_epoch" in ef:
            result["actor_stopped_epoch"] = int(ef["actor_stopped_epoch"])

        return 200, result

    # -- /api/trajectory/<idx>/epoch/<e>/frame/<f> -------------------------

    def _traj_epoch_frame(self, traj_idx: int, epoch: int, frame: int) -> Tuple[int, Dict[str, Any]]:
        status, overview = self._traj_epoch_overview(traj_idx, epoch)
        if status != 200:
            return status, overview
        T = overview["length"]
        if frame < 0 or frame >= T:
            return 404, {"error": f"frame {frame} out of range (len={T})"}

        result: Dict[str, Any] = {"traj_idx": traj_idx, "epoch": epoch, "frame": frame}
        for key in ["ratio", "clip_mask", "new_log_prob"]:
            if key in overview:
                result[key] = overview[key][frame]
        for ch in self.data.channel_names:
            nv_key = f"new_value_{ch}"
            if nv_key in overview:
                result[nv_key] = overview[nv_key][frame]

        return 200, result

    # -- /api/trajectory/<idx>/epoch_compare --------------------------------

    def _traj_epoch_compare(self, traj_idx: int) -> Tuple[int, Dict[str, Any]]:
        ef = self.data.epoch_frames_npz
        if ef is None:
            return 404, {"error": "epoch_frames.npz not found"}

        offsets = self.data.seg_offsets
        if traj_idx < 0 or traj_idx + 1 >= len(offsets):
            return 404, {"error": f"trajectory {traj_idx} out of range"}
        start = int(offsets[traj_idx])
        end = int(offsets[traj_idx + 1])
        T = end - start

        n_epochs = int(ef["n_epochs"]) if "n_epochs" in ef else 0
        result: Dict[str, Any] = {
            "traj_idx": traj_idx,
            "length": T,
            "n_epochs": n_epochs,
            "epochs": [],
        }

        for e in range(n_epochs):
            ratio_key = f"ratio.{e}"
            clip_key = f"clip_mask.{e}"
            entry: Dict[str, Any] = {"epoch": e}
            if ratio_key in ef:
                entry["ratio"] = _arr_to_list(ef[ratio_key][start:end])
            if clip_key in ef:
                entry["clip_mask"] = [bool(x) for x in ef[clip_key][start:end]]
            result["epochs"].append(entry)

        # actor_stopped_epoch
        if "actor_stopped_epoch" in ef:
            result["actor_stopped_epoch"] = int(ef["actor_stopped_epoch"])

        # clip_eps from timeline
        tl = self.data.timeline_npz
        if tl is not None and "clip_eps" in tl:
            result["clip_eps"] = float(tl["clip_eps"])
        else:
            result["clip_eps"] = 0.2

        return 200, result

    # -- /api/timeline/overview ---------------------------------------------

    def _timeline_overview(self) -> Dict[str, Any]:
        tl = self.data.timeline_npz
        if tl is None:
            return {"error": "timeline.npz not found", "available": False}

        result: Dict[str, Any] = {
            "available": True,
            "n_epochs": int(tl["n_epochs"]) if "n_epochs" in tl else 0,
            "n_batches": int(tl["n_batches"]) if "n_batches" in tl else 0,
            "n_steps": int(tl["n_steps"]) if "n_steps" in tl else 0,
            "early_stop_step": int(tl["early_stop_step"]) if "early_stop_step" in tl else -1,
            "target_kl": float(tl["target_kl"]) if "target_kl" in tl else float("nan"),
            "clip_eps": float(tl["clip_eps"]) if "clip_eps" in tl else 0.2,
        }

        n_steps = result["n_steps"]
        for key in ["epoch_idx", "mb_idx", "actor_active", "kl", "clip_frac",
                     "ratio_mean", "ratio_max", "policy_loss", "actor_grad",
                     "running_mean_kl"]:
            if key in tl:
                arr = tl[key]
                if arr.dtype == bool:
                    result[key] = [bool(x) for x in arr]
                else:
                    result[key] = _arr_to_list(np.asarray(arr, dtype=np.float32))

        # Per-channel critic_loss / critic_grad
        for ch in self.data.channel_names:
            cl_key = f"critic_loss"
            if cl_key in tl:
                d = _dict_item(tl[cl_key])
                if isinstance(d, dict) and ch in d:
                    result[f"critic_loss_{ch}"] = _arr_to_list(np.asarray(d[ch], dtype=np.float32))
            cg_key = f"critic_grad"
            if cg_key in tl:
                d = _dict_item(tl[cg_key])
                if isinstance(d, dict) and ch in d:
                    result[f"critic_grad_{ch}"] = _arr_to_list(np.asarray(d[ch], dtype=np.float32))

        return result

    # -- /api/timeline/step/<s> --------------------------------------------

    def _timeline_step(self, step: int) -> Tuple[int, Dict[str, Any]]:
        ov = self._timeline_overview()
        if not ov.get("available"):
            return 404, {"error": "timeline not available"}
        n_steps = ov["n_steps"]
        if step < 0 or step >= n_steps:
            return 404, {"error": f"step {step} out of range (n_steps={n_steps})"}

        result: Dict[str, Any] = {"step": step}
        for key in ["epoch_idx", "mb_idx", "actor_active", "kl", "clip_frac",
                     "ratio_mean", "ratio_max", "policy_loss", "actor_grad",
                     "running_mean_kl"]:
            if key in ov and step < len(ov[key]):
                result[key] = ov[key][step]

        for ch in self.data.channel_names:
            cl_key = f"critic_loss_{ch}"
            if cl_key in ov and step < len(ov[cl_key]):
                result[cl_key] = ov[cl_key][step]
            cg_key = f"critic_grad_{ch}"
            if cg_key in ov and step < len(ov[cg_key]):
                result[cg_key] = ov[cg_key][step]

        return 200, result


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _ViewerHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves the frontend + API + static images.

    The ``api`` and ``viewer_dir`` are accessed via the server instance
    (``self.server.api`` / ``self.server.viewer_dir``), injected by
    ``serve()``.
    """

    @property
    def api(self) -> ViewerAPI:
        return self.server.api  # type: ignore[attr-defined]

    @property
    def viewer_dir(self) -> Path:
        return self.server.viewer_dir  # type: ignore[attr-defined]

    def log_message(self, format, *args):  # noqa: A002
        return  # quiet

    def do_GET(self):
        path = self.path.split("?")[0]  # strip query

        # API routes
        if path.startswith("/api/"):
            self._handle_api(path)
            return

        # Image route (binary)
        if path.startswith("/img/"):
            self._handle_image(path)
            return

        # Static files
        self._handle_static(path)

    def _handle_api(self, path: str):
        status, body = self.api.handle(path)
        if status == 200 and isinstance(body, Path):
            # Image binary response
            self._serve_file(body, "image/png")
            return
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body, ensure_ascii=False).encode("utf-8"))

    def _handle_image(self, path: str):
        # /img/<ep_pos>/<frame>
        parts = path.strip("/").split("/")
        if len(parts) != 3:
            self.send_error(404)
            return
        try:
            ep_pos = int(parts[1])
            frame = int(parts[2])
        except ValueError:
            self.send_error(404)
            return
        png_path = self.api.data.image_path(ep_pos, frame)
        if png_path is None or not png_path.exists():
            self.send_error(404)
            return
        self._serve_file(png_path, "image/png")

    def _handle_static(self, path: str):
        if path == "/" or path == "":
            path = "/index.html"
        # Security: only serve files from viewer_dir
        file_path = (self.viewer_dir / path.lstrip("/")).resolve()
        try:
            file_path.relative_to(self.viewer_dir)
        except ValueError:
            self.send_error(403)
            return
        if not file_path.is_file():
            self.send_error(404)
            return
        content_type = "application/octet-stream"
        if file_path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif file_path.suffix == ".js":
            content_type = "application/javascript"
        elif file_path.suffix == ".css":
            content_type = "text/css"
        self._serve_file(file_path, content_type)

    def _serve_file(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(
    dump_dir: Path,
    port: int = 8766,
    open_browser: bool = True,
) -> None:
    """Start the viewer HTTP server.

    Args:
        dump_dir: Path to the dump directory (e.g. runs/.../dumps/u00008/).
        port: HTTP port (default 8766).
        open_browser: Auto-open the browser (default True).
    """
    dump_dir = Path(dump_dir).resolve()
    if not dump_dir.is_dir():
        raise NotADirectoryError(f"dump dir not found: {dump_dir}")

    if not _BUNDLED_HTML.exists():
        raise FileNotFoundError(f"bundled index.html not found: {_BUNDLED_HTML}")

    data = DumpData(dump_dir)
    api = ViewerAPI(data)

    # Verify manifest exists
    _ = data.manifest

    with _ThreadingServer(("", port), _ViewerHandler) as httpd:
        # Inject api and viewer_dir into the server instance so handlers
        # can access them via self.server.api / self.server.viewer_dir.
        httpd.api = api  # type: ignore[attr-defined]
        httpd.viewer_dir = _HERE  # type: ignore[attr-defined]
        url = f"http://localhost:{port}/"
        print(f"[viewer] serving dump: {dump_dir}", flush=True)
        print(f"[viewer] url: {url}", flush=True)
        print(f"[viewer] press Ctrl+C to stop", flush=True)
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[viewer] shutting down", flush=True)
            httpd.shutdown()


__all__ = ["DumpData", "ViewerAPI", "serve"]
