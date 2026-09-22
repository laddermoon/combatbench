"""DumpDataset — the single read entry for a captured update dump.

All dump readers — the viewer API, ``dump_analysis.py``,
``dump_delta.py``, ``dump_render.py``, the ``debug.py`` CLI and ad-hoc
analysis scripts — go through this layer.  See DESIGN_frame_access.md.

Design:

- **Row spaces, not files.**  ``ds.frames`` is the canonical
  trajectory-frame space (n = sum of traj_lengths); ``ds.trajs`` is the
  per-trajectory scalar space; ``ds.episodes`` is the per-episode space
  with per-frame slicing via ``ds.episodes[i]``; ``ds.timeline`` /
  ``ds.gradsig`` cover their own row counts.  Observer outputs join into
  ``ds.frames`` through ``traj_map``.
- **Columnar lazy loading.**  npz members are decompressed on first
  column access only (NpzFile member-level laziness) and cached.
  Dict-serialized members (``advs_all`` etc.) unwrap once.
- **No query DSL.**  Tables are ``dict[str, ndarray]`` — numpy boolean
  masks are the where-clause, numpy/pandas are the aggregators.
- **Schema knowledge lives here once**: key spelling, dict-of-array
  unwrapping, ep↔traj index math, flat vs object-array observer
  layouts, traj_map rebuild fallback, ``flat:`` unmapped frames.

Semantic contract (shared with dump_analysis.py): missing data surfaces
as ``None``/absent columns — never fabricated zeros; gradsig rows are a
<=2000-frame sample, not full-buffer statistics.
"""
from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# NpzSource — one npz file with member-level lazy loading
# ---------------------------------------------------------------------------

class NpzSource:
    """Lazy member-level reader for one ``*.npz`` (zip) file.

    ``np.load`` already decompresses members on demand — we hold the
    NpzFile open and cache loaded members.  ``keys()`` reads the zip
    central directory only, so schema discovery is cheap even for the
    180MB episodes.npz.  Thread-safe (the viewer is multi-threaded).
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._npz: Optional[Any] = None
        self._cols: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.path.is_file()

    def _open(self) -> Optional[Any]:
        if self._npz is None and self.available:
            self._npz = np.load(self.path, allow_pickle=True)
        return self._npz

    def keys(self) -> List[str]:
        """Member names — cheap (zip index only, no decompression)."""
        with self._lock:
            npz = self._open()
            return list(npz.files) if npz is not None else []

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def __getitem__(self, key: str) -> np.ndarray:
        arr = self.get(key)
        if arr is None:
            raise KeyError(key)
        return arr

    def get(self, key: str) -> Optional[np.ndarray]:
        """Member array or None.  First access decompresses this member."""
        with self._lock:
            npz = self._open()
            if npz is None or key not in npz.files:
                return None
            if key not in self._cols:
                self._cols[key] = npz[key]
            return self._cols[key]

    def get_dict(self, key: str) -> Optional[Dict[str, Any]]:
        """Object-serialized dict member → real dict (or None)."""
        arr = self.get(key)
        if arr is None:
            return None
        if arr.dtype == object and arr.size == 1:
            d = arr.item()
            return d if isinstance(d, dict) else None
        return None

    def scalar(self, key: str) -> Optional[Any]:
        """0-d/scalar member → Python scalar or None."""
        arr = self.get(key)
        if arr is None:
            return None
        try:
            return np.asarray(arr).item()
        except (TypeError, ValueError):
            return None

    def close(self) -> None:
        with self._lock:
            if self._npz is not None:
                self._npz.close()
                self._npz = None
            self._cols.clear()

    def clear_cache(self) -> None:
        with self._lock:
            self._cols.clear()


def _dict_item(arr: Any) -> Any:
    """0-d object array → dict payload; anything else passes through."""
    if isinstance(arr, np.ndarray) and arr.dtype == object \
            and arr.size == 1:
        return arr.item()
    return arr


def parse_frame_id(fid: Any) -> Optional[Dict[str, Any]]:
    """'ep0007:robot_a:123' → {episode, agent, env_frame}; 'flat:*' → None."""
    s = str(fid)
    if s.startswith("flat:"):
        return None
    parts = s.split(":")
    if len(parts) != 3 or not parts[0].startswith("ep"):
        return None
    try:
        return {
            "episode": int(parts[0][2:]),
            "agent": parts[1],
            "env_frame": int(parts[2]),
        }
    except ValueError:
        return None


def _agent_suffix(agent_id: str) -> str:
    """'robot_a' → 'a' — observer keys use the suffix (standing_balance_a)."""
    return agent_id.rsplit("_", 1)[-1]


def _build_traj_map_from_frame_ids(
    frame_ids: np.ndarray, traj_lengths: np.ndarray
) -> List[Dict[str, Any]]:
    """Reconstruct traj_map.json when the file is absent (older dumps).

    Parses the first frame_id of each trajectory; 'flat:*' segments are
    excluded (unmappable), matching dump_capture semantics.
    """
    episodes: Dict[int, List[Dict[str, Any]]] = {}
    offset = 0
    for t_idx, length in enumerate(np.asarray(traj_lengths).tolist()):
        length = int(length)
        if length and offset < len(frame_ids):
            parsed = parse_frame_id(frame_ids[offset])
            if parsed is not None:
                episodes.setdefault(parsed["episode"], []).append({
                    "traj_idx": int(t_idx),
                    "agent_id": parsed["agent"],
                    "t_start": int(parsed["env_frame"]),
                    "length": length,
                })
        offset += length
    # Fill the full range — episodes without mapped trajectories still
    # appear with an empty trajectories list (matches the old viewer's
    # _build_traj_map_from_frame_ids semantics).
    max_ep = max(episodes.keys()) if episodes else -1
    return [
        {"list_pos": i, "seed": None, "num_frames": None,
         "trajectories": episodes.get(i, [])}
        for i in range(max_ep + 1)
    ]


# ---------------------------------------------------------------------------
# Table — one row space: named columns, same row count, lazy resolution
# ---------------------------------------------------------------------------

class Table:
    """A named-column view over one row space.

    ``t["col"]`` resolves and caches the column; ``t[bool_mask]`` /
    ``t[int_array]`` / ``t[slice]`` produce a sub-view (columns still
    resolve lazily on the base table — the selection applies at access
    time, so derived columns computed on the base remain consistent).
    """

    def __init__(self, ds: "DumpDataset", sel: Optional[np.ndarray] = None):
        self._ds = ds
        self._sel = sel
        # base table owns the caches; sub-views share them
        self._base_table: "Table" = self
        self._col_cache: Dict[str, Optional[np.ndarray]] = {}
        self._derived: Dict[str, Callable[["Table"], np.ndarray]] = {}

    # -- subclass hooks -----------------------------------------------------

    def _n_rows_base(self) -> int:
        raise NotImplementedError

    def _resolve(self, name: str) -> Optional[np.ndarray]:
        """Full-length (base-space) column or None when unavailable."""
        raise NotImplementedError

    def _base_columns(self) -> List[str]:
        return []

    def _base(self) -> "Table":
        return self._base_table

    # -- public API ---------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return len(self._sel) if self._sel is not None \
            else self._n_rows_base()

    def __len__(self) -> int:
        return self.n_rows

    @property
    def columns(self) -> List[str]:
        cols = set(self._base_columns())
        cols.update(self._base_table._derived.keys())
        return sorted(cols)

    def __contains__(self, name: str) -> bool:
        return self.col(name) is not None

    def col(self, name: str) -> Optional[np.ndarray]:
        """Column ndarray (masked to this view) or None when unavailable."""
        cache = self._base_table._col_cache
        if name in cache:
            arr = cache[name]
        else:
            fn = self._base_table._derived.get(name)
            if fn is not None:
                r = fn(self._base_table)
                arr = None if r is None else np.asarray(r)
            else:
                arr = self._resolve(name)
            cache[name] = arr
        if arr is None:
            return None
        return arr if self._sel is None else arr[self._sel]

    def require(self, name: str) -> np.ndarray:
        """Column or KeyError — for paths where absence is an error."""
        arr = self.col(name)
        if arr is None:
            raise KeyError(f"column '{name}' unavailable")
        return arr

    def cols(self, names: List[str]) -> Dict[str, Optional[np.ndarray]]:
        return {n: self.col(n) for n in names}

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            arr = self.col(key)
            if arr is None:
                raise KeyError(key)
            return arr
        sel = np.asarray(key)
        if sel.dtype == bool:
            sel = np.flatnonzero(sel)
        if self._sel is not None:
            sel = self._sel[sel]  # compose into base-space indices
        return self._subview(sel)

    def _subview(self, sel: np.ndarray) -> "Table":
        sub = type(self)(self._ds, sel)
        sub._base_table = self._base_table
        return sub

    def __iter__(self) -> Iterator[int]:
        """Row *indices* — iteration yields positions, not row objects
        (column access is the fast path; rows are for convenience)."""
        return iter(range(self.n_rows))

    def row(self, i: int, cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """One row as a JSON-safe dict of scalars (trace convenience).

        Resolves every listed column — pass ``cols`` to stay cheap on
        wide dumps (default = all columns; heavy ndim>1 columns are
        included but only their i-th row leaves memory-wise what the
        column itself cost to load).
        """
        names = cols if cols is not None else self.columns
        out: Dict[str, Any] = {}
        for n in names:
            a = self.col(n)
            if a is None or i >= len(a):
                continue
            out[n] = self._ds.to_jsonable(a[i])
        return out

    def register(self, name: str, fn: Callable[["Table"], np.ndarray]) -> None:
        """Derived column: ``fn`` receives the UNMASKED base table so
        windowed/boundary-aware computations see full context."""
        self._base_table._derived[name] = fn

    def scalar(self, name: str) -> Optional[Any]:
        arr = self.col(name)
        if arr is None:
            return None
        try:
            return np.asarray(arr).item()
        except (TypeError, ValueError):
            return None

    def to_pandas(self, cols: Optional[List[str]] = None):
        """Selected columns → DataFrame (explicit; pandas optional)."""
        import pandas as pd
        return pd.DataFrame({
            n: self.col(n) for n in (cols or self.columns)})


# ---------------------------------------------------------------------------
# FrameTable — canonical trajectory-frame row space
# ---------------------------------------------------------------------------

_FRAME_FILES = ("trajectories", "buffer", "combine", "gae", "epoch_frames")

# normalized templates: public prefix → dict member in its npz
_GAE_DICT_KEYS = {
    "adv": "advs_all",
    "value": "values_all",
    "ret": "rets_all",
    "key_frame_mask": "key_frame_mask",
}
_COMBINE_DICT_KEYS = {
    "normed_adv": "normed_advs",
    "aw_normed": "aw_normed",
    "key_aw": "key_actor_weight_frame",
}
_DICT_MEMBERS = {  # file → member → public prefix (for `columns` listing)
    "gae": _GAE_DICT_KEYS,
    "combine": _COMBINE_DICT_KEYS,
}

_OBSERVER_RE = re.compile(r"^observer\.(.+?)(?:\.([^.]+))?$")
_EPOCH_RE = re.compile(r"^epoch\.(\d+)\.(.+)$")


class FrameTable(Table):
    """Trajectory-frame space (n = sum of traj_lengths).

    Resolution order for ``col(name)``:
      1. registered/derived columns
      2. verbatim member of a frame-space npz — ``reward.<ch>``,
         ``actor_weight.<ch>``, ``combined_adv``, ``frame_id``,
         ``ratio.0`` … (future dump fields appear automatically;
         size-1 object-dict members are excluded — use templates)
      3. normalized templates: ``adv.<ch>``/``value.<ch>``/``ret.<ch>``/
         ``key_frame_mask.<ch>`` (gae dicts),
         ``normed_adv.<ch>``/``aw_normed.<ch>``/``key_aw.<ch>``/
         ``contrib.<ch>`` (combine), ``epoch.<e>.{ratio,clip_mask,
         log_prob,value.<ch>}``, ``observer.<base>.<field>`` (joined
         from episodes.npz; base = observer key without _a/_b suffix)
      4. None
    """

    def _n_rows_base(self) -> int:
        tl = self._ds._traj_lengths()
        if tl is not None and len(tl):
            return int(np.asarray(tl).sum())
        fids = self._ds.npz("buffer").get("frame_id")
        if fids is not None:
            return int(len(fids))
        return 0

    # -- trajectory structure -------------------------------------------------

    def traj_slice(self, traj_idx: int) -> Tuple[int, int]:
        off = self._ds.seg_offsets
        if traj_idx < 0 or traj_idx + 1 >= len(off):
            raise IndexError(f"trajectory {traj_idx} out of range")
        return int(off[traj_idx]), int(off[traj_idx + 1])

    def traj(self, traj_idx: int) -> "FrameTable":
        s, e = self.traj_slice(traj_idx)
        return self._base()._subview(np.arange(s, e))  # type: ignore[return-value]

    def window_reduce(self, col: str, k: int, how: str = "max") -> np.ndarray:
        """Forward-looking reduce over [i, i+k) per frame, never crossing
        trajectory boundaries.  ``how``: 'max' | 'any' | 'mean'.
        Returned in this view's row space."""
        a = self._base().col(col)
        if a is None:
            raise KeyError(col)
        a = np.asarray(a, dtype=np.float64)
        out = np.full(len(a), np.nan)
        off = self._ds.seg_offsets
        for ti in range(len(off) - 1):
            s, e = int(off[ti]), int(off[ti + 1])
            seg = a[s:e]
            kk = min(k, len(seg))
            if kk <= 0:
                continue
            if how == "any":
                acc = np.isfinite(seg) & (seg != 0)
                for j in range(1, kk):
                    acc[:-j] |= np.isfinite(seg[j:]) & (seg[j:] != 0)
                out[s:e] = acc.astype(np.float64)
            else:
                acc = seg.copy()
                for j in range(1, kk):
                    if how == "mean":
                        acc[:-j] += seg[j:]
                    else:
                        acc[:-j] = np.fmax(acc[:-j], seg[j:])
                if how == "mean":
                    span = np.minimum(k, np.arange(len(seg), 0, -1))
                    acc = acc / span
                out[s:e] = acc
        return out if self._sel is None else out[self._sel]

    # -- resolution ------------------------------------------------------------

    def _resolve(self, name: str) -> Optional[np.ndarray]:
        if name.startswith("contrib."):
            return self._contrib(name[len("contrib."):])
        if name.startswith("aw."):
            name = "actor_weight." + name[3:]

        for f in _FRAME_FILES:
            arr = self._ds.npz(f).get(name)
            if arr is not None:
                if arr.dtype == object and arr.size == 1 \
                        and isinstance(_dict_item(arr), dict):
                    continue  # dict member — use normalized templates
                return arr

        prefix, _, rest = name.partition(".")
        if not rest:
            return None
        if prefix in _GAE_DICT_KEYS:
            d = self._ds.npz("gae").get_dict(_GAE_DICT_KEYS[prefix])
            v = d.get(rest) if isinstance(d, dict) else None
            return np.asarray(v) if v is not None else None
        if prefix in _COMBINE_DICT_KEYS:
            d = self._ds.npz("combine").get_dict(_COMBINE_DICT_KEYS[prefix])
            v = d.get(rest) if isinstance(d, dict) else None
            return np.asarray(v) if v is not None else None
        m = _EPOCH_RE.match(name)
        if m:
            epoch, field = int(m.group(1)), m.group(2)
            ef = self._ds.npz("epoch_frames")
            key = {
                "ratio": f"ratio.{epoch}",
                "clip_mask": f"clip_mask.{epoch}",
                "log_prob": f"new_log_prob.{epoch}",
            }.get(field)
            if key is None:
                vm = re.match(r"^value\.(.+)$", field)
                key = f"new_value.{epoch}.{vm.group(1)}" if vm else None
            return ef.get(key) if key else None
        m = _OBSERVER_RE.match(name)
        if m:
            return self._observer_frame_col(m.group(1), m.group(2))
        return None

    def _contrib(self, ch: str) -> Optional[np.ndarray]:
        """contrib.ch = aw_normed.ch × confidence.ch × normed_adv.ch —
        the per-channel summand of combined_adv (see trainer.py)."""
        awn = self._resolve(f"aw_normed.{ch}")
        nadv = self._resolve(f"normed_adv.{ch}")
        conf_d = self._ds.npz("combine").get_dict("confidences")
        if awn is None or nadv is None or not isinstance(conf_d, dict):
            return None
        conf = conf_d.get(ch)
        if conf is None:
            return None
        return np.asarray(awn, dtype=np.float64) * float(conf) \
            * np.asarray(nadv, dtype=np.float64)

    def _observer_frame_col(
        self, base: str, field: Optional[str]
    ) -> Optional[np.ndarray]:
        """Join one observer column from episodes.npz into frame space.

        For each trajectory, the agent id selects the suffixed key
        ``observer_outputs.{base}_{suffix}.{field}`` (suffix = last
        component of agent_id: robot_a → 'a'); the unsuffixed
        ``observer_outputs.{base}.{field}`` is a fallback for
        agent-agnostic observers.  Trajectories without episode mapping
        (``flat:`` frame_ids) yield NaN.  Both flat (n_ep_frames,) and
        object-array (n_episodes,) of per-episode arrays are supported.
        """
        ep = self._ds.npz("episodes")
        if not ep.available:
            return None
        prov = self._ds._traj_provenance
        suffix = field and f".{field}" or ""
        # An explicit ``_x`` tail on the observer name is an agent filter:
        # ``observer.foot_state_a.X`` resolves only on robot_a
        # trajectories (other agents' rows stay NaN).  The canonical
        # public name is unsuffixed ``observer.foot_state.X``, resolved
        # per-trajectory by that trajectory's agent.
        m2 = re.match(r"^(.*)_([a-z])$", base)
        req = m2.group(2) if m2 else None
        stem = m2.group(1) if m2 else base
        # Per-trajectory source keys
        keys: List[Optional[str]] = []
        proto: Optional[np.ndarray] = None
        ep_keys = ep.keys()
        for t in prov:
            key = None
            if t is not None:
                suff = _agent_suffix(str(t["agent_id"]))
                if req is not None:
                    cands = ([f"observer_outputs.{stem}_{req}{suffix}"]
                             if suff == req else [])
                else:
                    cands = [f"observer_outputs.{stem}_{suff}{suffix}",
                             f"observer_outputs.{stem}{suffix}"]
                for cand in cands:
                    if cand in ep_keys:
                        key = cand
                        break
            keys.append(key)
            if proto is None and key is not None:
                proto = ep.get(key)
        if proto is None:
            return None

        n = self._n_rows_base()
        flat = self._ds._ep_flat_idx()
        if proto.dtype == object:
            # object-array layout: one element per episode
            out = np.full(n, np.nan, dtype=np.float64)
            for t_idx, (t, key) in enumerate(zip(prov, keys)):
                if t is None or key is None:
                    continue
                arr = ep.get(key)
                if arr is None or arr.dtype != object \
                        or int(t["ep_pos"]) >= len(arr):
                    continue
                elem = arr[int(t["ep_pos"])]
                if elem is None:
                    continue
                elem = np.asarray(elem).reshape(-1)
                s, e = self.traj_slice(t_idx)
                L = min(e - s, len(elem) - int(t["t_start"]))
                if L > 0:
                    out[s:s + L] = elem[int(t["t_start"]):
                                        int(t["t_start"]) + L]
            return out
        # flat layout: gather by ep_flat_idx within each trajectory
        out = np.full((n,) + proto.shape[1:], np.nan, dtype=np.float64)
        for t_idx, (t, key) in enumerate(zip(prov, keys)):
            if t is None or key is None:
                continue
            arr = ep.get(key)
            if arr is None or arr.dtype == object:
                continue
            s, e = self.traj_slice(t_idx)
            idx = flat[s:e]
            ok = idx >= 0
            if ok.any():
                out[s:e][ok] = np.asarray(arr[idx[ok]], dtype=np.float64)
        return out

    def _base_columns(self) -> List[str]:
        cols: List[str] = []
        for f in _FRAME_FILES:
            src = self._ds.npz(f)
            dict_members = _DICT_MEMBERS.get(f, {})
            for k in src.keys():
                if k in dict_members.values():
                    continue  # expanded below
                cols.append(k)
            for pub, member in dict_members.items():
                d = src.get_dict(member)
                if isinstance(d, dict):
                    cols.extend(f"{pub}.{ch}" for ch in d)
        ef = self._ds.npz("epoch_frames")
        for k in ef.keys():
            m = re.match(r"^(ratio|clip_mask|new_log_prob)\.(\d+)$", k)
            if m:
                pub = {"ratio": "ratio", "clip_mask": "clip_mask",
                       "new_log_prob": "log_prob"}[m.group(1)]
                cols.append(f"epoch.{m.group(2)}.{pub}")
            vm = re.match(r"^new_value\.(\d+)\.(.+)$", k)
            if vm:
                cols.append(f"epoch.{vm.group(1)}.value.{vm.group(2)}")
        for k in self._ds.npz("episodes").keys():
            if k.startswith("observer_outputs."):
                rest = k[len("observer_outputs."):]
                # Canonical public name drops the agent suffix on the
                # observer-name component: foot_state_a.X → foot_state.X
                cols.append("observer." + re.sub(r"_[a-z](?=\.|$)", "", rest))
        # Derived per-channel contribution column (aw_normed×conf×normed_adv)
        comb = self._ds.npz("combine")
        if isinstance(comb.get_dict("confidences"), dict):
            for ch in self._ds.channel_names:
                cols.append(f"contrib.{ch}")
        return sorted(set(cols))


# ---------------------------------------------------------------------------
# TrajTable — per-trajectory scalars + provenance (row space = n_trajs)
# ---------------------------------------------------------------------------

class TrajTable(Table):
    """Per-trajectory space: provenance (ep_pos/agent_id/t_start/length/
    frame_start) + per-traj npz members (importance, is_terminated.<ch>)
    + gae per-traj dicts (key_seg_*.<ch>, bootstrap_value.<ch>)."""

    def _n_rows_base(self) -> int:
        tl = self._ds._traj_lengths()
        return len(tl) if tl is not None else 0

    def _resolve(self, name: str) -> Optional[np.ndarray]:
        ds = self._ds
        if name == "ep_pos":
            return ds._traj_ep_pos()
        if name == "agent_id":
            return ds._traj_agent_id()
        if name == "t_start":
            return ds._traj_t_start()
        if name == "length":
            return ds._traj_length()
        if name == "frame_start":
            return ds.seg_offsets[:-1]
        arr = ds.npz("trajectories").get(name)
        if arr is not None and not (
                arr.dtype == object and arr.size == 1) \
                and len(arr) == self._n_rows_base():
            return arr
        for prefix, key in (("key_seg_active", "key_seg_active"),
                            ("key_seg_terminated", "key_seg_terminated"),
                            ("bootstrap_value", "bootstrap_values")):
            if name.startswith(prefix + "."):
                d = ds.npz("gae").get_dict(key)
                v = d.get(name[len(prefix) + 1:]) if isinstance(d, dict) \
                    else None
                return np.asarray(v) if v is not None else None
        return None

    def _base_columns(self) -> List[str]:
        cols = ["ep_pos", "agent_id", "t_start", "length", "frame_start"]
        n = self._n_rows_base()
        for k in self._ds.npz("trajectories").keys():
            if k == "frame_id":
                continue
            a = self._ds.npz("trajectories").get(k)
            if a is not None and a.dtype != object and len(a) == n:
                cols.append(k)
        for member, pub in (("key_seg_active", "key_seg_active"),
                            ("key_seg_terminated", "key_seg_terminated"),
                            ("bootstrap_values", "bootstrap_value")):
            d = self._ds.npz("gae").get_dict(member)
            if isinstance(d, dict):
                cols.extend(f"{pub}.{ch}" for ch in d)
        return cols


# ---------------------------------------------------------------------------
# EpisodeTable / EpisodeView — episode space
# ---------------------------------------------------------------------------

class EpisodeTable(Table):
    """Per-episode scalars; ``ds.episodes[i]`` → EpisodeView for frames."""

    def _n_rows_base(self) -> int:
        ep = self._ds.npz("episodes")
        n = ep.get("n_episodes")
        if n is not None:
            return int(np.asarray(n).item())
        return len(self._ds.traj_map)

    def _resolve(self, name: str) -> Optional[np.ndarray]:
        ep = self._ds.npz("episodes")
        tm = self._ds.traj_map
        if name == "seed":
            a = ep.get("base_seeds")
            if a is not None:
                return np.asarray(a)
            return np.asarray([e.get("seed") for e in tm], dtype=object) \
                if tm else None
        if name == "num_frames":
            a = ep.get("num_frames")
            if a is not None:
                return np.asarray(a)
            return np.asarray(
                [e.get("num_frames") for e in tm], dtype=object) \
                if tm else None
        if name == "frame_offset":
            off = self._ds.episode_frame_offsets
            return off[:-1] if len(off) > 1 else None
        if name == "list_pos":
            return np.asarray([e.get("list_pos") for e in tm],
                              dtype=np.int64) if tm else None
        if name == "n_trajectories":
            return np.asarray(
                [len(e.get("trajectories", [])) for e in tm],
                dtype=np.int64) if tm else None
        if name == "rendered":
            return np.asarray(
                [self._ds.episode_rendered(i)
                 for i in range(self._n_rows_base())], dtype=bool)
        arr = ep.get(name)
        if arr is not None and arr.dtype != object:
            return arr
        return None

    def _base_columns(self) -> List[str]:
        return ["seed", "num_frames", "frame_offset", "list_pos",
                "n_trajectories", "rendered"]

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, (int, np.integer)):
            return EpisodeView(self._ds, int(key))
        return super().__getitem__(key)


class EpisodeView:
    """One episode's per-frame data in episode space (obs/actions/
    observer_outputs), plus provenance into the trajectory space."""

    def __init__(self, ds: "DumpDataset", pos: int):
        self._ds = ds
        self.pos = pos
        off = ds.episode_frame_offsets
        if pos < 0 or pos + 1 >= len(off):
            raise IndexError(f"episode {pos} out of range")
        self.start = int(off[pos])
        self.end = int(off[pos + 1])
        self.n_frames = self.end - self.start

    @property
    def seed(self) -> Optional[int]:
        s = self._ds.episodes.col("seed")
        if s is None or self.pos >= len(s):
            return None
        v = s[self.pos]
        return int(v) if v is not None else None

    def col(
        self, name: str, frame: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """``obs.<aid>``/``actions.<aid>``/``explore_factors.<aid>``/
        ``observer.<okey>.<field>`` → (T, ...) slice; ``frame`` picks one."""
        ep = self._ds.npz("episodes")
        if name.startswith("observer."):
            arr = ep.get(f"observer_outputs.{name[len('observer.'):]}")
            if arr is None:
                return None
            if arr.dtype == object:
                # object-array layout: element per episode
                if self.pos >= len(arr) or arr[self.pos] is None:
                    return None
                out = np.asarray(arr[self.pos])
            else:
                out = arr[self.start:self.end]
            return out if frame is None else out[frame]
        arr = ep.get(name)
        if arr is None or arr.dtype == object:
            return None
        out = arr[self.start:self.end]
        return out if frame is None else out[frame]

    @property
    def observer_columns(self) -> List[str]:
        return [
            "observer." + k[len("observer_outputs."):]
            for k in self._ds.npz("episodes").keys()
            if k.startswith("observer_outputs.")
        ]

    @property
    def trajectories(self) -> List[Dict[str, Any]]:
        return self._ds.trajs_of_episode(self.pos)

    @property
    def termination(self) -> Optional[Any]:
        arr = self._ds.npz("episodes").get("_termination_records")
        if arr is None or arr.size == 0:
            return None
        try:
            raw = arr.item() if (arr.dtype == object or arr.ndim == 0) \
                else arr[0]
            data = json.loads(str(raw))
            return data.get(f"ep{self.pos:04d}")
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            return None


# ---------------------------------------------------------------------------
# TimelineTable / GradsigTable — their own row spaces
# ---------------------------------------------------------------------------

class TimelineTable(Table):
    """Per-minibatch-step space: verbatim members + per-channel dict
    expansion (``critic_loss.<ch>``, ``critic_grad.<ch>``)."""

    def _n_rows_base(self) -> int:
        tl = self._ds.npz("timeline")
        n = tl.get("n_steps")
        if n is not None:
            return int(np.asarray(n).item())
        for k in ("kl_mean", "kl"):
            a = tl.get(k)
            if a is not None:
                return int(len(a))
        return 0

    def _resolve(self, name: str) -> Optional[np.ndarray]:
        tl = self._ds.npz("timeline")
        arr = tl.get(name)
        if arr is not None and not (
                arr.dtype == object and arr.size == 1):
            return arr
        for dk in ("critic_loss", "critic_grad"):
            if name.startswith(dk + "."):
                d = tl.get_dict(dk)
                v = d.get(name[len(dk) + 1:]) if isinstance(d, dict) \
                    else None
                return np.asarray(v) if v is not None else None
        return None

    def _base_columns(self) -> List[str]:
        tl = self._ds.npz("timeline")
        cols: List[str] = []
        for k in tl.keys():
            if k in ("critic_loss", "critic_grad"):
                d = tl.get_dict(k)
                if isinstance(d, dict):
                    cols.extend(f"{k}.{ch}" for ch in d)
                continue
            cols.append(k)
        return cols


class GradsigTable(Table):
    """Sampled gradient frames (≤2000 rows) with join columns back into
    the canonical frame space: ``frame_idx``, ``frame_id``, ``traj_idx``,
    ``traj_frame``, ``episode``, ``agent``, ``env_frame``."""

    _JOIN_COLS = ("frame_idx", "frame_id", "traj_idx", "traj_frame",
                  "episode", "agent", "env_frame")

    def _n_rows_base(self) -> int:
        g = self._ds.npz("gradsig")
        s = g.get("sampled_idx")
        if s is not None:
            return int(len(s))
        v = g.get("valid")
        return int(len(v)) if v is not None else 0

    def _resolve(self, name: str) -> Optional[np.ndarray]:
        if name in self._JOIN_COLS:
            return self._join_col(name)
        arr = self._ds.npz("gradsig").get(name)
        if arr is not None and arr.dtype != object:
            return arr
        return None

    def _join_col(self, name: str) -> Optional[np.ndarray]:
        sel = self._ds.npz("gradsig").get("sampled_idx")
        if sel is None:
            return None
        sel = np.asarray(sel, dtype=np.int64)
        n = len(sel)
        if name == "frame_idx":
            return sel
        ds = self._ds
        if name == "frame_id":
            fids = ds.frames.col("frame_id")
            if fids is None:
                return None
            return np.asarray(
                [str(fids[i]) if 0 <= i < len(fids) else ""
                 for i in sel], dtype=object)
        if name == "agent":
            out_obj = np.empty(n, dtype=object)
            out_obj[:] = ""
            for j, i in enumerate(sel):
                loc = ds.frame_provenance(int(i))
                out_obj[j] = loc.get("agent") or ""
            return out_obj
        out = np.full(n, -1, dtype=np.int64)
        key = {"traj_idx": "traj_idx", "traj_frame": "traj_frame",
               "episode": "episode", "env_frame": "env_frame"}[name]
        for j, i in enumerate(sel):
            loc = ds.frame_provenance(int(i))
            v = loc.get(key)
            if v is not None:
                out[j] = int(v)
        return out

    def _base_columns(self) -> List[str]:
        g = self._ds.npz("gradsig")
        cols = [k for k in g.keys()
                if not k.startswith("hist") ]
        cols += list(self._JOIN_COLS)
        return cols


# ---------------------------------------------------------------------------
# DumpDataset — root object
# ---------------------------------------------------------------------------

class DumpDataset:
    """Read entry for one ``dumps/uNNNNN/`` directory.

    Usage::

        ds = DumpDataset(dump_dir)
        f = ds.frames                        # canonical traj-frame table
        m = f["observer.foot_state.h_left_foot"] > 0.03
        f[m]["contrib.r_potential"].mean()
    """

    def __init__(self, dump_dir: Any):
        self.dump_dir = Path(dump_dir).resolve()
        if not self.dump_dir.is_dir():
            raise NotADirectoryError(f"dump dir not found: {self.dump_dir}")
        self._npz: Dict[str, NpzSource] = {}
        self._json: Dict[str, Any] = {}
        self._frames: Optional[FrameTable] = None
        self._trajs: Optional[TrajTable] = None
        self._episodes: Optional[EpisodeTable] = None
        self._timeline: Optional[TimelineTable] = None
        self._gradsig: Optional[GradsigTable] = None
        self._seg_offsets: Optional[np.ndarray] = None
        self._traj_prov: Optional[List[Optional[Dict[str, Any]]]] = None
        self._ep_flat: Optional[np.ndarray] = None
        self._ep_off: Optional[np.ndarray] = None

    def __enter__(self) -> "DumpDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- sources ---------------------------------------------------------------

    def npz(self, name: str) -> NpzSource:
        """Member-level lazy reader for ``<name>.npz`` (always returned;
        check ``.available``)."""
        if name not in self._npz:
            self._npz[name] = NpzSource(self.dump_dir / f"{name}.npz")
        return self._npz[name]

    def has(self, name: str) -> bool:
        return self.npz(name).available

    def json(self, name: str) -> Optional[Any]:
        if name not in self._json:
            p = self.dump_dir / f"{name}.json"
            obj = None
            if p.is_file():
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    obj = None
            self._json[name] = obj
        return self._json[name]

    def close(self) -> None:
        for s in self._npz.values():
            s.close()

    def clear_cache(self) -> None:
        """Release all decoded columns (sources stay open)."""
        for s in self._npz.values():
            s.clear_cache()
        for t in (self._frames, self._trajs, self._episodes,
                  self._timeline, self._gradsig):
            if t is not None:
                t._base_table._col_cache.clear()

    # -- metadata ------------------------------------------------------------

    @property
    def manifest(self) -> Dict[str, Any]:
        m = self.json("manifest")
        if m is None:
            raise FileNotFoundError(
                f"manifest.json not found in {self.dump_dir}")
        return m

    @property
    def traj_map(self) -> List[Dict[str, Any]]:
        tm = self.json("traj_map")
        if tm is not None:
            return tm
        fids = self.npz("trajectories").get("frame_id")
        if fids is None:
            fids = self.npz("buffer").get("frame_id")
        tl = self._traj_lengths()
        if fids is None or tl is None or len(tl) == 0:
            return []
        tm = _build_traj_map_from_frame_ids(fids, tl)
        self._json["traj_map"] = tm
        return tm

    @property
    def channel_names(self) -> List[str]:
        cn = self.npz("trajectories").get("channel_names")
        if cn is not None:
            return [str(x) for x in cn.tolist()]
        names: set = set()
        for dk in ("advs_all", "values_all", "rets_all"):
            d = self.npz("gae").get_dict(dk)
            if isinstance(d, dict):
                names.update(map(str, d.keys()))
        for dk in ("normed_advs", "aw_normed", "key_actor_weight_frame",
                   "confidences"):
            d = self.npz("combine").get_dict(dk)
            if isinstance(d, dict):
                names.update(map(str, d.keys()))
        return sorted(names)

    @property
    def agent_ids(self) -> List[str]:
        ids = {k[len("obs."):] for k in self.npz("episodes").keys()
               if k.startswith("obs.")}
        if not ids:
            ids = {t["agent_id"] for e in self.traj_map
                   for t in e.get("trajectories", [])
                   if t.get("agent_id")}
        return sorted(ids)

    @property
    def observer_keys(self) -> List[str]:
        keys = set()
        for k in self.npz("episodes").keys():
            if k.startswith("observer_outputs."):
                keys.add(k[len("observer_outputs."):].split(".")[0])
        return sorted(keys)

    @property
    def n_trajectories(self) -> int:
        tl = self._traj_lengths()
        return len(tl) if tl is not None else 0

    @property
    def n_episodes(self) -> int:
        return self.episodes.n_rows

    @property
    def n_frames(self) -> int:
        return self.frames.n_rows

    @property
    def n_epochs(self) -> int:
        n = self.npz("epoch_frames").get("n_epochs")
        return int(np.asarray(n).item()) if n is not None else 0

    # -- tables ----------------------------------------------------------------

    @property
    def frames(self) -> FrameTable:
        if self._frames is None:
            self._frames = FrameTable(self)
        return self._frames

    @property
    def trajs(self) -> TrajTable:
        if self._trajs is None:
            self._trajs = TrajTable(self)
        return self._trajs

    @property
    def episodes(self) -> EpisodeTable:
        if self._episodes is None:
            self._episodes = EpisodeTable(self)
        return self._episodes

    @property
    def timeline(self) -> TimelineTable:
        if self._timeline is None:
            self._timeline = TimelineTable(self)
        return self._timeline

    @property
    def gradsig(self) -> GradsigTable:
        if self._gradsig is None:
            self._gradsig = GradsigTable(self)
        return self._gradsig

    @property
    def update(self) -> NpzSource:
        """update.npz scalar access: ``ds.update.scalar("kl_mean")``."""
        return self.npz("update")

    def capabilities(self) -> Dict[str, Any]:
        """Per-file availability map (inspect endpoint)."""
        caps = {f: self.has(f) for f in (
            "episodes", "trajectories", "buffer", "gae", "combine",
            "update", "timeline", "epoch_frames")}
        g = self.npz("gradsig")
        caps["gradsig"] = (
            False if not g.available
            else ("full" if "hist" in g.keys() else "partial"))
        return caps

    # -- index structures ------------------------------------------------------

    def _traj_lengths(self) -> Optional[np.ndarray]:
        for f in ("trajectories", "buffer"):
            tl = self.npz(f).get("traj_lengths")
            if tl is None:
                tl = self.npz(f).get("ep_lengths")  # older dumps
            if tl is not None:
                return np.asarray(tl, dtype=np.int64)
        return None

    @property
    def seg_offsets(self) -> np.ndarray:
        """Cumulative trajectory offsets into the frame row space."""
        if self._seg_offsets is None:
            tl = self._traj_lengths()
            off = np.zeros(len(tl) + 1 if tl is not None else 1,
                           dtype=np.int64)
            if tl is not None and len(tl):
                off[1:] = np.cumsum(tl)
            self._seg_offsets = off
        return self._seg_offsets

    @property
    def episode_frame_offsets(self) -> np.ndarray:
        if self._ep_off is None:
            off = self.npz("episodes").get("episode_frame_offsets")
            if off is not None:
                self._ep_off = np.asarray(off, dtype=np.int64)
            else:
                lens = [e.get("num_frames") or 0 for e in self.traj_map]
                out = np.zeros(len(lens) + 1, dtype=np.int64)
                out[1:] = np.cumsum(lens)
                self._ep_off = out
        return self._ep_off

    @property
    def _traj_provenance(self) -> List[Optional[Dict[str, Any]]]:
        """traj_idx → {ep_pos, agent_id, t_start, length} or None."""
        if self._traj_prov is None:
            n = self.n_trajectories
            prov: List[Optional[Dict[str, Any]]] = [None] * n
            for e in self.traj_map:
                for t in e.get("trajectories", []):
                    ti = t.get("traj_idx")
                    if ti is not None and 0 <= ti < n:
                        prov[ti] = {
                            "ep_pos": e.get("list_pos"),
                            "agent_id": t.get("agent_id"),
                            "t_start": int(t.get("t_start", 0)),
                            "length": int(t.get("length", 0)),
                        }
            self._traj_prov = prov
        return self._traj_prov

    def _traj_ep_pos(self) -> np.ndarray:
        return np.asarray(
            [p["ep_pos"] if p is not None and p["ep_pos"] is not None
             else -1 for p in self._traj_provenance], dtype=np.int64)

    def _traj_agent_id(self) -> np.ndarray:
        return np.asarray(
            [p["agent_id"] if p is not None else ""
             for p in self._traj_provenance], dtype=object)

    def _traj_t_start(self) -> np.ndarray:
        return np.asarray(
            [p["t_start"] if p is not None else 0
             for p in self._traj_provenance], dtype=np.int64)

    def _traj_length(self) -> np.ndarray:
        tl = self._traj_lengths()
        if tl is not None:
            return tl
        return np.asarray(
            [p["length"] if p is not None else 0
             for p in self._traj_provenance], dtype=np.int64)

    def _ep_flat_idx(self) -> np.ndarray:
        """frame row → flat episode-frame index (for flat observer
        arrays); -1 where the trajectory has no episode mapping."""
        if self._ep_flat is None:
            ep_off = self.episode_frame_offsets
            n = self.frames._n_rows_base()
            out = np.full(n, -1, dtype=np.int64)
            for t_idx, p in enumerate(self._traj_provenance):
                if p is None or p["ep_pos"] is None \
                        or p["ep_pos"] + 1 >= len(ep_off):
                    continue
                s, e = self.frames.traj_slice(t_idx)
                base = int(ep_off[int(p["ep_pos"])]) + int(p["t_start"])
                out[s:e] = base + np.arange(e - s)
            self._ep_flat = out
        return self._ep_flat

    # -- provenance API ----------------------------------------------------------

    def provenance(self, traj_idx: int) -> Optional[Dict[str, Any]]:
        p = self._traj_provenance
        return p[traj_idx] if 0 <= traj_idx < len(p) else None

    def trajs_of_episode(self, ep_pos: int) -> List[Dict[str, Any]]:
        if ep_pos < 0 or ep_pos >= len(self.traj_map):
            return []
        return list(self.traj_map[ep_pos].get("trajectories", []))

    def frame_traj(self, frame_idx: int) -> Tuple[Optional[int], Optional[int]]:
        """frame row → (traj_idx, frame offset within trajectory)."""
        off = self.seg_offsets
        if frame_idx < 0 or len(off) < 2 or frame_idx >= int(off[-1]):
            return None, None
        t = int(np.searchsorted(off, frame_idx, side="right") - 1)
        return t, int(frame_idx - off[t])

    def frame_provenance(self, frame_idx: int) -> Dict[str, Any]:
        """Full location record for one frame row."""
        loc: Dict[str, Any] = {"buffer_idx": frame_idx, "mapped": False}
        fids = self.frames.col("frame_id")
        n = self.frames._n_rows_base()
        if fids is None or frame_idx < 0 or frame_idx >= n:
            loc["reason"] = \
                "frame_id unavailable or index out of range"
            return loc
        traj_idx, traj_frame = self.frame_traj(frame_idx)
        if traj_idx is not None:
            loc["traj_idx"] = traj_idx
            loc["traj_frame"] = traj_frame
        parsed = parse_frame_id(fids[frame_idx])
        if parsed is None:
            loc["reason"] = \
                "frame_id is flat:* or unparsable — not an env episode frame"
            return loc
        loc.update({
            "mapped": True,
            "episode": parsed["episode"],
            "agent": parsed["agent"],
            "env_frame": parsed["env_frame"],
        })
        return loc

    # -- derived artifacts (record/ delta/) --------------------------------------

    @property
    def has_images(self) -> bool:
        rd = self.dump_dir / "record"
        return rd.is_dir() and (rd / "index.json").exists()

    def episode_rendered(self, episode_pos: int) -> bool:
        ep_dir = self.dump_dir / "record" / f"episode_{episode_pos:05d}"
        if not ep_dir.is_dir():
            return False
        try:
            next(ep_dir.glob("step_*.png"))
            return True
        except StopIteration:
            return False

    def image_path(self, episode_pos: int, frame: int) -> Optional[Path]:
        """PNG for (episode, frame): recorded step N+1 = dump frame N."""
        ep_dir = self.dump_dir / "record" / f"episode_{episode_pos:05d}"
        if not ep_dir.is_dir():
            return None
        png = ep_dir / f"step_{frame + 1:05d}.png"
        return png if png.exists() else None

    def delta_dir(self, episode_pos: int) -> Path:
        return self.dump_dir / "delta" / f"episode_{episode_pos:05d}"

    # -- misc -----------------------------------------------------------------

    @staticmethod
    def to_jsonable(v: Any) -> Any:
        """ndarray/scalar → JSON-safe; non-finite floats → None."""
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                return DumpDataset.to_jsonable(v.item())
            if v.dtype == object:
                return [str(x) for x in v.tolist()]
            lst = v.tolist()
            if np.issubdtype(v.dtype, np.floating):
                return [x if (x is not None and math.isfinite(x)) else None
                        for x in lst]
            return lst
        if isinstance(v, (np.floating, float)):
            return float(v) if math.isfinite(float(v)) else None
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v
