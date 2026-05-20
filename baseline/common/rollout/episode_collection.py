"""``EpisodeCollection``: in-memory container for many :class:`Episode`s.

See ``baseline/common/rollout/DESIGN.md`` §2.2 / §2.3 for the schema and
on-disk layout. This module owns the **collection-level** invariants:

* All episodes share the same blueprint (verified via blueprint hash).
* save/load uses the directory layout::

      <path>/
          collection.json
          blueprint.yaml
          episodes/
              episode_00000.npz
              episode_00000.json
              ...
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence

import numpy as np

from envs.framework.blueprint import EnvBlueprint

from .episode import Episode, blueprint_hash


COLLECTION_FORMAT_VERSION = 1


class EpisodeCollection:
    """Sequence of :class:`Episode`s sharing one :class:`EnvBlueprint`.

    The blueprint is the only invariant: every episode appended must
    have a matching ``blueprint_hash``. Anything else (varying lengths,
    different agents, different terminations) is allowed and is the
    consumer's problem.
    """

    def __init__(
        self,
        blueprint: EnvBlueprint,
        episodes: Sequence[Episode] = (),
    ) -> None:
        self._blueprint = blueprint
        self._blueprint_hash = blueprint_hash(blueprint)
        self._episodes: List[Episode] = []
        for episode in episodes:
            self.append(episode)

    # ------------------------------------------------------------------
    # Container interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> Episode:
        return self._episodes[index]

    def __iter__(self) -> Iterator[Episode]:
        return iter(self._episodes)

    def append(self, episode: Episode) -> None:
        if episode.blueprint_hash != self._blueprint_hash:
            raise ValueError(
                f"episode.blueprint_hash {episode.blueprint_hash[:12]}... "
                f"does not match collection blueprint hash "
                f"{self._blueprint_hash[:12]}..."
            )
        self._episodes.append(episode)

    def extend(self, episodes: Iterable[Episode]) -> None:
        for episode in episodes:
            self.append(episode)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @property
    def blueprint(self) -> EnvBlueprint:
        return self._blueprint

    @property
    def blueprint_hash(self) -> str:
        return self._blueprint_hash

    @property
    def total_frames(self) -> int:
        return sum(int(episode.num_frames) for episode in self._episodes)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def stack_field(
        self,
        getter: Callable[[Episode], np.ndarray],
        axis: int = 0,
    ) -> np.ndarray:
        """Concatenate ``getter(episode)`` across episodes.

        Useful for e.g. ``coll.stack_field(lambda e: e.observations["robot_a"])``
        to get ``(sum_T, obs_dim)``. Empty collections return ``np.empty((0,))``.
        """
        if not self._episodes:
            return np.empty((0,))
        parts = [np.asarray(getter(episode)) for episode in self._episodes]
        return np.concatenate(parts, axis=axis)

    def split_by_termination(self) -> tuple["EpisodeCollection", "EpisodeCollection"]:
        """Return ``(terminated, truncated)`` collections."""
        terminated = EpisodeCollection(self._blueprint)
        truncated = EpisodeCollection(self._blueprint)
        for episode in self._episodes:
            (terminated if episode.is_terminated else truncated).append(episode)
        return terminated, truncated

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Any) -> None:
        """Write to ``<path>/`` per the layout in this module's docstring.

        The directory is created if missing. Existing files are overwritten;
        existing extra files are NOT deleted, which makes resumed / appended
        runs harmless but means manual cleanup is the caller's job.
        """
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        episodes_dir = root / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        # Blueprint.
        self._blueprint.save(root / "blueprint.yaml")

        # Episodes.
        for index, episode in enumerate(self._episodes):
            stem = episodes_dir / f"episode_{index:05d}"
            episode.save(stem)

        # Top-level metadata last (so partial writes leave the dir incomplete).
        metadata = {
            "format_version": COLLECTION_FORMAT_VERSION,
            "blueprint_hash": self._blueprint_hash,
            "num_episodes": len(self._episodes),
            "total_frames": self.total_frames,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "episodes": [
                {
                    "index": index,
                    "stem": f"episodes/episode_{index:05d}",
                    "num_frames": int(self._episodes[index].num_frames),
                    "base_seed": int(self._episodes[index].base_seed),
                }
                for index in range(len(self._episodes))
            ],
        }
        (root / "collection.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Any) -> "EpisodeCollection":
        root = Path(path)
        meta_path = root / "collection.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found; not an EpisodeCollection directory"
            )
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(metadata.get("format_version", 0)) != COLLECTION_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported collection format_version "
                f"{metadata.get('format_version')}; expected "
                f"{COLLECTION_FORMAT_VERSION}"
            )

        blueprint = EnvBlueprint.load(root / "blueprint.yaml")
        expected_hash = metadata["blueprint_hash"]
        actual_hash = blueprint_hash(blueprint)
        if actual_hash != expected_hash:
            raise ValueError(
                f"blueprint.yaml in {root} hashes to {actual_hash[:12]}... "
                f"but collection.json says {expected_hash[:12]}..."
            )

        collection = cls(blueprint)
        for entry in metadata.get("episodes", []):
            stem = root / entry["stem"]
            collection.append(Episode.load(stem))
        return collection


__all__ = ["EpisodeCollection", "COLLECTION_FORMAT_VERSION"]
