"""S5 + S2 debug infrastructure.

S5: behavior probes + metric verifiers (``DEBUG_GUIDE.md`` §3.6/§3.7,
``DESIGN_debug_system.md`` §5.2/§5.3):
- :mod:`probes` — run behavior probe suites on deterministic rollouts
- :mod:`metrics` — verify metric definitions with strict alternatives

S5 does **not** depend on S1–S4 (provenance, sink, snapshots, chain
reports).  It answers ①环 (did the behavior ever happen?) and ⑨环
(is the metric measuring what I think?).

S2: DebugSink + snapshot + replay (``DEBUG_GUIDE.md`` §3.11,
``DESIGN_debug_system.md`` §3.2/§4):
- :mod:`sink` — write-only per-frame data capture (DebugSink + NpzSink)
- :mod:`snapshot` — sentinel-file trigger + on-disk snapshot writer
- :mod:`replay` — offline re-run of ppo_update with self-verification

S2 depends on S1 (provenance) for frame-level data alignment.
"""
