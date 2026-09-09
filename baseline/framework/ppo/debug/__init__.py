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

S3: chain / attribute / frame (``DEBUG_GUIDE.md`` §3.2/§3.3/§3.4,
``DESIGN_debug_system.md`` §7):
- :mod:`where` — ``--where`` expression parser for frame filtering
- :mod:`attribute` — update attribution from training log S0 aggregates
- :mod:`chain` — nine-ring signal chain profile (log + snapshot + probe)
- :mod:`frame` — frame-level inspector (``--id`` + ``--where``)

S3 depends on S1 (provenance), S2 (snapshot/replay), S0 (aggregates),
and S5 (probes).  It is a pure consumer of existing data — no new
training-time instrumentation.

S6: intervene-check / compare / noise / timeline
(``DEBUG_GUIDE.md`` §3.8–§3.10, ``DESIGN_debug_system.md`` §5.4):
- :mod:`knobs` — KnobCheck registry + 5 built-in knob checks
- :mod:`intervene` — verify configured knobs entered the data pathway
- :mod:`compare` — cross-run comparison with optional noise band
- :mod:`noise` — launch multi-seed training to establish noise baseline
- :mod:`timeline` — event timeline with metric sparklines

S6 has no dependencies on S1–S5 (but reuses log parsing patterns from
S3's ``attribute`` module).  ``intervene-check`` optionally uses S2
snapshots for observer/per-frame knobs.
"""
