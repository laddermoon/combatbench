"""S5 debug infrastructure: behavior probes + metric verifiers.

This package implements the S5 stage of the debug system
(``DEBUG_GUIDE.md`` §3.6/§3.7, ``DESIGN_debug_system.md`` §5.2/§5.3):

- :mod:`probes` — run behavior probe suites on deterministic rollouts
- :mod:`metrics` — verify metric definitions with strict alternatives

S5 does **not** depend on S1–S4 (provenance, sink, snapshots, chain
reports).  It answers ①环 (did the behavior ever happen?) and ⑨环
(is the metric measuring what I think?).
"""
