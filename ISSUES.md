# Known Issues

## ISSUE-001: mujoco.Renderer EGL context leak on simulator without close()

**Status:** Partially fixed (close() added to Humanoid21Simulator; underlying
BaseSimulator.close() still a no-op).

**Severity:** Medium — affects long-running batch workloads (hundreds of
sequential rounds/matches), not single-round runs.

**Description:**

`Humanoid21Simulator.get_broadcastview_image()` lazily creates a
`mujoco.Renderer` and caches it on `self._renderer` for reuse across frames
(perf optimization). The `Renderer` holds an EGL context and GPU framebuffer.

`BaseSimulator.close()` (in `envs/framework/backend.py`) is a no-op (`pass`).
`_RuntimeCore.close()` calls `self.simulator.close()`, and `RoundRunner.close()`
calls `runtime.close()`, so the call chain exists — but before the fix
`Humanoid21Simulator` did not override `close()`, so the cached Renderer was
never explicitly released. It relied on Python GC `__del__` for cleanup.

In batch scenarios (e.g. `MatchRunner` running many rounds in one process),
GC may not run between rounds, causing EGL contexts and GPU memory to
accumulate. On the 8-GPU server this can exhaust EGL contexts or VRAM after
~50-100 rounds depending on frame resolution.

**Fix applied:**

`Humanoid21Simulator.close()` now sets `self._renderer = None` to release
the Renderer reference promptly.

**Remaining work:**

- `BaseSimulator.close()` is still a no-op. Other simulator backends
  (e.g. T800) may have similar cached GPU resources. Consider making
  `close()` abstract or adding a default resource-tracking mechanism.
- `mujoco.Renderer` does not expose an explicit `close()`/`free()` method
  in the Python bindings (as of mujoco 3.5.0). Release relies on `__del__`.
  Setting the reference to `None` is the best available approach. If future
  mujoco versions add an explicit close method, use it instead.
