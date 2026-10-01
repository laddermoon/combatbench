"""Centralized batched inference server (Unix domain socket).

This module is the GPU half of the optional remote-rollout path.  When
``ParallelRollouter`` is created with ``rollout_inference="gpu"`` it spawns
one :func:`serve` process per training run; rollout workers wrap their
policies in :class:`~baseline.framework.rollout.remote_policy.
RemoteSamplingPolicy`, which forwards every ``act()`` to this server.

Design contract
---------------

1. **Whole-semantics remote.**  The server owns the complete
   ``SamplingPolicy`` computation: reference-policy ensemble, delta-floor
   context, and stochastic sampling.  Workers send only ``obs`` +
   per-request noise + the resolved ``explore_factor``; they receive the
   action, the log_prob, and the reference action (for ``sctx__`` extras).

2. **Deterministic under fixed shape.**  Forward batches are padded to a
   fixed capacity so cuBLAS kernel selection — and therefore the exact
   floating-point reduction order — is invariant to request count and
   arrival order.  Combined with per-request injected noise
   (``uniform=`` kwarg on the exported ``sample_action``) each reply is a
   pure function of (weights, obs, noise) — GPU↔GPU bit-identical
   regardless of batching.  CPU↔GPU stays merely ~1e-7-equivalent
   (different math libraries); the CPU path remains the golden
   reference.

3. **Spec registry.**  Workers REGISTER a ``(policy blueprint, sampling
   spec)`` pair once per wrapper construction; the server caches the
   built networks under a content hash (``spec_id``) and subsequent
   registrations are free.  Registrations are lazy — the server needs no
   knowledge of the experiment layout.

Wire protocol (little-endian, 4-byte length-prefixed frames)
-------------------------------------------------------------

Frame = ``u32 payload_len`` + ``payload``; ``payload[0]`` is the type byte.

- ``b"R"`` register — payload ``b"R" + pickle({policy_bp, spec})`` → reply
  ``b"r" + pickle({spec_id, obs_dim, action_dim, supports_uniform,
  supports_delta})``.  On failure: ``b"r" + pickle({"error": str})``.
- ``b"A"`` act — payload
  ``b"A" + <qdBHH>(spec_id, ef, want_extra, obs_len, noise_len) + obs + noise``
  → reply ``b"a" + <dBB>(log_prob, has_ref, has_delta) + action
  + ref_action + delta``.  ``ref_action`` is present iff ``has_ref``
  (suppressed under ``delta_mode="frozen"`` — the recorded payload is
  Δ, not a_ref); ``delta`` iff ``has_delta`` (frozen mode only; the
  action-level ``det_action(current) − a_ref``, ``(D,)`` flat).
- ``b"P"`` ping — reply ``b"p"`` (readiness probe).
- ``b"X"`` shutdown — server exits its loop.
"""
from __future__ import annotations

import hashlib
import inspect
import logging
import os
import pickle
import selectors
import socket
import struct
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

_ACT_HDR = struct.Struct("<QdBHH")
_ACT_REPLY_HDR = struct.Struct("<dBB")

#: Default padded batch capacity.  Every forward is executed at exactly
#: this many rows (dead rows carry zeroed obs / midpoint noise) so the
#: chosen GEMM kernels — and hence the bit-level reduction order — never
#: depend on how many requests happened to arrive together.  Bigger than
#: needed wastes padding compute (the K reference nets all forward the
#: padded block); smaller than the in-flight population just splits into
#: more chunks — determinism is unaffected either way.
DEFAULT_CAPACITY = 128

#: Max cached specs on the server.  Each spec keeps an inner net plus its
#: reference ensemble resident in GPU memory; eviction is oldest-first.
_MAX_SPECS = 256

#: Set False once any CUDA-graph capture fails — a capture error can
#: leave the process-wide stream-capture state poisoned (subsequent
#: ``torch.cuda.graph`` entries crash inside ``empty_cache``), so the
#: whole server falls back to the eager path permanently.
_CAPTURE_OK = True


def _spec_id_of(policy_bp_dict: Dict[str, Any], spec_dict: Dict[str, Any]) -> int:
    """Content-hash of the registration payload → stable int64 spec id."""
    blob = pickle.dumps((policy_bp_dict, spec_dict))
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


# ---------------------------------------------------------------------------
# Frame helpers (shared with remote_policy.py)
# ---------------------------------------------------------------------------
def _send_frame(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack("<I", len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("inference server closed the connection")
        buf += chunk
    return bytes(buf)


def _recv_frame(sock: socket.socket) -> bytes:
    (n,) = struct.unpack("<I", _recv_exact(sock, 4))
    return _recv_exact(sock, n)


def send_register(
    sock: socket.socket,
    policy_bp_dict: Dict[str, Any],
    spec_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Client-side: REGISTER round-trip, returns the server reply dict."""
    _send_frame(sock, b"R" + pickle.dumps({
        "policy_bp": policy_bp_dict, "spec": spec_dict,
    }))
    reply = _recv_frame(sock)
    if not reply or reply[0:1] != b"r":
        raise RuntimeError("inference server: malformed register reply")
    data = pickle.loads(reply[1:])
    if "error" in data:
        raise RuntimeError(f"inference server rejected spec: {data['error']}")
    return data


def send_act(
    sock: socket.socket,
    spec_id: int,
    ef: float,
    want_extra: bool,
    obs: np.ndarray,
    noise: np.ndarray,
) -> Tuple[np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Client-side ACT round-trip → (action, log_prob, ref|None, delta|None)."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    noise = np.asarray(noise, dtype=np.float32).reshape(-1)
    hdr = _ACT_HDR.pack(spec_id, float(ef), int(bool(want_extra)),
                      len(obs), len(noise))
    _send_frame(sock, b"A" + hdr + obs.tobytes() + noise.tobytes())
    reply = _recv_frame(sock)
    if reply and reply[0:1] == b"!":
        # Server-side forward failure — fail fast instead of blocking
        # the rollout on the socket timeout.
        raise RuntimeError(f"inference server: {pickle.loads(reply[1:])}")
    if not reply or reply[0:1] != b"a":
        raise RuntimeError("inference server: malformed act reply")
    log_prob, has_ref, has_delta = _ACT_REPLY_HDR.unpack(
        reply[1:1 + _ACT_REPLY_HDR.size])
    rest = reply[1 + _ACT_REPLY_HDR.size:]
    d = len(noise) - 1  # action_dim
    action = np.frombuffer(rest[: d * 4], dtype=np.float32).copy()
    pos = d * 4
    ref = None
    if has_ref:
        ref = np.frombuffer(rest[pos: pos + d * 4], dtype=np.float32).copy()
        pos += d * 4
    delta = None
    if has_delta:
        delta = np.frombuffer(rest[pos:], dtype=np.float32).copy()
        if delta.size > d:
            delta = delta.reshape(-1, d)  # per-component (K, D)
    return action, float(log_prob), ref, delta


# ---------------------------------------------------------------------------
# Server internals
# ---------------------------------------------------------------------------
class _SpecSession:
    """Server-side state for one registered (policy, spec) pair."""

    __slots__ = (
        "spec_id", "inner_net", "inner_policy", "refs", "delta_factor",
        "delta_frozen", "obs_dim", "action_dim", "uses_ctx",
        "supports_uniform", "supports_delta", "supports_delta_out",
        "_ref_stack", "_ref_weights",
        "_graph", "_g_obs", "_g_noise", "_g_ef", "_g_out", "_graph_done",
    )

    def __init__(
        self, spec_id: int, policy_bp_dict: Dict[str, Any],
        spec_dict: Dict[str, Any], device: Any,
    ) -> None:
        import torch  # local import — server process only

        from envs.framework.policy import PolicyBlueprint

        self.spec_id = spec_id
        policy = PolicyBlueprint.from_dict(policy_bp_dict).build()
        net = getattr(policy, "_policy", None)
        if net is None or not callable(getattr(net, "sample_action", None)):
            raise TypeError(
                f"remote inference requires an exported policy exposing "
                f"`._policy.sample_action`; got "
                f"{policy_bp_dict.get('cls')!r} → {type(policy).__name__}"
            )
        self.inner_policy = policy
        self.inner_net = net.to(device).eval()
        self.obs_dim = int(getattr(net, "obs_dim"))
        self.action_dim = int(getattr(net, "action_dim"))
        self.supports_delta = bool(
            getattr(policy, "SUPPORTS_REFERENCE_DELTA", False)
        )

        sig = inspect.signature(net.sample_action)
        self.uses_ctx = "ctx" in sig.parameters
        self.supports_uniform = "uniform" in sig.parameters
        # Frozen-delta needs the inner net's deterministic action to
        # compute Δ = det_action(current policy) − a_ref server-side.
        self.supports_delta_out = (
            self.uses_ctx
            and callable(getattr(net, "deterministic_action", None))
        )
        if not self.uses_ctx and "explore_factor" not in sig.parameters:
            raise TypeError(
                f"{type(net).__name__}.sample_action accepts neither "
                f"`ctx` nor `explore_factor` — unsupported signature"
            )

        ref = spec_dict.get("reference")
        self.refs: List[Tuple[float, Any]] = []
        if ref:
            for w, bp_d in zip(ref["weights"], ref["policies"]):
                pol = PolicyBlueprint.from_dict(bp_d).build()
                rnet = getattr(pol, "_policy", None)
                if rnet is not None and callable(
                    getattr(rnet, "deterministic_action", None)
                ):
                    self.refs.append((float(w), rnet.to(device).eval()))
                else:
                    # Scripted (non-export) reference: keep the policy and
                    # take the serial act() path per request.
                    self.refs.append((float(w), pol))
        self.delta_factor = float(spec_dict.get("delta_factor", 0.0))
        self.delta_frozen = spec_dict.get("delta_mode") == "frozen"
        if self.delta_frozen and not self.supports_delta_out:
            raise TypeError(
                f"{type(net).__name__} cannot supply the frozen-delta "
                f"payload (missing `ctx` support or "
                f"`deterministic_action`) — re-export the policy with "
                f"current code"
            )

        # Stack the reference ensemble into one vmapped forward when all
        # refs share the same module class — a delta run with K=10 refs
        # would otherwise pay 10 sequential forwards (≈100 kernel
        # launches) per batch.  ``None`` → per-net serial fallback.
        self._ref_stack = None
        self._build_ref_stack(device)

        # CUDA-graph state — the fixed-capacity padded batch gives a
        # *constant* input/output shape per spec, so the whole pipeline
        # (ref ensemble → ctx σ floor → stochastic sample) can be captured
        # once and replayed with a single launch.  This both collapses
        # per-request Python dispatch overhead and hardens determinism:
        # a replayed graph can never deviate from its captured kernel
        # sequence.
        self._graph = None
        self._g_obs = None
        self._g_noise = None
        self._g_ef = None
        self._g_out = None
        self._graph_done = False  # capture attempted (success or failed)

    def graphable(self, device: Any) -> bool:
        """Can this spec's forward be captured as a CUDA graph?"""
        import torch

        if (
            not _CAPTURE_OK
            or device.type != "cuda"
            or not self.supports_uniform
        ):
            return False
        if not all(
            callable(getattr(rn, "deterministic_action", None))
            for _, rn in self.refs
        ):
            return False  # scripted serial refs are not capturable
        return hasattr(torch.cuda, "CUDAGraph")

    def _compute(
        self, obs_t: Any, ef_t: Any, noise_t: Any,
    ) -> Tuple[Any, Any, Optional[Any], Optional[Any]]:
        """Full sampling semantics on already-padded device tensors.

        Returns ``(action, log_prob, ref_action_or_None, delta_or_None)``.
        ``delta`` is the per-frame Δ the σ floor consumed — non-``None``
        only in frozen-delta mode, where ``ref_action`` is withheld from
        the reply (the two payloads are mutually exclusive on the wire).
        Kept free of CPU→GPU syncs so it is safe to run under CUDA-graph
        capture.
        """
        import torch

        ref_pad: Optional[torch.Tensor] = None
        if self.refs:
            # IMPORTANT: run the reference ensemble on the *padded*
            # batch.  GEMM kernel choice depends on batch shape — a
            # (B,·) slice forward would make ref values (and hence the
            # delta-floored σ) drift at the ULP level with batch
            # composition, breaking the fixed-shape determinism
            # contract.
            if self._ref_stack is not None:
                params, buffers, det_fn = self._ref_stack
                k = len(self.refs)
                obs_k = obs_t.unsqueeze(0).expand(k, -1, -1).contiguous()
                from torch.func import vmap
                acts = vmap(det_fn)(params, buffers, obs_k)  # (K, C, D)
                ref_pad = (self._ref_weights.view(k, 1, 1) * acts).sum(0)
            else:
                acc: Optional[torch.Tensor] = None
                for w, rn in self.refs:
                    if callable(getattr(rn, "deterministic_action", None)):
                        a = rn.deterministic_action(obs_t)
                    else:
                        serial = [
                            rn.act(obs_t[i].cpu().numpy())[0]
                            for i in range(obs_t.shape[0])
                        ]
                        a = torch.as_tensor(
                            np.asarray(serial, dtype=np.float32)
                        ).to(obs_t.device)
                    acc = w * a if acc is None else acc + w * a
                ref_pad = acc

        delta_pad: Optional[torch.Tensor] = None
        if self.delta_frozen and ref_pad is not None:
            # Δ = det_action(current policy) − a_ref — an action-level
            # quantity computed server-side (the inner net IS the
            # current policy); it enters ctx as an input field.
            delta_pad = (
                self.inner_net.deterministic_action(obs_t) - ref_pad
            )
        if self.uses_ctx:
            ctx = SimpleNamespace(
                explore_factor=ef_t,
                # Mutual exclusion at the input contract: frozen mode
                # feeds `delta`, dynamic feeds `reference_action`.
                reference_action=(
                    None if delta_pad is not None else ref_pad
                ),
                delta=delta_pad,
                delta_factor=self.delta_factor,
            )
            action_all, lp_all = self.inner_net.sample_action(
                obs_t, ctx=ctx,
                **({"uniform": noise_t} if self.supports_uniform else {}),
            )
        else:
            action_all, lp_all = self.inner_net.sample_action(
                obs_t, explore_factor=ef_t,
                **({"uniform": noise_t} if self.supports_uniform else {}),
            )
        return action_all, lp_all, ref_pad, delta_pad

    def _ensure_graph(
        self, capacity: int, device: Any, obs_t: Any, ef_t: Any,
        noise_t: Any,
    ) -> None:
        """Warm up and capture the padded forward as a CUDA graph."""
        import torch

        self._graph_done = True
        try:
            self._g_obs = torch.zeros(
                capacity, self.obs_dim, dtype=torch.float32, device=device,
            )
            self._g_noise = torch.zeros(
                capacity, self.action_dim + 1, dtype=torch.float32,
                device=device,
            )
            self._g_ef = torch.zeros(
                capacity, dtype=torch.float32, device=device,
            )
            self._g_obs.copy_(obs_t)
            self._g_noise.copy_(noise_t)
            self._g_ef.copy_(ef_t)

            # Canonical recipe: warmup iterations on a side stream so
            # cuBLAS workspaces / autotuning settle, then capture.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._compute(self._g_obs, self._g_ef, self._g_noise)
            torch.cuda.current_stream().wait_stream(s)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._g_out = self._compute(
                    self._g_obs, self._g_ef, self._g_noise,
                )
            self._graph = graph
            _logger.debug("captured CUDA graph for spec %d", self.spec_id)
        except Exception:
            global _CAPTURE_OK
            _CAPTURE_OK = False  # poisoned capture state — never retry
            _logger.warning(
                "CUDA-graph capture failed for spec %d; disabling "
                "capture for the whole server (eager path)",
                self.spec_id, exc_info=True,
            )
            self._graph = None
            self._g_out = None

    def replay_inputs(
        self, obs_t: Any, ef_t: Any, noise_t: Any,
    ) -> None:
        """Stage new inputs into the graph's static buffers and replay."""
        self._g_obs.copy_(obs_t)
        self._g_noise.copy_(noise_t)
        self._g_ef.copy_(ef_t)
        self._graph.replay()

    def _build_ref_stack(self, device: Any) -> None:
        """Try to stack ref nets into a single vmap-batched forward."""
        import torch
        from torch import nn

        nets = [rn for _, rn in self.refs]
        if not nets or not all(
            type(n) is type(nets[0])
            and callable(getattr(n, "deterministic_action", None))
            for n in nets
        ):
            self._ref_stack = None
            return

        class _DetShim(nn.Module):
            """Adapter so ``functional_call`` hits ``deterministic_action``
            through ``forward`` — family-agnostic (the per-cell tail like
            argmax-gather lives inside the method itself)."""

            def __init__(self, net):
                super().__init__()
                self.inner = net

            def forward(self, x):
                return self.inner.deterministic_action(x)

        try:
            from torch.func import (
                functional_call, stack_module_state, vmap,
            )
            shims = [_DetShim(n) for n in nets]
            params, buffers = stack_module_state(shims)
            template = shims[0]

            def _det(p, b, x):
                return functional_call(template, (p, b), (x,))

            # Smoke-check the stacked path against the serial one once;
            # on any mismatch silently fall back to per-net calls.
            probe = torch.zeros(1, self.obs_dim, device=device)
            probe_k = probe.expand(len(nets), -1, -1).contiguous()
            stacked = vmap(_det)(params, buffers, probe_k)
            serial = torch.stack(
                [n.deterministic_action(probe) for n in nets],
            ).squeeze(1)
            if not torch.allclose(stacked.squeeze(1), serial, atol=1e-5):
                self._ref_stack = None
                return
            self._ref_weights = torch.as_tensor(
                [w for w, _ in self.refs],
                dtype=torch.float32, device=device,
            )
            self._ref_stack = (params, buffers, _det)
        except Exception:
            self._ref_stack = None


class _Conn:
    """Per-connection receive buffer."""

    __slots__ = ("sock", "buf")

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buf = bytearray()


class _Request:
    __slots__ = ("conn", "spec_id", "ef", "want_extra", "obs", "noise")

    def __init__(self, conn, spec_id, ef, want_extra, obs, noise) -> None:
        self.conn = conn
        self.spec_id = spec_id
        self.ef = ef
        self.want_extra = want_extra
        self.obs = obs
        self.noise = noise


def _forward_group(
    sess: _SpecSession, reqs: List[_Request], capacity: int, device: Any,
) -> None:
    """Run one (possibly padded) batched forward for same-spec requests
    and write replies on each request's connection."""
    import torch

    B = len(reqs)
    assert B <= capacity
    obs_dim, act_dim = sess.obs_dim, sess.action_dim

    obs_pad = np.zeros((capacity, obs_dim), dtype=np.float32)
    noise_pad = np.full((capacity, act_dim + 1), 0.5, dtype=np.float32)
    ef_pad = np.zeros(capacity, dtype=np.float32)
    for i, r in enumerate(reqs):
        obs_pad[i] = r.obs
        noise_pad[i] = r.noise
        ef_pad[i] = r.ef

    obs_t = torch.from_numpy(obs_pad).to(device)
    noise_t = torch.from_numpy(noise_pad).to(device)
    ef_t = torch.from_numpy(ef_pad).to(device)

    with torch.no_grad():
        if sess._graph is not None:
            sess.replay_inputs(obs_t, ef_t, noise_t)
            action_all, lp_all, ref_pad, delta_pad = sess._g_out
        else:
            if not sess._graph_done and sess.graphable(device):
                sess._ensure_graph(
                    capacity, device, obs_t, ef_t, noise_t,
                )
                if sess._graph is not None:
                    # Capture does not execute — replay once to produce
                    # outputs for the just-staged inputs.
                    sess._graph.replay()
                    action_all, lp_all, ref_pad, delta_pad = sess._g_out
                else:
                    action_all, lp_all, ref_pad, delta_pad = sess._compute(
                        obs_t, ef_t, noise_t,
                    )
            else:
                action_all, lp_all, ref_pad, delta_pad = sess._compute(
                    obs_t, ef_t, noise_t,
                )

    action_np = action_all[:B].detach().cpu().numpy().astype(np.float32)
    lp_np = lp_all[:B].detach().cpu().numpy().astype(np.float64)
    # Frozen mode withholds ref from the wire — the recorded payload is
    # Δ, not a_ref (mutually exclusive).
    ref_np = (
        ref_pad[:B].detach().cpu().numpy().astype(np.float32)
        if ref_pad is not None and not sess.delta_frozen else None
    )
    delta_np = (
        delta_pad[:B].detach().cpu().numpy().astype(np.float32)
        if delta_pad is not None else None
    )

    for i, r in enumerate(reqs):
        hdr = _ACT_REPLY_HDR.pack(
            float(lp_np[i]), int(ref_np is not None),
            int(delta_np is not None),
        )
        payload = (
            b"a" + hdr + action_np[i].tobytes()
            + (ref_np[i].tobytes() if ref_np is not None else b"")
            + (delta_np[i].tobytes() if delta_np is not None else b"")
        )
        try:
            _send_frame(r.conn.sock, payload)
        except OSError:
            continue  # worker died — drop the reply


def serve(
    sock_path: str,
    device: str = "cuda",
    capacity: int = DEFAULT_CAPACITY,
) -> None:
    """Server main loop — runs inside a spawned process."""
    # Deterministic-kernel environment — must be set before CUDA init.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - old torch
        pass
    dev = torch.device(device)

    sessions: Dict[int, _SpecSession] = {}
    sessions_order: List[int] = []  # insertion order for LRU eviction
    pending: List[_Request] = []
    sel = selectors.DefaultSelector()

    listen = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listen.bind(sock_path)
    listen.listen()
    listen.setblocking(False)
    sel.register(listen, selectors.EVENT_READ, "listen")

    def _drop(conn: _Conn) -> None:
        try:
            sel.unregister(conn.sock)
        except Exception:
            pass
        try:
            conn.sock.close()
        except OSError:
            pass

    running = True
    while running:
        # Block only when nothing is pending — once a request is queued we
        # poll so the batch ships immediately after draining arrivals.
        events = sel.select(timeout=None if not pending else 0)
        for key, _ in events:
            if key.data == "listen":
                csock, _ = listen.accept()
                csock.setblocking(False)
                conn = _Conn(csock)
                sel.register(csock, selectors.EVENT_READ, conn)
                continue

            conn: _Conn = key.data
            try:
                data = conn.sock.recv(1 << 16)
            except OSError:
                data = b""
            if not data:
                _drop(conn)
                continue
            conn.buf += data
            # Parse complete frames out of the buffer.
            while len(conn.buf) >= 4:
                (flen,) = struct.unpack("<I", conn.buf[:4])
                if len(conn.buf) < 4 + flen:
                    break
                payload = bytes(conn.buf[4:4 + flen])
                del conn.buf[:4 + flen]
                t = payload[:1]
                if t == b"P":
                    conn.sock.sendall(struct.pack("<I", 1) + b"p")
                elif t == b"X":
                    running = False
                elif t == b"R":
                    try:
                        reg = pickle.loads(payload[1:])
                        sid = _spec_id_of(reg["policy_bp"], reg["spec"])
                        if sid not in sessions:
                            if len(sessions_order) >= _MAX_SPECS:
                                evicted = sessions_order.pop(0)
                                sessions.pop(evicted, None)
                            sessions[sid] = _SpecSession(
                                sid, reg["policy_bp"], reg["spec"], dev,
                            )
                            sessions_order.append(sid)
                        sess = sessions[sid]
                        _send_frame(conn.sock, b"r" + pickle.dumps({
                            "spec_id": sid,
                            "obs_dim": sess.obs_dim,
                            "action_dim": sess.action_dim,
                            "supports_uniform": sess.supports_uniform,
                            "supports_delta": sess.supports_delta,
                            "supports_delta_out": sess.supports_delta_out,
                        }))
                    except Exception as exc:  # keep server alive
                        _logger.warning("register failed: %s", exc)
                        try:
                            _send_frame(conn.sock, b"r" + pickle.dumps(
                                {"error": str(exc)}))
                        except OSError:
                            pass
                elif t == b"A":
                    try:
                        spec_id, ef, want_extra, n_obs, n_noise = \
                            _ACT_HDR.unpack(payload[1:1 + _ACT_HDR.size])
                        body = payload[1 + _ACT_HDR.size:]
                        obs = np.frombuffer(
                            body[: n_obs * 4], dtype=np.float32).copy()
                        noise = np.frombuffer(
                            body[n_obs * 4: n_obs * 4 + n_noise * 4],
                            dtype=np.float32,
                        ).copy()
                        pending.append(
                            _Request(conn, spec_id, ef, want_extra, obs, noise)
                        )
                    except Exception:
                        _drop(conn)
                        break
                else:
                    _drop(conn)
                    break

        # Greedy drain → one forward per spec group.
        if pending:
            batch, pending = pending, []
            by_spec: Dict[int, List[_Request]] = {}
            for r in batch:
                by_spec.setdefault(r.spec_id, []).append(r)
            for sid, reqs in by_spec.items():
                sess = sessions.get(sid)
                if sess is None:
                    # Spec was evicted (or never registered) — fail the
                    # workers fast instead of letting them block on the
                    # socket timeout.
                    err = b"!" + pickle.dumps(
                        f"spec {sid} not registered (evicted?)"
                    )
                    for r in reqs:
                        try:
                            _send_frame(r.conn.sock, err)
                        except OSError:
                            pass
                    continue
                for i in range(0, len(reqs), capacity):
                    try:
                        _forward_group(sess, reqs[i:i + capacity],
                                       capacity, dev)
                    except Exception as exc:
                        _logger.exception("forward failed for spec %d", sid)
                        # Answer every pending request in this chunk with
                        # an error frame so workers fail fast instead of
                        # hanging on the socket timeout.
                        err = b"!" + pickle.dumps(
                            f"forward failed (spec {sid}): {exc}"
                        )
                        for r in reqs[i:i + capacity]:
                            try:
                                _send_frame(r.conn.sock, err)
                            except OSError:
                                pass

    listen.close()
    try:
        os.unlink(sock_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Parent-side handle
# ---------------------------------------------------------------------------
class InferenceServerHandle:
    """Owns a spawned :func:`serve` process; created per training run."""

    def __init__(self, device: str = "cuda", capacity: int = DEFAULT_CAPACITY):
        import multiprocessing as mp

        self._sock_path = os.path.join(
            tempfile.gettempdir(),
            f"cb_infer_{os.getpid()}_{time.time_ns()}.sock",
        )
        self._proc = mp.get_context("spawn").Process(
            target=serve,
            args=(self._sock_path, device, capacity),
            daemon=True,
        )
        self._proc.start()

    @property
    def address(self) -> str:
        return self._sock_path

    def wait_ready(self, timeout: float = 60.0) -> None:
        """Block until the server accepts a ping (or fail fast)."""
        deadline = time.monotonic() + timeout
        while True:
            if not self._proc.is_alive():
                raise RuntimeError(
                    "inference server exited during startup "
                    f"(exitcode={self._proc.exitcode})"
                )
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.settimeout(2.0)
                s.connect(self._sock_path)
                _send_frame(s, b"P")
                if _recv_frame(s) == b"p":
                    return
            except OSError:
                pass
            finally:
                s.close()
            if time.monotonic() > deadline:
                raise TimeoutError("inference server did not become ready")
            time.sleep(0.05)

    def close(self) -> None:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect(self._sock_path)
            _send_frame(s, b"X")
            s.close()
        except OSError:
            pass
        if self._proc is not None:
            self._proc.join(timeout=10.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=5.0)
            self._proc = None
        try:
            os.unlink(self._sock_path)
        except OSError:
            pass
