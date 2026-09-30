"""RemoteSamplingPolicy — worker-side Policy shell over the UDS server.

This is the remote-inference counterpart of
:class:`~baseline.framework.rollout.exploratory_policy.SamplingPolicy`.
When ``ParallelRollouter(rollout_inference="gpu")`` injects a
``"_remote_addr"`` into the serialized :class:`SamplingSpec` dict,
``_wrap_policy`` builds this shell instead of a local ``SamplingPolicy``.

Per ``act()`` it:

1. Resolves ``explore_factor`` exactly like ``SamplingPolicy`` (constant
   or ``(obs, step)`` callable — evaluated *client-side* so callables
   never have to reach the server).
2. Draws this frame's noise from a **per-episode RNG** reseeded by
   ``reset(seed)`` — the noise travels with the request, so the action is
   a deterministic function of (weights, obs, noise) independent of how
   the server batches the request.
3. Sends a blocking ACT request over the shared per-process UDS
   connection; the server runs the whole sampling semantics (reference
   ensemble → delta-mix ctx → truncated-normal sampling) batched on GPU.
4. Reconstructs the extras dict with the exact same keys/values as
   ``SamplingPolicy.act`` so ``action_extras`` → ``sctx__*`` → dump npz
   contracts are preserved verbatim.

The class is duck-type compatible with :class:`envs.framework.policy.Policy`
(``act``/``reset``/``close``); ``EpisodeRunner`` requires no changes.
"""
from __future__ import annotations

import logging
import socket
from typing import Any, Dict, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

from envs.framework.policy import Policy

from baseline.framework.rollout.inference_server import send_act, send_register

#: UDS request/response round-trip timeout.  Generous because a batch
#: forward can be queued behind other spec groups — a stuck server should
#: still surface as an error rather than a hung rollout forever.
_SOCKET_TIMEOUT_S = 300.0

# Per-process connection cache: worker processes host many
# RemoteSamplingPolicy instances (one per spec × agent) that all share a
# single socket per server address.  Requests on one connection are
# strictly sequential — ``act()`` blocks until its own reply arrives.
_CONNS: Dict[str, socket.socket] = {}


_ATEXIT_REGISTERED = False


def _get_conn(addr: str) -> socket.socket:
    global _ATEXIT_REGISTERED
    conn = _CONNS.get(addr)
    if conn is None:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(_SOCKET_TIMEOUT_S)
        conn.connect(addr)
        _CONNS[addr] = conn
        if not _ATEXIT_REGISTERED:
            # Spawned pool workers run a normal interpreter teardown, so
            # atexit gives the server a clean EOF on worker exit instead
            # of a leaked connection.
            import atexit

            atexit.register(drop_conns)
            _ATEXIT_REGISTERED = True
    return conn


def drop_conns() -> None:
    """Close all cached connections (called on worker teardown)."""
    for conn in _CONNS.values():
        try:
            conn.close()
        except OSError:
            pass
    _CONNS.clear()


class RemoteSamplingPolicy(Policy):
    """Policy whose ``act()`` is a blocking UDS round-trip to the
    centralized batched-inference server.

    Parameters
    ----------
    policy_bp_dict:
        Serialized :class:`PolicyBlueprint` dict for the *inner* policy —
        forwarded to the server at registration time; the worker never
        builds the local network.
    spec_dict:
        Serialized :class:`SamplingSpec` dict carrying ``"_remote_addr"``
        (the server socket path) plus ``explore_factor`` / ``reference`` /
        ``delta_factor`` / ``delta_mix`` — the same dict
        ``SamplingSpec.to_dict()`` produces, with the address injected by
        ``ParallelRollouter.collect``.
    """

    def __init__(
        self,
        policy_bp_dict: Dict[str, Any],
        spec_dict: Dict[str, Any],
    ) -> None:
        spec_dict = dict(spec_dict)
        addr = spec_dict.pop("_remote_addr", None)
        if not addr:
            raise ValueError(
                "RemoteSamplingPolicy requires '_remote_addr' in spec_dict"
            )
        self._addr = str(addr)
        self._conn = _get_conn(self._addr)

        # The spec fields the worker itself needs: explore_factor is
        # evaluated client-side (may be a callable); delta fields and the
        # reference presence flag feed the extras contract.
        self._ef_spec = spec_dict.get("explore_factor", 0.0)
        self._delta_factor = float(spec_dict.get("delta_factor", 0.0))
        self._delta_mix = float(spec_dict.get("delta_mix", 0.0))
        self._delta_frozen = spec_dict.get("delta_mode") == "frozen"
        self._has_ref = spec_dict.get("reference") is not None

        reply = send_register(self._conn, policy_bp_dict, spec_dict)
        self._spec_id = int(reply["spec_id"])
        self._obs_dim = int(reply["obs_dim"])
        self._action_dim = int(reply["action_dim"])
        # Mirror SamplingPolicy's intent-vs-capability handshake.
        if (
            self._has_ref
            and self._delta_mix != 0.0
            and not reply.get("supports_delta", False)
        ):
            raise TypeError(
                "SamplingSpec demands the reference-delta σ mix "
                f"(reference set, delta_mix={self._delta_mix}) but the "
                "remote policy does not declare SUPPORTS_REFERENCE_DELTA "
                "— export the policy with current code or pick a "
                "supported policy cell"
            )

        if self._delta_frozen and not reply.get("supports_delta_out", False):
            raise TypeError(
                "SamplingSpec demands frozen-delta mode but the remote "
                "policy cannot supply the Δ payload (missing `ctx` "
                "support or `deterministic_action`) — re-export the "
                "policy with current code"
            )

        if not reply.get("supports_uniform", False):
            _logger.warning(
                "remote policy export predates the `uniform` injection "
                "parameter — noise will be drawn server-side and results "
                "will NOT be reproducible across batch compositions. "
                "Re-export the policy with current code to fix this."
            )

        self._step = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------
    def act(
        self,
        observation: Any,
        *,
        want_extra: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        # Same ef resolution order as SamplingPolicy.act: resolve with the
        # current step index, then advance.
        ef = (
            float(self._ef_spec(observation, self._step))
            if callable(self._ef_spec)
            else float(self._ef_spec)
        )
        self._step += 1

        # (D+1,) uniforms: [0] = mixture-component selector, [1:] = action
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        noise = self._rng.random(self._action_dim + 1).astype(np.float32)

        action, log_prob, ref, delta = send_act(
            self._conn, self._spec_id, ef, want_extra, obs, noise,
        )

        extra: Dict[str, Any] = {}
        if want_extra:
            extra["log_prob"] = float(log_prob)
        # Reproduce ctx.record_fields() ordering: reference_action,
        # delta_factor, delta_mix, delta (non-None fields), then
        # legacy ef.  Frozen mode records `sctx__delta` (the
        # action-level Δ the server computed as det_action(current)
        # − a_ref) instead of reference_action — the two payloads
        # never coexist; the payload itself is the mode marker.
        if ref is not None:
            extra["sctx__reference_action"] = np.asarray(
                ref, dtype=np.float32,
            )
        extra["sctx__delta_factor"] = np.asarray(
            self._delta_factor, dtype=np.float32,
        )
        extra["sctx__delta_mix"] = np.asarray(
            self._delta_mix, dtype=np.float32,
        )
        if delta is not None:
            extra["sctx__delta"] = np.asarray(delta, dtype=np.float32)
        extra["explore_factor"] = ef
        return action, extra

    def reset(self, seed: Optional[int] = None) -> None:
        """Per-episode reseed — the runner's derived per-agent seed drives
        this policy's private noise stream, so the episode's actions are a
        deterministic function of its seed regardless of server batching."""
        self._step = 0
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

    def close(self) -> None:
        """No-op: the connection is shared per-process and dies with the
        worker; closing it per-policy would break siblings."""
        return None
