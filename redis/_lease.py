from enum import Enum
from typing import Optional


class LeasePhase(str, Enum):
    """The phase of a connection lease ("in-flight" lifecycle).

    A lease covers the time between acquiring a connection from a pool
    (``ConnectionPool``) or a cluster node (``ClusterNode._free``) and
    returning it. The phase is advanced by the real acquire/send/read
    lifecycle of the connection:

    - ``LEASED``: the connection was checked out and registered as in-use.
    - ``WRITE_INFLIGHT``: a command is being packed/written to the socket.
    - ``AWAIT_READ``: the connection is about to read a reply from the
      socket.
    - ``COMPLETED``: terminal phase. Every expected reply was consumed and
      the connection may be returned and reused without disconnecting.
    - ``ABORTED_DIRTY``: terminal phase. The command/pipeline failed before
      every expected reply was effectively read (a write-side failure, a
      connection error mid-read, or a cancellation with a possibly unread
      reply). The connection must be disconnected (or otherwise marked as
      not reusable) before it is returned.

    A single lease may contain several write/read round trips (e.g. a
    transaction preflight or a folded HIMPORT PREPARE); ``WRITE_INFLIGHT``
    and ``AWAIT_READ`` may therefore alternate. Once a terminal phase is
    reached the lease is no longer active; only the next checkout
    (``ConnectionLease.begin()``) starts a new lease. Terminal phases are
    kept on the connection until that next checkout, so the outcome of a
    lease stays observable after release.
    """

    LEASED = "leased"
    WRITE_INFLIGHT = "write_inflight"
    AWAIT_READ = "await_read"
    COMPLETED = "completed"
    ABORTED_DIRTY = "aborted_dirty"


_ACTIVE_PHASES = frozenset(
    (LeasePhase.LEASED, LeasePhase.WRITE_INFLIGHT, LeasePhase.AWAIT_READ)
)


class ConnectionLease:
    """Read-only observable state of a single connection lease.

    One instance lives on each connection for the lifetime of that
    connection; it is advanced exclusively by the connection primitives
    (acquire/send/read) and by the command/pipeline exits that decide
    whether a lease ended cleanly or dirty.

    The phase attributes are read-only to outsiders. This class
    intentionally provides only transition primitives, never a single
    "effective disposition" helper: clean vs. dirty return is decided
    separately by every exit (standalone ``Pipeline.reset``, cluster
    ``TransactionStrategy.reset``, ``ClusterNode.execute_command``,
    ``ClusterNode.execute_pipeline`` and the synchronous
    ``ClusterPipeline._send_cluster_commands``).
    """

    __slots__ = ("_phase",)

    def __init__(self) -> None:
        # None means the connection has never been checked out (or it is
        # used directly, without a pool, before its first send).
        self._phase: Optional[LeasePhase] = None

    def begin(self) -> None:
        """Start a new lease when the connection is checked out."""
        self._phase = LeasePhase.LEASED

    def mark_write_inflight(self) -> None:
        """Advance to WRITE_INFLIGHT.

        When the connection is used directly (no pool/acquire), a send
        lazily starts the lease in this transition.
        """
        # Allow WRITE_INFLIGHT <-> AWAIT_READ cycling for the multiple
        # write/read round trips a single lease may contain.
        if self._phase in _ACTIVE_PHASES or self._phase is None:
            self._phase = LeasePhase.WRITE_INFLIGHT

    def mark_await_read(self) -> None:
        """Advance to AWAIT_READ before reading a reply from the socket."""
        # A read without a preceding mark_write_inflight (defensive; e.g.
        # direct connection usage) lazily starts the lease too.
        if self._phase in _ACTIVE_PHASES or self._phase is None:
            self._phase = LeasePhase.AWAIT_READ

    def mark_completed(self) -> None:
        """Mark the terminal clean phase: every expected reply was read."""
        if self._phase in _ACTIVE_PHASES:
            self._phase = LeasePhase.COMPLETED

    def mark_aborted_dirty(self) -> None:
        """Mark the terminal dirty phase.

        Used by the exits when a failure or a cancellation happened before
        every expected reply was effectively read; the connection must not
        be reused without reconnecting.
        """
        if self._phase in _ACTIVE_PHASES:
            self._phase = LeasePhase.ABORTED_DIRTY

    @property
    def phase(self) -> Optional[LeasePhase]:
        """Current phase, or ``None`` before the first checkout/send."""
        return self._phase

    @property
    def is_active(self) -> bool:
        """Whether the lease is between checkout and a terminal phase."""
        return self._phase in _ACTIVE_PHASES

    @property
    def is_completed(self) -> bool:
        """Whether the lease terminated cleanly."""
        return self._phase is LeasePhase.COMPLETED

    @property
    def is_dirty(self) -> bool:
        """Whether the lease terminated dirty (must not be reused)."""
        return self._phase is LeasePhase.ABORTED_DIRTY

    def __repr__(self) -> str:
        return f"ConnectionLease(phase={self._phase!r})"
