import pytest

import redis
from redis import ConnectionLease, LeasePhase
from redis._lease import _ACTIVE_PHASES
import redis.asyncio as aioredis


class TestLeasePhase:
    def test_exact_phase_members(self):
        assert {p.name for p in LeasePhase} == {
            "LEASED",
            "WRITE_INFLIGHT",
            "AWAIT_READ",
            "COMPLETED",
            "ABORTED_DIRTY",
        }

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            (LeasePhase.LEASED, "leased"),
            (LeasePhase.WRITE_INFLIGHT, "write_inflight"),
            (LeasePhase.AWAIT_READ, "await_read"),
            (LeasePhase.COMPLETED, "completed"),
            (LeasePhase.ABORTED_DIRTY, "aborted_dirty"),
        ],
    )
    def test_phase_values(self, member, value):
        assert member.value == value
        assert isinstance(member, str)

    def test_exports_are_shared_between_stacks(self):
        assert aioredis.LeasePhase is LeasePhase
        assert aioredis.ConnectionLease is ConnectionLease
        assert redis.LeasePhase is LeasePhase
        assert redis.ConnectionLease is ConnectionLease


class TestConnectionLease:
    def test_initial_phase_is_none(self):
        lease = ConnectionLease()
        assert lease.phase is None
        assert not lease.is_active
        assert not lease.is_dirty
        assert not lease.is_completed

    def test_full_success_sequence(self):
        lease = ConnectionLease()
        lease.begin()
        assert lease.phase is LeasePhase.LEASED
        assert lease.is_active
        lease.mark_write_inflight()
        assert lease.phase is LeasePhase.WRITE_INFLIGHT
        lease.mark_await_read()
        assert lease.phase is LeasePhase.AWAIT_READ
        lease.mark_completed()
        assert lease.phase is LeasePhase.COMPLETED
        assert lease.is_completed
        assert not lease.is_active

    def test_dirty_sequence(self):
        lease = ConnectionLease()
        lease.begin()
        lease.mark_write_inflight()
        lease.mark_aborted_dirty()
        assert lease.phase is LeasePhase.ABORTED_DIRTY
        assert lease.is_dirty
        assert not lease.is_active

    def test_lazy_begin_on_send_without_acquire(self):
        # Direct (pool-less) connection usage starts the lease at send.
        lease = ConnectionLease()
        lease.mark_write_inflight()
        assert lease.phase is LeasePhase.WRITE_INFLIGHT
        assert lease.is_active

    def test_lazy_begin_on_read_without_send_mark(self):
        lease = ConnectionLease()
        lease.mark_await_read()
        assert lease.phase is LeasePhase.AWAIT_READ

    def test_write_read_round_trips_cycle_within_one_lease(self):
        lease = ConnectionLease()
        lease.begin()
        lease.mark_write_inflight()
        lease.mark_await_read()
        # e.g. a folded preflight followed by the main batch
        lease.mark_write_inflight()
        assert lease.phase is LeasePhase.WRITE_INFLIGHT
        lease.mark_await_read()
        assert lease.phase is LeasePhase.AWAIT_READ
        lease.mark_completed()
        assert lease.is_completed

    def test_terminal_phases_are_sticky(self):
        lease = ConnectionLease()
        lease.begin()
        lease.mark_write_inflight()
        lease.mark_completed()
        lease.mark_write_inflight()
        lease.mark_await_read()
        lease.mark_aborted_dirty()
        assert lease.phase is LeasePhase.COMPLETED

        lease = ConnectionLease()
        lease.begin()
        lease.mark_await_read()
        lease.mark_aborted_dirty()
        lease.mark_await_read()
        lease.mark_write_inflight()
        lease.mark_completed()
        assert lease.phase is LeasePhase.ABORTED_DIRTY

    def test_begin_restarts_the_lease(self):
        lease = ConnectionLease()
        lease.begin()
        lease.mark_write_inflight()
        lease.mark_aborted_dirty()
        # Terminal phase stays observable after release ...
        assert lease.is_dirty
        # ... until the connection is checked out again.
        lease.begin()
        assert lease.phase is LeasePhase.LEASED
        assert lease.is_active
        assert not lease.is_dirty

    def test_terminal_transitions_are_noop_before_begin(self):
        # Nothing can "complete" or go "dirty" if the lease never started.
        lease = ConnectionLease()
        lease.mark_completed()
        assert lease.phase is None
        lease.mark_aborted_dirty()
        assert lease.phase is None

    def test_active_phase_set_constant(self):
        assert _ACTIVE_PHASES == frozenset(
            (
                LeasePhase.LEASED,
                LeasePhase.WRITE_INFLIGHT,
                LeasePhase.AWAIT_READ,
            )
        )

    def test_repr_does_not_raise(self):
        lease = ConnectionLease()
        assert "ConnectionLease" in repr(lease)
