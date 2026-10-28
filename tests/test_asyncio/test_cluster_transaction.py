import asyncio
from typing import Tuple
from unittest import mock
from unittest.mock import patch, Mock

import pytest

import redis
from redis import CrossSlotTransactionError, RedisClusterException
from redis._lease import ConnectionLease, LeasePhase
from redis.asyncio import RedisCluster, Connection
from redis.asyncio.cluster import ClusterNode, NodesManager, TransactionStrategy
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.cluster import PRIMARY
from tests.conftest import skip_if_server_version_lt


def _find_source_and_target_node_for_slot(
    r: RedisCluster, slot: int
) -> Tuple[ClusterNode, ClusterNode]:
    """Returns a pair of ClusterNodes, where the first node is the
    one that owns the slot and the second is a possible target
    for that slot, i.e. a primary node different from the first
    one.
    """
    node_migrating = r.nodes_manager.get_node_from_slot(slot)
    assert node_migrating, f"No node could be found that owns slot #{slot}"

    available_targets = [
        n
        for n in r.nodes_manager.startup_nodes.values()
        if node_migrating.name != n.name and n.server_type == PRIMARY
    ]

    assert available_targets, f"No possible target nodes for slot #{slot}"
    return node_migrating, available_targets[0]


@pytest.mark.onlycluster
class TestClusterTransaction:
    @pytest.mark.onlycluster
    async def test_pipeline_is_true(self, r) -> None:
        "Ensure pipeline instances are not false-y"
        async with r.pipeline(transaction=True) as pipe:
            assert pipe

    @pytest.mark.onlycluster
    async def test_pipeline_empty_transaction(self, r):
        await r.set("a", 0)

        async with r.pipeline(transaction=True) as pipe:
            assert await pipe.execute() == []

    @pytest.mark.onlycluster
    async def test_executes_transaction_against_cluster(self, r) -> None:
        async with r.pipeline(transaction=True) as tx:
            tx.set("{foo}bar", "value1")
            tx.set("{foo}baz", "value2")
            tx.set("{foo}bad", "value3")
            tx.get("{foo}bar")
            tx.get("{foo}baz")
            tx.get("{foo}bad")
            assert await tx.execute() == [
                True,
                True,
                True,
                b"value1",
                b"value2",
                b"value3",
            ]

        await r.flushall()

        tx = r.pipeline(transaction=True)
        tx.set("{foo}bar", "value1")
        tx.set("{foo}baz", "value2")
        tx.set("{foo}bad", "value3")
        tx.get("{foo}bar")
        tx.get("{foo}baz")
        tx.get("{foo}bad")
        assert await tx.execute() == [True, True, True, b"value1", b"value2", b"value3"]

    @pytest.mark.onlycluster
    async def test_throws_exception_on_different_hash_slots(self, r):
        async with r.pipeline(transaction=True) as tx:
            tx.set("{foo}bar", "value1")
            tx.set("{foobar}baz", "value2")

            with pytest.raises(
                CrossSlotTransactionError,
                match="All keys involved in a cluster transaction must map to the same slot",
            ):
                await tx.execute()

    @pytest.mark.onlycluster
    async def test_throws_exception_with_watch_on_different_hash_slots(self, r):
        async with r.pipeline(transaction=True) as tx:
            with pytest.raises(
                RedisClusterException,
                match="WATCH - all keys must map to the same key slot",
            ):
                await tx.watch("key1", "key2")

    @pytest.mark.onlycluster
    async def test_transaction_with_watched_keys(self, r):
        await r.set("a", 0)

        async with r.pipeline(transaction=True) as pipe:
            await pipe.watch("a")
            a = await pipe.get("a")

            pipe.multi()
            pipe.set("a", int(a) + 1)
            assert await pipe.execute() == [True]

    @pytest.mark.onlycluster
    async def test_pipeline_reset_releases_connection_when_cancelled(self, r):
        # Regression test for https://github.com/redis/redis-py/issues/4121
        # (cluster variant): cancelling reset() while it awaits the UNWATCH
        # reply must still return the node-pooled connection.
        await r.set("a", 0)

        pipe = r.pipeline(transaction=True)
        await pipe.watch("a")

        strategy = pipe._execution_strategy
        leased = strategy._transaction_connection
        node = strategy._transaction_node
        assert leased is not None
        assert node is not None
        assert leased not in node._free

        started = asyncio.Event()

        async def hang(*_args, **_kwargs):
            started.set()
            await asyncio.sleep(60)

        with mock.patch.object(leased, "read_response", side_effect=hang):
            task = asyncio.ensure_future(pipe.reset())
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert strategy._transaction_connection is None
        assert strategy._transaction_node is None
        assert strategy._watching is False
        assert leased in node._free
        # Cancellation while the UNWATCH reply may be unread terminates the
        # lease dirty and the socket must have been cleared before return.
        assert leased.lease.phase is redis.LeasePhase.ABORTED_DIRTY
        assert leased.is_connected is False

    @pytest.mark.onlycluster
    async def test_retry_transaction_during_unfinished_slot_migration(self, r):
        """
        When a transaction is triggered during a migration, MovedError
        or AskError may appear (depends on the key being already migrated
        or the key not existing already). The patch on parse_response
        simulates such an error, but the slot cache is not updated
        (meaning the migration is still ongoing) so the pipeline eventually
        fails as if it was retried but the migration is not yet complete.
        """
        key = "book"
        slot = r.keyslot(key)
        node_migrating, node_importing = _find_source_and_target_node_for_slot(r, slot)

        original_parse_response = ClusterNode.parse_response
        with (
            patch.object(
                ClusterNode, "parse_response", autospec=True
            ) as parse_response,
            patch.object(NodesManager, "move_slot") as manager_move_slot,
        ):

            async def ask_redirect_effect(redis_node, connection, command, **options):
                if command == "MULTI":
                    return
                elif command == "EXEC":
                    raise redis.exceptions.ExecAbortError()
                elif command in ("_", "SET"):
                    raise redis.exceptions.AskError(f"{slot} {node_importing.name}")

                return await original_parse_response(
                    redis_node, connection, command, **options
                )

            parse_response.side_effect = ask_redirect_effect

            async with r.pipeline(transaction=True) as pipe:
                pipe.set(key, "val")
                with pytest.raises(redis.exceptions.AskError) as ex:
                    await pipe.execute()

                assert str(ex.value).startswith(
                    "Command # 1 (SET book val) of pipeline caused error:"
                    f" {slot} {node_importing.name}"
                )

            manager_move_slot.assert_called()

    @pytest.mark.onlycluster
    async def test_retry_transaction_during_slot_migration_successful(
        self, create_redis
    ):
        """
        If a MovedError or AskError appears when calling EXEC and no key is watched,
        the pipeline is retried after updating the node manager slot table. If the
        migration was completed, the transaction may then complete successfully.
        """
        r = await create_redis(flushdb=False)
        key = "book"
        slot = r.keyslot(key)
        node_migrating, node_importing = _find_source_and_target_node_for_slot(r, slot)

        with (
            patch.object(ClusterNode, "parse_response") as parse_response,
        ):

            def ask_redirect_effect(conn, *args, **options):
                # first call should go here, we trigger an AskError
                if f"{conn.host}:{conn.port}" == node_migrating.name:
                    if "MULTI" in args:
                        return
                    elif "EXEC" in args:
                        raise redis.exceptions.ExecAbortError()

                    raise redis.exceptions.AskError(f"{slot} {node_importing.name}")
                # if the slot table is updated, the next call will go here
                elif f"{conn.host}:{conn.port}" == node_importing.name:
                    if "EXEC" in args:
                        return ["OK"]  # mock value to validate this section was called
                    return
                else:
                    assert False, f"unexpected node {conn.host}:{conn.port} was called"

            parse_response.side_effect = ask_redirect_effect

            result = None
            async with r.pipeline(transaction=True) as pipe:
                pipe.multi()
                pipe.set(key, "val")
                result = await pipe.execute()

            assert result and True in result, "Target node was not called"

    @pytest.mark.onlycluster
    async def test_retry_transaction_with_watch_after_slot_migration(self, r):
        """
        If a MovedError or AskError appears when calling WATCH, the client
        must attempt to recover itself before proceeding and no WatchError
        should appear.
        """
        key = "book"
        slot = r.keyslot(key)
        r.reinitialize_steps = 1

        # force a MovedError on the first call to pipe.watch()
        # by switching the node that owns the slot to another one
        _node_migrating, node_importing = _find_source_and_target_node_for_slot(r, slot)
        r.nodes_manager.slots_cache[slot] = [node_importing]

        async with r.pipeline(transaction=True) as pipe:
            await pipe.watch(key)
            pipe.multi()
            pipe.set(key, "val")
            assert await pipe.execute() == [True]

    @pytest.mark.onlycluster
    async def test_retry_transaction_with_watch_during_slot_migration(self, r):
        """
        If a MovedError or AskError appears when calling EXEC and keys were
        being watched before the migration started, a WatchError should appear.
        These errors imply resetting the connection and connecting to a new node,
        so watches are lost anyway and the client code must be notified.
        """
        key = "book"
        slot = r.keyslot(key)
        node_migrating, node_importing = _find_source_and_target_node_for_slot(r, slot)

        with patch.object(ClusterNode, "parse_response") as parse_response:

            def ask_redirect_effect(conn, *args, **options):
                if f"{conn.host}:{conn.port}" == node_migrating.name:
                    # we simulate the watch was sent before the migration started
                    if "WATCH" in args:
                        return b"OK"
                    # but the pipeline was triggered after the migration started
                    elif "MULTI" in args:
                        return
                    elif "EXEC" in args:
                        raise redis.exceptions.ExecAbortError()

                    raise redis.exceptions.AskError(f"{slot} {node_importing.name}")
                # we should not try to connect to any other node
                else:
                    assert False, f"unexpected node {conn.host}:{conn.port} was called"

            parse_response.side_effect = ask_redirect_effect

            async with r.pipeline(transaction=True) as pipe:
                await pipe.watch(key)

                pipe.multi()
                pipe.set(key, "val")
                with pytest.raises(redis.exceptions.WatchError) as ex:
                    await pipe.execute()

                assert str(ex.value).startswith(
                    "Slot rebalancing occurred while watching keys"
                )

    @pytest.mark.onlycluster
    async def test_retry_transaction_on_connection_error(self, r):
        key = "book"
        slot = r.keyslot(key)

        _node_migrating, node_importing = _find_source_and_target_node_for_slot(r, slot)
        original_slots_cache = r.nodes_manager.slots_cache[slot]

        mock_connection = Mock(spec=Connection)
        mock_connection.read_response.side_effect = redis.exceptions.ConnectionError(
            "Conn error"
        )
        mock_connection.retry = Retry(NoBackoff(), 0)
        mock_connection.host = node_importing.host
        mock_connection.port = node_importing.port
        mock_connection.db = 0

        node_importing._free.append(mock_connection)
        r.nodes_manager.slots_cache[slot] = [node_importing]
        r.reinitialize_steps = 1

        try:
            async with r.pipeline(transaction=True) as pipe:
                pipe.set(key, "val")
                assert await pipe.execute() == [True]

            assert mock_connection.read_response.call_count == 1
        finally:
            # Clean up mock connection from node so teardown can work
            if mock_connection in node_importing._free:
                node_importing._free.remove(mock_connection)
            r.nodes_manager.slots_cache[slot] = original_slots_cache

    @pytest.mark.onlycluster
    async def test_retry_transaction_on_connection_error_with_watched_keys(self, r):
        key = "book"
        slot = r.keyslot(key)

        _node_migrating, node_importing = _find_source_and_target_node_for_slot(r, slot)
        original_slots_cache = r.nodes_manager.slots_cache[slot]

        mock_connection = Mock(spec=Connection)
        mock_connection.read_response.side_effect = redis.exceptions.ConnectionError(
            "Conn error"
        )
        mock_connection.retry = Retry(NoBackoff(), 0)
        mock_connection.host = node_importing.host
        mock_connection.port = node_importing.port
        mock_connection.db = 0

        node_importing._free.append(mock_connection)
        r.nodes_manager.slots_cache[slot] = [node_importing]
        r.reinitialize_steps = 1

        try:
            async with r.pipeline(transaction=True) as pipe:
                await pipe.watch(key)

                pipe.multi()
                pipe.set(key, "val")
                assert await pipe.execute() == [True]

            assert mock_connection.read_response.call_count == 1
        finally:
            # Clean up mock connection from node so teardown can work
            if mock_connection in node_importing._free:
                node_importing._free.remove(mock_connection)
            r.nodes_manager.slots_cache[slot] = original_slots_cache

    @pytest.mark.onlycluster
    async def test_exec_error_raised(self, r):
        hashkey = "{key}"
        await r.set(f"{hashkey}:c", "a")

        async with r.pipeline(transaction=True) as pipe:
            pipe.set(f"{hashkey}:a", 1).set(f"{hashkey}:b", 2)
            pipe.lpush(f"{hashkey}:c", 3).set(f"{hashkey}:d", 4)
            with pytest.raises(redis.ResponseError) as ex:
                await pipe.execute()
            assert str(ex.value).startswith(
                "Command # 3 (LPUSH {key}:c 3) of pipeline caused error: "
            )

            # make sure the pipe was restored to a working state
            assert await pipe.set(f"{hashkey}:z", "zzz").execute() == [True]
            assert await r.get(f"{hashkey}:z") == b"zzz"

    @pytest.mark.onlycluster
    async def test_parse_error_raised(self, r):
        hashkey = "{key}"
        async with r.pipeline(transaction=True) as pipe:
            # the zrem is invalid because we don't pass any keys to it
            pipe.set(f"{hashkey}:a", 1).zrem(f"{hashkey}:b").set(f"{hashkey}:b", 2)
            with pytest.raises(redis.ResponseError) as ex:
                await pipe.execute()

            assert str(ex.value).startswith(
                "Command # 2 (ZREM {key}:b) of pipeline caused error: wrong number"
            )

            # make sure the pipe was restored to a working state
            assert await pipe.set(f"{hashkey}:z", "zzz").execute() == [True]
            assert await r.get(f"{hashkey}:z") == b"zzz"

    @pytest.mark.onlycluster
    async def test_transaction_callable(self, r):
        hashkey = "{key}"
        await r.set(f"{hashkey}:a", 1)
        await r.set(f"{hashkey}:b", 2)
        has_run = []

        async def my_transaction(pipe):
            a_value = await pipe.get(f"{hashkey}:a")
            assert a_value in (b"1", b"2")
            b_value = await pipe.get(f"{hashkey}:b")
            assert b_value == b"2"

            # silly run-once code... incr's "a" so WatchError should be raised
            # forcing this all to run again. this should incr "a" once to "2"
            if not has_run:
                await r.incr(f"{hashkey}:a")
                has_run.append("it has")

            pipe.multi()
            pipe.set(f"{hashkey}:c", int(a_value) + int(b_value))

        result = await r.transaction(my_transaction, f"{hashkey}:a", f"{hashkey}:b")
        assert result == [True]
        assert await r.get(f"{hashkey}:c") == b"4"

    @pytest.mark.onlycluster
    @skip_if_server_version_lt("2.0.0")
    async def test_transaction_discard(self, r):
        hashkey = "{key}"

        # pipelines enabled as transactions can be discarded at any point
        async with r.pipeline(transaction=True) as pipe:
            await pipe.watch(f"{hashkey}:key")
            await pipe.set(f"{hashkey}:key", "someval")
            await pipe.discard()

            assert not pipe._execution_strategy._watching
            assert not len(pipe)


class _ResetFakeConnection:
    """Stand-in connection covering just the surface used by
    TransactionStrategy.reset()."""

    def __init__(
        self,
        read_error: BaseException | None = None,
        send_error: BaseException | None = None,
        hang_event: asyncio.Event | None = None,
        **kwargs,
    ) -> None:
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = kwargs.get("port", 7000)
        self.db = kwargs.get("db", 0)
        self.lease = ConnectionLease()
        self.is_connected = True
        self.disconnect_count = 0
        self._should_reconnect = False
        self._read_error = read_error
        self._send_error = send_error
        self._hang_event = hang_event

    def should_reconnect(self) -> bool:
        return self._should_reconnect

    def mark_for_reconnect(self) -> None:
        self._should_reconnect = True

    async def disconnect(self) -> None:
        self.disconnect_count += 1
        self.is_connected = False
        self._should_reconnect = False

    async def send_command(self, *args, **kwargs) -> None:
        self.lease.mark_write_inflight()
        if self._send_error is not None:
            raise self._send_error

    async def read_response(self):
        self.lease.mark_await_read()
        if self._hang_event is not None:
            self._hang_event.set()
            await asyncio.sleep(60)
        if self._read_error is not None:
            raise self._read_error
        return True


def _make_node_with_connection(connection: _ResetFakeConnection) -> ClusterNode:
    return ClusterNode(
        "127.0.0.1",
        7000,
        max_connections=1,
        connection_class=lambda **kwargs: connection,
    )


def _make_strategy(
    node: ClusterNode, connection: _ResetFakeConnection, watching: bool
) -> TransactionStrategy:
    strategy = TransactionStrategy.__new__(TransactionStrategy)
    strategy._command_queue = []
    strategy._watching = watching
    strategy._transaction_node = node
    strategy._transaction_connection = connection
    strategy._explicit_transaction = False
    strategy._pipeline_slots = set()
    strategy._executing = False
    return strategy


class TestTransactionStrategyResetLease:
    """Offline lease-flight tests for the cluster transaction reset exit.

    These deliberately drive TransactionStrategy.reset() directly instead
    of going through a shared cleanup helper, to keep this exit forked and
    independently testable (AC-7/AC-8)."""

    async def test_reset_unwatch_success_completed_clean(self) -> None:
        connection = _ResetFakeConnection()
        node = _make_node_with_connection(connection)
        leased = node.acquire_connection()
        strategy = _make_strategy(node, leased, watching=True)

        await strategy.reset()

        assert leased.lease.phase is LeasePhase.COMPLETED
        assert leased.is_connected is True
        assert leased.disconnect_count == 0
        assert leased in node._free
        assert strategy._transaction_connection is None
        assert strategy._watching is False

    async def test_reset_unwatch_connection_error_dirty(self) -> None:
        connection = _ResetFakeConnection(
            read_error=redis.exceptions.ConnectionError("simulated read failure")
        )
        node = _make_node_with_connection(connection)
        leased = node.acquire_connection()
        strategy = _make_strategy(node, leased, watching=True)

        await strategy.reset()

        assert leased.lease.phase is LeasePhase.ABORTED_DIRTY
        assert leased.is_connected is False
        assert leased.disconnect_count == 1
        assert leased in node._free
        assert strategy._transaction_connection is None
        assert strategy._watching is False

    async def test_reset_cancelled_during_unwatch_read_dirty(self) -> None:
        started = asyncio.Event()
        connection = _ResetFakeConnection(hang_event=started)
        node = _make_node_with_connection(connection)
        leased = node.acquire_connection()
        strategy = _make_strategy(node, leased, watching=True)

        task = asyncio.ensure_future(strategy.reset())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert leased.lease.phase is LeasePhase.ABORTED_DIRTY
        assert leased.is_connected is False
        assert leased.disconnect_count == 1
        # The slot is returned exactly once even though the release path
        # was reached via cancellation; node.release is synchronous, so a
        # second CancelledError raised by the loop cannot skip the append.
        assert node._free.count(leased) == 1
        assert strategy._transaction_connection is None
        assert strategy._watching is False

    async def test_reset_cancelled_during_unwatch_send_dirty(self) -> None:
        connection = _ResetFakeConnection(send_error=asyncio.CancelledError())
        node = _make_node_with_connection(connection)
        leased = node.acquire_connection()
        strategy = _make_strategy(node, leased, watching=True)

        with pytest.raises(asyncio.CancelledError):
            await strategy.reset()

        assert leased.lease.phase is LeasePhase.ABORTED_DIRTY
        assert leased.is_connected is False
        assert leased in node._free

    async def test_reset_non_watching_held_connection_completed_clean(self) -> None:
        # After a batched MULTI/EXEC the pinned connection is returned
        # through reset() without an UNWATCH round trip.
        connection = _ResetFakeConnection()
        node = _make_node_with_connection(connection)
        leased = node.acquire_connection()
        leased.lease.mark_write_inflight()
        leased.lease.mark_await_read()
        strategy = _make_strategy(node, leased, watching=False)

        await strategy.reset()

        assert leased.lease.phase is LeasePhase.COMPLETED
        assert leased.is_connected is True
        assert leased.disconnect_count == 0
        assert leased in node._free
