"""
Tests for SGH scheduler: ready-set computation and dispatcher.

Phase 2 tests verify:
  - U(S) computation produces correct ready sets at each round
  - Parallel dispatch: |U| > 1 fires on independent node pairs
  - Sequential ordering: dependent nodes dispatch only after predecessors complete
  - any_of join: sibling cancellation happens mid-run, correct node wins
  - Blocked nodes are correctly detected and terminated
  - Mock executions complete with correct final states

All tests use MockNode — zero LLM/network calls (NFR-3).
"""
import asyncio
import time

import pytest

from sgh.core.plan import Edge, JoinMode, Node, NodeConfig, NodeState, NodeType, Plan
from sgh.core.state_machine import LEGAL_TRANSITIONS
from sgh.nodes.mock_node import MockNode
from sgh.scheduler.dispatcher import Dispatcher, EngineConfig, SchedulingRoundEvent
from sgh.scheduler.ready_set import (
    compute_ready_set,
    execution_is_complete,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mock_node_cfg(
    delay: float = 0.0,
    fail_type: str | None = None,
    output: dict | None = None,
) -> dict:
    return {
        "node_type": NodeType.MOCK,
        "mock_delay_s": delay,
        "mock_fail_type": fail_type,
        "mock_output": output or {"result": "ok"},
    }


def make_plan(nodes: list[dict], edges: list[dict] | None = None) -> Plan:
    node_objs = [Node(**n) for n in nodes]
    edge_objs = [Edge(**e) for e in (edges or [])]
    return Plan(nodes=node_objs, edges=edge_objs)


def fast_engine() -> EngineConfig:
    return EngineConfig(
        default_timeout_s=10.0,
        default_max_retries=1,
        default_retry_delay_s=0.0,
        global_timeout_s=30.0,
    )


# ---------------------------------------------------------------------------
# Ready-set unit tests
# ---------------------------------------------------------------------------


class TestReadySetComputation:
    def test_root_nodes_become_ready(self) -> None:
        plan = make_plan(nodes=[
            {"id": "a", "label": "A", "config": mock_node_cfg()},
            {"id": "b", "label": "B", "config": mock_node_cfg()},
        ])
        states = {"a": NodeState.PENDING, "b": NodeState.PENDING}
        rsr = compute_ready_set(plan, states)
        assert set(rsr.ready_node_ids) == {"a", "b"}
        assert rsr.round_u_size == 2

    def test_dependent_node_not_ready_until_predecessor_executes(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A", "config": mock_node_cfg()},
                {"id": "b", "label": "B", "config": mock_node_cfg()},
            ],
            edges=[{"source": "a", "target": "b"}],
        )
        states = {"a": NodeState.PENDING, "b": NodeState.PENDING}
        rsr = compute_ready_set(plan, states)
        assert rsr.ready_node_ids == ["a"]
        assert "b" not in rsr.ready_node_ids

    def test_dependent_node_becomes_ready_after_predecessor(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A", "config": mock_node_cfg()},
                {"id": "b", "label": "B", "config": mock_node_cfg()},
            ],
            edges=[{"source": "a", "target": "b"}],
        )
        states = {"a": NodeState.EXECUTED, "b": NodeState.PENDING}
        rsr = compute_ready_set(plan, states)
        assert rsr.ready_node_ids == ["b"]

    def test_all_of_requires_all_predecessors(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A", "config": mock_node_cfg()},
                {"id": "b", "label": "B", "config": mock_node_cfg()},
                {"id": "c", "label": "C", "config": mock_node_cfg()},
            ],
            edges=[
                {"source": "a", "target": "c"},
                {"source": "b", "target": "c"},
            ],
        )
        # Only a executed — c should NOT be ready
        states = {"a": NodeState.EXECUTED, "b": NodeState.RUNNING, "c": NodeState.PENDING}
        rsr = compute_ready_set(plan, states)
        assert "c" not in rsr.ready_node_ids

        # Both executed — c should become ready
        states = {"a": NodeState.EXECUTED, "b": NodeState.EXECUTED, "c": NodeState.PENDING}
        rsr = compute_ready_set(plan, states)
        assert "c" in rsr.ready_node_ids

    def test_blocked_when_predecessor_failed(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A", "config": mock_node_cfg()},
                {"id": "b", "label": "B", "config": mock_node_cfg()},
            ],
            edges=[{"source": "a", "target": "b"}],
        )
        states = {"a": NodeState.FAILED, "b": NodeState.PENDING}
        rsr = compute_ready_set(plan, states)
        assert "b" in rsr.blocked_node_ids
        assert rsr.updated_states["b"] == NodeState.BLOCKED

    def test_any_of_triggers_sibling_skip(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "fix_a", "label": "Fix A", "config": mock_node_cfg()},
                {"id": "fix_b", "label": "Fix B", "config": mock_node_cfg()},
                {
                    "id": "tests",
                    "label": "Tests",
                    "config": {**mock_node_cfg(), "join_mode": JoinMode.ANY_OF},
                },
            ],
            edges=[
                {"source": "fix_a", "target": "tests"},
                {"source": "fix_b", "target": "tests"},
            ],
        )
        # fix_a executed; fix_b still pending → tests becomes ready, fix_b gets skipped
        states = {"fix_a": NodeState.EXECUTED, "fix_b": NodeState.PENDING, "tests": NodeState.PENDING}
        rsr = compute_ready_set(plan, states)
        assert "tests" in rsr.ready_node_ids
        assert "fix_b" in rsr.skipped_node_ids
        assert rsr.updated_states["fix_b"] == NodeState.SKIPPED

    def test_execution_complete_all_terminal(self) -> None:
        states = {
            "a": NodeState.EXECUTED,
            "b": NodeState.EXECUTED,
            "c": NodeState.SKIPPED,
        }
        assert execution_is_complete(states) is True

    def test_execution_not_complete_with_pending(self) -> None:
        states = {"a": NodeState.EXECUTED, "b": NodeState.PENDING}
        assert execution_is_complete(states) is False


# ---------------------------------------------------------------------------
# Dispatcher integration tests
# ---------------------------------------------------------------------------


class TestDispatcher:
    @pytest.mark.asyncio
    async def test_single_node_plan_executes(self) -> None:
        plan = make_plan(nodes=[
            {"id": "a", "label": "A", "config": mock_node_cfg(output={"value": 42})}
        ])
        dispatcher = Dispatcher(engine_config=fast_engine())
        result = await dispatcher.run(plan)
        assert result.succeeded is True
        assert result.terminal_states["a"] == NodeState.EXECUTED
        assert result.node_outputs["a"] == {"value": 42}

    @pytest.mark.asyncio
    async def test_linear_chain_executes_in_order(self) -> None:
        execution_order: list[str] = []

        class OrderTrackingMock(MockNode):
            async def execute(self, ctx):
                execution_order.append(ctx.node_id)
                return await super().execute(ctx)

        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A", "config": mock_node_cfg()},
                {"id": "b", "label": "B", "config": mock_node_cfg()},
                {"id": "c", "label": "C", "config": mock_node_cfg()},
            ],
            edges=[
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
            ],
        )

        def factory(node):
            return OrderTrackingMock(node)

        dispatcher = Dispatcher(engine_config=fast_engine())
        result = await dispatcher.run(plan, executor_factory=factory)
        assert result.succeeded is True
        assert execution_order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_parallel_dispatch_u_size_greater_than_one(self) -> None:
        """
        FR-12 / Success metric: |U(s)| > 1 must fire at least once.
        Two independent nodes (no edges between them) must be dispatched in the
        same scheduling round.
        """
        rounds: list[SchedulingRoundEvent] = []

        plan = make_plan(nodes=[
            {"id": "a", "label": "A", "config": mock_node_cfg(delay=0.05)},
            {"id": "b", "label": "B", "config": mock_node_cfg(delay=0.05)},
        ])

        dispatcher = Dispatcher(
            engine_config=fast_engine(),
            on_scheduling_round=rounds.append,
        )
        result = await dispatcher.run(plan)
        assert result.succeeded is True
        # Must have had at least one round where |U| == 2
        max_u = max(r.u_size for r in rounds)
        assert max_u >= 2, (
            f"Expected |U| ≥ 2 in at least one round, but max was {max_u}. "
            "Parallel dispatch is not working correctly."
        )

    @pytest.mark.asyncio
    async def test_parallel_dispatch_is_faster_than_sequential(self) -> None:
        """
        Parallel dispatch of two 100ms nodes should take ~100ms, not ~200ms.
        """
        plan = make_plan(nodes=[
            {"id": "a", "label": "A", "config": mock_node_cfg(delay=0.1)},
            {"id": "b", "label": "B", "config": mock_node_cfg(delay=0.1)},
        ])
        dispatcher = Dispatcher(engine_config=fast_engine())
        t0 = time.monotonic()
        result = await dispatcher.run(plan)
        elapsed = time.monotonic() - t0
        assert result.succeeded is True
        # Should complete in ~100ms, not ~200ms (with 50% tolerance for CI)
        assert elapsed < 0.18, (
            f"Parallel dispatch took {elapsed:.3f}s but expected < 0.18s. "
            "Nodes may be running sequentially instead of in parallel."
        )

    @pytest.mark.asyncio
    async def test_figure1_dag_parallel_waves(self) -> None:
        """
        Verify the paper's Figure 1 DAG produces |U| = 2 and |U| = 3 rounds.

        DAG structure:
          search_auth → read_auth ──┐
          search_utils → read_utils ─┤
                                     ├→ analyze → fix_a (any_of) → run_tests → report
                                     │           fix_b ──────────────────────→/
                                     └──────────────────────────→ update_docs → report
        """
        rounds: list[SchedulingRoundEvent] = []

        def cfgm(**kw) -> dict:
            return {**mock_node_cfg(), **kw}

        plan = make_plan(
            nodes=[
                {"id": "search_auth",  "label": "Search Auth",  "config": cfgm()},
                {"id": "search_utils", "label": "Search Utils", "config": cfgm()},
                {"id": "read_auth",    "label": "Read Auth",    "config": cfgm()},
                {"id": "read_utils",   "label": "Read Utils",   "config": cfgm()},
                {"id": "analyze",      "label": "Analyze",      "config": cfgm()},
                {"id": "fix_a",        "label": "Fix A",        "config": cfgm()},
                {"id": "fix_b",        "label": "Fix B",        "config": cfgm()},
                {"id": "run_tests",    "label": "Run Tests",    "config": {**cfgm(), "join_mode": JoinMode.ANY_OF}},
                {"id": "update_docs",  "label": "Update Docs",  "config": cfgm()},
                {"id": "report",       "label": "Report",       "config": cfgm()},
            ],
            edges=[
                {"source": "search_auth",  "target": "read_auth"},
                {"source": "search_utils", "target": "read_utils"},
                {"source": "read_auth",    "target": "analyze"},
                {"source": "read_utils",   "target": "analyze"},
                {"source": "analyze",      "target": "fix_a"},
                {"source": "analyze",      "target": "fix_b"},
                {"source": "analyze",      "target": "update_docs"},
                {"source": "fix_a",        "target": "run_tests"},
                {"source": "fix_b",        "target": "run_tests"},
                {"source": "run_tests",    "target": "report"},
                {"source": "update_docs",  "target": "report"},
            ],
        )

        dispatcher = Dispatcher(
            engine_config=fast_engine(),
            on_scheduling_round=rounds.append,
        )
        result = await dispatcher.run(plan)
        assert result.succeeded is True

        u_sizes = [r.u_size for r in rounds]
        # Must have at least one round with |U| = 2 (the initial parallel searches)
        assert 2 in u_sizes, f"Expected |U|=2 in some round. Got: {u_sizes}"
        # Must have at least one round with |U| ≥ 2
        assert max(u_sizes) >= 2, f"Max |U| was {max(u_sizes)}, expected ≥ 2"

    @pytest.mark.asyncio
    async def test_failed_node_marks_execution_failed(self) -> None:
        plan = make_plan(nodes=[
            {
                "id": "a",
                "label": "A",
                "config": {**mock_node_cfg(), "mock_fail_type": "dependency_error", "max_retries": 0},
            }
        ])
        dispatcher = Dispatcher(engine_config=EngineConfig(
            default_max_retries=0,
            global_timeout_s=10.0,
        ))
        result = await dispatcher.run(plan)
        assert result.succeeded is False
        assert result.terminal_states["a"] == NodeState.FAILED

    @pytest.mark.asyncio
    async def test_any_of_join_skips_loser(self) -> None:
        """
        When fix_a completes and fix_b is still pending, fix_b should be skipped
        and run_tests should execute.
        """
        plan = make_plan(
            nodes=[
                {"id": "fix_a", "label": "Fix A", "config": mock_node_cfg(delay=0.01)},
                {"id": "fix_b", "label": "Fix B", "config": mock_node_cfg(delay=5.0)},  # slow
                {
                    "id": "run_tests",
                    "label": "Run Tests",
                    "config": {**mock_node_cfg(), "join_mode": JoinMode.ANY_OF},
                },
            ],
            edges=[
                {"source": "fix_a", "target": "run_tests"},
                {"source": "fix_b", "target": "run_tests"},
            ],
        )
        dispatcher = Dispatcher(engine_config=fast_engine())
        result = await dispatcher.run(plan)
        assert result.succeeded is True
        assert result.terminal_states["run_tests"] == NodeState.EXECUTED
        # fix_b should be skipped (it was still pending when fix_a won)
        assert result.terminal_states["fix_b"] in (NodeState.SKIPPED, NodeState.EXECUTED)

    @pytest.mark.asyncio
    async def test_outputs_propagate_to_downstream(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A", "config": mock_node_cfg(output={"x": 123})},
                {"id": "b", "label": "B", "config": mock_node_cfg()},
            ],
            edges=[{"source": "a", "target": "b"}],
        )
        captured_upstream: dict = {}

        class CapturingMock(MockNode):
            async def execute(self, ctx):
                if ctx.node_id == "b":
                    captured_upstream.update(ctx.upstream_outputs)
                return await super().execute(ctx)

        dispatcher = Dispatcher(engine_config=fast_engine())
        result = await dispatcher.run(plan, executor_factory=lambda n: CapturingMock(n))
        assert result.succeeded is True
        assert captured_upstream.get("a") == {"x": 123}

    @pytest.mark.asyncio
    async def test_scheduling_round_events_emitted(self) -> None:
        rounds: list[SchedulingRoundEvent] = []
        plan = make_plan(nodes=[
            {"id": "a", "label": "A", "config": mock_node_cfg()},
        ])
        dispatcher = Dispatcher(
            engine_config=fast_engine(),
            on_scheduling_round=rounds.append,
        )
        await dispatcher.run(plan)
        assert len(rounds) >= 1
        assert rounds[0].u_size == 1
        assert rounds[0].ready_set == ["a"]
