"""
Tests for SGH recovery layer — Phase 3.

Covers:
  1. Diagnoser: correct failure type classification
  2. Escalation manager: precondition enforcement, escalation ladder
  3. Failure injection: InjectedNode wraps correctly, fires on correct attempt
  4. Dispatcher integration: transient→retry, contract_violation→patch, full escalation

All tests use MockNode or InjectedNode — zero LLM calls (NFR-3).
"""
from __future__ import annotations

import pytest
import asyncio

from sgh.core.plan import Edge, JoinMode, Node, NodeConfig, NodeState, NodeType, Plan, RecoveryLevel
from sgh.nodes.base import NodeOutput
from sgh.nodes.failure_injection import FailureInjector, FailureSpec, InjectedNode
from sgh.nodes.mock_node import MockNode
from sgh.recovery.diagnoser import FailureType, diagnose
from sgh.recovery.escalation import (
    EscalationViolationError,
    RecoveryActionType,
    RecoveryManager,
)
from sgh.scheduler.dispatcher import Dispatcher, EngineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plan(nodes: list[dict], edges: list[dict] | None = None) -> Plan:
    node_objs = [Node(**n) for n in nodes]
    edge_objs = [Edge(**e) for e in (edges or [])]
    return Plan(nodes=node_objs, edges=edge_objs)


def mock_cfg(**kw) -> dict:
    return {"node_type": NodeType.MOCK, **kw}


def fast_engine(max_retries: int = 1) -> EngineConfig:
    return EngineConfig(
        default_timeout_s=10.0,
        default_max_retries=max_retries,
        default_retry_delay_s=0.0,
        global_timeout_s=30.0,
    )


# ---------------------------------------------------------------------------
# Diagnoser tests
# ---------------------------------------------------------------------------


class TestDiagnoser:
    def test_transient_from_retry_outcome(self) -> None:
        output = NodeOutput(outcome="retry", failure_type="transient")
        assert diagnose(output) == FailureType.TRANSIENT

    def test_contract_violation_from_failure_type(self) -> None:
        output = NodeOutput(outcome="failure", failure_type="contract_violation")
        assert diagnose(output) == FailureType.CONTRACT_VIOLATION

    def test_dependency_error_from_failure_type(self) -> None:
        output = NodeOutput(outcome="failure", failure_type="dependency_error")
        assert diagnose(output) == FailureType.DEPENDENCY_ERROR

    def test_retry_outcome_no_explicit_type_defaults_transient(self) -> None:
        output = NodeOutput(outcome="retry", failure_type="")
        assert diagnose(output) == FailureType.TRANSIENT

    def test_unknown_failure_type_returns_unknown(self) -> None:
        output = NodeOutput(outcome="failure", failure_type="bizarre_error")
        assert diagnose(output) == FailureType.UNKNOWN

    def test_diagnose_raises_on_success(self) -> None:
        output = NodeOutput(outcome="success")
        with pytest.raises(ValueError, match="successful"):
            diagnose(output)


# ---------------------------------------------------------------------------
# RecoveryManager escalation tests
# ---------------------------------------------------------------------------


class TestRecoveryManager:
    def _make_manager(self, node_id: str = "node_a") -> tuple[RecoveryManager, Plan]:
        plan = make_plan([{"id": node_id, "label": "A", "config": mock_cfg()}])
        return RecoveryManager(plan), plan

    def test_initial_state_is_pristine(self) -> None:
        manager, _ = self._make_manager("a")
        assert manager.recovery_state("a") == RecoveryLevel.PRISTINE

    def test_transient_on_pristine_selects_retry(self) -> None:
        manager, _ = self._make_manager("a")
        action = manager.select_action(
            node_id="a",
            failure_type=FailureType.TRANSIENT,
            current_node_state=NodeState.FAILED_RETRYABLE,
            all_states={"a": NodeState.FAILED_RETRYABLE},
        )
        assert action.action_type == RecoveryActionType.LOCAL_RETRY
        assert manager.recovery_state("a") == RecoveryLevel.RETRIED

    def test_retry_transitions_recovery_state_to_retried(self) -> None:
        manager, _ = self._make_manager("a")
        manager.select_action(
            node_id="a",
            failure_type=FailureType.TRANSIENT,
            current_node_state=NodeState.FAILED_RETRYABLE,
            all_states={"a": NodeState.FAILED_RETRYABLE},
        )
        assert manager.recovery_state("a") == RecoveryLevel.RETRIED

    def test_contract_violation_on_pristine_selects_patch(self) -> None:
        """CONTRACT_VIOLATION on first attempt should skip retry (won't help) and go to patch."""
        manager, _ = self._make_manager("a")
        action = manager.select_action(
            node_id="a",
            failure_type=FailureType.CONTRACT_VIOLATION,
            current_node_state=NodeState.FAILED,
            all_states={"a": NodeState.FAILED},
        )
        assert action.action_type == RecoveryActionType.LOCAL_PATCH
        assert manager.recovery_state("a") == RecoveryLevel.PATCHED

    def test_failure_on_retried_selects_patch(self) -> None:
        manager, _ = self._make_manager("a")
        # First: transient → local_retry → RETRIED
        manager.select_action(
            node_id="a",
            failure_type=FailureType.TRANSIENT,
            current_node_state=NodeState.FAILED_RETRYABLE,
            all_states={"a": NodeState.FAILED_RETRYABLE},
        )
        assert manager.recovery_state("a") == RecoveryLevel.RETRIED
        # Second failure → local_patch
        action = manager.select_action(
            node_id="a",
            failure_type=FailureType.TRANSIENT,
            current_node_state=NodeState.FAILED_RETRYABLE,
            all_states={"a": NodeState.FAILED_RETRYABLE},
        )
        assert action.action_type == RecoveryActionType.LOCAL_PATCH
        assert manager.recovery_state("a") == RecoveryLevel.PATCHED

    def test_patched_config_clears_mock_fail_type(self) -> None:
        plan = make_plan([{
            "id": "a",
            "label": "A",
            "config": mock_cfg(mock_fail_type="transient"),
        }])
        manager = RecoveryManager(plan)
        action = manager.select_action(
            node_id="a",
            failure_type=FailureType.CONTRACT_VIOLATION,
            current_node_state=NodeState.FAILED,
            all_states={"a": NodeState.FAILED},
        )
        assert action.should_patch
        assert action.patched_config is not None
        # Patched config must clear mock_fail_type so the next attempt succeeds
        assert action.patched_config.mock_fail_type is None

    def test_replan_when_all_failed_nodes_patched(self) -> None:
        plan = make_plan([{"id": "a", "label": "A", "config": mock_cfg()}])
        manager = RecoveryManager(plan)
        # Get to PATCHED state
        manager.select_action(
            node_id="a",
            failure_type=FailureType.CONTRACT_VIOLATION,
            current_node_state=NodeState.FAILED,
            all_states={"a": NodeState.FAILED},
        )
        assert manager.recovery_state("a") == RecoveryLevel.PATCHED
        # Third failure — all patched → request_replan
        action = manager.select_action(
            node_id="a",
            failure_type=FailureType.CONTRACT_VIOLATION,
            current_node_state=NodeState.FAILED,
            all_states={"a": NodeState.FAILED},
        )
        assert action.action_type == RecoveryActionType.REQUEST_REPLAN

    def test_escalation_violation_retry_on_retried_state(self) -> None:
        """Calling _apply_retry when already RETRIED must raise EscalationViolationError."""
        manager, _ = self._make_manager("a")
        manager._recovery_state["a"] = RecoveryLevel.RETRIED
        with pytest.raises(EscalationViolationError) as exc_info:
            manager._apply_retry("a", reason="test")
        err = exc_info.value
        assert err.node_id == "a"
        assert "local_retry" in str(err)

    def test_escalation_violation_patch_on_patched_state(self) -> None:
        """Calling _apply_patch when already PATCHED must raise EscalationViolationError."""
        manager, _ = self._make_manager("a")
        manager._recovery_state["a"] = RecoveryLevel.PATCHED
        with pytest.raises(EscalationViolationError) as exc_info:
            manager._apply_patch("a", failure_type=FailureType.TRANSIENT, reason="test")
        err = exc_info.value
        assert err.node_id == "a"
        assert "local_patch" in str(err)

    def test_dependency_error_selects_no_action(self) -> None:
        manager, _ = self._make_manager("a")
        action = manager.select_action(
            node_id="a",
            failure_type=FailureType.DEPENDENCY_ERROR,
            current_node_state=NodeState.FAILED,
            all_states={"a": NodeState.FAILED},
        )
        assert action.is_exhausted

    def test_replan_precondition_requires_all_patched(self) -> None:
        plan = make_plan([
            {"id": "a", "label": "A", "config": mock_cfg()},
            {"id": "b", "label": "B", "config": mock_cfg()},
        ])
        manager = RecoveryManager(plan)
        # Only 'a' is PATCHED; 'b' is still PRISTINE
        manager._recovery_state["a"] = RecoveryLevel.PATCHED
        assert manager.replan_precondition_met(["a", "b"]) is False
        manager._recovery_state["b"] = RecoveryLevel.PATCHED
        assert manager.replan_precondition_met(["a", "b"]) is True


# ---------------------------------------------------------------------------
# Failure injection tests
# ---------------------------------------------------------------------------


class TestFailureInjector:
    def _make_node(self, node_id: str = "fix_a") -> Node:
        return Node(
            id=node_id,
            label="Fix A",
            config=NodeConfig(node_type=NodeType.MOCK, mock_output={"result": "ok"}),
        )

    @pytest.mark.asyncio
    async def test_inject_transient_on_attempt_1(self) -> None:
        node = self._make_node("fix_a")
        injector = FailureInjector([FailureSpec("fix_a", "transient", on_attempt=1)])
        wrapped = injector.wrap_executor(node, MockNode(node))
        from sgh.nodes.base import ExecContext
        ctx = ExecContext(
            plan_id="test", plan_version=1, node_id="fix_a", node_label="Fix A",
            config=node.config, upstream_outputs={}, attempt_number=1,
        )
        result = await wrapped.execute(ctx)
        assert result.outcome == "retry"
        assert result.failure_type == "transient"
        assert "[INJECTED]" in result.error_message

    @pytest.mark.asyncio
    async def test_no_injection_on_attempt_2_when_spec_says_1(self) -> None:
        node = self._make_node("fix_a")
        injector = FailureInjector([FailureSpec("fix_a", "transient", on_attempt=1)])
        wrapped = injector.wrap_executor(node, MockNode(node))
        from sgh.nodes.base import ExecContext
        ctx = ExecContext(
            plan_id="test", plan_version=1, node_id="fix_a", node_label="Fix A",
            config=node.config, upstream_outputs={}, attempt_number=2,
        )
        result = await wrapped.execute(ctx)
        # Attempt 2 should pass through to real MockNode
        assert result.outcome == "success"

    @pytest.mark.asyncio
    async def test_inject_every_attempt_when_on_attempt_is_none(self) -> None:
        node = self._make_node("fix_a")
        injector = FailureInjector([FailureSpec("fix_a", "contract_violation", on_attempt=None)])
        wrapped = injector.wrap_executor(node, MockNode(node))
        from sgh.nodes.base import ExecContext
        for attempt in [1, 2, 3]:
            ctx = ExecContext(
                plan_id="test", plan_version=1, node_id="fix_a", node_label="Fix A",
                config=node.config, upstream_outputs={}, attempt_number=attempt,
            )
            result = await wrapped.execute(ctx)
            assert result.outcome == "failure", f"Expected failure on attempt {attempt}"
            assert result.failure_type == "contract_violation"

    def test_no_injection_for_unregistered_node(self) -> None:
        node = self._make_node("other_node")
        injector = FailureInjector([FailureSpec("fix_a", "transient", on_attempt=1)])
        # wrap_executor returns the original node for unregistered nodes
        wrapped = injector.wrap_executor(node, MockNode(node))
        assert not isinstance(wrapped, InjectedNode)

    def test_from_dict(self) -> None:
        injector = FailureInjector.from_dict({
            "fix_a": "transient",
            "analyze": {"type": "contract_violation", "attempt": 2},
        })
        assert injector.should_inject("fix_a", 1) == "transient"
        assert injector.should_inject("fix_a", 2) is None  # spec says attempt=1
        assert injector.should_inject("analyze", 2) == "contract_violation"
        assert injector.should_inject("analyze", 1) is None


# ---------------------------------------------------------------------------
# Dispatcher end-to-end recovery integration tests
# ---------------------------------------------------------------------------


class TestDispatcherRecovery:
    @pytest.mark.asyncio
    async def test_transient_failure_triggers_local_retry_and_succeeds(self) -> None:
        """
        Node fails with transient error on attempt 1, succeeds on attempt 2.
        Expected: local_retry fires, node ends in EXECUTED.
        """
        plan = make_plan([{
            "id": "a",
            "label": "A",
            "config": mock_cfg(mock_output={"result": "ok"}),
        }])

        # Inject transient failure on attempt 1 only
        injector = FailureInjector([FailureSpec("a", "transient", on_attempt=1)])
        dispatcher = Dispatcher(
            engine_config=fast_engine(max_retries=1),
            failure_injector=injector,
        )
        result = await dispatcher.run(plan)
        assert result.succeeded is True
        assert result.terminal_states["a"] == NodeState.EXECUTED

    @pytest.mark.asyncio
    async def test_contract_violation_triggers_local_patch_and_succeeds(self) -> None:
        """
        Node fails with contract_violation on attempt 1, succeeds on attempt 2 (patched).
        local_patch clears mock_fail_type so MockNode succeeds on next attempt.
        Expected: node ends in EXECUTED after patch.
        """
        plan = make_plan([{
            "id": "a",
            "label": "A",
            "config": mock_cfg(mock_output={"result": "patched_ok"}),
        }])

        # Contract violation on attempt 1 only
        injector = FailureInjector([FailureSpec("a", "contract_violation", on_attempt=1)])
        dispatcher = Dispatcher(
            engine_config=fast_engine(),
            failure_injector=injector,
        )
        result = await dispatcher.run(plan)
        # After patch, attempt 2 should succeed
        assert result.terminal_states["a"] == NodeState.EXECUTED
        assert result.succeeded is True

    @pytest.mark.asyncio
    async def test_permanent_failure_after_all_recovery_exhausted(self) -> None:
        """
        Node fails on ALL attempts (on_attempt=None).
        Expected: local_retry → local_patch → request_replan → FAILED (v1).
        """
        plan = make_plan([{
            "id": "a",
            "label": "A",
            "config": mock_cfg(mock_output={"result": "ok"}),
        }])

        # Always inject contract violation
        injector = FailureInjector([FailureSpec("a", "contract_violation", on_attempt=None)])
        dispatcher = Dispatcher(
            engine_config=fast_engine(max_retries=0),
            failure_injector=injector,
        )
        result = await dispatcher.run(plan)
        # After all recovery exhausted, node must be FAILED
        assert result.terminal_states["a"] == NodeState.FAILED
        assert result.succeeded is False

    @pytest.mark.asyncio
    async def test_transient_then_patch_full_escalation(self) -> None:
        """
        Attempt 1: transient → local_retry
        Attempt 2: transient again → local_patch
        Attempt 3: success (patch cleared the fail)
        """
        plan = make_plan([{
            "id": "a",
            "label": "A",
            "config": mock_cfg(mock_output={"result": "ok"}),
        }])

        # Transient on attempts 1 and 2, success on 3+
        injector = FailureInjector([
            FailureSpec("a", "transient", on_attempt=1),
            FailureSpec("a", "transient", on_attempt=2),
        ])
        dispatcher = Dispatcher(
            engine_config=fast_engine(max_retries=1),
            failure_injector=injector,
        )
        result = await dispatcher.run(plan)
        assert result.terminal_states["a"] == NodeState.EXECUTED
        assert result.succeeded is True

    @pytest.mark.asyncio
    async def test_recovery_does_not_affect_unrelated_nodes(self) -> None:
        """
        Node 'a' fails and recovers. Node 'b' (independent) runs without any intervention.
        Both should end in EXECUTED.
        """
        plan = make_plan(nodes=[
            {"id": "a", "label": "A", "config": mock_cfg(mock_output={"a": 1})},
            {"id": "b", "label": "B", "config": mock_cfg(mock_output={"b": 2})},
        ])

        injector = FailureInjector([FailureSpec("a", "transient", on_attempt=1)])
        dispatcher = Dispatcher(engine_config=fast_engine(), failure_injector=injector)
        result = await dispatcher.run(plan)
        assert result.terminal_states["a"] == NodeState.EXECUTED
        assert result.terminal_states["b"] == NodeState.EXECUTED
        assert result.succeeded is True

    @pytest.mark.asyncio
    async def test_failed_node_blocks_downstream(self) -> None:
        """
        When 'a' permanently fails after all recovery, downstream 'b' becomes blocked→failed.
        """
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A", "config": mock_cfg()},
                {"id": "b", "label": "B", "config": mock_cfg()},
            ],
            edges=[{"source": "a", "target": "b"}],
        )

        # 'a' always fails
        injector = FailureInjector([FailureSpec("a", "contract_violation", on_attempt=None)])
        dispatcher = Dispatcher(
            engine_config=fast_engine(max_retries=0),
            failure_injector=injector,
        )
        result = await dispatcher.run(plan)
        assert result.terminal_states["a"] == NodeState.FAILED
        # 'b' should be blocked (pending→blocked, then blocked→failed or stays blocked)
        assert result.terminal_states["b"] in (NodeState.BLOCKED, NodeState.FAILED)
        assert result.succeeded is False
