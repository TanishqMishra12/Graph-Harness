"""
Tests for the SGH node state machine (NFR-1).

Covers every legal transition present in LEGAL_TRANSITIONS and every
illegal transition that should raise IllegalTransitionError.

All tests are zero-dependency (no LLM, no network, no DB).
"""
import pytest

from sgh.core.state_machine import (
    LEGAL_TRANSITIONS,
    IllegalTransitionError,
    allows_patch,
    allows_retry,
    can_be_dispatched,
    is_terminal,
    transition,
)
from sgh.core.plan import NodeState


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

ALL_STATES = list(NodeState)


# ---------------------------------------------------------------------------
# Legal transitions — every pair in LEGAL_TRANSITIONS must succeed
# ---------------------------------------------------------------------------


class TestLegalTransitions:
    """Every pair in LEGAL_TRANSITIONS must pass through transition() cleanly."""

    @pytest.mark.parametrize("from_state,to_state", list(LEGAL_TRANSITIONS))
    def test_legal_transition_succeeds(
        self, from_state: NodeState, to_state: NodeState
    ) -> None:
        result = transition(from_state, to_state, node_id="test_node")
        assert result == to_state, (
            f"transition({from_state.value!r} → {to_state.value!r}) "
            f"returned {result.value!r} instead of {to_state.value!r}"
        )

    def test_legal_transitions_are_nonempty(self) -> None:
        """Sanity check: there must be a reasonable number of legal transitions."""
        # Paper has ~20+ transitions; ensure we didn't accidentally empty the set
        assert len(LEGAL_TRANSITIONS) >= 18, (
            f"Expected ≥18 legal transitions, got {len(LEGAL_TRANSITIONS)}. "
            "Did you accidentally truncate LEGAL_TRANSITIONS?"
        )


# ---------------------------------------------------------------------------
# Illegal transitions — every pair NOT in LEGAL_TRANSITIONS must raise
# ---------------------------------------------------------------------------


class TestIllegalTransitions:
    """Specific illegal transitions that must raise IllegalTransitionError."""

    ILLEGAL_PAIRS = [
        # Terminal → anything
        (NodeState.EXECUTED,         NodeState.RUNNING),
        (NodeState.EXECUTED,         NodeState.PENDING),
        (NodeState.EXECUTED,         NodeState.FAILED),
        (NodeState.CANCELLED,        NodeState.RUNNING),
        (NodeState.CANCELLED,        NodeState.READY),
        (NodeState.SKIPPED,          NodeState.RUNNING),
        (NodeState.SKIPPED,          NodeState.EXECUTED),
        # Backwards transitions
        (NodeState.RUNNING,          NodeState.PENDING),
        (NodeState.RUNNING,          NodeState.READY),
        (NodeState.EXECUTED,         NodeState.READY),
        # Skipping intermediate states
        (NodeState.PENDING,          NodeState.EXECUTED),   # must go through running
        (NodeState.PENDING,          NodeState.RUNNING),    # must be ready first
        (NodeState.PENDING,          NodeState.FAILED),     # must run first
        (NodeState.READY,            NodeState.EXECUTED),   # must be running first
        (NodeState.READY,            NodeState.FAILED),     # must be running first
        # Recovery bypassing order
        (NodeState.PENDING,          NodeState.FAILED_RETRYABLE),
        (NodeState.READY,            NodeState.FAILED_RETRYABLE),
        # Waiting human → arbitrary
        (NodeState.WAITING_HUMAN,    NodeState.EXECUTED),
        (NodeState.WAITING_HUMAN,    NodeState.FAILED),
        (NodeState.WAITING_HUMAN,    NodeState.SKIPPED),
    ]

    @pytest.mark.parametrize("from_state,to_state", ILLEGAL_PAIRS)
    def test_illegal_transition_raises(
        self, from_state: NodeState, to_state: NodeState
    ) -> None:
        with pytest.raises(IllegalTransitionError) as exc_info:
            transition(from_state, to_state, node_id="test_node")
        err = exc_info.value
        assert err.node_id == "test_node"
        assert err.current == from_state
        assert err.target == to_state

    def test_all_non_legal_pairs_raise(self) -> None:
        """
        Exhaustive: every (from, to) pair NOT in LEGAL_TRANSITIONS must raise.

        This is the strongest form of NFR-1 coverage.
        """
        for from_state in ALL_STATES:
            for to_state in ALL_STATES:
                if (from_state, to_state) in LEGAL_TRANSITIONS:
                    continue  # legal — skip
                with pytest.raises(IllegalTransitionError):
                    transition(from_state, to_state, node_id="exhaustive_test")


# ---------------------------------------------------------------------------
# Error message quality
# ---------------------------------------------------------------------------


class TestErrorMessage:
    def test_error_message_contains_node_id(self) -> None:
        with pytest.raises(IllegalTransitionError) as exc_info:
            transition(NodeState.EXECUTED, NodeState.RUNNING, node_id="my_special_node")
        assert "my_special_node" in str(exc_info.value)

    def test_error_message_contains_state_names(self) -> None:
        with pytest.raises(IllegalTransitionError) as exc_info:
            transition(NodeState.EXECUTED, NodeState.RUNNING, node_id="n")
        msg = str(exc_info.value)
        assert "executed" in msg
        assert "running" in msg


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------


class TestPredicates:
    def test_is_terminal_executed(self) -> None:
        assert is_terminal(NodeState.EXECUTED) is True

    def test_is_terminal_failed(self) -> None:
        assert is_terminal(NodeState.FAILED) is True

    def test_is_terminal_cancelled(self) -> None:
        assert is_terminal(NodeState.CANCELLED) is True

    def test_is_terminal_skipped(self) -> None:
        assert is_terminal(NodeState.SKIPPED) is True

    def test_non_terminal_states(self) -> None:
        non_terminal = [
            NodeState.PENDING,
            NodeState.READY,
            NodeState.RUNNING,
            NodeState.WAITING_HUMAN,
            NodeState.BLOCKED,
            NodeState.FAILED_RETRYABLE,
        ]
        for state in non_terminal:
            assert is_terminal(state) is False, f"Expected {state.value!r} to be non-terminal"

    def test_can_be_dispatched_only_ready(self) -> None:
        assert can_be_dispatched(NodeState.READY) is True
        for state in ALL_STATES:
            if state != NodeState.READY:
                assert can_be_dispatched(state) is False

    def test_allows_retry_only_failed_retryable(self) -> None:
        assert allows_retry(NodeState.FAILED_RETRYABLE) is True
        for state in ALL_STATES:
            if state != NodeState.FAILED_RETRYABLE:
                assert allows_retry(state) is False

    def test_allows_patch_only_failed(self) -> None:
        assert allows_patch(NodeState.FAILED) is True
        for state in ALL_STATES:
            if state != NodeState.FAILED:
                assert allows_patch(state) is False


# ---------------------------------------------------------------------------
# NodeState enum properties
# ---------------------------------------------------------------------------


class TestNodeStateEnum:
    def test_all_terminal_states(self) -> None:
        expected_terminal = {
            NodeState.EXECUTED,
            NodeState.FAILED,
            NodeState.CANCELLED,
            NodeState.SKIPPED,
        }
        actual_terminal = {s for s in ALL_STATES if s.is_terminal}
        assert actual_terminal == expected_terminal

    def test_state_values_are_strings(self) -> None:
        for state in ALL_STATES:
            assert isinstance(state.value, str)

    def test_ten_states_total(self) -> None:
        assert len(ALL_STATES) == 10, (
            f"Expected exactly 10 node states (§6.1), found {len(ALL_STATES)}: {ALL_STATES}"
        )
