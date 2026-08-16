"""
Tests for SGH join semantics (§7.1 all_of, §7.2 any_of).

All tests are zero-dependency (no LLM, no network, no DB).
"""
import pytest

from sgh.core.joins import JoinEvaluator, JoinResult
from sgh.core.plan import Edge, JoinMode, Node, NodeConfig, NodeState, Plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_plan(
    nodes: list[dict],
    edges: list[dict] | None = None,
) -> Plan:
    node_objs = [Node(**n) if isinstance(n, dict) else n for n in nodes]
    edge_objs = [Edge(**e) if isinstance(e, dict) else e for e in (edges or [])]
    return Plan(nodes=node_objs, edges=edge_objs)


E = NodeState.EXECUTED
P = NodeState.PENDING
R = NodeState.RUNNING
F = NodeState.FAILED
C = NodeState.CANCELLED
S = NodeState.SKIPPED
FR = NodeState.FAILED_RETRYABLE


# ---------------------------------------------------------------------------
# ALL_OF join
# ---------------------------------------------------------------------------


class TestAllOf:
    def test_satisfied_when_all_executed(self) -> None:
        states = {"a": E, "b": E}
        assert JoinEvaluator.all_of_satisfied(states) is True

    def test_not_satisfied_when_one_pending(self) -> None:
        states = {"a": E, "b": P}
        assert JoinEvaluator.all_of_satisfied(states) is False

    def test_not_satisfied_when_all_running(self) -> None:
        states = {"a": R, "b": R}
        assert JoinEvaluator.all_of_satisfied(states) is False

    def test_not_satisfied_when_one_failed(self) -> None:
        states = {"a": E, "b": F}
        assert JoinEvaluator.all_of_satisfied(states) is False

    def test_blocked_when_one_failed(self) -> None:
        states = {"a": E, "b": F}
        assert JoinEvaluator.all_of_blocked(states) is True

    def test_blocked_when_one_cancelled(self) -> None:
        states = {"a": E, "b": C}
        assert JoinEvaluator.all_of_blocked(states) is True

    def test_not_blocked_when_all_pending(self) -> None:
        states = {"a": P, "b": P}
        assert JoinEvaluator.all_of_blocked(states) is False

    def test_single_predecessor_all_of(self) -> None:
        assert JoinEvaluator.all_of_satisfied({"a": E}) is True
        assert JoinEvaluator.all_of_satisfied({"a": P}) is False

    def test_three_predecessors_satisfied(self) -> None:
        states = {"a": E, "b": E, "c": E}
        assert JoinEvaluator.all_of_satisfied(states) is True

    def test_three_predecessors_one_running(self) -> None:
        states = {"a": E, "b": E, "c": R}
        assert JoinEvaluator.all_of_satisfied(states) is False
        assert JoinEvaluator.all_of_blocked(states) is False


# ---------------------------------------------------------------------------
# ANY_OF join
# ---------------------------------------------------------------------------


class TestAnyOf:
    def test_winner_when_one_executed(self) -> None:
        states = {"a": P, "b": E}
        winner = JoinEvaluator.any_of_winner(states)
        assert winner == "b"

    def test_winner_is_lexicographically_smallest_when_tie(self) -> None:
        # Both executed simultaneously — lexicographic min wins
        states = {"fix_b": E, "fix_a": E}
        winner = JoinEvaluator.any_of_winner(states)
        assert winner == "fix_a"

    def test_no_winner_when_all_pending(self) -> None:
        states = {"a": P, "b": P}
        assert JoinEvaluator.any_of_winner(states) is None

    def test_no_winner_when_all_running(self) -> None:
        states = {"a": R, "b": R}
        assert JoinEvaluator.any_of_winner(states) is None

    def test_satisfied_when_one_executed(self) -> None:
        states = {"a": E, "b": P}
        assert JoinEvaluator.any_of_satisfied(states) is True

    def test_not_satisfied_when_all_pending(self) -> None:
        states = {"a": P, "b": P}
        assert JoinEvaluator.any_of_satisfied(states) is False

    def test_blocked_when_all_failed_or_cancelled(self) -> None:
        states = {"a": F, "b": C}
        assert JoinEvaluator.any_of_blocked(states) is True

    def test_not_blocked_when_one_still_running(self) -> None:
        states = {"a": F, "b": R}
        assert JoinEvaluator.any_of_blocked(states) is False

    def test_not_blocked_when_one_executed(self) -> None:
        # One succeeded — not blocked (satisfied)
        states = {"a": E, "b": F}
        assert JoinEvaluator.any_of_blocked(states) is False

    def test_three_way_any_of_winner(self) -> None:
        states = {"fix_a": R, "fix_b": E, "fix_c": R}
        winner = JoinEvaluator.any_of_winner(states)
        assert winner == "fix_b"


# ---------------------------------------------------------------------------
# Sibling skip computation
# ---------------------------------------------------------------------------


class TestSiblingSkips:
    def _make_any_of_plan(self) -> Plan:
        """
        Creates: fix_a → run_tests
                 fix_b → run_tests   (any_of)
                 fix_c → run_tests
        """
        return make_plan(
            nodes=[
                {"id": "fix_a", "label": "Fix A"},
                {"id": "fix_b", "label": "Fix B"},
                {"id": "fix_c", "label": "Fix C"},
                {
                    "id": "run_tests",
                    "label": "Run Tests",
                    "config": {"join_mode": JoinMode.ANY_OF},
                },
            ],
            edges=[
                {"source": "fix_a", "target": "run_tests"},
                {"source": "fix_b", "target": "run_tests"},
                {"source": "fix_c", "target": "run_tests"},
            ],
        )

    def test_siblings_to_skip_excludes_winner(self) -> None:
        plan = self._make_any_of_plan()
        siblings = JoinEvaluator.get_siblings_to_skip(plan, "fix_a", "run_tests")
        assert "fix_a" not in siblings
        assert "fix_b" in siblings
        assert "fix_c" in siblings

    def test_siblings_to_skip_all_except_winner(self) -> None:
        plan = self._make_any_of_plan()
        siblings = JoinEvaluator.get_siblings_to_skip(plan, "fix_b", "run_tests")
        assert set(siblings) == {"fix_a", "fix_c"}

    def test_two_node_any_of_one_sibling(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "fix_a", "label": "Fix A"},
                {"id": "fix_b", "label": "Fix B"},
                {
                    "id": "run_tests",
                    "label": "Run Tests",
                    "config": {"join_mode": JoinMode.ANY_OF},
                },
            ],
            edges=[
                {"source": "fix_a", "target": "run_tests"},
                {"source": "fix_b", "target": "run_tests"},
            ],
        )
        siblings = JoinEvaluator.get_siblings_to_skip(plan, "fix_a", "run_tests")
        assert siblings == ["fix_b"]


# ---------------------------------------------------------------------------
# Unified evaluate() — integration
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_root_node_always_satisfied(self) -> None:
        plan = make_plan(nodes=[{"id": "root", "label": "Root"}])
        states = {"root": P}
        result = JoinEvaluator.evaluate(plan, "root", states)
        assert result.satisfied is True
        assert result.blocked is False

    def test_all_of_two_predecessors_both_executed(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A"},
                {"id": "b", "label": "B"},
                {"id": "c", "label": "C"},
            ],
            edges=[
                {"source": "a", "target": "c"},
                {"source": "b", "target": "c"},
            ],
        )
        states = {"a": E, "b": E, "c": P}
        result = JoinEvaluator.evaluate(plan, "c", states)
        assert result.satisfied is True

    def test_all_of_one_still_running(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A"},
                {"id": "b", "label": "B"},
                {"id": "c", "label": "C"},
            ],
            edges=[
                {"source": "a", "target": "c"},
                {"source": "b", "target": "c"},
            ],
        )
        states = {"a": E, "b": R, "c": P}
        result = JoinEvaluator.evaluate(plan, "c", states)
        assert result.satisfied is False
        assert result.blocked is False

    def test_all_of_blocked_by_failed_predecessor(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A"},
                {"id": "b", "label": "B"},
                {"id": "c", "label": "C"},
            ],
            edges=[
                {"source": "a", "target": "c"},
                {"source": "b", "target": "c"},
            ],
        )
        states = {"a": E, "b": F, "c": P}
        result = JoinEvaluator.evaluate(plan, "c", states)
        assert result.satisfied is False
        assert result.blocked is True

    def test_any_of_fires_and_returns_siblings(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "fix_a", "label": "Fix A"},
                {"id": "fix_b", "label": "Fix B"},
                {
                    "id": "run_tests",
                    "label": "Run Tests",
                    "config": {"join_mode": JoinMode.ANY_OF},
                },
            ],
            edges=[
                {"source": "fix_a", "target": "run_tests"},
                {"source": "fix_b", "target": "run_tests"},
            ],
        )
        states = {"fix_a": R, "fix_b": E, "run_tests": P}
        result = JoinEvaluator.evaluate(plan, "run_tests", states)
        assert result.satisfied is True
        assert result.winner_id == "fix_b"
        assert result.siblings_to_skip == ["fix_a"]

    def test_any_of_all_failed_is_blocked(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "fix_a", "label": "Fix A"},
                {"id": "fix_b", "label": "Fix B"},
                {
                    "id": "run_tests",
                    "label": "Run Tests",
                    "config": {"join_mode": JoinMode.ANY_OF},
                },
            ],
            edges=[
                {"source": "fix_a", "target": "run_tests"},
                {"source": "fix_b", "target": "run_tests"},
            ],
        )
        states = {"fix_a": F, "fix_b": C, "run_tests": P}
        result = JoinEvaluator.evaluate(plan, "run_tests", states)
        assert result.satisfied is False
        assert result.blocked is True
