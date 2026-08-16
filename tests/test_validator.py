"""
Tests for the SGH plan validator (FR-1).

All tests are zero-dependency (no LLM, no network, no DB).
"""
import pytest

from sgh.core.plan import Edge, JoinMode, Node, NodeConfig, Plan
from sgh.core.validator import PlanValidationError, validate_plan


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_plan(
    nodes: list[dict],
    edges: list[dict] | None = None,
    plan_id: str = "test-plan",
) -> Plan:
    """Convenience factory for test plans."""
    node_objs = [Node(**n) if isinstance(n, dict) else n for n in nodes]
    edge_objs = [Edge(**e) if isinstance(e, dict) else e for e in (edges or [])]
    return Plan(plan_id=plan_id, nodes=node_objs, edges=edge_objs)


# ---------------------------------------------------------------------------
# Valid plans
# ---------------------------------------------------------------------------


class TestValidPlans:
    def test_single_node_plan_is_valid(self) -> None:
        plan = make_plan(nodes=[{"id": "a", "label": "A"}])
        validate_plan(plan)  # must not raise

    def test_linear_chain_is_valid(self) -> None:
        plan = make_plan(
            nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            edges=[{"source": "a", "target": "b"}],
        )
        validate_plan(plan)

    def test_diamond_dag_is_valid(self) -> None:
        # a → b → d
        #   → c → d
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A"},
                {"id": "b", "label": "B"},
                {"id": "c", "label": "C"},
                {"id": "d", "label": "D"},
            ],
            edges=[
                {"source": "a", "target": "b"},
                {"source": "a", "target": "c"},
                {"source": "b", "target": "d"},
                {"source": "c", "target": "d"},
            ],
        )
        validate_plan(plan)

    def test_any_of_with_two_predecessors_is_valid(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A"},
                {"id": "b", "label": "B"},
                {
                    "id": "c",
                    "label": "C",
                    "config": {"join_mode": JoinMode.ANY_OF},
                },
            ],
            edges=[
                {"source": "a", "target": "c"},
                {"source": "b", "target": "c"},
            ],
        )
        validate_plan(plan)

    def test_parallel_roots_are_valid(self) -> None:
        # Two independent roots, both feeding into a join
        plan = make_plan(
            nodes=[
                {"id": "root1", "label": "Root1"},
                {"id": "root2", "label": "Root2"},
                {"id": "join", "label": "Join"},
            ],
            edges=[
                {"source": "root1", "target": "join"},
                {"source": "root2", "target": "join"},
            ],
        )
        validate_plan(plan)


# ---------------------------------------------------------------------------
# Cyclic graphs
# ---------------------------------------------------------------------------


class TestCycleDetection:
    def test_simple_cycle_raises(self) -> None:
        # a → b → a
        plan = make_plan(
            nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            edges=[
                {"source": "a", "target": "b"},
                {"source": "b", "target": "a"},
            ],
        )
        with pytest.raises(PlanValidationError, match="[Cc]ycle"):
            validate_plan(plan)

    def test_self_loop_raises(self) -> None:
        plan = make_plan(
            nodes=[{"id": "a", "label": "A"}],
            edges=[{"source": "a", "target": "a"}],
        )
        with pytest.raises(PlanValidationError, match="[Ss]elf"):
            validate_plan(plan)

    def test_three_node_cycle_raises(self) -> None:
        # a → b → c → a
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A"},
                {"id": "b", "label": "B"},
                {"id": "c", "label": "C"},
            ],
            edges=[
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
                {"source": "c", "target": "a"},
            ],
        )
        with pytest.raises(PlanValidationError, match="[Cc]ycle"):
            validate_plan(plan)


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------


class TestEdgeIntegrity:
    def test_missing_source_node_raises(self) -> None:
        plan = make_plan(
            nodes=[{"id": "a", "label": "A"}],
            edges=[{"source": "nonexistent", "target": "a"}],
        )
        with pytest.raises(PlanValidationError, match="source"):
            validate_plan(plan)

    def test_missing_target_node_raises(self) -> None:
        plan = make_plan(
            nodes=[{"id": "a", "label": "A"}],
            edges=[{"source": "a", "target": "nonexistent"}],
        )
        with pytest.raises(PlanValidationError, match="target"):
            validate_plan(plan)

    def test_duplicate_edges_raises(self) -> None:
        plan = make_plan(
            nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            edges=[
                {"source": "a", "target": "b"},
                {"source": "a", "target": "b"},  # duplicate
            ],
        )
        with pytest.raises(PlanValidationError, match="[Dd]uplicate"):
            validate_plan(plan)


# ---------------------------------------------------------------------------
# Node ID uniqueness
# ---------------------------------------------------------------------------


class TestNodeIdUniqueness:
    def test_duplicate_node_id_raises(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "First"},
                {"id": "a", "label": "Second"},  # duplicate ID
            ],
        )
        with pytest.raises(PlanValidationError, match="[Dd]uplicate"):
            validate_plan(plan)


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


class TestReachability:
    def test_disconnected_node_raises(self) -> None:
        # 'orphan' has no edges to/from the main chain
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A"},
                {"id": "b", "label": "B"},
                {"id": "orphan", "label": "Orphan"},
            ],
            edges=[{"source": "a", "target": "b"}],
        )
        with pytest.raises(PlanValidationError, match="[Uu]nreachable|[Ii]solated"):
            validate_plan(plan)


# ---------------------------------------------------------------------------
# any_of join arity
# ---------------------------------------------------------------------------


class TestAnyOfArity:
    def test_any_of_with_single_predecessor_raises(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A"},
                {
                    "id": "b",
                    "label": "B",
                    "config": {"join_mode": JoinMode.ANY_OF},
                },
            ],
            edges=[{"source": "a", "target": "b"}],
        )
        with pytest.raises(PlanValidationError, match="any_of"):
            validate_plan(plan)

    def test_any_of_with_zero_predecessors_raises(self) -> None:
        # Root node with any_of — makes no sense (no predecessors to join)
        plan = make_plan(
            nodes=[
                {
                    "id": "a",
                    "label": "A",
                    "config": {"join_mode": JoinMode.ANY_OF},
                }
            ],
        )
        with pytest.raises(PlanValidationError, match="any_of"):
            validate_plan(plan)

    def test_any_of_with_three_predecessors_is_valid(self) -> None:
        plan = make_plan(
            nodes=[
                {"id": "a", "label": "A"},
                {"id": "b", "label": "B"},
                {"id": "c", "label": "C"},
                {
                    "id": "d",
                    "label": "D",
                    "config": {"join_mode": JoinMode.ANY_OF},
                },
            ],
            edges=[
                {"source": "a", "target": "d"},
                {"source": "b", "target": "d"},
                {"source": "c", "target": "d"},
            ],
        )
        validate_plan(plan)  # must not raise


# ---------------------------------------------------------------------------
# Plan immutability
# ---------------------------------------------------------------------------


class TestPlanImmutability:
    def test_plan_is_frozen_in_model_config(self) -> None:
        """Plan's Pydantic model_config must declare frozen=True (FR-9 / §5.2)."""
        from pydantic import ConfigDict
        plan = make_plan(nodes=[{"id": "a", "label": "A"}])
        # Pydantic v2 stores config in model_config class attribute
        assert plan.model_config.get("frozen") is True, (
            "Plan must be frozen (model_config = ConfigDict(frozen=True)) "
            "to enforce plan-version immutability (FR-9, §5.2)."
        )

    def test_plan_nodes_are_tuple(self) -> None:
        """Nodes must be stored as a tuple (immutable) not a list."""
        plan = make_plan(nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}])
        assert isinstance(plan.nodes, tuple), (
            f"plan.nodes should be a tuple, got {type(plan.nodes).__name__}"
        )

    def test_plan_edges_are_tuple(self) -> None:
        """Edges must be stored as a tuple (immutable) not a list."""
        plan = make_plan(
            nodes=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            edges=[{"source": "a", "target": "b"}],
        )
        assert isinstance(plan.edges, tuple)

    def test_plan_version_in_error_messages(self) -> None:
        plan = make_plan(
            nodes=[{"id": "a", "label": "A"}, {"id": "a", "label": "B"}],
        )
        plan = Plan(plan_id="my-plan", version=3, nodes=plan.nodes, edges=plan.edges)
        with pytest.raises(PlanValidationError) as exc_info:
            validate_plan(plan)
        assert "v3" in str(exc_info.value)
        assert "my-plan" in str(exc_info.value)
