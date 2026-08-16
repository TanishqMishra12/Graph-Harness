"""
DAG plan validator for SGH.

Validates an execution plan Π before any execution begins (FR-1, §5.1 / Appendix A.2).

Checks performed (in order):
  1. Node ID uniqueness
  2. Edge referential integrity (both endpoints must exist as node IDs)
  3. Acyclicity — uses networkx DFS; rejects any cycle (§5.1 requires a DAG)
  4. Reachability — every non-root node must be reachable from at least one root
  5. Any_of join consistency — a node with join_mode=ANY_OF must have ≥2 predecessors
     (a single-predecessor any_of is degenerate and likely a misconfiguration)
  6. Edge set coherence — no duplicate edges, no self-loops

These checks mirror the DAG validity pre-conditions required before the ready-set
function U(s) can be safely computed (Definition 3.1, §3.1).
"""
from __future__ import annotations

import networkx as nx

from sgh.core.plan import JoinMode, Plan


class PlanValidationError(ValueError):
    """Raised when a Plan fails structural validation before execution."""


def validate_plan(plan: Plan) -> None:
    """
    Validate plan Π.  Raises PlanValidationError on the first detected problem.

    This function must be called before the Dispatcher accepts a plan.
    It is also used by `sgh validate` (CLI) and the planner layer.

    Args:
        plan: The Plan to validate.

    Raises:
        PlanValidationError: With a descriptive message indicating which check failed.
    """
    _check_node_id_uniqueness(plan)
    _check_edge_integrity(plan)
    _check_no_self_loops(plan)
    _check_no_duplicate_edges(plan)
    g = _build_digraph(plan)
    _check_acyclicity(plan, g)
    _check_reachability(plan, g)
    _check_any_of_join_arity(plan)
    _check_any_of_consistency(plan)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_node_id_uniqueness(plan: Plan) -> None:
    seen: set[str] = set()
    for node in plan.nodes:
        if node.id in seen:
            raise PlanValidationError(
                f"Plan {plan.plan_id!r} v{plan.version}: "
                f"Duplicate node ID {node.id!r}. Node IDs must be unique within a plan."
            )
        seen.add(node.id)


def _check_edge_integrity(plan: Plan) -> None:
    node_ids = plan.node_ids()
    for edge in plan.edges:
        if edge.source not in node_ids:
            raise PlanValidationError(
                f"Plan {plan.plan_id!r} v{plan.version}: "
                f"Edge source {edge.source!r} does not exist as a node ID."
            )
        if edge.target not in node_ids:
            raise PlanValidationError(
                f"Plan {plan.plan_id!r} v{plan.version}: "
                f"Edge target {edge.target!r} does not exist as a node ID."
            )


def _check_no_self_loops(plan: Plan) -> None:
    for edge in plan.edges:
        if edge.source == edge.target:
            raise PlanValidationError(
                f"Plan {plan.plan_id!r} v{plan.version}: "
                f"Self-loop detected on node {edge.source!r}. "
                "A node cannot depend on itself."
            )


def _check_no_duplicate_edges(plan: Plan) -> None:
    seen: set[tuple[str, str]] = set()
    for edge in plan.edges:
        key = (edge.source, edge.target)
        if key in seen:
            raise PlanValidationError(
                f"Plan {plan.plan_id!r} v{plan.version}: "
                f"Duplicate edge {edge.source!r} → {edge.target!r}."
            )
        seen.add(key)


def _build_digraph(plan: Plan) -> nx.DiGraph:
    g: nx.DiGraph = nx.DiGraph()
    g.add_nodes_from(n.id for n in plan.nodes)
    g.add_edges_from((e.source, e.target) for e in plan.edges)
    return g


def _check_acyclicity(plan: Plan, g: nx.DiGraph) -> None:
    if not nx.is_directed_acyclic_graph(g):
        try:
            cycle = nx.find_cycle(g)
            cycle_str = " → ".join(f"{u!r}" for u, _ in cycle) + f" → {cycle[0][0]!r}"
        except nx.NetworkXNoCycle:
            cycle_str = "<unable to extract cycle>"
        raise PlanValidationError(
            f"Plan {plan.plan_id!r} v{plan.version}: "
            f"Cycle detected: {cycle_str}. "
            "Execution plans must be acyclic (DAG)."
        )


def _check_reachability(plan: Plan, g: nx.DiGraph) -> None:
    """
    Detect truly isolated nodes — nodes with no edges at all when other nodes
    have edges. These would sit in `pending` forever (NFR-2 violation).

    What is VALID:
      - Multiple independent root nodes (no edges between them) — these all
        become ready in Round 1 and are dispatched in parallel. This is the
        key SGH parallel dispatch pattern.
      - A mix of independent chains sharing no common root.

    What is INVALID:
      - A node with zero edges in a plan where edges exist elsewhere and the
        node shares no connection (directly or indirectly) with any other node.
        Example: plan has a→b chain plus an isolated node 'orphan' with no edges.
    """
    if len(g.nodes) <= 1:
        return  # trivially valid

    # If the plan has no edges at all, every node is an independent root.
    # That is valid — they all dispatch in Round 1 as parallel tasks.
    if g.number_of_edges() == 0:
        return

    # With edges present: every node must appear in at least one edge
    # (either as source or target). A node that has no edges in a plan
    # that has edges elsewhere is an isolated orphan.
    nodes_in_edges: set[str] = set()
    for e in plan.edges:
        nodes_in_edges.add(e.source)
        nodes_in_edges.add(e.target)

    isolated = set(g.nodes) - nodes_in_edges
    if isolated:
        raise PlanValidationError(
            f"Plan {plan.plan_id!r} v{plan.version}: "
            f"Isolated node(s) detected: {sorted(isolated)}. "
            "These nodes have no edges in a plan where other nodes do. "
            "An isolated node can never transition from pending → ready (NFR-2)."
        )


def _check_any_of_join_arity(plan: Plan) -> None:
    """
    A node with join_mode=ANY_OF must have ≥2 predecessors.

    A single-predecessor any_of is semantically equivalent to all_of and
    is almost certainly a misconfiguration.  Reject it early.
    """
    for node in plan.nodes:
        if node.config.join_mode == JoinMode.ANY_OF:
            preds = plan.predecessors(node.id)
            if len(preds) < 2:
                raise PlanValidationError(
                    f"Plan {plan.plan_id!r} v{plan.version}: "
                    f"Node {node.id!r} has join_mode=any_of but only "
                    f"{len(preds)} predecessor(s). any_of requires ≥2 predecessors. "
                    "Use join_mode=all_of for nodes with a single predecessor."
                )


def _check_any_of_consistency(plan: Plan) -> None:
    """
    Ensure that all predecessors of an any_of node belong to the same "fan-in group".

    This is a semantic consistency check: the paper's any_of semantics assume that
    the sibling branches feeding an any_of join are genuinely alternative paths.
    We enforce that an any_of node's predecessors all share the same set of
    *their* predecessors (i.e., they form a true fan-out/fan-in bracket), OR
    that they have no shared predecessor (they are independent alternatives).

    This check is advisory — it warns but does not block if the structure is unusual.
    Future versions may tighten this.  For v1 we skip the advisory and only enforce
    arity (done above).  Placeholder for future tightening.
    """
    # Intentionally minimal in v1: arity check above is the hard requirement.
    # Structural consistency (same upstream source) is validated at demo time.
    pass
