"""
Join semantics for SGH.

Implements the two join modes from §7.1 (all_of) and §7.2 (any_of) of arXiv:2604.11378v1.

ALL_OF join (§7.1):
  A node v with join_mode=all_of becomes ready when ALL predecessor nodes
  have reached the `executed` state. This encodes constructive parallelism:
  every branch must complete before v can proceed.

ANY_OF join (§7.2):
  A node v with join_mode=any_of becomes ready when ANY ONE predecessor node
  reaches `executed`. The first predecessor to execute "wins". All sibling
  predecessor nodes (those that haven't yet reached a terminal state) are
  transitioned to `skipped`.

  Determinism guarantee: when multiple predecessors are simultaneously in
  `executed` at the moment of evaluation (e.g., they completed in the same
  scheduling round), the "winner" is deterministically chosen as the one with
  the lexicographically smallest node ID. This matches classical tie-breaking
  in DAG schedulers and ensures reproducible behavior across runs.

Excluded by design (§6.3 / non_goals):
  - first_of / competitive parallelism: starting all branches and cancelling
    the rest after one fires. SGH explicitly excludes this for controllability.
"""
from __future__ import annotations

from sgh.core.plan import JoinMode, NodeState, Plan


class JoinEvaluator:
    """
    Stateless evaluator for all_of and any_of join conditions.

    All methods are pure functions: they inspect state but never mutate it.
    The dispatcher uses the return values to drive state transitions via
    state_machine.transition().
    """

    # ------------------------------------------------------------------
    # all_of
    # ------------------------------------------------------------------

    @staticmethod
    def all_of_satisfied(
        predecessor_states: dict[str, NodeState],
    ) -> bool:
        """
        Return True iff every predecessor is in `executed` (§7.1).

        Args:
            predecessor_states: Mapping {node_id: NodeState} for all predecessors.

        Returns:
            True when all predecessors are executed; False otherwise.
        """
        return all(s == NodeState.EXECUTED for s in predecessor_states.values())

    @staticmethod
    def all_of_blocked(
        predecessor_states: dict[str, NodeState],
    ) -> bool:
        """
        Return True iff at least one predecessor is in a terminal failure state
        (failed or cancelled), making this node permanently unable to proceed.

        A blocked node transitions pending → blocked and then blocked → failed.
        """
        blocking = {NodeState.FAILED, NodeState.CANCELLED}
        return any(s in blocking for s in predecessor_states.values())

    # ------------------------------------------------------------------
    # any_of
    # ------------------------------------------------------------------

    @staticmethod
    def any_of_winner(
        predecessor_states: dict[str, NodeState],
    ) -> str | None:
        """
        Return the winning predecessor node ID if any_of is satisfied (§7.2).

        A predecessor "wins" if it has reached `executed`. If multiple predecessors
        are simultaneously executed (same scheduling round), the lexicographically
        smallest ID is chosen for determinism.

        Returns:
            The winning node ID, or None if no predecessor has executed yet.
        """
        executed = [
            nid for nid, s in predecessor_states.items() if s == NodeState.EXECUTED
        ]
        if not executed:
            return None
        return min(executed)  # deterministic tie-breaking

    @staticmethod
    def any_of_satisfied(
        predecessor_states: dict[str, NodeState],
    ) -> bool:
        """Return True iff at least one predecessor has executed (§7.2)."""
        return any(s == NodeState.EXECUTED for s in predecessor_states.values())

    @staticmethod
    def any_of_blocked(
        predecessor_states: dict[str, NodeState],
    ) -> bool:
        """
        Return True iff ALL predecessors have reached a terminal non-executed state.

        This is the degenerate failure case: every alternative path has failed or
        been cancelled. The any_of node itself becomes blocked → failed.
        """
        terminal_failures = {NodeState.FAILED, NodeState.CANCELLED, NodeState.SKIPPED}
        return all(
            s in terminal_failures or s == NodeState.EXECUTED
            for s in predecessor_states.values()
        ) and not any(s == NodeState.EXECUTED for s in predecessor_states.values())

    # ------------------------------------------------------------------
    # Sibling skip computation
    # ------------------------------------------------------------------

    @staticmethod
    def get_siblings_to_skip(
        plan: Plan,
        winner_id: str,
        join_node_id: str,
    ) -> list[str]:
        """
        Return the node IDs of sibling predecessors that should be skipped
        after an any_of join fires (§7.2).

        Siblings are all predecessors of `join_node_id` that are NOT the winner
        and have NOT yet reached a terminal state. They will be transitioned to
        `skipped` by the dispatcher.

        Args:
            plan: The execution plan (used for predecessor lookup).
            winner_id: The node ID that won the any_of join.
            join_node_id: The node ID of the any_of join node.

        Returns:
            List of node IDs to skip (may be empty if all siblings are terminal).
        """
        predecessors = plan.predecessors(join_node_id)
        return [nid for nid in predecessors if nid != winner_id]

    # ------------------------------------------------------------------
    # Unified join evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(
        plan: Plan,
        node_id: str,
        states: dict[str, NodeState],
    ) -> "JoinResult":
        """
        Evaluate the join condition for `node_id` given the current node states.

        This is the main entry point used by ready_set.compute_ready_set().

        Args:
            plan: The execution plan.
            node_id: The node whose join condition is being evaluated.
            states: The complete current node state mapping {node_id: NodeState}.

        Returns:
            A JoinResult indicating whether the node should transition to ready,
            remain pending, become blocked, or trigger sibling skips.
        """
        node = plan.node_by_id(node_id)
        predecessor_ids = plan.predecessors(node_id)

        if not predecessor_ids:
            # Root node: no predecessors → immediately satisfies any join
            return JoinResult(satisfied=True, winner_id=None, siblings_to_skip=[])

        predecessor_states = {nid: states[nid] for nid in predecessor_ids}

        if node.config.join_mode == JoinMode.ALL_OF:
            if JoinEvaluator.all_of_satisfied(predecessor_states):
                return JoinResult(satisfied=True, winner_id=None, siblings_to_skip=[])
            if JoinEvaluator.all_of_blocked(predecessor_states):
                return JoinResult(satisfied=False, blocked=True, winner_id=None, siblings_to_skip=[])
            return JoinResult(satisfied=False, winner_id=None, siblings_to_skip=[])

        else:  # ANY_OF
            winner_id = JoinEvaluator.any_of_winner(predecessor_states)
            if winner_id is not None:
                siblings = JoinEvaluator.get_siblings_to_skip(plan, winner_id, node_id)
                return JoinResult(satisfied=True, winner_id=winner_id, siblings_to_skip=siblings)
            if JoinEvaluator.any_of_blocked(predecessor_states):
                return JoinResult(satisfied=False, blocked=True, winner_id=None, siblings_to_skip=[])
            return JoinResult(satisfied=False, winner_id=None, siblings_to_skip=[])


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class JoinResult:
    """
    Result of a join condition evaluation.

    Attributes:
        satisfied:        True iff the join condition is met and the node should
                          become ready.
        blocked:          True iff the join can never be satisfied (all paths failed).
        winner_id:        For any_of joins: the node ID of the winning predecessor.
                          None for all_of joins or unsatisfied any_of.
        siblings_to_skip: For any_of joins: list of predecessor IDs to skip.
    """

    __slots__ = ("satisfied", "blocked", "winner_id", "siblings_to_skip")

    def __init__(
        self,
        satisfied: bool,
        blocked: bool = False,
        winner_id: str | None = None,
        siblings_to_skip: list[str] | None = None,
    ) -> None:
        self.satisfied = satisfied
        self.blocked = blocked
        self.winner_id = winner_id
        self.siblings_to_skip: list[str] = siblings_to_skip or []

    def __repr__(self) -> str:
        return (
            f"JoinResult(satisfied={self.satisfied}, blocked={self.blocked}, "
            f"winner={self.winner_id!r}, skip={self.siblings_to_skip})"
        )
