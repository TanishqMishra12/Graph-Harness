"""
Ready-set computation for SGH.

Implements U(S) = {v ∈ V | s_v = ready ∧ ∀(u,v)∈E: s_u = executed}

from Definition 3.1 (§3.1) of arXiv:2604.11378v1, extended with join semantics
(§7.1, §7.2) and blocked-state detection.

The ready set is computed from scratch at each scheduling round:
  1. For every node in `pending` state, evaluate its join condition.
  2. Nodes whose join is satisfied transition pending → ready.
  3. Nodes whose join is permanently blocked transition pending → blocked.
  4. For any_of joins that fire: sibling predecessors transition to skipped.
  5. The returned ready set is the list of node IDs now in `ready` state.

This function is pure (no I/O, no side effects on its own) — it returns
a list of state updates for the dispatcher to apply via state_machine.transition().
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from sgh.core.joins import JoinEvaluator
from sgh.core.plan import NodeState, Plan
from sgh.core.state_machine import transition

if TYPE_CHECKING:
    pass


@dataclasses.dataclass
class ReadySetResult:
    """
    Output of compute_ready_set() for a single scheduling round.

    Attributes:
        ready_node_ids:   Nodes that transitioned pending → ready (eligible for dispatch).
        blocked_node_ids: Nodes that transitioned pending → blocked (all paths failed).
        skipped_node_ids: Nodes that transitioned pending/ready → skipped (any_of fired).
        updated_states:   The new state dict after all transitions (copy of input + changes).
        round_u_size:     |U(s)| — the cardinality of the ready set this round.
    """
    ready_node_ids: list[str]
    blocked_node_ids: list[str]
    skipped_node_ids: list[str]
    updated_states: dict[str, NodeState]
    round_u_size: int


def compute_ready_set(
    plan: Plan,
    states: dict[str, NodeState],
) -> ReadySetResult:
    """
    Compute U(S) — the set of nodes eligible for dispatch this scheduling round.

    Evaluates all join conditions for pending nodes and applies the resulting
    state transitions. Returns a ReadySetResult describing what changed.

    The caller (Dispatcher) is responsible for:
      1. Starting coroutines for each node in `ready_node_ids`.
      2. Applying `updated_states` as the new canonical state.
      3. Persisting the resulting state transitions to the event log.

    Args:
        plan:   The frozen execution plan.
        states: Current node state mapping {node_id: NodeState}.

    Returns:
        ReadySetResult with the computed ready set and all state changes.
    """
    new_states = dict(states)  # mutable working copy
    ready_ids: list[str] = []
    blocked_ids: list[str] = []
    skipped_ids: list[str] = []

    # Collect pending nodes to evaluate
    pending_nodes = [
        node_id for node_id, state in states.items()
        if state == NodeState.PENDING
    ]

    for node_id in pending_nodes:
        join_result = JoinEvaluator.evaluate(plan, node_id, new_states)

        if join_result.satisfied:
            # pending → ready
            new_states[node_id] = transition(
                NodeState.PENDING, NodeState.READY, node_id
            )
            ready_ids.append(node_id)

            # Handle any_of sibling skips
            for sibling_id in join_result.siblings_to_skip:
                sibling_state = new_states[sibling_id]
                # Siblings may be in pending, ready, or running — each has a
                # valid path to skipped via the legal transitions.
                if sibling_state == NodeState.PENDING:
                    new_states[sibling_id] = transition(
                        NodeState.PENDING, NodeState.SKIPPED, sibling_id
                    )
                    skipped_ids.append(sibling_id)
                elif sibling_state == NodeState.READY:
                    new_states[sibling_id] = transition(
                        NodeState.READY, NodeState.SKIPPED, sibling_id
                    )
                    skipped_ids.append(sibling_id)
                # Running siblings: cannot skip mid-execution in SGH v1.
                # The dispatcher cancels them via their running → cancelled path
                # when they return. See dispatcher.py for handling.

        elif join_result.blocked:
            # pending → blocked (a required predecessor is failed/cancelled)
            new_states[node_id] = transition(
                NodeState.PENDING, NodeState.BLOCKED, node_id
            )
            blocked_ids.append(node_id)

        # else: join not yet satisfied — node stays pending

    # Also collect nodes already in `ready` state from a previous round
    # (shouldn't happen in normal flow, but defensive check)
    already_ready = [
        node_id for node_id, state in new_states.items()
        if state == NodeState.READY and node_id not in ready_ids
    ]
    ready_ids.extend(already_ready)

    return ReadySetResult(
        ready_node_ids=ready_ids,
        blocked_node_ids=blocked_ids,
        skipped_node_ids=skipped_ids,
        updated_states=new_states,
        round_u_size=len(ready_ids),
    )


def execution_is_complete(states: dict[str, NodeState]) -> bool:
    """
    Return True iff every node has reached a terminal state.

    This is the termination condition for the dispatcher's main loop (NFR-2).
    """
    return all(state.is_terminal for state in states.values())


def execution_has_unrecoverable_failure(states: dict[str, NodeState]) -> bool:
    """
    Return True iff there are nodes stuck in `blocked` or `failed` with no
    remaining non-terminal, non-failed nodes to drive execution forward.

    Used by the dispatcher to detect deadlock: all non-terminal nodes are
    blocked, and no running/ready nodes can unblock them.
    """
    non_terminal = {
        node_id for node_id, s in states.items() if not s.is_terminal
    }
    if not non_terminal:
        return False
    # Check if any non-terminal node is not blocked
    active = {
        node_id for node_id in non_terminal
        if states[node_id] not in (NodeState.BLOCKED,)
    }
    return len(active) == 0
