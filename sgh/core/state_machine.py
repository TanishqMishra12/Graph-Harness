"""
Node state machine for SGH.

Implements the full transition table from §6.1 (Table 15) of arXiv:2604.11378v1.
This module is the SINGLE GATEWAY for all state changes — no code outside this
module may change a node's state. This is the mechanical enforcement of NFR-1.

Legal transitions (read as: current_state → target_state):

  pending          → ready              (join condition satisfied)
  pending          → blocked            (a required predecessor failed)
  pending          → cancelled          (plan cancelled / any_of sibling won)
  ready            → running            (dispatcher picks up the node)
  ready            → cancelled          (plan cancelled mid-ready)
  ready            → skipped            (any_of sibling wins before dispatch)
  running          → executed           (successful execution + contract passes)
  running          → failed_retryable   (execution failed; retry budget remains)
  running          → failed             (execution failed; no budget remaining)
  running          → waiting_human      (node requests human approval)
  running          → cancelled          (plan cancelled mid-run)
  waiting_human    → running            (human approves; execution continues)
  waiting_human    → cancelled          (human rejects / plan cancelled)
  failed_retryable → running            (local_retry action applied)
  failed_retryable → failed             (retry budget exhausted during recovery eval)
  failed_retryable → cancelled          (plan cancelled before retry fires)
  failed           → running            (local_patch action applied, new attempt)
  failed           → cancelled          (plan cancelled)
  blocked          → failed             (propagated failure after diagnosis)
  blocked          → cancelled          (plan cancelled)
  cancelled        → (terminal — no outgoing transitions)
  executed         → (terminal — no outgoing transitions)
  failed           → (terminal after patch exhausted — see above)
  skipped          → (terminal — no outgoing transitions)

Note on `failed` re-entry:
  The paper allows local_patch to place a `failed` node back into `running` for
  one more attempt (the patch modifies the node's config/prompt, not the topology).
  This is the only case where a terminal-appearing state has an outbound edge.
  `failed` is fully terminal only after local_patch + the new attempt also fails.
"""
from __future__ import annotations

from sgh.core.plan import NodeState


# ---------------------------------------------------------------------------
# Legal transition table — frozenset of (from, to) pairs
# ---------------------------------------------------------------------------

LEGAL_TRANSITIONS: frozenset[tuple[NodeState, NodeState]] = frozenset(
    {
        # pending outgoing
        (NodeState.PENDING,          NodeState.READY),
        (NodeState.PENDING,          NodeState.BLOCKED),
        (NodeState.PENDING,          NodeState.CANCELLED),
        (NodeState.PENDING,          NodeState.SKIPPED),    # any_of wins before this node queues
        # ready outgoing
        (NodeState.READY,            NodeState.RUNNING),
        (NodeState.READY,            NodeState.CANCELLED),
        (NodeState.READY,            NodeState.SKIPPED),
        # running outgoing
        (NodeState.RUNNING,          NodeState.EXECUTED),
        (NodeState.RUNNING,          NodeState.FAILED_RETRYABLE),
        (NodeState.RUNNING,          NodeState.FAILED),
        (NodeState.RUNNING,          NodeState.WAITING_HUMAN),
        (NodeState.RUNNING,          NodeState.CANCELLED),
        # waiting_human outgoing
        (NodeState.WAITING_HUMAN,    NodeState.RUNNING),
        (NodeState.WAITING_HUMAN,    NodeState.CANCELLED),
        # failed_retryable outgoing
        (NodeState.FAILED_RETRYABLE, NodeState.RUNNING),
        (NodeState.FAILED_RETRYABLE, NodeState.FAILED),
        (NodeState.FAILED_RETRYABLE, NodeState.CANCELLED),
        # failed outgoing (local_patch can re-enter running for one more attempt)
        (NodeState.FAILED,           NodeState.RUNNING),
        (NodeState.FAILED,           NodeState.CANCELLED),
        # blocked outgoing
        (NodeState.BLOCKED,          NodeState.FAILED),
        (NodeState.BLOCKED,          NodeState.CANCELLED),
        # terminal states: executed, cancelled, skipped have NO outgoing edges
    }
)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class IllegalTransitionError(ValueError):
    """
    Raised when code attempts a state transition not present in Table 15.

    This exception is the mechanical enforcement of NFR-1:
    "State machine must reject any transition not present in Table 15."
    """

    def __init__(
        self,
        node_id: str,
        current: NodeState,
        target: NodeState,
    ) -> None:
        super().__init__(
            f"Illegal state transition for node '{node_id}': "
            f"{current.value!r} → {target.value!r}. "
            f"This pair is not present in the legal transition table (Table 15, §6.1)."
        )
        self.node_id = node_id
        self.current = current
        self.target = target


# ---------------------------------------------------------------------------
# The single transition gateway
# ---------------------------------------------------------------------------


def transition(
    current: NodeState,
    target: NodeState,
    node_id: str = "<unknown>",
) -> NodeState:
    """
    Attempt a state transition and return the new state if legal.

    This is the ONLY function that may change a node's state. All callers
    (dispatcher, recovery manager, join evaluator) must route through here.

    Raises:
        IllegalTransitionError: If (current, target) is not in LEGAL_TRANSITIONS.

    Returns:
        target — the new (valid) state.
    """
    if (current, target) not in LEGAL_TRANSITIONS:
        raise IllegalTransitionError(node_id, current, target)
    return target


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------


def is_terminal(state: NodeState) -> bool:
    """True iff the node has reached a terminal state (Σ_term)."""
    return state.is_terminal


def can_be_dispatched(state: NodeState) -> bool:
    """True iff the node is in `ready` state and eligible for the ready set U(s)."""
    return state == NodeState.READY


def allows_retry(state: NodeState) -> bool:
    """True iff local_retry is a valid action for this state."""
    return state == NodeState.FAILED_RETRYABLE


def allows_patch(state: NodeState) -> bool:
    """True iff local_patch is a valid action for this state (requires prior retry)."""
    return state == NodeState.FAILED
