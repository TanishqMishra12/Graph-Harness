"""
Failure type classifier (diagnoser) for SGH recovery layer.

Classifies a NodeOutput failure into one of three types that determine
which recovery escalation level is appropriate (§6.2):

  TRANSIENT          → infrastructure/timeout class; LLM call dropped,
                       rate-limited, timed out. Appropriate action: local_retry.

  CONTRACT_VIOLATION → the node ran but produced output that failed syntactic
                       or semantic contract validation. The reasoning was wrong,
                       not the infrastructure. Appropriate action: local_patch
                       (retry alone is unlikely to fix a reasoning error).

  DEPENDENCY_ERROR   → a required upstream node is in a failed state. The node
                       itself cannot succeed regardless of retries. Appropriate
                       action: escalate to request_replan or mark as blocked.

These map to the three-level escalation ladder:
  TRANSIENT          → Level 1: local_retry
  CONTRACT_VIOLATION → Level 2: local_patch  (after retry budget used)
  DEPENDENCY_ERROR   → Level 3: request_replan  (structural issue)

From the paper §6.2 and Table 3: "Three-level recovery distinguishing error types"
"""
from __future__ import annotations

from enum import Enum

from sgh.nodes.base import NodeOutput


class FailureType(str, Enum):
    """Classified failure type, determines the recovery escalation path."""
    TRANSIENT          = "transient"
    CONTRACT_VIOLATION = "contract_violation"
    DEPENDENCY_ERROR   = "dependency_error"
    UNKNOWN            = "unknown"


def diagnose(node_output: NodeOutput) -> FailureType:
    """
    Classify a NodeOutput failure into a FailureType.

    Classification logic (in priority order):
      1. If node_output.failure_type is already set by the executor, use it.
      2. If outcome == 'retry', classify as TRANSIENT.
      3. Otherwise classify as UNKNOWN (treated as TRANSIENT for safety).

    Args:
        node_output: The output from a failed node execution.

    Returns:
        FailureType indicating the appropriate recovery path.
    """
    if node_output.succeeded:
        raise ValueError(
            "diagnose() called on a successful NodeOutput. "
            "Only call this for failed/retry outcomes."
        )

    # Executor-set failure_type takes priority
    ft = node_output.failure_type
    if ft == FailureType.TRANSIENT or ft == "transient":
        return FailureType.TRANSIENT
    if ft == FailureType.CONTRACT_VIOLATION or ft == "contract_violation":
        return FailureType.CONTRACT_VIOLATION
    if ft == FailureType.DEPENDENCY_ERROR or ft == "dependency_error":
        return FailureType.DEPENDENCY_ERROR

    # outcome-based fallback
    if node_output.should_retry:
        return FailureType.TRANSIENT

    # Unrecognised failure type — treat as transient (safest default)
    return FailureType.UNKNOWN
