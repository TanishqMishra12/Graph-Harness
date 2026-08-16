"""
Recovery escalation manager for SGH (FR-5, §6.2).

Implements the three-level escalation protocol with mechanically enforced
preconditions. The escalation invariants (from the paper §6.2) are:

  Level 1 — local_retry:
    Precondition: recovery_state[v] == PRISTINE
    Effect: recovery_state[v] → RETRIED
    Semantics: re-dispatch the node with the same config. Appropriate for
               TRANSIENT failures (timeouts, rate limits, infrastructure errors).

  Level 2 — local_patch:
    Precondition: recovery_state[v] == RETRIED
    Effect: recovery_state[v] → PATCHED
    Semantics: re-dispatch the node with a modified config (patched prompt or
               parameters). Appropriate after retry failed, indicating the error
               is not purely transient.

  Level 3 — request_replan:
    Precondition: ALL nodes with recovery_state != PRISTINE are in PATCHED state
    Effect: triggers plan versioning; new Plan v+1 produced with lineage pointer
    Semantics: structural error that cannot be resolved by node-local actions.

EscalationViolationError is raised if a caller attempts to skip a level
(e.g., calling local_patch before local_retry has been tried). This is the
mechanical enforcement of NFR-1 extended to the recovery layer.

The RecoveryManager is stateful per-execution (not per-node). The dispatcher
holds one instance for the duration of a plan execution.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

from sgh.core.plan import NodeConfig, NodeState, Plan, RecoveryLevel
from sgh.recovery.diagnoser import FailureType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EscalationViolationError(ValueError):
    """
    Raised when a recovery action is requested out of escalation order.

    Examples of violations:
      - Calling local_patch when recovery_state[v] == PRISTINE (retry not yet tried)
      - Calling request_replan when some failed nodes are not yet PATCHED
    """
    def __init__(self, node_id: str, requested: str, current_level: RecoveryLevel) -> None:
        super().__init__(
            f"Escalation violation for node {node_id!r}: "
            f"cannot apply '{requested}' when recovery_state is {current_level.value!r}. "
            f"Recovery must follow the strict order: "
            f"pristine → local_retry → local_patch → request_replan."
        )
        self.node_id = node_id
        self.requested = requested
        self.current_level = current_level


# ---------------------------------------------------------------------------
# Recovery action
# ---------------------------------------------------------------------------


class RecoveryActionType(str):
    LOCAL_RETRY     = "local_retry"
    LOCAL_PATCH     = "local_patch"
    REQUEST_REPLAN  = "request_replan"
    NO_ACTION       = "no_action"      # node already exhausted all recovery


@dataclasses.dataclass
class RecoveryAction:
    """
    The output of RecoveryManager.select_action().

    Attributes:
        action_type:    What the dispatcher should do.
        node_id:        Which node this action applies to.
        patched_config: For local_patch: the modified NodeConfig to use.
                        None for all other action types.
        reason:         Human-readable explanation (for event log).
    """
    action_type: str
    node_id: str
    patched_config: NodeConfig | None = None
    reason: str = ""

    @property
    def should_retry(self) -> bool:
        return self.action_type == RecoveryActionType.LOCAL_RETRY

    @property
    def should_patch(self) -> bool:
        return self.action_type == RecoveryActionType.LOCAL_PATCH

    @property
    def should_replan(self) -> bool:
        return self.action_type == RecoveryActionType.REQUEST_REPLAN

    @property
    def is_exhausted(self) -> bool:
        return self.action_type == RecoveryActionType.NO_ACTION


# ---------------------------------------------------------------------------
# Recovery manager
# ---------------------------------------------------------------------------


class RecoveryManager:
    """
    Stateful recovery manager for a single plan execution.

    Tracks recovery_state[v] ∈ {PRISTINE, RETRIED, PATCHED} per node and
    enforces the escalation preconditions from §6.2.

    Usage:
        manager = RecoveryManager(plan)
        action = manager.select_action(node_id, failure_type, node_output)
        # Dispatcher applies the action
    """

    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        # recovery_state[v]: starts PRISTINE for every node
        self._recovery_state: dict[str, RecoveryLevel] = {
            n.id: RecoveryLevel.PRISTINE for n in plan.nodes
        }
        # patched configs: stored when local_patch is applied
        self._patched_configs: dict[str, NodeConfig] = {}

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def recovery_state(self, node_id: str) -> RecoveryLevel:
        return self._recovery_state[node_id]

    def all_recovery_states(self) -> dict[str, RecoveryLevel]:
        return dict(self._recovery_state)

    def patched_config(self, node_id: str) -> NodeConfig | None:
        return self._patched_configs.get(node_id)

    def replan_precondition_met(self, failed_node_ids: list[str]) -> bool:
        """
        Return True iff all failed nodes have been PATCHED (replan precondition §6.2).

        A replan is only valid once every failed node has exhausted local_retry
        and local_patch. This prevents premature replanning.
        """
        return all(
            self._recovery_state.get(nid) == RecoveryLevel.PATCHED
            for nid in failed_node_ids
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def select_action(
        self,
        node_id: str,
        failure_type: FailureType,
        current_node_state: NodeState,
        all_states: dict[str, NodeState],
        raw_error: str = "",
    ) -> RecoveryAction:
        """
        Select the appropriate recovery action for a failed node.

        Enforces the escalation invariants from §6.2:
          - PRISTINE → local_retry  (for TRANSIENT failures)
          - PRISTINE → local_patch  (for CONTRACT_VIOLATION; retry won't help)
          - RETRIED  → local_patch
          - PATCHED  → request_replan (if all failed nodes are patched)
          - PATCHED  → no_action     (if other nodes are still unpatched)

        Args:
            node_id:           The failing node.
            failure_type:      Classified failure type from diagnoser.
            current_node_state: The node's current state (FAILED or FAILED_RETRYABLE).
            all_states:        Full node state map (needed for replan precondition).
            raw_error:         Original error message (for logging/audit).

        Returns:
            RecoveryAction describing what the dispatcher should do next.

        Raises:
            EscalationViolationError: If a precondition is violated.
        """
        level = self._recovery_state[node_id]

        logger.debug(
            "RecoveryManager.select_action: node=%r, failure_type=%r, "
            "recovery_level=%r, node_state=%r",
            node_id, failure_type, level, current_node_state,
        )

        # PRISTINE: first failure — decide based on failure type
        if level == RecoveryLevel.PRISTINE:
            if failure_type == FailureType.TRANSIENT:
                return self._apply_retry(node_id, reason=f"Transient failure: {raw_error}")
            elif failure_type in (FailureType.CONTRACT_VIOLATION, FailureType.UNKNOWN):
                # Contract violation won't be fixed by a plain retry — go straight to patch
                return self._apply_patch(
                    node_id,
                    failure_type=failure_type,
                    reason=f"Contract violation on first attempt: {raw_error}",
                )
            else:  # DEPENDENCY_ERROR
                return self._apply_no_action(
                    node_id,
                    reason=f"Dependency error — cannot recover locally: {raw_error}",
                )

        # RETRIED: retry was used — escalate to patch regardless of failure type
        elif level == RecoveryLevel.RETRIED:
            return self._apply_patch(
                node_id,
                failure_type=failure_type,
                reason=f"Retry exhausted, applying local_patch. Error: {raw_error}",
            )

        # PATCHED: patch was used — escalate to replan if preconditions met
        elif level == RecoveryLevel.PATCHED:
            failed_nodes = [
                nid for nid, s in all_states.items()
                if s in (NodeState.FAILED, NodeState.FAILED_RETRYABLE)
            ]
            # Include the current node (it may not yet be in FAILED state in dict)
            if node_id not in failed_nodes:
                failed_nodes.append(node_id)

            if self.replan_precondition_met(failed_nodes):
                return RecoveryAction(
                    action_type=RecoveryActionType.REQUEST_REPLAN,
                    node_id=node_id,
                    reason=(
                        f"All failed nodes are PATCHED — escalating to request_replan. "
                        f"Failed nodes: {failed_nodes}. Error: {raw_error}"
                    ),
                )
            else:
                # Some other failed nodes not yet patched — wait for them
                unpatched = [
                    nid for nid in failed_nodes
                    if self._recovery_state.get(nid) != RecoveryLevel.PATCHED
                    and nid != node_id
                ]
                return RecoveryAction(
                    action_type=RecoveryActionType.NO_ACTION,
                    node_id=node_id,
                    reason=(
                        f"Cannot replan yet — unpatched failed nodes: {unpatched}. "
                        f"Marking {node_id!r} as permanently failed."
                    ),
                )

        raise AssertionError(f"Unreachable recovery_level={level!r}")

    # ------------------------------------------------------------------
    # Internal helpers (each enforces its precondition)
    # ------------------------------------------------------------------

    def _apply_retry(self, node_id: str, reason: str = "") -> RecoveryAction:
        """Apply local_retry. Precondition: recovery_state == PRISTINE."""
        level = self._recovery_state[node_id]
        if level != RecoveryLevel.PRISTINE:
            raise EscalationViolationError(node_id, "local_retry", level)
        self._recovery_state[node_id] = RecoveryLevel.RETRIED
        logger.info("Recovery: local_retry applied to %r. %s", node_id, reason)
        return RecoveryAction(
            action_type=RecoveryActionType.LOCAL_RETRY,
            node_id=node_id,
            reason=reason,
        )

    def _apply_patch(
        self,
        node_id: str,
        failure_type: FailureType,
        reason: str = "",
    ) -> RecoveryAction:
        """
        Apply local_patch. Precondition: recovery_state in {PRISTINE, RETRIED}.

        The patch modifies the node's config to hint at the failure:
          - Appends a recovery note to prompt_template
          - Disables any caching (not applicable in v1)
          - For contract violations: adds the schema reminder to the prompt
        """
        level = self._recovery_state[node_id]
        if level == RecoveryLevel.PATCHED:
            raise EscalationViolationError(node_id, "local_patch", level)

        original_node = self.plan.node_by_id(node_id)
        patched = self._build_patch(original_node.config, failure_type)
        self._patched_configs[node_id] = patched
        self._recovery_state[node_id] = RecoveryLevel.PATCHED
        logger.info("Recovery: local_patch applied to %r. %s", node_id, reason)
        return RecoveryAction(
            action_type=RecoveryActionType.LOCAL_PATCH,
            node_id=node_id,
            patched_config=patched,
            reason=reason,
        )

    def _apply_no_action(self, node_id: str, reason: str = "") -> RecoveryAction:
        logger.warning("Recovery: no_action for %r (exhausted or ineligible). %s", node_id, reason)
        return RecoveryAction(
            action_type=RecoveryActionType.NO_ACTION,
            node_id=node_id,
            reason=reason,
        )

    def _build_patch(self, config: NodeConfig, failure_type: FailureType) -> NodeConfig:
        """
        Build a patched NodeConfig for a local_patch recovery attempt.

        The patch is intentionally minimal in v1:
          - For CONTRACT_VIOLATION: append a structured-output reminder to the prompt.
          - For TRANSIENT: no prompt change — just marks it as a retry with patch.
          - mock_fail_type is cleared so a MockNode won't keep failing after a patch.
        """
        updates: dict[str, Any] = {}

        # Clear mock failure injection so the patched attempt can succeed
        if config.mock_fail_type is not None:
            updates["mock_fail_type"] = None

        # Append a recovery hint to the prompt template
        if failure_type == FailureType.CONTRACT_VIOLATION and config.prompt_template:
            updates["prompt_template"] = (
                config.prompt_template
                + "\n\n[RECOVERY PATCH: Previous attempt failed contract validation. "
                "Ensure your output strictly matches the required JSON schema.]"
            )
        elif failure_type == FailureType.CONTRACT_VIOLATION and not config.prompt_template:
            updates["prompt_template"] = (
                "[RECOVERY PATCH: Ensure output strictly matches the required JSON schema.]"
            )

        return config.model_copy(update=updates)
