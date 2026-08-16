"""
Base node executor for SGH.

Defines the abstract interface that all node types must implement, plus the
ExecContext dataclass that enforces the execution/diagnostic context separation
required by §5.3-5.4 (FR-7) of arXiv:2604.11378v1.

Context partition (§5.4):
  C_exec — the execution context passed to a running node. Contains:
    - The node's own configuration (prompt, tools, model, etc.)
    - Outputs from upstream `executed` predecessor nodes
    - Task metadata (plan ID, version, node ID, label)
  C_diag — the diagnostic context, visible ONLY to the recovery/planner layers.
    Contains failure history, retry counts, error messages from past attempts.

  INVARIANT: A node executor never receives C_diag. If it did, failure history
  could "leak" into the LLM's context and corrupt subsequent reasoning —
  exactly the pathology the paper identifies in §5.4.

NodeOutput encodes the paper's outcome space O = {success, failure, retry, escalate}.
"""
from __future__ import annotations

import abc
import dataclasses
from typing import Any

from sgh.core.plan import Node, NodeConfig, NodeState


# ---------------------------------------------------------------------------
# Execution context  C_exec
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExecContext:
    """
    Execution context C_exec (§5.3).

    This is the ONLY context object passed to node executors. It deliberately
    excludes diagnostic/failure history (C_diag) — see module docstring.

    Attributes:
        plan_id:           ID of the owning plan.
        plan_version:      Version of the owning plan.
        node_id:           ID of the node being executed.
        node_label:        Human-readable node label.
        config:            The node's configuration (σ(v)).
        upstream_outputs:  Mapping of predecessor node IDs → their output dicts.
                           Only includes `executed` predecessors (FR-7).
        task_description:  Top-level task description (may be None).
        attempt_number:    Which attempt this is (1-indexed). Exposed to allow
                           prompt variation between attempts, but does NOT include
                           why previous attempts failed (that's C_diag).
    """
    plan_id: str
    plan_version: int
    node_id: str
    node_label: str
    config: NodeConfig
    upstream_outputs: dict[str, dict[str, Any]]
    task_description: str | None = None
    attempt_number: int = 1


@dataclasses.dataclass(frozen=True)
class DiagContext:
    """
    Diagnostic context C_diag (§5.4).

    NEVER passed to node executors. Only the recovery layer and planner
    layer have access to this. Keeping it a separate type makes it impossible
    to accidentally pass it as ExecContext.

    Attributes:
        node_id:       The failing node.
        failure_type:  Classified failure type.
        error_message: Human-readable error description.
        attempts:      How many execution attempts have been made.
        raw_outputs:   List of raw outputs from each attempt (may be partial).
        recovery_level: Current escalation state.
    """
    node_id: str
    failure_type: str
    error_message: str
    attempts: int
    raw_outputs: list[dict[str, Any]]
    recovery_level: str  # RecoveryLevel value


# ---------------------------------------------------------------------------
# Node output  O = {success, failure, retry, escalate}
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class NodeOutput:
    """
    The result of a node execution attempt.

    Maps to the outcome space O = {success, failure, retry, escalate} (§3.1, Def 3.1).

    Attributes:
        outcome:       One of 'success', 'failure', 'retry', 'escalate'.
        payload:       The node's output dict (meaningful only on 'success').
        error_message: Human-readable failure description (on failure/retry/escalate).
        failure_type:  Classified failure type for the recovery layer.
                       Values: 'transient', 'contract_violation', 'dependency_error'.
        token_usage:   Token counts from the LLM call (for benchmarking).
        latency_s:     Wall-clock seconds the execution took.
    """
    outcome: str  # 'success' | 'failure' | 'retry' | 'escalate'
    payload: dict[str, Any] = dataclasses.field(default_factory=dict)
    error_message: str = ""
    failure_type: str = ""          # 'transient' | 'contract_violation' | 'dependency_error'
    token_usage: dict[str, int] = dataclasses.field(default_factory=dict)
    latency_s: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.outcome == "success"

    @property
    def failed(self) -> bool:
        return self.outcome in ("failure", "escalate")

    @property
    def should_retry(self) -> bool:
        return self.outcome == "retry"

    @property
    def should_wait(self) -> bool:
        return self.outcome == "wait"


# ---------------------------------------------------------------------------
# Abstract base node
# ---------------------------------------------------------------------------


class BaseNode(abc.ABC):
    """
    Abstract base class for all SGH node executors.

    Subclasses implement `execute()` and must:
      1. Accept only an ExecContext (never DiagContext).
      2. Return a NodeOutput with outcome in O.
      3. Respect the timeout_s in config (via asyncio.wait_for in async implementations).
      4. Never mutate the plan or state machine directly.
    """

    def __init__(self, node: Node) -> None:
        self.node = node

    @abc.abstractmethod
    async def execute(self, ctx: ExecContext) -> NodeOutput:
        """
        Execute this node and return its outcome.

        Args:
            ctx: The execution context C_exec. MUST NOT include diagnostic history.

        Returns:
            NodeOutput describing the outcome (success/failure/retry/escalate).
        """
        ...

    @property
    def node_id(self) -> str:
        return self.node.id

    @property
    def config(self) -> NodeConfig:
        return self.node.config

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.node.id!r})"
