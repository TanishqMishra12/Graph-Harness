"""
Core data models for SGH execution plans.

Formal foundations (arXiv:2604.11378v1):
  - Plan Π = (id, version, V, E, σ, κ)              §5.1
  - Node state set Σ, terminal states Σ_term         §6.1, Def 6.1
  - Recovery level set R                             §6.2, Def 6.3
  - Output contract κ_v                              §6.1, Theorem 6.3
  - Execution/diagnostic context C_exec, C_diag      §5.3 / §5.4

Plan-version immutability is enforced by Pydantic's frozen=True model config:
once a Plan object is created, its (V, E) cannot be mutated. Any structural
change must produce a new Plan with an incremented version and a lineage pointer.
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class NodeState(str, Enum):
    """
    The full node state set Σ from §6.1 (Definition 6.1).

    Terminal states Σ_term = {executed, failed, cancelled, skipped}.
    All other states are non-terminal; the scheduler loops until every node
    reaches a terminal state (or a global timeout fires).
    """
    PENDING          = "pending"           # Awaiting predecessor completion
    READY            = "ready"             # Eligible for dispatch (in U(s))
    RUNNING          = "running"           # Currently executing
    WAITING_HUMAN    = "waiting_human"     # Paused for human-in-the-loop approval
    BLOCKED          = "blocked"           # A required predecessor is in `failed`
    EXECUTED         = "executed"          # Completed successfully; output validated
    FAILED_RETRYABLE = "failed_retryable"  # Failed but retry budget remains
    FAILED           = "failed"            # Exhausted all recovery options; terminal
    CANCELLED        = "cancelled"         # Cancelled by the scheduler; terminal
    SKIPPED          = "skipped"           # Skipped by any_of sibling win; terminal

    @property
    def is_terminal(self) -> bool:
        """Returns True iff this state is in Σ_term."""
        return self in _TERMINAL_STATES


_TERMINAL_STATES: frozenset[NodeState] = frozenset(
    {NodeState.EXECUTED, NodeState.FAILED, NodeState.CANCELLED, NodeState.SKIPPED}
)


class JoinMode(str, Enum):
    """
    Join semantics for a node with multiple predecessors (§7.1, §7.2).

    ALL_OF  — All predecessors must be in `executed` before this node becomes ready.
              Constructive parallelism: every branch must complete.

    ANY_OF  — The first predecessor to reach `executed` satisfies the join.
              Sibling predecessor branches transition to `skipped`.
              (Note: competitive parallelism / first_of is excluded by design; §6.3.)
    """
    ALL_OF = "all_of"
    ANY_OF = "any_of"


class NodeType(str, Enum):
    """Node executor type, determines which NodeExecutor implementation is used."""
    LLM   = "llm"    # Calls a real LLM (Anthropic Claude by default)
    TOOL  = "tool"   # Calls a deterministic Python tool function
    MOCK  = "mock"   # Returns a fixed payload; used in tests (NFR-3)
    HUMAN = "human"  # Pauses for human-in-the-loop input


class RecoveryLevel(str, Enum):
    """
    Escalation state for a node, tracking how much recovery has been spent (§6.2).

    The escalation invariants (FR-5) are:
      - local_retry  requires: recovery_state[v] == PRISTINE
      - local_patch  requires: recovery_state[v] == RETRIED
      - request_replan requires: ALL failed nodes have recovery_state == PATCHED

    These are mechanically enforced in sgh.recovery.escalation.RecoveryManager.
    """
    PRISTINE = "pristine"   # No recovery attempted yet
    RETRIED  = "retried"    # local_retry has been applied once
    PATCHED  = "patched"    # local_patch has been applied; next action is replan


# ---------------------------------------------------------------------------
# Output contract (κ_v)
# ---------------------------------------------------------------------------


class OutputContract(BaseModel):
    """
    Per-node output contract κ_v (§6.1, Theorem 6.3).

    Syntactic validation uses JSON Schema and is always deterministic.
    Semantic validation uses an LLM-as-judge and is explicitly marked unreliable
    (the "validation gap" noted in LIMITATIONS.md).
    """
    model_config = ConfigDict(frozen=True)

    json_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "JSON Schema dict the node output must satisfy. "
            "None means no syntactic check is performed."
        ),
    )
    semantic_check: bool = Field(
        default=False,
        description=(
            "If True, the runtime calls an LLM-as-judge after syntactic validation. "
            "Results are advisory—pass/fail from semantic check alone does not block "
            "execution unless explicitly configured. See LIMITATIONS.md §2."
        ),
    )


# ---------------------------------------------------------------------------
# Node configuration (σ : V → NodeConfig)
# ---------------------------------------------------------------------------


class NodeConfig(BaseModel):
    """
    Per-node configuration mapping σ(v) (§5.1, Definition 5.1).

    Timeout and retry values default to the engine-level EngineConfig when None,
    implementing per-node override with global defaults (resolved open question #1).
    """
    model_config = ConfigDict(frozen=True)

    node_type: NodeType = NodeType.LLM

    # Scheduling / dependency semantics
    join_mode: JoinMode = Field(
        default=JoinMode.ALL_OF,
        description=(
            "How this node's dependency join is evaluated. "
            "Relevant only when the node has ≥2 predecessors."
        ),
    )

    # Execution budget (None = inherit engine-level default)
    timeout_s: float | None = Field(
        default=None,
        ge=0.1,
        description="Maximum wall-clock seconds for a single node execution attempt.",
    )
    max_retries: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Maximum number of local_retry attempts before escalating to local_patch. "
            "Each retry consumes one recovery budget unit."
        ),
    )
    retry_delay_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Seconds to wait between retry attempts (simple fixed delay).",
    )

    # LLM-specific configuration (used when node_type == LLM)
    prompt_template: str | None = Field(
        default=None,
        description=(
            "Jinja2 / f-string template for the LLM prompt. "
            "Available variables: {task}, {upstream_outputs}, {node_label}."
        ),
    )
    model: str = Field(
        default="anthropic/claude-3-5-sonnet-20240620",
        description="LLM model identifier compatible with litellm (e.g. 'anthropic/claude-3.5-sonnet', 'nvidia/meta/llama-3.1-70b-instruct').",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Tool names available to this node during execution.",
    )

    # Mock-node configuration (used when node_type == MOCK)
    mock_output: dict[str, Any] | None = Field(
        default=None,
        description="Fixed output payload returned by MockNode.",
    )
    mock_delay_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Simulated execution latency for MockNode.",
    )
    mock_fail_type: str | None = Field(
        default=None,
        description=(
            "If set, MockNode raises this failure type instead of returning mock_output. "
            "Values: 'transient', 'contract_violation', 'dependency_error'."
        ),
    )

    # Human-in-the-loop (used when node_type == HUMAN)
    human_prompt: str | None = Field(
        default=None,
        description="Message displayed to the human operator when waiting for approval.",
    )

    # Contract
    contract: OutputContract = Field(default_factory=OutputContract)

    # Side-effect classification (Principle 4 from §4 / Table 3)
    idempotent: bool = Field(
        default=False,
        description=(
            "True if re-executing this node with the same inputs is safe (idempotent). "
            "Non-idempotent nodes have retry_delay_s enforced and are flagged in logs."
        ),
    )


# ---------------------------------------------------------------------------
# Graph elements
# ---------------------------------------------------------------------------


class Node(BaseModel):
    """A node v ∈ V in the execution plan."""
    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique node identifier within the plan.")
    label: str = Field(description="Human-readable name for display and logging.")
    config: NodeConfig = Field(default_factory=NodeConfig)


class Edge(BaseModel):
    """A directed dependency edge (source → target) in E ⊆ V × V."""
    model_config = ConfigDict(frozen=True)

    source: str = Field(description="Node ID of the predecessor.")
    target: str = Field(description="Node ID of the successor.")


# ---------------------------------------------------------------------------
# Execution Plan  Π = (id, version, V, E, σ, κ)
# ---------------------------------------------------------------------------


class Plan(BaseModel):
    """
    An immutable execution plan (§5.1, Definition 5.1).

    Π = (id, version, V, E, σ, κ) where:
      id      — globally unique plan identifier
      version — monotonically increasing integer; starts at 1
      V       — frozen tuple of Node objects
      E       — frozen tuple of Edge objects
      σ       — encoded in each Node's `.config` field (σ : V → NodeConfig)
      κ       — plan-level output contract (optional)

    Plan-version immutability (FR-9, §5.2):
      Once a Plan is created, its nodes and edges cannot change. Any structural
      modification (topology change during replan) must produce a NEW Plan with
      version = parent.version + 1 and parent_version_id = parent.plan_id.

    The (nodes, edges) fields are stored as tuples to communicate intent; Pydantic
    frozen=True prevents attribute reassignment after construction.
    """
    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Globally unique plan identifier.",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Monotonically increasing plan version (1-indexed).",
    )
    parent_version_id: str | None = Field(
        default=None,
        description=(
            "plan_id of the plan that triggered a replan to produce this version. "
            "None for the initial plan (version 1). Forms the lineage chain."
        ),
    )

    nodes: tuple[Node, ...] = Field(description="Immutable ordered tuple of nodes V.")
    edges: tuple[Edge, ...] = Field(description="Immutable ordered tuple of dependency edges E.")

    # Plan-level output contract κ (optional — most constraints are per-node)
    contract: OutputContract = Field(
        default_factory=OutputContract,
        description="Plan-level output contract κ applied after all nodes are terminal.",
    )

    task_description: str | None = Field(
        default=None,
        description="Natural-language description of the task this plan solves.",
    )

    # Convenience accessors --------------------------------------------------

    def node_ids(self) -> frozenset[str]:
        return frozenset(n.id for n in self.nodes)

    def node_by_id(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"Node '{node_id}' not found in plan {self.plan_id!r}")

    def predecessors(self, node_id: str) -> list[str]:
        """Return IDs of all direct predecessor nodes."""
        return [e.source for e in self.edges if e.target == node_id]

    def successors(self, node_id: str) -> list[str]:
        """Return IDs of all direct successor nodes."""
        return [e.target for e in self.edges if e.source == node_id]

    def root_nodes(self) -> list[str]:
        """Return node IDs with no incoming edges (entry points of the DAG)."""
        targets = {e.target for e in self.edges}
        return [n.id for n in self.nodes if n.id not in targets]

    def leaf_nodes(self) -> list[str]:
        """Return node IDs with no outgoing edges (exit points of the DAG)."""
        sources = {e.source for e in self.edges}
        return [n.id for n in self.nodes if n.id not in sources]

    @model_validator(mode="before")
    @classmethod
    def _coerce_sequences_to_tuples(cls, data: Any) -> Any:
        """Allow callers to pass lists for nodes/edges; internally stored as tuples."""
        if isinstance(data, dict):
            if "nodes" in data and isinstance(data["nodes"], list):
                data["nodes"] = tuple(data["nodes"])
            if "edges" in data and isinstance(data["edges"], list):
                data["edges"] = tuple(data["edges"])
        return data
