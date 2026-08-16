"""
SGH — Structured Graph Harness
A DAG-based execution engine for LLM agents, implementing the scheduler-theoretic
design proposed in arXiv:2604.11378v1 (Hu Wei, 2026).

This is an independent implementation; the paper provides no reference code.
"""
from sgh.core.plan import (
    Edge,
    JoinMode,
    Node,
    NodeConfig,
    NodeState,
    NodeType,
    OutputContract,
    Plan,
    RecoveryLevel,
)
from sgh.core.validator import PlanValidationError, validate_plan

__all__ = [
    "Edge",
    "JoinMode",
    "Node",
    "NodeConfig",
    "NodeState",
    "NodeType",
    "OutputContract",
    "Plan",
    "PlanValidationError",
    "RecoveryLevel",
    "validate_plan",
]

__version__ = "0.1.0"
