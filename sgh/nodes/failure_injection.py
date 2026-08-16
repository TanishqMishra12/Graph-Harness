"""
Failure injection harness for SGH (FR-15).

Allows deliberate, controlled injection of specific failure types into node
executions. This is the primary tool for demonstrating the recovery escalation
ladder without relying on incidental real-world failures.

Injection is configured via:
  1. A dict of specs (programmatic, used in tests)
  2. An env var SGH_INJECT_FAILURES=node_id:type:attempt,...
     e.g. SGH_INJECT_FAILURES=fix_a:transient:1,analyze:contract_violation:2

The injected failure types map directly to the FailureType enum:
  transient           → NodeOutput(outcome='retry', failure_type='transient')
  contract_violation  → NodeOutput(outcome='failure', failure_type='contract_violation')
  dependency_error    → NodeOutput(outcome='failure', failure_type='dependency_error')

The `on_attempt` field controls which attempt triggers the injection:
  on_attempt=1  → fail on the first attempt only (test local_retry)
  on_attempt=2  → fail on second attempt (test local_patch after retry)
  on_attempt=None → fail on ALL attempts (test full escalation to replan)

The FailureInjector wraps a BaseNode executor. The wrapper intercepts execute()
and either returns the injected failure or delegates to the real executor.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

from sgh.core.plan import Node
from sgh.nodes.base import BaseNode, ExecContext, NodeOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure specification
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FailureSpec:
    """
    A single failure injection specification.

    Attributes:
        node_id:      Which node to intercept.
        failure_type: 'transient', 'contract_violation', or 'dependency_error'.
        on_attempt:   Which attempt number to inject on (1-indexed).
                      None means inject on every attempt.
    """
    node_id: str
    failure_type: str         # 'transient' | 'contract_violation' | 'dependency_error'
    on_attempt: int | None = 1   # 1-indexed; None = every attempt


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------


class FailureInjector:
    """
    Controls deliberate failure injection during execution.

    Usage (programmatic):
        injector = FailureInjector([
            FailureSpec("fix_a", "transient", on_attempt=1),
        ])

    Usage (from env var):
        injector = FailureInjector.from_env()
        # SGH_INJECT_FAILURES=fix_a:transient:1,analyze:contract_violation:2
    """

    def __init__(self, specs: list[FailureSpec] | None = None) -> None:
        self._specs: dict[str, list[FailureSpec]] = {}
        for spec in (specs or []):
            self._specs.setdefault(spec.node_id, []).append(spec)
        self._attempt_counts: dict[str, int] = {}

    @classmethod
    def from_env(cls, env_var: str = "SGH_INJECT_FAILURES") -> "FailureInjector":
        """
        Parse failure specs from an environment variable.

        Format: node_id:failure_type:attempt[,node_id:failure_type:attempt,...]
        Attempt is optional; defaults to 1.

        Examples:
          SGH_INJECT_FAILURES=fix_a:transient:1
          SGH_INJECT_FAILURES=fix_a:transient:1,analyze:contract_violation:2
          SGH_INJECT_FAILURES=fix_a:transient  (defaults to attempt=1)
        """
        raw = os.environ.get(env_var, "").strip()
        if not raw:
            return cls()

        specs = []
        for entry in raw.split(","):
            parts = entry.strip().split(":")
            if len(parts) < 2:
                logger.warning("Skipping malformed injection spec %r", entry)
                continue
            node_id = parts[0]
            failure_type = parts[1]
            on_attempt = int(parts[2]) if len(parts) >= 3 else 1
            specs.append(FailureSpec(node_id, failure_type, on_attempt))

        logger.info(
            "FailureInjector loaded %d spec(s) from %s: %s",
            len(specs), env_var, [(s.node_id, s.failure_type, s.on_attempt) for s in specs],
        )
        return cls(specs)

    @classmethod
    def from_dict(cls, specs: dict[str, str | dict]) -> "FailureInjector":
        """
        Parse failure specs from a plain dict.

        Format:
          {"node_id": "failure_type"}  (attempt=1)
          {"node_id": {"type": "failure_type", "attempt": N}}

        Example:
          FailureInjector.from_dict({"fix_a": "transient", "analyze": {"type": "contract_violation", "attempt": 2}})
        """
        result = []
        for node_id, spec in specs.items():
            if isinstance(spec, str):
                result.append(FailureSpec(node_id, spec, on_attempt=1))
            elif isinstance(spec, dict):
                result.append(FailureSpec(
                    node_id=node_id,
                    failure_type=spec.get("type", "transient"),
                    on_attempt=spec.get("attempt", 1),
                ))
        return cls(result)

    def has_injection(self, node_id: str) -> bool:
        return node_id in self._specs

    def should_inject(self, node_id: str, attempt: int) -> str | None:
        """
        Return the failure_type to inject, or None if no injection applies.

        Args:
            node_id: The node being executed.
            attempt: Which attempt this is (1-indexed).

        Returns:
            failure_type string, or None.
        """
        specs = self._specs.get(node_id, [])
        for spec in specs:
            if spec.on_attempt is None or spec.on_attempt == attempt:
                logger.info(
                    "FAILURE INJECTION: node=%r attempt=%d type=%r",
                    node_id, attempt, spec.failure_type,
                )
                return spec.failure_type
        return None

    def wrap_executor(self, node: Node, base_executor: BaseNode) -> "InjectedNode":
        """
        Wrap a BaseNode executor with failure injection capability.

        If this injector has no spec for node.id, returns base_executor unchanged.
        """
        if not self.has_injection(node.id):
            return base_executor  # type: ignore[return-value]
        return InjectedNode(node, base_executor, self)


# ---------------------------------------------------------------------------
# Injected node wrapper
# ---------------------------------------------------------------------------


class InjectedNode(BaseNode):
    """
    A BaseNode wrapper that injects failures before delegating to the real executor.

    The injection is transparent to the caller — it returns a NodeOutput with
    the injected failure type rather than calling the real executor.
    """

    def __init__(
        self,
        node: Node,
        real_executor: BaseNode,
        injector: FailureInjector,
    ) -> None:
        super().__init__(node)
        self._real = real_executor
        self._injector = injector

    async def execute(self, ctx: ExecContext) -> NodeOutput:
        failure_type = self._injector.should_inject(ctx.node_id, ctx.attempt_number)

        if failure_type is None:
            # No injection — delegate to real executor
            return await self._real.execute(ctx)

        # Inject the specified failure
        if failure_type == "transient":
            return NodeOutput(
                outcome="retry",
                error_message=(
                    f"[INJECTED] Transient failure on {ctx.node_id!r} "
                    f"attempt {ctx.attempt_number}"
                ),
                failure_type="transient",
            )
        elif failure_type == "contract_violation":
            return NodeOutput(
                outcome="failure",
                error_message=(
                    f"[INJECTED] Contract violation on {ctx.node_id!r} "
                    f"attempt {ctx.attempt_number}"
                ),
                failure_type="contract_violation",
                payload={"injected": True, "bad_field": "this_fails_schema"},
            )
        elif failure_type == "dependency_error":
            return NodeOutput(
                outcome="failure",
                error_message=(
                    f"[INJECTED] Dependency error on {ctx.node_id!r} "
                    f"attempt {ctx.attempt_number}"
                ),
                failure_type="dependency_error",
            )
        else:
            return NodeOutput(
                outcome="failure",
                error_message=f"[INJECTED] Unknown failure type {failure_type!r}",
                failure_type=failure_type,
            )
