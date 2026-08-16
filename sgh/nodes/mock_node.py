"""
Mock node executor for SGH testing.

MockNode returns a configurable output after a configurable delay.
Failure injection is supported via config.mock_fail_type.

This is the primary testing tool for Phases 2–3:
  - Tests run with zero LLM/network calls (NFR-3)
  - Latency can be simulated to verify parallel dispatch timing
  - Failures can be injected to verify recovery escalation

Configuration (via NodeConfig):
  mock_output:    The dict to return on success. Defaults to {"result": "mock_ok"}.
  mock_delay_s:   Seconds to sleep before returning (simulates LLM latency).
  mock_fail_type: If set, raises this failure type instead of returning output.
                  Values: 'transient', 'contract_violation'
"""
from __future__ import annotations

import asyncio
import time

from sgh.core.plan import Node
from sgh.nodes.base import BaseNode, ExecContext, NodeOutput


class MockNode(BaseNode):
    """
    Mock node that returns a fixed payload after a simulated delay.

    Used exclusively in tests and demos. Never makes real LLM or tool calls.
    """

    def __init__(self, node: Node) -> None:
        super().__init__(node)

    async def execute(self, ctx: ExecContext) -> NodeOutput:
        config = ctx.config
        delay = config.mock_delay_s

        # Simulate execution latency
        if delay > 0:
            await asyncio.sleep(delay)

        start = time.monotonic()

        # Failure injection via config
        fail_type = config.mock_fail_type
        if fail_type:
            latency = time.monotonic() - start + delay
            if fail_type == "transient":
                return NodeOutput(
                    outcome="retry",
                    error_message=f"Mock transient error injected for node {ctx.node_id!r}",
                    failure_type="transient",
                    latency_s=latency,
                )
            elif fail_type == "contract_violation":
                return NodeOutput(
                    outcome="failure",
                    error_message=f"Mock contract violation for node {ctx.node_id!r}",
                    failure_type="contract_violation",
                    payload={"invalid": "output"},  # won't pass any schema
                    latency_s=latency,
                )
            else:
                return NodeOutput(
                    outcome="failure",
                    error_message=f"Mock failure type {fail_type!r} for node {ctx.node_id!r}",
                    failure_type=fail_type,
                    latency_s=latency,
                )

        # Success path
        output = dict(config.mock_output) if config.mock_output else {"result": "mock_ok"}
        latency = time.monotonic() - start + delay
        return NodeOutput(
            outcome="success",
            payload=output,
            latency_s=latency,
        )
