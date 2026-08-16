"""
Tool Node Executor for SGH.

Executes deterministic Python functions without an LLM.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable

from sgh.nodes.base import BaseNode, ExecContext, NodeOutput

logger = logging.getLogger(__name__)

# Global registry for tool functions
_TOOL_REGISTRY: dict[str, Callable] = {}


def register_tool(name: str, func: Callable) -> None:
    """Register a tool function by name."""
    _TOOL_REGISTRY[name] = func


def get_tool(name: str) -> Callable | None:
    return _TOOL_REGISTRY.get(name)


class ToolNode(BaseNode):
    """
    Executes a deterministic tool function.
    
    The tool name is expected to be the first element in ctx.config.tools.
    """

    async def execute(self, ctx: ExecContext) -> NodeOutput:
        if not ctx.config.tools:
            return NodeOutput(
                outcome="failure",
                error_message="ToolNode requires at least one tool name in config.tools.",
                failure_type="contract_violation"
            )

        tool_name = ctx.config.tools[0]
        func = get_tool(tool_name)
        if not func:
            return NodeOutput(
                outcome="failure",
                error_message=f"Tool {tool_name!r} not found in registry.",
                failure_type="dependency_error"
            )

        try:
            if inspect.iscoroutinefunction(func):
                result = await func(ctx)
            else:
                # Run sync functions in threadpool so they don't block event loop
                result = await asyncio.to_thread(func, ctx)
        except Exception as e:
            # Tool crash -> transient or contract? Treat as transient for retry
            return NodeOutput(
                outcome="retry",
                error_message=f"Tool {tool_name!r} raised Exception: {e}",
                failure_type="transient"
            )

        # Output payload is the dictionary returned by the tool, or a wrapper
        if isinstance(result, dict):
            payload = result
        else:
            payload = {"result": result}

        return NodeOutput(
            outcome="success",
            payload=payload
        )
