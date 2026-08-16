"""
Node Executor Factory for SGH.

Instantiates the correct BaseNode subclass based on the node's NodeType.
"""
from __future__ import annotations

from typing import Any

from sgh.core.plan import Node, NodeType
from sgh.nodes.base import BaseNode
from sgh.nodes.human_node import HumanNode
from sgh.nodes.llm_node import LLMNode
from sgh.nodes.mock_node import MockNode
from sgh.nodes.tool_node import ToolNode


def default_executor_factory(node: Node, kwargs: dict[str, Any] | None = None) -> BaseNode:
    """
    Default factory to create a Node executor from a Node definition.
    
    Args:
        node: The Node from the plan.
        kwargs: Optional dict of kwargs to pass to litellm in LLMNode.
        
    Returns:
        An instantiated BaseNode subclass ready for execution.
    """
    if node.config.node_type == NodeType.LLM:
        return LLMNode(node, kwargs=kwargs)
    elif node.config.node_type == NodeType.TOOL:
        return ToolNode(node)
    elif node.config.node_type == NodeType.MOCK:
        return MockNode(node)
    elif node.config.node_type == NodeType.HUMAN:
        return HumanNode(node)
    else:
        raise ValueError(f"Unknown node_type: {node.config.node_type}")
