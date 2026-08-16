"""
Tests for Phase 4: Node Executors (LLMNode, ToolNode, HumanNode).

Focuses on context partition (ExecContext vs DiagContext) and outcome mapping.
"""
import asyncio
import pytest
from typing import Any

from sgh.core.plan import Node, NodeConfig, OutputContract, NodeType
from sgh.nodes.base import ExecContext
from sgh.nodes.human_node import HumanNode
from sgh.nodes.tool_node import ToolNode, register_tool
from sgh.nodes.llm_node import LLMNode


# ---------------------------------------------------------------------------
# Mocks & Helpers
# ---------------------------------------------------------------------------

class MockAnthropicMessage:
    def __init__(self, text: str):
        self.text = text

class MockAnthropicUsage:
    def __init__(self):
        self.input_tokens = 10
        self.output_tokens = 20

class MockAnthropicResponse:
    def __init__(self, content_text: str):
        self.content = [MockAnthropicMessage(content_text)]
        self.usage = MockAnthropicUsage()

class MockAnthropicMessages:
    def __init__(self, response_text: str, should_fail: bool = False):
        self._response_text = response_text
        self._should_fail = should_fail
        self.last_kwargs = {}

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._should_fail:
            raise RuntimeError("API Error")
        return MockAnthropicResponse(self._response_text)

class MockAsyncAnthropic:
    def __init__(self, response_text: str = "hello", should_fail: bool = False):
        self.messages = MockAnthropicMessages(response_text, should_fail)


def make_ctx(node_type: NodeType, **config_kwargs) -> ExecContext:
    config = NodeConfig(node_type=node_type, **config_kwargs)
    node = Node(id="test_node", label="Test Node", config=config)
    return ExecContext(
        plan_id="p1",
        plan_version=1,
        node_id=node.id,
        node_label=node.label,
        config=config,
        upstream_outputs={"prev": {"val": 42}},
        task_description="Global task"
    )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_human_node_returns_wait_outcome():
    ctx = make_ctx(NodeType.HUMAN)
    node = HumanNode(Node(id="test_node", label="Test", config=ctx.config))
    result = await node.execute(ctx)
    assert result.outcome == "wait"
    assert result.should_wait is True

@pytest.mark.asyncio
async def test_tool_node_executes_sync_function():
    def my_tool(ctx: ExecContext) -> dict:
        return {"sync_done": True, "task": ctx.task_description}
    
    register_tool("my_sync_tool", my_tool)
    
    ctx = make_ctx(NodeType.TOOL, tools=["my_sync_tool"])
    node = ToolNode(Node(id="test_node", label="Test", config=ctx.config))
    result = await node.execute(ctx)
    
    assert result.succeeded is True
    assert result.payload == {"sync_done": True, "task": "Global task"}

@pytest.mark.asyncio
async def test_tool_node_executes_async_function():
    async def my_async_tool(ctx: ExecContext) -> str:
        await asyncio.sleep(0.01)
        return "async_done"
    
    register_tool("my_async_tool", my_async_tool)
    
    ctx = make_ctx(NodeType.TOOL, tools=["my_async_tool"])
    node = ToolNode(Node(id="test_node", label="Test", config=ctx.config))
    result = await node.execute(ctx)
    
    assert result.succeeded is True
    # ToolNode wraps raw string in a dict
    assert result.payload == {"result": "async_done"}

@pytest.mark.asyncio
async def test_tool_node_missing_tool_fails():
    ctx = make_ctx(NodeType.TOOL, tools=["non_existent_tool"])
    node = ToolNode(Node(id="test_node", label="Test", config=ctx.config))
    result = await node.execute(ctx)
    
    assert result.failed is True
    assert result.failure_type == "dependency_error"
    assert "not found in registry" in result.error_message

@pytest.mark.asyncio
async def test_llm_node_formats_prompt_correctly():
    mock_client = MockAsyncAnthropic(response_text="ok")
    ctx = make_ctx(NodeType.LLM, prompt_template="Label: {node_label}, Upstream: {upstream_outputs}")
    node = LLMNode(Node(id="test_node", label="Test", config=ctx.config), client=mock_client)
    
    result = await node.execute(ctx)
    assert result.succeeded is True
    
    # Check that the prompt was rendered correctly
    sent_messages = mock_client.messages.last_kwargs["messages"]
    prompt = sent_messages[0]["content"]
    assert "Label: Test Node" in prompt
    assert "val" in prompt
    assert "42" in prompt

@pytest.mark.asyncio
async def test_llm_node_validates_json_schema():
    mock_client = MockAsyncAnthropic(response_text='{"answer": 42}')
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"]
    }
    ctx = make_ctx(NodeType.LLM, contract=OutputContract(json_schema=schema))
    node = LLMNode(Node(id="test_node", label="Test", config=ctx.config), client=mock_client)
    
    result = await node.execute(ctx)
    assert result.succeeded is True
    assert result.payload == {"answer": 42}
    
    # Check that system prompt included the schema constraint
    system_prompt = mock_client.messages.last_kwargs["system"]
    assert "OUTPUT CONTRACT" in system_prompt
    assert "answer" in system_prompt

@pytest.mark.asyncio
async def test_llm_node_invalid_json_violates_contract():
    mock_client = MockAsyncAnthropic(response_text='{"answer": "not_an_int"}')
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"]
    }
    ctx = make_ctx(NodeType.LLM, contract=OutputContract(json_schema=schema))
    node = LLMNode(Node(id="test_node", label="Test", config=ctx.config), client=mock_client)
    
    result = await node.execute(ctx)
    assert result.failed is True
    assert result.failure_type == "contract_violation"
    assert "JSON Schema validation failed" in result.error_message

@pytest.mark.asyncio
async def test_llm_node_api_failure_is_transient():
    mock_client = MockAsyncAnthropic(should_fail=True)
    ctx = make_ctx(NodeType.LLM)
    node = LLMNode(Node(id="test_node", label="Test", config=ctx.config), client=mock_client)
    
    result = await node.execute(ctx)
    assert result.should_retry is True
    assert result.failure_type == "transient"
    assert "API Error" in result.error_message
