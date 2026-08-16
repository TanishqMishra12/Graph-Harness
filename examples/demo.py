"""
Demo for SGH: Executing a simple code generation and validation DAG.

Models Figure 1 from the paper: replacing an unbounded ReAct loop with a 
static, structured DAG.

Graph:
          write_code
         /          \
  lint_code        test_code
         \          /
       deploy_code (Human)

"""
import asyncio
import json
import logging
from pathlib import Path

from sgh.core.plan import Plan, Node, NodeConfig, OutputContract, NodeType, Edge
from sgh.nodes.tool_node import register_tool
from sgh.nodes.base import ExecContext
from sgh.scheduler.dispatcher import Dispatcher, EngineConfig
from sgh.persistence.models import init_db
from sgh.persistence.plan_store import PlanStore
from sgh.persistence.event_log import EventLog
from sgh.viz.live_view import LiveDashboard

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# --- Define Tools ---

def lint_code_tool(ctx: ExecContext) -> dict:
    code = ctx.upstream_outputs.get("write_code", {}).get("code", "")
    logging.info(f"Linting code:\n{code}")
    if "import os" in code:
        raise ValueError("Linter Error: 'import os' is forbidden.")
    return {"lint_passed": True}

def test_code_tool(ctx: ExecContext) -> dict:
    code = ctx.upstream_outputs.get("write_code", {}).get("code", "")
    logging.info(f"Testing code:\n{code}")
    if "def " not in code:
        raise ValueError("Test Error: No function defined.")
    return {"test_passed": True}

register_tool("lint_code_tool", lint_code_tool)
register_tool("test_code_tool", test_code_tool)


async def main():
    # 1. Setup Persistence
    db_path = "sgh_demo.db"
    await init_db(db_path)
    plan_store = PlanStore(db_path)
    event_log = EventLog(db_path)
    event_log.start()

    # 2. Build DAG Plan
    write_code_node = Node(
        id="write_code",
        label="Generate Python function",
        config=NodeConfig(
            node_type=NodeType.LLM,
            model="nvidia_nim/meta/llama-3.1-70b-instruct",
            prompt_template="Write a python function called `hello_world` that returns 'hello'. Return as JSON with key 'code'.",
            contract=OutputContract(
                json_schema={
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"]
                }
            )
        )
    )

    lint_node = Node(
        id="lint_code",
        label="Lint the code",
        config=NodeConfig(
            node_type=NodeType.TOOL,
            tools=["lint_code_tool"]
        )
    )

    test_node = Node(
        id="test_code",
        label="Test the code",
        config=NodeConfig(
            node_type=NodeType.TOOL,
            tools=["test_code_tool"]
        )
    )

    deploy_node = Node(
        id="deploy_code",
        label="Approve deployment",
        config=NodeConfig(
            node_type=NodeType.HUMAN
        )
    )

    plan = Plan(
        nodes=[write_code_node, lint_node, test_node, deploy_node],
        edges=[
            Edge(source="write_code", target="lint_code"),
            Edge(source="write_code", target="test_code"),
            Edge(source="lint_code", target="deploy_code"),
            Edge(source="test_code", target="deploy_code"),
        ],
        task_description="Generate, validate, and deploy a hello world function."
    )

    # Save to immutable store
    try:
        await plan_store.save_plan(plan)
        logging.info(f"Saved Plan {plan.plan_id} v{plan.version} to store.")
    except Exception as e:
        logging.info(f"Plan already saved: {e}")

    # Setup live dashboard
    dashboard = LiveDashboard(plan)
    dashboard.start()

    def handle_transition(event):
        event_log.on_transition(event)
        dashboard.on_transition(event)

    def handle_round(event):
        dashboard.on_round(event)

    # 3. Execute
    dispatcher = Dispatcher(
        engine_config=EngineConfig(global_timeout_s=30.0),
        on_node_transition=handle_transition,
        on_scheduling_round=handle_round
    )

    logging.info("Starting DAG execution...")
    result = await dispatcher.run(plan)

    logging.info(f"Execution complete. Succeeded: {result.succeeded}")
    logging.info(f"Terminal states: {result.terminal_states}")
    if "write_code" in result.node_outputs:
        logging.info(f"Generated code: {result.node_outputs['write_code']['code']}")

    # Stop background logger and dashboard
    dashboard.stop()
    await event_log.stop()

    # 4. Read back audit trail
    history = await event_log.get_history(plan.plan_id, plan.version)
    logging.info(f"Total transition events recorded: {len(history)}")


if __name__ == "__main__":
    asyncio.run(main())
