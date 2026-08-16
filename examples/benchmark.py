"""
Empirical benchmark comparing SGH vs Agent Loop.

Proves the core hypothesis: |U| > 1 reduces wall-clock latency.
We construct a perfectly parallel DAG (10 independent mock nodes).
- SGH should execute it in ~0.5s.
- Agent Loop should execute it in ~5.0s.
"""
import asyncio
import time
import logging

from rich.console import Console
from rich.table import Table

from sgh.core.plan import Plan, Node, NodeConfig, NodeType
from sgh.nodes.base import BaseNode, NodeOutput
from sgh.scheduler.dispatcher import Dispatcher, EngineConfig
from sgh.baselines.structured_loop import AgentLoopDispatcher

# Silence normal logging for clean benchmark output
logging.getLogger().setLevel(logging.ERROR)
console = Console()

class SleepyMockNode(BaseNode):
    """A mock node that simulates 0.5s of work."""
    async def execute(self, ctx) -> NodeOutput:
        await asyncio.sleep(0.5)
        return NodeOutput(outcome="success", payload={"done": True})

def sleepy_factory(node: Node) -> BaseNode:
    return SleepyMockNode(node)

async def main():
    console.print("[bold]SGH Benchmark: Structured Graphs vs Agent Loops[/bold]")
    console.print("Workload: 10 purely parallel tasks (each takes 0.5s)\n")

    # Build the parallel DAG
    nodes = []
    for i in range(10):
        nodes.append(Node(
            id=f"task_{i}",
            label=f"Task {i}",
            config=NodeConfig(node_type=NodeType.MOCK)
        ))
    
    plan = Plan(nodes=nodes, edges=[])

    # 1. Run Agent Loop (Baseline)
    console.print("[yellow]Running Agent Loop Baseline (|U| <= 1)...[/yellow]")
    agent_dispatcher = AgentLoopDispatcher(engine_config=EngineConfig())
    t0 = time.monotonic()
    agent_res = await agent_dispatcher.run(plan, executor_factory=sleepy_factory)
    agent_time = time.monotonic() - t0
    
    # Calculate average U size
    agent_u_sum = sum(r.u_size for r in agent_res.scheduling_rounds)
    agent_avg_u = agent_u_sum / len(agent_res.scheduling_rounds) if agent_res.scheduling_rounds else 0

    # 2. Run SGH
    console.print("[green]Running SGH (Concurrent)...[/green]")
    sgh_dispatcher = Dispatcher(engine_config=EngineConfig())
    t0 = time.monotonic()
    sgh_res = await sgh_dispatcher.run(plan, executor_factory=sleepy_factory)
    sgh_time = time.monotonic() - t0
    
    sgh_u_sum = sum(r.u_size for r in sgh_res.scheduling_rounds)
    sgh_avg_u = sgh_u_sum / len(sgh_res.scheduling_rounds) if sgh_res.scheduling_rounds else 0

    # 3. Print Results
    table = Table(title="Benchmark Results")
    table.add_column("Engine", style="cyan")
    table.add_column("Avg Concurrency |U|", justify="right")
    table.add_column("Wall-clock Latency", justify="right")
    table.add_column("Speedup", justify="right")

    speedup = agent_time / sgh_time

    table.add_row(
        "Agent Loop Baseline",
        f"{agent_avg_u:.1f}",
        f"{agent_time:.2f}s",
        "1.0x"
    )
    table.add_row(
        "Structured Graph Harness",
        f"{sgh_avg_u:.1f}",
        f"{sgh_time:.2f}s",
        f"[bold green]{speedup:.1f}x[/bold green]"
    )

    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
