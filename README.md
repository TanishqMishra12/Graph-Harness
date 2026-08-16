# Structured Graph Harness (SGH)

SGH is a DAG-based execution engine for LLM Agents, serving as an unofficial reference implementation of the concepts proposed in Hu Wei's position paper: *"From Agent Loops to Structured Graphs: A Scheduler-Theoretic Framework for LLM Agent Execution"* (arXiv:2604.11378v1).

## Overview

The standard approach to orchestrating LLM agents relies on an iterative "Agent Loop" (e.g., ReAct), where an LLM repeatedly determines the next action by analyzing an execution context. In scheduler-theoretic terms, this functions as a single-ready-unit scheduler ($|U| \leq 1$), which introduces structural constraints:
- **Sequential Bottlenecks**: The system cannot execute independent tools concurrently.
- **Unbounded Recovery**: Failures can result in infinite retry loops without strict termination limits.
- **Context Pollution**: Combining execution instructions ($C_{exec}$) with diagnostic history ($C_{diag}$) increases token consumption and degrades reasoning performance.

SGH replaces the implicit loop with a formal Directed Acyclic Graph (DAG) scheduler.

## The Research Paper & How We Achieved It

This project is the direct implementation of **"From Agent Loops to Structured Graphs: A Scheduler-Theoretic Framework for LLM Agent Execution"** (arXiv:2604.11378v1) by Hu Wei. 

The paper argued that Agent Loops (like ReAct) suffer from three fatal flaws: they cannot execute tools concurrently, they get stuck in infinite recovery loops, and they pollute the LLM's context window with failure history. 

**How we solved it:**
We built a deterministic Python execution engine that completely decouples the LLM from the routing logic. 
1. **Concurrency**: Instead of asking the LLM what to do next, we strictly author a topological DAG (`Plan`). Our `Dispatcher` evaluates the graph mathematically, identifying all independent nodes and executing them concurrently via `asyncio`.
2. **Context Partitioning**: We physically separated the Execution Context ($C_{exec}$) from the Diagnostic Context ($C_{diag}$). The LLM only receives the pristine execution instructions, while the Engine handles the failure history.
3. **Deterministic Recovery**: We built a strict 3-tier failure cascade (`local_retry` -> `local_patch` -> `request_replan`). If an LLM node fails, the Engine attempts bounded retries and semantic patching. If it still fails, the Engine halts the graph gracefully instead of looping forever.

By offloading the scheduling burden from the LLM back to classical computer science graph theory, we achieved a more robust, auditable, and **10x faster** multi-agent execution system.

## Core Features

- **Concurrent Execution**: Resolves topological dependencies to maximize concurrency ($|U| > 1$). Independent nodes execute in parallel natively.
- **Context Partitioning**: Node executors receive isolated execution prompts ($C_{exec}$). Failure history ($C_{diag}$) is managed by the scheduler, not the node's LLM context.
- **Deterministic Recovery**: Implements a three-tier recovery escalation:
  1. `local_retry`: Transient failure retry.
  2. `local_patch`: Budget-constrained semantic patch.
  3. `request_replan`: Global execution halt on permanent failure.
- **State Machine**: Adheres to a strict 10-state lifecycle (e.g., Pending, Ready, Running, Blocked, Failed).
- **Immutable Audit Trail**: All state transitions are appended to a SQLite-backed event log.

## Installation

```bash
git clone https://github.com/TanishqMishra12/Graph-Harness.git sgh
cd sgh
pip install -e .
```

## Usage

### Command Line Interface

The Typer-based CLI provides commands for validation, execution, and auditing.

```bash
# Validate the structural integrity of a DAG plan
sgh validate examples/plan.json

# Execute a plan using the live terminal visualizer
sgh run examples/plan.json --db sgh.db

# Query the immutable audit history for a plan execution
sgh history <plan_id> --db sgh.db
```

### Python API

```python
import asyncio
from pathlib import Path
from sgh.core.plan import Plan
from sgh.scheduler.dispatcher import Dispatcher, EngineConfig
from sgh.nodes.factory import default_executor_factory

async def main():
    # Load and validate the plan
    plan_data = Path("plan.json").read_text()
    plan = Plan.model_validate_json(plan_data)
    
    # Initialize the dispatcher and execute
    dispatcher = Dispatcher(engine_config=EngineConfig())
    result = await dispatcher.run(plan, executor_factory=default_executor_factory)
    
    print(f"Execution succeeded: {result.succeeded}")
    if not result.succeeded:
        print(f"Failure summary: {result.failure_summary}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Performance Benchmark

By decoupling scheduling from LLM inference, SGH enables parallel execution of independent tasks.

A benchmark of 10 independent tasks (simulating 0.5s of work each) demonstrates the latency reduction compared to a constrained sequential Agent Loop ($|U| \leq 1$):

| Execution Engine       | Avg Concurrency ($|U|$) | Wall-clock Latency | Speedup |
|------------------------|-------------------------|--------------------|---------|
| Agent Loop Baseline    | 1.0                     | 5.06s              | 1.0x    |
| SGH (Concurrent DAG)   | 10.0                    | 0.50s              | 10.1x   |

*(See `examples/benchmark.py` for the implementation.)*

## Architecture

The system is composed of four primary layers:
1. **Core**: Pydantic models defining the `Plan`, `Node`, `Edge`, and `NodeState`.
2. **Scheduler**: The `Dispatcher` and `ReadySet` logic resolving execution order.
3. **Recovery**: The subsystem handling exponential backoffs and retry budgets.
4. **Persistence**: The SQLite `EventLog` and `PlanStore` maintaining immutability.

For known limitations and divergences from the reference paper (such as the absence of a dynamic LLM planner), see `docs/LIMITATIONS.md`.

## License

MIT License.
