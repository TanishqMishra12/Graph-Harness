# Structured Graph Harness (SGH)

A highly concurrent, DAG-based execution engine for LLM Agents, heavily inspired by Hu Wei's position paper: *"From Agent Loops to Structured Graphs: A Scheduler-Theoretic Framework for LLM Agent Execution"* (arXiv:2604.11378v1).

## ⚠️ Disclaimer
**This project is a standalone, unofficial reference implementation.** It implements the exact `Structured Graph Harness` architecture proposed in the paper, but under an original name, and is completely unaffiliated with the author.

---

## 📖 The Paradigm Shift: From Loops to Graphs

Most modern LLM agents operate on the **Agent Loop** paradigm (e.g., ReAct). A single LLM orchestrates actions by iteratively reading a growing context window.

According to Hu Wei's scheduler-theoretic framework, Agent Loops are **single-ready-unit schedulers** ($|U| \leq 1$). At any given time, only one action can be executed, leading to three structural weaknesses:
1. **Implicit Dependencies**: The LLM must deduce execution order at runtime, which scales poorly as tasks grow complex.
2. **Unbounded Recovery**: Failures often trigger endless loops of ad-hoc retries.
3. **Mutable History**: Debugging is a nightmare because execution traces are deeply embedded in messy LLM context windows.

**SGH** replaces the Agent Loop with a **DAG (Directed Acyclic Graph) Scheduler**.

### SGH Architecture
- **Nodes**: Represent distinct units of work (`LLM`, `TOOL`, `HUMAN`).
- **Edges**: Represent explicit execution dependencies.
- **Dispatcher**: The core engine that resolves the DAG topologically, maximizing concurrent execution ($|U| > 1$).
- **Strict Context Partitioning**: $C_{exec}$ (execution prompts) vs $C_{diag}$ (diagnostic history). Executors only ever receive pristine $C_{exec}$. The engine abstracts away failure handling.
- **Three-Level Recovery**: Deterministic escalation from `local_retry` -> `local_patch` -> `request_replan` (global failure).
- **Immutable Audit Logs**: All state transitions are saved to a SQLite `EventLog`, providing a perfectly auditable execution trace.

---

## 🚀 Benchmark: Proving $|U| > 1$ Speedups

Because SGH decouples scheduling from the LLM, it natively supports parallel execution of independent nodes.

We ran a benchmark comparing SGH to a constrained Baseline Agent Loop (where $|U| \leq 1$):
```
Workload: 10 purely parallel tasks (0.5s each)

+-----------------------------------------------------------------------------+
| Engine                 | Avg Concurrency |U| | Wall-clock Latency | Speedup |
|------------------------+---------------------+--------------------+---------|
| Agent Loop Baseline    |                 1.0 |              5.06s |    1.0x |
| Structured Graph       |                10.0 |              0.50s |   10.1x |
+-----------------------------------------------------------------------------+
```
**Conclusion:** Moving from Agent Loops to Structured Graphs yields massive latency reductions for concurrent tasks.

---

## 📦 Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/TanishqMishra12/Graph-Harness.git sgh
cd sgh

# Install via pip
pip install -e .
```

### The Typer CLI

SGH comes with a fully-featured Typer CLI for managing and executing plans.

```bash
# Validate a JSON/YAML plan statically
python -m sgh.cli.main validate examples/plan.json

# Run a DAG plan with a Live Rich Dashboard
python -m sgh.cli.main run examples/plan.json --db sgh.db

# Query immutable audit history
python -m sgh.cli.main history "plan-123" --db sgh.db
```

### Python API Usage

```python
import asyncio
from sgh.core.plan import Plan
from sgh.scheduler.dispatcher import Dispatcher, EngineConfig
from sgh.nodes.factory import default_executor_factory

async def main():
    plan = Plan.model_validate_json(Path("plan.json").read_text())
    
    dispatcher = Dispatcher(engine_config=EngineConfig())
    result = await dispatcher.run(plan, executor_factory=default_executor_factory)
    
    print(f"Success: {result.succeeded}")

asyncio.run(main())
```

---

## 🧩 Built-In Features
- `NodeState` State Machine: A rigorous 10-state lifecycle (Pending -> Ready -> Running -> Executed/Failed).
- `JoinEvaluator`: Supports `all_of` and `any_of` dependency joins for complex workflows.
- `LiveDashboard`: A `rich`-powered live terminal UI visualizing active states and the $|U|$ metric.

## 📄 License
MIT License. Created by Tanishq Mishra.
