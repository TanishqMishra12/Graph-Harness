# Limitations and Divergences

While SGH strictly adheres to the core architecture proposed in *From Agent Loops to Structured Graphs*, there are several known limitations and practical divergences in this implementation.

## 1. Dynamic Plan Generation (Phase 10)
**Paper Claim:** The paper envisions a macro-LLM "Planner" that receives a user prompt and generates the DAG dynamically before execution. 

**Our Divergence:** SGH currently focuses strictly on the *Scheduler/Execution* layer. Plans must be authored statically (e.g., via `Plan.model_validate_json()` or manually building Pydantic objects). Generating deterministic graphs via LLMs (e.g., forcing JSON schemas or using function calling for DAG topologies) is highly complex and error-prone with current models. While the architecture supports an LLM planner frontend (as a FastAPI route), it is not implemented out of the box in this release.

## 2. Global Re-Planning (`request_replan`)
**Paper Claim:** When a node completely exhausts its local retry and patch budgets, it triggers a `request_replan`. The execution halts, the Planner reads the diagnostic context ($C_{diag}$), and generates an entirely new DAG (a new `version` of the plan) to resume execution.

**Our Divergence:** In SGH, when a node triggers `request_replan`, the engine correctly cascades a terminal `FAILED` state to all downstream nodes and halts execution safely. However, we do not have an automated Planner to catch this failure and synthesize a v2 Plan. The recovery layer currently serves as an architectural boundary where an external macro-agent *could* intervene, but the engine itself just stops and returns `succeeded=False`.

## 3. Data Flow Between Nodes
**Paper Claim:** The paper focuses on control flow (execution topologies) and leaves data flow abstract.

**Our Divergence:** SGH passes a dictionary of `upstream_outputs` to the `execute()` method of `NodeExecutor`. While functional, this lacks a rigorous type-safe data schema between nodes (like Pydantic guarantees). If a downstream node expects a specific key that the upstream node did not emit, it fails at runtime rather than static validation time.

## 4. Subgraphs and Composition
**Paper Claim:** The paper hints at hierarchical graph execution (nodes that themselves represent entire DAGs).

**Our Divergence:** SGH does not support subgraph nodes. The engine is entirely flat. While you can author complex topologies, you cannot encapsulate a sub-plan into a single `NodeType.SUBGRAPH`.

## Summary
This implementation of SGH focuses on the execution scheduler, demonstrating that decoupling the scheduler from the LLM context loop mitigates unbounded retries, enables $|U| > 1$ concurrency, and supports immutable audit trails.

Future iterations may introduce the Macro-Planner and automated global replanning components.
