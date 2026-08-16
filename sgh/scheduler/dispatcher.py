"""
Async dispatcher for SGH — the main scheduling loop.

Implements the execution system E = (S, U, P, O, Δ) from §3.1 of arXiv:2604.11378v1.

The dispatcher loop:
  1. Compute the ready set U(S) via ready_set.compute_ready_set().
  2. Apply the pending → ready / pending → blocked transitions.
  3. Dispatch all ready nodes concurrently via asyncio.gather().
  4. For each completed node, determine the next state via the outcome O:
       success  → running → executed  (+ contract validation)
       retry    → running → failed_retryable  (retry budget checked)
       failure  → running → failed    (or failed_retryable if budget remains)
       escalate → recovery layer invoked
  5. Recovery layer (Phase 3):
       failed_retryable → RecoveryManager → local_retry → re-queued as ready
       failed           → RecoveryManager → local_patch → re-queued as ready (patched config)
       failed (patched) → RecoveryManager → request_replan → new Plan version
  6. Repeat until all nodes are terminal or global timeout fires.

Policy P (§3.3):
  "dispatch all ready nodes" — deterministic, multi-ready-unit (|U| ≥ 1 when
  independent nodes are available).

NFR-2 (Termination):
  Every node reaches a terminal state within its budget + global_timeout_s.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import Any, Callable

from sgh.core.plan import Node, NodeConfig, NodeState, Plan
from sgh.core.state_machine import transition
from sgh.core.validator import validate_plan
from sgh.nodes.base import BaseNode, DiagContext, ExecContext, NodeOutput
from sgh.nodes.mock_node import MockNode
from sgh.nodes.factory import default_executor_factory
from sgh.recovery.diagnoser import FailureType, diagnose
from sgh.recovery.escalation import (
    EscalationViolationError,
    RecoveryAction,
    RecoveryActionType,
    RecoveryManager,
)
from sgh.scheduler.ready_set import (
    ReadySetResult,
    compute_ready_set,
    execution_is_complete,
)

logger = logging.getLogger(__name__)

NodeExecutorFactory = Callable[[Node], BaseNode]


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EngineConfig:
    """
    Global defaults applied when a node's NodeConfig leaves a field as None.

    These are the "global defaults with per-node override" from open question #1.
    """
    default_timeout_s: float = 120.0      # 2 minutes per node attempt
    default_max_retries: int = 1          # one local_retry before escalating to patch
    default_retry_delay_s: float = 2.0   # seconds between retry attempts
    global_timeout_s: float = 3600.0     # 1 hour total execution wall-clock limit
    max_scheduling_rounds: int = 1000    # safety limit on the main loop iterations


def _resolve_config(node_config: NodeConfig, engine: EngineConfig) -> NodeConfig:
    """Return a NodeConfig with None values filled from engine-level defaults."""
    return node_config.model_copy(update={
        "timeout_s": node_config.timeout_s if node_config.timeout_s is not None
                     else engine.default_timeout_s,
        "max_retries": node_config.max_retries if node_config.max_retries is not None
                       else engine.default_max_retries,
        "retry_delay_s": node_config.retry_delay_s if node_config.retry_delay_s is not None
                         else engine.default_retry_delay_s,
    })


# ---------------------------------------------------------------------------
# Scheduling round event (for visualization / event log)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SchedulingRoundEvent:
    """Emitted before each dispatch batch. Used by live_view and event_log."""
    round_number: int
    ready_set: list[str]
    u_size: int               # |U(s)| — the key SGH metric
    state_snapshot: dict[str, NodeState]


@dataclasses.dataclass
class NodeTransitionEvent:
    """Emitted for every node state transition."""
    plan_id: str
    plan_version: int
    round_number: int
    node_id: str
    from_state: NodeState
    to_state: NodeState
    payload: dict[str, Any] = dataclasses.field(default_factory=dict)
    error_message: str = ""
    latency_s: float = 0.0


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ExecutionResult:
    """Final result of a complete plan execution."""
    plan_id: str
    plan_version: int
    terminal_states: dict[str, NodeState]
    scheduling_rounds: list[SchedulingRoundEvent]
    node_outputs: dict[str, dict[str, Any]]   # node_id → output payload
    wall_clock_s: float
    succeeded: bool                            # True iff all nodes are `executed`
    failure_summary: str = ""


# ---------------------------------------------------------------------------
# Node executor factory
# ---------------------------------------------------------------------------


NodeExecutorFactory = Callable[[Node], BaseNode]


def default_executor_factory(node: Node) -> BaseNode:
    """
    Default factory: returns a MockNode for MOCK type, raises for others.

    The LLM executor factory is wired in Phase 4.
    """
    from sgh.core.plan import NodeType
    if node.config.node_type == NodeType.MOCK:
        return MockNode(node)
    raise NotImplementedError(
        f"No executor available for node_type={node.config.node_type.value!r}. "
        "Wire a full executor factory when using LLM or TOOL nodes."
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class Dispatcher:
    """
    The SGH execution engine — implements the main scheduling loop.

    Usage:
        dispatcher = Dispatcher(engine_config=EngineConfig())
        result = await dispatcher.run(plan, executor_factory=my_factory)
    """

    def __init__(
        self,
        engine_config: EngineConfig | None = None,
        on_scheduling_round: Callable[[SchedulingRoundEvent], None] | None = None,
        on_node_transition: Callable[[NodeTransitionEvent], None] | None = None,
        failure_injector: Any | None = None,  # FailureInjector | None
    ) -> None:
        self.engine = engine_config or EngineConfig()
        self._on_round = on_scheduling_round
        self._on_transition = on_node_transition
        self._failure_injector = failure_injector  # injected in Phase 3+

    async def run(
        self,
        plan: Plan,
        executor_factory: NodeExecutorFactory | None = None,
        upstream_outputs: dict[str, dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        """
        Execute a validated plan to completion.

        Args:
            plan:             The plan to execute (must pass validate_plan first).
            executor_factory: Factory function Node → BaseNode.
                              Defaults to default_executor_factory (mock-only).
            upstream_outputs: Pre-seeded outputs for nodes already in `executed` state
                              (used when resuming from a checkpoint or replan).

        Returns:
            ExecutionResult with final states, outputs, and scheduling history.
        """
        # Validate before accepting
        validate_plan(plan)
        factory = executor_factory or default_executor_factory

        # Initialize state
        states: dict[str, NodeState] = {n.id: NodeState.PENDING for n in plan.nodes}
        node_outputs: dict[str, dict[str, Any]] = dict(upstream_outputs or {})
        retry_counts: dict[str, int] = {n.id: 0 for n in plan.nodes}
        scheduling_rounds: list[SchedulingRoundEvent] = []

        # Phase 3: recovery manager tracks escalation state per node
        recovery_manager = RecoveryManager(plan)
        # Per-node patched configs (set by local_patch; used on re-dispatch)
        patched_configs: dict[str, NodeConfig] = {}
        # Nodes queued for re-dispatch after recovery action
        recovery_queue: dict[str, NodeConfig | None] = {}  # node_id → patched_config or None
        # Always-incrementing attempt counter (separate from retry budget)
        attempt_counts: dict[str, int] = {n.id: 0 for n in plan.nodes}

        start_wall = time.monotonic()
        global_deadline = start_wall + self.engine.global_timeout_s
        round_number = 0

        logger.info(
            "Dispatcher starting plan %r v%d (%d nodes, %d edges)",
            plan.plan_id, plan.version, len(plan.nodes), len(plan.edges),
        )

        while not execution_is_complete(states):
            round_number += 1

            if round_number > self.engine.max_scheduling_rounds:
                raise RuntimeError(
                    f"Exceeded max_scheduling_rounds={self.engine.max_scheduling_rounds}. "
                    "Possible infinite loop or deadlock."
                )
            if time.monotonic() > global_deadline:
                raise TimeoutError(
                    f"Global execution timeout ({self.engine.global_timeout_s}s) exceeded."
                )

            # 1. Compute ready set
            rsr: ReadySetResult = compute_ready_set(plan, states)
            states = rsr.updated_states

            # Emit blocked/skipped transitions
            for node_id in rsr.blocked_node_ids:
                self._emit_transition(plan, round_number, node_id, NodeState.PENDING, NodeState.BLOCKED)
            for node_id in rsr.skipped_node_ids:
                prev = NodeState.PENDING  # may also be READY; logged as best-effort
                self._emit_transition(plan, round_number, node_id, prev, NodeState.SKIPPED)

            # 3. Drain recovery_queue: nodes queued by the recovery layer are
            #    dispatched in the SAME round that produced them (or next round).
            #    CRITICAL: this must happen BEFORE the early-continue guard so that
            #    recovery nodes dispatch even when rsr.ready_node_ids is empty.
            recovery_ready = list(recovery_queue.keys())
            for node_id in recovery_ready:
                patched_cfg = recovery_queue.pop(node_id)
                if patched_cfg is not None:
                    patched_configs[node_id] = patched_cfg
                current = states[node_id]
                states[node_id] = transition(current, NodeState.RUNNING, node_id)
                self._emit_transition(plan, round_number, node_id, current, NodeState.RUNNING,
                                      payload={"recovery": "re-dispatched"})

            all_dispatch = list(rsr.ready_node_ids) + recovery_ready

            if not all_dispatch:
                # Nothing to dispatch this round
                if all(s.is_terminal or s == NodeState.BLOCKED for s in states.values()):
                    break
                await asyncio.sleep(0.01)
                continue

            # 2. Emit scheduling round event (only when dispatching something)
            if rsr.ready_node_ids:
                round_event = SchedulingRoundEvent(
                    round_number=round_number,
                    ready_set=list(rsr.ready_node_ids),
                    u_size=rsr.round_u_size,
                    state_snapshot=dict(states),
                )
                scheduling_rounds.append(round_event)
                if self._on_round:
                    self._on_round(round_event)

            logger.info(
                "Round %d: dispatch %d node(s) (%d fresh, %d recovery)",
                round_number, len(all_dispatch), len(rsr.ready_node_ids), len(recovery_ready),
            )

            # Transition fresh-ready nodes to running
            for node_id in rsr.ready_node_ids:
                self._emit_transition(plan, round_number, node_id, NodeState.READY, NodeState.RUNNING)
                states[node_id] = transition(NodeState.READY, NodeState.RUNNING, node_id)

            tasks = {
                node_id: asyncio.create_task(
                    self._execute_node(
                        plan=plan,
                        node_id=node_id,
                        states=states,
                        node_outputs=node_outputs,
                        factory=factory,
                        attempt_counts=attempt_counts,
                        round_number=round_number,
                        override_config=patched_configs.get(node_id),
                    )
                )
                for node_id in all_dispatch
            }

            # 4. Await all dispatched nodes (parallel execution)
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)

            for node_id, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.error("Node %r raised unexpected exception: %s", node_id, result)
                    states[node_id] = transition(NodeState.RUNNING, NodeState.FAILED, node_id)
                    self._emit_transition(
                        plan, round_number, node_id, NodeState.RUNNING, NodeState.FAILED,
                        error_message=str(result),
                    )
                else:
                    new_state, output, raw_output = result
                    old_state = NodeState.RUNNING

                    # --- Recovery integration (Phase 3) ---
                    if new_state in (NodeState.FAILED_RETRYABLE, NodeState.FAILED):
                        failure_type = diagnose(raw_output) if raw_output else FailureType.UNKNOWN
                        try:
                            action = recovery_manager.select_action(
                                node_id=node_id,
                                failure_type=failure_type,
                                current_node_state=new_state,
                                all_states=states,
                                raw_error=raw_output.error_message if raw_output else "",
                            )
                        except EscalationViolationError as e:
                            logger.error("Escalation violation: %s", e)
                            action = RecoveryAction(
                                action_type=RecoveryActionType.NO_ACTION,
                                node_id=node_id,
                                reason=str(e),
                            )

                        if action.should_retry:
                            # local_retry: keep as FAILED_RETRYABLE (non-terminal) so the
                            # while loop continues into the next round and processes the queue.
                            states[node_id] = NodeState.FAILED_RETRYABLE
                            recovery_queue[node_id] = None  # no config change
                            logger.info(
                                "Recovery: local_retry queued for %r (round %d)",
                                node_id, round_number,
                            )
                        elif action.should_patch:
                            # local_patch: use FAILED_RETRYABLE (non-terminal) so the
                            # while loop continues. The patched config is applied on re-dispatch.
                            states[node_id] = NodeState.FAILED_RETRYABLE
                            recovery_queue[node_id] = action.patched_config
                            logger.info(
                                "Recovery: local_patch queued for %r (round %d)",
                                node_id, round_number,
                            )
                        elif action.should_replan:
                            # request_replan: in v1, mark as permanently failed.
                            # Full LLM replan is Phase 10.
                            states[node_id] = NodeState.FAILED
                            logger.warning(
                                "Recovery: request_replan triggered by %r. "
                                "v1 marks node as permanently failed (full replan in Phase 10).",
                                node_id,
                            )
                        else:  # NO_ACTION — exhausted all recovery
                            states[node_id] = NodeState.FAILED

                        self._emit_transition(
                            plan, round_number, node_id, old_state, states[node_id],
                            error_message=action.reason,
                        )
                    else:
                        states[node_id] = new_state
                        if output:
                            node_outputs[node_id] = output
                        self._emit_transition(
                            plan, round_number, node_id, old_state, new_state,
                            payload=output or {},
                        )

            # The recovery_queue break guard is now unnecessary since we use
            # FAILED_RETRYABLE (non-terminal) to keep the loop alive.

        wall_clock = time.monotonic() - start_wall
        succeeded = all(s == NodeState.EXECUTED for s in states.values())

        if not succeeded:
            failures = [nid for nid, s in states.items() if s == NodeState.FAILED]
            failure_summary = f"Nodes failed: {failures}" if failures else "Execution incomplete."
        else:
            failure_summary = ""

        logger.info(
            "Plan %r v%d complete in %.2fs | succeeded=%s | rounds=%d",
            plan.plan_id, plan.version, wall_clock, succeeded, round_number,
        )

        return ExecutionResult(
            plan_id=plan.plan_id,
            plan_version=plan.version,
            terminal_states=dict(states),
            scheduling_rounds=scheduling_rounds,
            node_outputs=node_outputs,
            wall_clock_s=wall_clock,
            succeeded=succeeded,
            failure_summary=failure_summary,
        )

    async def _execute_node(
        self,
        plan: Plan,
        node_id: str,
        states: dict[str, NodeState],
        node_outputs: dict[str, dict[str, Any]],
        factory: NodeExecutorFactory,
        attempt_counts: dict[str, int],
        round_number: int,
        override_config: NodeConfig | None = None,
    ) -> tuple[NodeState, dict[str, Any] | None, NodeOutput | None]:
        """
        Execute a single node and return (new_state, output_payload, raw_node_output).

        Uses `attempt_counts` (always-incrementing) for ExecContext.attempt_number
        so that failure injection specs fire on the correct attempt regardless of
        whether the previous failure was a retry or a patch.
        """
        node = plan.node_by_id(node_id)
        base_config = override_config or node.config
        resolved_config = _resolve_config(base_config, self.engine)

        # Always increment attempt counter before execution
        attempt_counts[node_id] = attempt_counts.get(node_id, 0) + 1
        attempt_number = attempt_counts[node_id]

        # Build ExecContext (C_exec — no diagnostic history)
        upstream = {
            pred_id: node_outputs[pred_id]
            for pred_id in plan.predecessors(node_id)
            if pred_id in node_outputs
        }
        ctx = ExecContext(
            plan_id=plan.plan_id,
            plan_version=plan.version,
            node_id=node_id,
            node_label=node.label,
            config=resolved_config,
            upstream_outputs=upstream,
            task_description=plan.task_description,
            attempt_number=attempt_number,
        )

        # Apply failure injection if configured
        base_executor = factory(node)
        if self._failure_injector is not None:
            executor = self._failure_injector.wrap_executor(node, base_executor)
        else:
            executor = base_executor

        try:
            node_output: NodeOutput = await asyncio.wait_for(
                executor.execute(ctx),
                timeout=resolved_config.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("Node %r timed out after %.1fs", node_id, resolved_config.timeout_s)
            node_output = NodeOutput(
                outcome="retry",
                error_message=f"Node {node_id!r} timed out after {resolved_config.timeout_s}s",
                failure_type="transient",
            )

        # Map outcome → state transition
        if node_output.succeeded:
            return NodeState.EXECUTED, node_output.payload, node_output

        elif node_output.should_retry:
            return NodeState.FAILED_RETRYABLE, None, node_output

        elif node_output.should_wait:
            return NodeState.WAITING_HUMAN, None, node_output

        else:  # failure or escalate
            return NodeState.FAILED, None, node_output

    def _emit_transition(
        self,
        plan,
        round_number: int,
        node_id: str,
        from_state: NodeState,
        to_state: NodeState,
        payload: dict[str, Any] | None = None,
        error_message: str = "",
        latency_s: float = 0.0,
    ) -> None:
        if self._on_transition:
            self._on_transition(NodeTransitionEvent(plan_id=plan.plan_id, plan_version=plan.version, 
                round_number=round_number,
                node_id=node_id,
                from_state=from_state,
                to_state=to_state,
                payload=payload or {},
                error_message=error_message,
                latency_s=latency_s,
            ))
