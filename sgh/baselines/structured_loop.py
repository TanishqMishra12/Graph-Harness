"""
Baseline Agent Loop implementation.

Restricts SGH to act like a standard ReAct loop by enforcing |U(s)| <= 1.
"""
import asyncio
import time
from typing import Any, Callable

from sgh.core.plan import Plan, NodeState
from sgh.scheduler.dispatcher import Dispatcher, EngineConfig, NodeTransitionEvent, SchedulingRoundEvent, ExecutionResult, NodeExecutorFactory
from sgh.scheduler.ready_set import compute_ready_set, ReadySetResult

class AgentLoopDispatcher(Dispatcher):
    """
    A strictly sequential dispatcher modeling a classic Agent Loop.
    
    Forces |U(s)| <= 1 at all times by artificially starving the ready set.
    Used for empirical benchmarking against the concurrent SGH Dispatcher.
    """
    
    async def run(
        self,
        plan: Plan,
        executor_factory: NodeExecutorFactory | None = None,
        upstream_outputs: dict[str, dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        
        # Intercept the engine's compute_ready_set with a decorator-like patch
        from sgh.scheduler import dispatcher as disp_module
        
        original_compute = disp_module.compute_ready_set
        
        def mock_compute(*args, **kwargs) -> ReadySetResult:
            rsr = original_compute(*args, **kwargs)
            # Artificial restriction: only allow 1 node to be ready at a time
            if len(rsr.ready_node_ids) > 1:
                # Keep the lexicographically first one
                winner = sorted(rsr.ready_node_ids)[0]
                losers = [n for n in rsr.ready_node_ids if n != winner]
                
                # Revert losers back to PENDING in the updated_states map
                for loser in losers:
                    rsr.updated_states[loser] = NodeState.PENDING
                
                rsr.ready_node_ids = [winner]
                rsr.round_u_size = 1
                
            return rsr
            
        disp_module.compute_ready_set = mock_compute
        
        try:
            return await super().run(plan, executor_factory, upstream_outputs)
        finally:
            disp_module.compute_ready_set = original_compute
