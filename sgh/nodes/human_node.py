"""
Human Node Executor for SGH.

Pauses execution to allow a human-in-the-loop to approve or provide data.
In a real system, this would block on an external event, wait for an HTTP callback,
or queue an asynchronous notification.
"""
from __future__ import annotations

import logging
from typing import Any

from sgh.nodes.base import BaseNode, ExecContext, NodeOutput

logger = logging.getLogger(__name__)


class HumanNode(BaseNode):
    """
    Simulates a human-in-the-loop pause.
    
    In Phase 4, we don't have a full async callback system yet, so this
    simply returns a WAITING_HUMAN transition outcome.
    """

    async def execute(self, ctx: ExecContext) -> NodeOutput:
        # We signal that we are waiting for human input.
        # The dispatcher handles WAITING_HUMAN in the state machine.
        # Returning 'retry' would cause the recovery manager to kick in,
        # so we need a dedicated outcome for this, or we just map it in dispatcher.
        # Actually, the base outcome space O = {success, failure, retry, escalate}
        # doesn't natively have "waiting", so in the dispatcher we mapped
        # WAITING_HUMAN as a possible outcome. 
        # But wait! NodeOutput doesn't have "wait".
        # If we return outcome="wait", we need to update dispatcher to handle it.
        # For now, we will return outcome="wait" and let the dispatcher handle it if needed,
        # or we could just consider human-in-the-loop as an external async event that
        # resolves a future.
        
        # In a real async runner, this might look like:
        # payload = await human_callback_queue.get(ctx.node_id)
        
        return NodeOutput(
            outcome="wait",  # Note: Dispatcher needs to recognize this
            error_message="Waiting for human input."
        )
