"""
Event log for SGH.

Handles append-only recording of NodeTransitionEvents to SQLite.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiosqlite

from sgh.scheduler.dispatcher import NodeTransitionEvent

logger = logging.getLogger(__name__)


class EventLog:
    """
    Append-only logger for state transitions.
    
    Can be used directly or provided as a callback to the Dispatcher.
    Uses an internal async queue to avoid blocking the scheduler loop
    with database writes.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._queue: asyncio.Queue[NodeTransitionEvent | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the background worker to flush events to SQLite."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        """Stop the background worker and flush remaining events."""
        if self._worker_task is not None:
            await self._queue.put(None)  # Sentinel to stop
            await self._worker_task
            self._worker_task = None

    def on_transition(self, event: NodeTransitionEvent) -> None:
        """
        Callback for Dispatcher._on_transition.
        Enqueues the event for background writing.
        """
        try:
            self._queue.put_nowait(event)
        except Exception as e:
            logger.error("Failed to enqueue transition event: %s", e)

    async def _worker_loop(self) -> None:
        """Background task that reads from the queue and writes to the DB."""
        async with aiosqlite.connect(self.db_path) as db:
            while True:
                event = await self._queue.get()
                if event is None:
                    self._queue.task_done()
                    break
                    
                payload_json = json.dumps(event.payload) if event.payload else None
                
                try:
                    await db.execute(
                        """
                        INSERT INTO events (
                            plan_id, plan_version, round_number, node_id, 
                            old_state, new_state, payload_json, error_message
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.plan_id,
                            event.plan_version,
                            event.round_number,
                            event.node_id,
                            event.from_state.value,
                            event.to_state.value,
                            payload_json,
                            event.error_message
                        )
                    )
                    await db.commit()
                except Exception as e:
                    logger.error("Failed to write event to DB: %s", e)
                finally:
                    self._queue.task_done()

    async def get_history(
        self, plan_id: str, version: int, node_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Query the event history for a plan.
        
        Args:
            plan_id: The ID of the plan.
            version: The version of the plan.
            node_id: If provided, filter events to just this node.
            
        Returns:
            A list of dictionary records representing the events, ordered by ID.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            if node_id:
                cursor = await db.execute(
                    """
                    SELECT * FROM events 
                    WHERE plan_id = ? AND plan_version = ? AND node_id = ?
                    ORDER BY event_id ASC
                    """,
                    (plan_id, version, node_id)
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT * FROM events 
                    WHERE plan_id = ? AND plan_version = ?
                    ORDER BY event_id ASC
                    """,
                    (plan_id, version)
                )
                
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
