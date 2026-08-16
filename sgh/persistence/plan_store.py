"""
Plan store for SGH.

Handles saving and loading of immutable Plan objects.
"""
from __future__ import annotations

import json
from typing import Any

import aiosqlite

from sgh.core.plan import Plan


class PlanStore:
    """
    Manages Plan persistence. Plans are immutable; saving an existing
    (plan_id, version) raises an error or is ignored.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def save_plan(self, plan: Plan) -> None:
        """
        Save a plan to the database.
        
        Args:
            plan: The Plan object to persist.
            
        Raises:
            aiosqlite.IntegrityError if the plan_id/version already exists.
        """
        json_data = json.dumps(plan.model_dump(mode="json"))
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO plans (plan_id, version, task_description, json_data)
                VALUES (?, ?, ?, ?)
                """,
                (plan.plan_id, plan.version, plan.task_description, json_data)
            )
            await db.commit()

    async def load_plan(self, plan_id: str, version: int | None = None) -> Plan | None:
        """
        Load a plan from the database.
        
        Args:
            plan_id: The ID of the plan.
            version: The specific version to load. If None, loads the latest version.
            
        Returns:
            The loaded Plan, or None if not found.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if version is not None:
                cursor = await db.execute(
                    "SELECT json_data FROM plans WHERE plan_id = ? AND version = ?",
                    (plan_id, version)
                )
            else:
                cursor = await db.execute(
                    "SELECT json_data FROM plans WHERE plan_id = ? ORDER BY version DESC LIMIT 1",
                    (plan_id,)
                )
                
            row = await cursor.fetchone()
            if not row:
                return None
                
            return Plan.model_validate_json(row["json_data"])
