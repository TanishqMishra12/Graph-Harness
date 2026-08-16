"""
SQLite models and schema for SGH persistence.
"""
from __future__ import annotations

import logging
import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA = """
-- Plans table (Immutable)
CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    task_description TEXT,
    json_data TEXT NOT NULL,  -- The serialized Plan object
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (plan_id, version)
);

-- Events table (Append-only)
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL,
    plan_version INTEGER NOT NULL,
    round_number INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    old_state TEXT NOT NULL,
    new_state TEXT NOT NULL,
    payload_json TEXT,        -- Serialized output payload or diagnostic payload
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (plan_id, plan_version) REFERENCES plans (plan_id, version)
);

CREATE INDEX IF NOT EXISTS idx_events_plan ON events (plan_id, plan_version);
CREATE INDEX IF NOT EXISTS idx_events_node ON events (plan_id, plan_version, node_id);
"""

async def init_db(db_path: str) -> None:
    """Initialize the SQLite database with the SGH schema."""
    logger.info("Initializing SGH database at %s", db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA)
        await db.commit()
