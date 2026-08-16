"""
Tests for Phase 5: Persistence & audit trail.
"""
import asyncio
import sqlite3
import pytest
import aiosqlite
import json
from pathlib import Path

from sgh.core.plan import Plan, Node, NodeConfig, NodeState, NodeType
from sgh.scheduler.dispatcher import NodeTransitionEvent
from sgh.persistence.models import init_db
from sgh.persistence.plan_store import PlanStore
from sgh.persistence.event_log import EventLog

@pytest.fixture
async def temp_db(tmp_path: Path):
    db_path = tmp_path / "test.db"
    await init_db(str(db_path))
    yield str(db_path)

@pytest.mark.asyncio
async def test_init_db_creates_tables(temp_db):
    async with aiosqlite.connect(temp_db) as db:
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in await cursor.fetchall()]
        assert "plans" in tables
        assert "events" in tables

@pytest.mark.asyncio
async def test_plan_store_save_and_load(temp_db):
    store = PlanStore(temp_db)
    
    config = NodeConfig(node_type=NodeType.MOCK)
    node = Node(id="a", label="A", config=config)
    plan = Plan(nodes=[node], edges=[])
    
    # Save plan
    await store.save_plan(plan)
    
    # Load plan
    loaded = await store.load_plan(plan.plan_id, plan.version)
    assert loaded is not None
    assert loaded.plan_id == plan.plan_id
    assert loaded.version == plan.version
    assert loaded.nodes[0].id == "a"

@pytest.mark.asyncio
async def test_plan_store_load_latest_version(temp_db):
    store = PlanStore(temp_db)
    
    plan1 = Plan(nodes=[], edges=[], version=1)
    plan2 = Plan(nodes=[], edges=[], version=2, plan_id=plan1.plan_id)
    
    await store.save_plan(plan1)
    await store.save_plan(plan2)
    
    # Load without specifying version should get the highest version
    loaded = await store.load_plan(plan1.plan_id)
    assert loaded is not None
    assert loaded.version == 2

@pytest.mark.asyncio
async def test_plan_store_immutability(temp_db):
    store = PlanStore(temp_db)
    plan = Plan(nodes=[], edges=[])
    
    await store.save_plan(plan)
    
    # Saving same plan_id and version again should raise IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        await store.save_plan(plan)

@pytest.mark.asyncio
async def test_event_log_records_transitions(temp_db):
    log = EventLog(temp_db)
    log.start()
    
    try:
        event = NodeTransitionEvent(
            plan_id="p1",
            plan_version=1,
            round_number=2,
            node_id="a",
            from_state=NodeState.READY,
            to_state=NodeState.RUNNING,
            payload={"test": 123},
            error_message=None
        )
        
        log.on_transition(event)
        
        # Give worker a moment to process
        await asyncio.sleep(0.05)
        
        history = await log.get_history("p1", 1)
        assert len(history) == 1
        record = history[0]
        
        assert record["plan_id"] == "p1"
        assert record["node_id"] == "a"
        assert record["old_state"] == "ready"
        assert record["new_state"] == "running"
        assert json.loads(record["payload_json"]) == {"test": 123}
    finally:
        await log.stop()

@pytest.mark.asyncio
async def test_event_log_filters_by_node(temp_db):
    log = EventLog(temp_db)
    log.start()
    
    try:
        e1 = NodeTransitionEvent(plan_id="p1", plan_version=1, round_number=1, node_id="a", from_state=NodeState.PENDING, to_state=NodeState.READY)
        e2 = NodeTransitionEvent(plan_id="p1", plan_version=1, round_number=1, node_id="b", from_state=NodeState.PENDING, to_state=NodeState.READY)
        e3 = NodeTransitionEvent(plan_id="p1", plan_version=1, round_number=2, node_id="a", from_state=NodeState.READY, to_state=NodeState.RUNNING)
        
        for e in [e1, e2, e3]:
            log.on_transition(e)
            
        await asyncio.sleep(0.05)
        
        # Filter by node "a"
        history_a = await log.get_history("p1", 1, "a")
        assert len(history_a) == 2
        assert history_a[0]["new_state"] == "ready"
        assert history_a[1]["new_state"] == "running"
        
        # Filter by node "b"
        history_b = await log.get_history("p1", 1, "b")
        assert len(history_b) == 1
        assert history_b[0]["new_state"] == "ready"
    finally:
        await log.stop()
