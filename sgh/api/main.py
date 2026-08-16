"""
FastAPI application for SGH.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
import logging

from sgh.core.plan import Plan
from sgh.core.validator import validate_plan, PlanValidationError
from sgh.persistence.models import init_db
from sgh.persistence.plan_store import PlanStore
from sgh.persistence.event_log import EventLog
from sgh.scheduler.dispatcher import Dispatcher, EngineConfig
from sgh.nodes.factory import default_executor_factory

app = FastAPI(
    title="Structured Graph Harness API",
    description="Execution engine for DAG-based LLM agents",
    version="0.1.0"
)

DB_PATH = "sgh_api.db"

@app.on_event("startup")
async def startup_event():
    await init_db(DB_PATH)

class PlanResponse(BaseModel):
    plan_id: str
    version: int
    status: str

@app.post("/plan", response_model=PlanResponse)
async def upload_plan(plan: Plan):
    """
    Validate and save a new DAG plan to the immutable store.
    """
    try:
        validate_plan(plan)
    except PlanValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    store = PlanStore(DB_PATH)
    try:
        await store.save_plan(plan)
    except Exception as e:
        # Ignore already exists
        pass
        
    return PlanResponse(plan_id=plan.plan_id, version=plan.version, status="saved")


class ExecutionResponse(BaseModel):
    plan_id: str
    succeeded: bool
    failure_summary: str | None


@app.post("/plan/{plan_id}/run", response_model=ExecutionResponse)
async def run_plan(plan_id: str, version: int = 1):
    """
    Execute a previously saved plan. (Blocking for MVP simplicity)
    """
    store = PlanStore(DB_PATH)
    plan = await store.get_plan(plan_id, version)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
        
    event_log = EventLog(DB_PATH)
    event_log.start()
    
    dispatcher = Dispatcher(
        engine_config=EngineConfig(),
        on_node_transition=event_log.on_transition
    )
    
    try:
        # Wait for execution to finish
        result = await dispatcher.run(plan, executor_factory=default_executor_factory)
        
        # Stop log drainer cleanly
        await event_log.stop()
        
        return ExecutionResponse(
            plan_id=plan_id,
            succeeded=result.succeeded,
            failure_summary=result.failure_summary
        )
    except Exception as e:
        await event_log.stop()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/plan/{plan_id}/events")
async def get_plan_events(plan_id: str, version: int = 1):
    """
    Fetch the immutable audit trail for a plan execution.
    """
    event_log = EventLog(DB_PATH)
    events = await event_log.get_history(plan_id, version)
    return {"events": events}
