"""
Typer CLI for SGH.
"""
import asyncio
import typer
import logging
from pathlib import Path
from rich.console import Console
from rich.table import Table

from sgh.core.plan import Plan
from sgh.core.validator import validate_plan as _validate_plan, PlanValidationError
from sgh.scheduler.dispatcher import Dispatcher, EngineConfig
from sgh.persistence.models import init_db
from sgh.persistence.plan_store import PlanStore
from sgh.persistence.event_log import EventLog
from sgh.viz.live_view import LiveDashboard
from sgh.nodes.factory import default_executor_factory

app = typer.Typer(help="Structured Graph Harness (SGH) execution engine CLI")
console = Console()

@app.command()
def validate(plan_path: Path):
    """Validate a DAG plan's syntax and topological integrity."""
    try:
        plan_json = plan_path.read_text()
        plan = Plan.model_validate_json(plan_json)
        _validate_plan(plan)
        console.print(f"[green]Plan {plan.plan_id} (v{plan.version}) is valid![/green]")
    except Exception as e:
        console.print(f"[bold red]Validation failed:[/bold red] {e}")
        raise typer.Exit(1)

@app.command()
def run(plan_path: Path, db_path: str = "sgh.db", live: bool = True):
    """Execute a DAG plan."""
    try:
        plan_json = plan_path.read_text()
        plan = Plan.model_validate_json(plan_json)
        _validate_plan(plan)
    except Exception as e:
        console.print(f"[bold red]Invalid plan:[/bold red] {e}")
        raise typer.Exit(1)

    async def _run():
        await init_db(db_path)
        plan_store = PlanStore(db_path)
        event_log = EventLog(db_path)
        event_log.start()
        
        try:
            await plan_store.save_plan(plan)
        except Exception:
            pass  # Already exists

        dashboard = LiveDashboard(plan) if live else None
        if dashboard:
            dashboard.start()
            
        def handle_transition(event):
            event_log.on_transition(event)
            if dashboard:
                dashboard.on_transition(event)
                
        def handle_round(event):
            if dashboard:
                dashboard.on_round(event)

        dispatcher = Dispatcher(
            engine_config=EngineConfig(),
            on_node_transition=handle_transition,
            on_scheduling_round=handle_round
        )

        try:
            # Note: The factory needs to be fully wired for LLM/TOOL nodes.
            # In a real CLI run, users would register tools before this.
            # For this MVP CLI, we'll default to the default_executor_factory.
            result = await dispatcher.run(plan, executor_factory=default_executor_factory)
            
            if dashboard:
                dashboard.stop()
            
            if result.succeeded:
                console.print("\n[bold green]Execution succeeded![/bold green]")
            else:
                console.print(f"\n[bold red]Execution failed:[/bold red] {result.failure_summary}")
        finally:
            if dashboard:
                dashboard.stop()
            await event_log.stop()

    asyncio.run(_run())


@app.command()
def history(plan_id: str, version: int = 1, db_path: str = "sgh.db"):
    """Query the audit trail for a specific plan run."""
    async def _history():
        event_log = EventLog(db_path)
        events = await event_log.get_history(plan_id, version)
        
        if not events:
            console.print(f"[yellow]No history found for plan {plan_id} v{version}[/yellow]")
            return
            
        table = Table(title=f"Audit History: {plan_id} (v{version})")
        table.add_column("Round")
        table.add_column("Node ID", style="bold")
        table.add_column("From State")
        table.add_column("To State")
        table.add_column("Error Message", style="red")
        
        for ev in events:
            table.add_row(
                str(ev["round_number"]),
                ev["node_id"],
                ev["old_state"],
                ev["new_state"],
                ev["error_message"] or ""
            )
            
        console.print(table)
        
    asyncio.run(_history())

if __name__ == "__main__":
    app()
