"""
Live terminal visualization for SGH DAG execution using Rich.
"""
from __future__ import annotations

import logging
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.console import Group

from sgh.core.plan import Plan, NodeState
from sgh.scheduler.dispatcher import NodeTransitionEvent, SchedulingRoundEvent

# Map states to rich colors
STATE_COLORS = {
    NodeState.PENDING: "dim",
    NodeState.READY: "blue",
    NodeState.RUNNING: "cyan",
    NodeState.WAITING_HUMAN: "magenta",
    NodeState.BLOCKED: "yellow",
    NodeState.EXECUTED: "green",
    NodeState.FAILED_RETRYABLE: "bold yellow",
    NodeState.FAILED: "bold red",
    NodeState.CANCELLED: "red",
    NodeState.SKIPPED: "dim white",
}

class LiveDashboard:
    """
    Renders a live terminal dashboard showing node statuses and the |U| metric.
    """

    def __init__(self, plan: Plan):
        self.plan = plan
        self.states: dict[str, NodeState] = {n.id: NodeState.PENDING for n in plan.nodes}
        self.round_number = 0
        self.u_size = 0
        self.live: Live | None = None
        
        # Suppress logging to avoid messing up the UI
        # Only errors will break through
        logging.getLogger().setLevel(logging.ERROR)

    def start(self) -> None:
        """Start the live rendering."""
        self.live = Live(self._generate_layout(), refresh_per_second=10)
        self.live.start()

    def stop(self) -> None:
        """Stop the live rendering."""
        if self.live:
            self.live.stop()
            # Restore logging
            logging.getLogger().setLevel(logging.INFO)

    def _generate_layout(self) -> Layout:
        """Generate the rich layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )
        
        # Header
        header = Panel(
            f"SGH Execution Dashboard - Plan: [bold]{self.plan.plan_id}[/bold] (v{self.plan.version})",
            style="white on dark_blue"
        )
        layout["header"].update(header)
        
        # Main Node Table
        table = Table(title=None, expand=True, show_header=True)
        table.add_column("Node ID", style="bold")
        table.add_column("Type")
        table.add_column("Status")
        
        for node in self.plan.nodes:
            state = self.states.get(node.id, NodeState.PENDING)
            color = STATE_COLORS.get(state, "white")
            table.add_row(
                node.id,
                node.config.node_type.value.upper(),
                f"[{color}]{state.value.upper()}[/{color}]"
            )
            
        layout["main"].update(Panel(table, title="DAG Nodes"))
        
        # Footer
        footer_text = Text.from_markup(
            f"Scheduling Round: [bold]{self.round_number}[/bold] | "
            f"Concurrency Metric |U|: [bold cyan]{self.u_size}[/bold cyan]"
        )
        layout["footer"].update(Panel(footer_text, style="white on dark_green"))
        
        return layout

    def update(self) -> None:
        """Refresh the UI."""
        if self.live:
            self.live.update(self._generate_layout())

    def on_transition(self, event: NodeTransitionEvent) -> None:
        """Callback for Dispatcher."""
        self.states[event.node_id] = event.to_state
        self.update()

    def on_round(self, event: SchedulingRoundEvent) -> None:
        """Callback for Dispatcher scheduling rounds."""
        self.round_number = event.round_number
        self.u_size = event.u_size
        self.states = event.state_snapshot
        self.update()
