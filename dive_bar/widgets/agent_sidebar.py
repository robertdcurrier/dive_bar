#!/usr/bin/env python3
"""Agent sidebar widget for Dive Bar TUI."""

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static


class AgentStatus(Static):
    """Status display for a single agent."""

    status = reactive("idle")

    STATUS_ICONS = {
        "idle": "[dim]o[/]",
        "thinking": "[yellow]*[/]",
        "talking": "[green]>[/]",
    }

    def __init__(
        self,
        agent_name: str,
        drink: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent_name = agent_name
        self.drink = drink

    def render(self) -> str:
        """Render the agent status line."""
        icon = self.STATUS_ICONS.get(
            self.status, "?"
        )
        return (
            f" {icon} {self.agent_name}"
            f"  [dim]{self.drink}[/]"
        )

    def set_status(self, status: str):
        """Update the agent's status."""
        self.status = status


class AgentSidebar(Widget):
    """Sidebar showing all agents and their status."""

    DEFAULT_CSS = """
    AgentSidebar {
        width: 32;
        border: solid $accent;
        padding: 1;
    }
    AgentSidebar Label {
        text-style: bold;
        width: 100%;
        content-align: center middle;
    }
    AgentSidebar #stats {
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        agent_names: list[tuple[str, str]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._agent_names = agent_names
        self._status_widgets: dict[
            str, AgentStatus
        ] = {}
        self.turn_count = 0

    def compose(self) -> ComposeResult:
        """Build the sidebar layout."""
        yield Label("AGENTS")
        for name, drink in self._agent_names:
            widget = AgentStatus(name, drink)
            self._status_widgets[name] = widget
            yield widget
        yield Static("", id="stats")

    def set_agent_status(
        self, name: str, status: str
    ):
        """Update an agent's status indicator."""
        if name in self._status_widgets:
            self._status_widgets[name].set_status(
                status
            )

    def update_stats(
        self, turn: int, speed: float
    ):
        """Update the stats display."""
        self.turn_count = turn
        stats = self.query_one("#stats", Static)
        stats.update(
            f"\n Turn: {turn}\n"
            f" Speed: {speed:.1f}x"
        )
