#!/usr/bin/env python3
"""Chat panel widget for Dive Bar TUI."""

from rich.text import Text
from textual.widgets import RichLog

# Colors assigned to agents by index
AGENT_COLORS = [
    "bright_cyan",
    "bright_magenta",
    "bright_yellow",
    "bright_green",
    "bright_red",
    "bright_blue",
    "orange1",
    "orchid",
]


class ChatPanel(RichLog):
    """Scrolling chat display for bar conversation."""

    DEFAULT_CSS = """
    ChatPanel {
        border: solid $accent;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(
            highlight=True,
            markup=True,
            wrap=True,
            **kwargs,
        )
        self._color_map: dict[str, str] = {}
        self._color_index = 0

    def _get_color(self, agent_name: str) -> str:
        """Get or assign a color for an agent."""
        if agent_name not in self._color_map:
            color = AGENT_COLORS[
                self._color_index % len(AGENT_COLORS)
            ]
            self._color_map[agent_name] = color
            self._color_index += 1
        return self._color_map[agent_name]

    def add_message(
        self,
        agent_name: str,
        content: str,
        timestamp: str = "",
    ):
        """Add a colored message to the chat log."""
        color = self._get_color(agent_name)
        line = Text()
        if timestamp:
            line.append(
                f"[{timestamp}] ",
                style="dim",
            )
        line.append(
            f"{agent_name}: ",
            style=f"bold {color}",
        )
        line.append(content)
        self.write(line)

    def add_system_message(self, content: str):
        """Add a system/narrator message."""
        line = Text()
        line.append(f"* {content} *", style="dim italic")
        self.write(line)
