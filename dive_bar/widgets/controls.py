#!/usr/bin/env python3
"""Bottom control bar for Dive Bar TUI."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Input, Static


class PauseToggled(Message):
    """Emitted when pause is toggled."""

    def __init__(self, paused: bool):
        super().__init__()
        self.paused = paused


class SpeedChanged(Message):
    """Emitted when speed changes."""

    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta


class StrangerMessage(Message):
    """Emitted when user injects a message."""

    def __init__(self, text: str):
        super().__init__()
        self.text = text


class BarControls(Static):
    """Bottom bar with pause, speed, and inject."""

    DEFAULT_CSS = """
    BarControls {
        dock: bottom;
        height: 4;
        border-top: solid $accent;
        padding: 0 1;
    }
    BarControls Horizontal {
        height: 1;
        margin-bottom: 0;
    }
    BarControls Button {
        min-width: 10;
        margin-right: 1;
    }
    BarControls Input {
        width: 1fr;
    }
    BarControls #status-line {
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._paused = False

    def compose(self) -> ComposeResult:
        """Build the controls layout."""
        with Horizontal():
            yield Button(
                "Pause", id="btn-pause",
                variant="default",
            )
            yield Button(
                "[-]", id="btn-slow",
                variant="default",
            )
            yield Button(
                "[+]", id="btn-fast",
                variant="default",
            )
            yield Input(
                placeholder=(
                    "Type as a stranger..."
                ),
                id="stranger-input",
            )
        yield Static(
            " Ready.", id="status-line"
        )

    def on_button_pressed(
        self, event: Button.Pressed
    ) -> None:
        """Handle button clicks."""
        if event.button.id == "btn-pause":
            self._toggle_pause(event.button)
        elif event.button.id == "btn-slow":
            self.post_message(SpeedChanged(-0.25))
        elif event.button.id == "btn-fast":
            self.post_message(SpeedChanged(0.25))

    def _toggle_pause(self, button: Button):
        """Toggle pause state."""
        self._paused = not self._paused
        button.label = (
            "Resume" if self._paused else "Pause"
        )
        self.post_message(
            PauseToggled(self._paused)
        )

    def on_input_submitted(
        self, event: Input.Submitted
    ) -> None:
        """Handle stranger message injection."""
        text = event.value.strip()
        if text:
            self.post_message(StrangerMessage(text))
            event.input.value = ""

    def set_status(self, text: str):
        """Update the status line."""
        status = self.query_one(
            "#status-line", Static
        )
        status.update(f" {text}")
