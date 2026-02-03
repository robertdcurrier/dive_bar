#!/usr/bin/env python3
"""Dive Bar -- AI agents chatting at a bar."""

import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)

from dive_bar.app import DiveBarApp
from dive_bar.config import load_config


def main():
    """Launch the Dive Bar."""
    config = load_config()
    app = DiveBarApp(config)
    app.run()


if __name__ == "__main__":
    main()
