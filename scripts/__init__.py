"""Consolidated data, operations, diagnostics, and evaluation commands."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping


def dispatch(description: str, commands: Mapping[str, Callable[[], None]]) -> None:
    """Dispatch a subcommand while leaving its arguments untouched."""
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        choices = " | ".join(commands)
        print(f"{description}\n\nUsage: {sys.argv[0]} <{choices}> [options]")
        return
    command = sys.argv[1]
    if command not in commands:
        choices = ", ".join(commands)
        raise SystemExit(f"unknown command {command!r}; choose one of: {choices}")
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    commands[command]()
