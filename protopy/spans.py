from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Position:
    """A concrete source position.

    Offsets are 0-based; line/column are 1-based for user-facing messages.
    """

    offset: int = 0
    line: int = 0
    column: int = 0


@dataclass(frozen=True, slots=True)
class Span:
    """Half-open span [start, end) in a single file."""

    file: str = ""
    start: Position = Position()
    end: Position = Position()

    def format(self) -> str:
        if self.file == "":
            return ""

        return f"{self.file}:{self.start.line}:{self.start.column}"

