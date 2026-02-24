"""
TuringTape3D — A sparse, infinite 3D Turing tape.

The tape is a dict mapping (x, y, z) coordinates to arbitrary string values.
A read/write head tracks the current position with 6-directional movement.
State can be serialized to / deserialized from JSON for persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Direction vectors for 6-axis movement
DIRECTIONS: Dict[str, Tuple[int, int, int]] = {
    "+x": (1, 0, 0),
    "-x": (-1, 0, 0),
    "+y": (0, 1, 0),
    "-y": (0, -1, 0),
    "+z": (0, 0, 1),
    "-z": (0, 0, -1),
}

VALID_DIRECTIONS = frozenset(DIRECTIONS.keys())


class TuringTape3D:
    """Sparse infinite 3D Turing tape with a movable read/write head."""

    def __init__(self) -> None:
        self._cells: Dict[Tuple[int, int, int], str] = {}
        self._head: Tuple[int, int, int] = (0, 0, 0)

    # ── Properties ───────────────────────────────────────────────────

    @property
    def head(self) -> Tuple[int, int, int]:
        """Current head position."""
        return self._head

    @property
    def cells(self) -> Dict[Tuple[int, int, int], str]:
        """Read-only view of all non-empty cells."""
        return dict(self._cells)

    # ── Core Operations ──────────────────────────────────────────────

    def read(self) -> str:
        """Read the value at the current head position. Empty cells return ''."""
        return self._cells.get(self._head, "")

    def write(self, value: str) -> str:
        """Write a value to the current head position. Returns confirmation."""
        if not isinstance(value, str):
            raise TypeError(f"Tape values must be strings, got {type(value).__name__}")
        if value == "":
            self._cells.pop(self._head, None)
        else:
            self._cells[self._head] = value
        return f"Wrote '{value}' at {self._head}"

    def move(self, direction: str) -> str:
        """Move the head one step in the given direction. Returns new position."""
        direction = direction.strip().lower()
        # Normalize common aliases
        aliases = {
            "up": "+y", "down": "-y",
            "right": "+x", "left": "-x",
            "forward": "+z", "backward": "-z",
            "back": "-z",
        }
        direction = aliases.get(direction, direction)

        if direction not in DIRECTIONS:
            raise ValueError(
                f"Invalid direction '{direction}'. "
                f"Valid: {sorted(VALID_DIRECTIONS)} or up/down/left/right/forward/backward"
            )
        dx, dy, dz = DIRECTIONS[direction]
        x, y, z = self._head
        self._head = (x + dx, y + dy, z + dz)
        return f"Moved {direction} → head now at {self._head}"

    def jump(self, x: int, y: int, z: int) -> str:
        """Jump the head to an arbitrary (x, y, z) coordinate."""
        self._head = (int(x), int(y), int(z))
        return f"Jumped to {self._head}"

    def scan_neighborhood(self, radius: int = 1) -> Dict[str, str]:
        """
        Read all non-empty cells within `radius` steps of the head.
        Returns a dict mapping "x,y,z" strings to values.
        """
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        hx, hy, hz = self._head
        result: Dict[str, str] = {}
        for (cx, cy, cz), val in self._cells.items():
            if (abs(cx - hx) <= radius and
                    abs(cy - hy) <= radius and
                    abs(cz - hz) <= radius):
                result[f"{cx},{cy},{cz}"] = val
        return result

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the tape state to a JSON-compatible dict."""
        serialized_cells: Dict[str, str] = {}
        for (x, y, z), val in self._cells.items():
            serialized_cells[f"{x},{y},{z}"] = val
        return {
            "head": list(self._head),
            "cells": serialized_cells,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TuringTape3D":
        """Deserialize a tape from a dict (as produced by to_dict)."""
        tape = cls()
        head = data.get("head", [0, 0, 0])
        tape._head = (int(head[0]), int(head[1]), int(head[2]))
        for key, val in data.get("cells", {}).items():
            parts = key.split(",")
            coord = (int(parts[0]), int(parts[1]), int(parts[2]))
            tape._cells[coord] = val
        return tape

    def save(self, path: Path) -> None:
        """Persist tape state to a JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "TuringTape3D":
        """Load tape state from a JSON file. Returns new tape if file missing."""
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    # ── Display ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"TuringTape3D(head={self._head}, cells={len(self._cells)})"

    def status(self) -> str:
        """Human-readable status string."""
        current = self.read()
        return (
            f"Head: {self._head} | "
            f"Current cell: '{current}' | "
            f"Total written cells: {len(self._cells)}"
        )
