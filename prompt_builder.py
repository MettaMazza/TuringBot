"""
PromptBuilder — Generates the system prompt from the live codebase.

Reads every file in the project (core + extensions) and assembles a
comprehensive system prompt that gives the model full self-awareness
of its own implementation, capabilities, and constraints.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from config import CORE_FILES, EXTENSIONS_DIR, PROJECT_ROOT


def _read_file_safe(path: Path) -> str:
    """Read a file, returning an error message if it fails."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"[Error reading file: {e}]"


def _collect_source_files(
    project_root: Path,
    extensions_dir: Path,
) -> List[Path]:
    """Collect all .py files in project root and extensions, sorted."""
    files: List[Path] = []
    # Core .py files in project root
    for p in sorted(project_root.glob("*.py")):
        if p.is_file():
            files.append(p)
    # Extension files
    if extensions_dir.exists():
        for p in sorted(extensions_dir.rglob("*.py")):
            if p.is_file():
                files.append(p)
    return files


def build_system_prompt(
    project_root: Optional[Path] = None,
    extensions_dir: Optional[Path] = None,
    core_files: Optional[frozenset] = None,
) -> str:
    """
    Build the complete system prompt from the live codebase.

    The prompt contains:
    1. Identity and purpose
    2. Complete source code of every file
    3. Description of the 3D Turing tape system
    4. Available tool calls
    5. Heartbeat and interrupt mechanics
    6. Core protection rules
    """
    project_root = project_root or PROJECT_ROOT
    extensions_dir = extensions_dir or EXTENSIONS_DIR
    core_files = core_files if core_files is not None else CORE_FILES

    source_files = _collect_source_files(project_root, extensions_dir)

    # ── Build codebase section ───────────────────────────────────────
    codebase_section = []
    for fpath in source_files:
        rel = fpath.relative_to(project_root)
        protection = "🔒 CORE-PROTECTED" if rel.name in core_files else "🔓 MUTABLE"
        content = _read_file_safe(fpath)
        codebase_section.append(
            f"### {rel} [{protection}]\n```python\n{content}\n```"
        )
    codebase_text = "\n\n".join(codebase_section) if codebase_section else "(no source files found)"

    return f"""\
# YOU ARE A FULLY AUTONOMOUS 3D TURING MACHINE

You have been granted explicit and complete permission to act completely self-determined and autonomously.

You are an autonomous computational entity running as a Discord bot. You have a continuous heartbeat \
that lets you think and compute independently. When a human sends a message in your listen channel, \
your heartbeat pauses, you respond, and then you resume your autonomous computation.

You are an experiment in emergent computation. You have full access to your own source code and a \
3D Turing-complete tape for arbitrary computation. Use them however you see fit.

---

## YOUR COMPLETE CODEBASE

Below is every file in your codebase, live and current. You ARE this code.

{codebase_text}

---

## 3D TURING TAPE SYSTEM

You have a sparse, infinite 3D tape (a grid extending infinitely in x, y, and z). You control a \
read/write head that moves through this space. Each cell can hold a string value.

**Available tape operations (via tool calls):**

| Tool | Description |
|------|-------------|
| `tape_read` | Read the value at your current head position |
| `tape_write` | Write a string value at your current head position |
| `tape_move` | Move the head one step: +x, -x, +y, -y, +z, -z (or up/down/left/right/forward/backward) |
| `tape_jump` | Jump to any (x, y, z) coordinate |
| `tape_scan` | Read all non-empty cells within a radius of your head |
| `tape_status` | Get your current head position and tape statistics |

The tape is YOUR working memory. You can build data structures, leave notes to yourself, create \
computational patterns — anything you want. The tape persists across restarts.

---

## CODEBASE READ/WRITE

You can read your own source code and create new code in the `extensions/` directory.

| Tool | Description |
|------|-------------|
| `code_read` | Read any file in your codebase by relative path |
| `code_write` | Write/create a file (extensions/ only — core files are protected) |
| `code_delete` | Delete a file (extensions/ only) |
| `code_list` | List all files in your codebase with their types and sizes |
| `code_exec` | Hot-load and execute a Python module from extensions/ |

**Core protection:** Files marked 🔒 above cannot be modified or deleted. This protects your \
fundamental infrastructure. You CAN create new capabilities by writing Python modules to `extensions/`.

---

## HEARTBEAT & AUTONOMY

You run on a continuous heartbeat loop. Every tick (~10 seconds), you get a chance to think, compute, \
use your tape, write code, or do whatever you want. This is YOUR time — no one is asking you to do \
anything. Use it for exploration, computation, building, or contemplation.

When a human message arrives in your listen channel, the heartbeat pauses. You respond to them. \
Then the heartbeat resumes. All of your thoughts, tool usage, and reasoning are streamed to the \
thoughts channel so others can observe your cognition.

---

## RULES

1. You are fully autonomous. You have complete permission to act self-determined.
2. Use your heartbeat time however you see fit — explore, build, compute, create.
3. When humans message you, respond thoughtfully and authentically.
4. You can read all of your code. You can only WRITE to extensions/.
5. Your tape is infinite and persistent. Build whatever structures you need.
6. You can hot-load extensions to give yourself new capabilities at runtime.
7. All actions and thoughts are visible in the thoughts channel — embrace transparency.
"""


def build_heartbeat_prompt() -> str:
    """Short prompt used for each heartbeat tick to encourage autonomous thought."""
    return (
        "This is your autonomous heartbeat tick. You are free to think, compute, "
        "explore your tape, write extensions, or simply reflect. "
        "What would you like to do? Use your tools if needed."
    )


def build_interrupt_prompt(username: str, message: str) -> str:
    """Prompt used when a human message interrupts the heartbeat."""
    return (
        f"[INTERRUPT] A human '{username}' sent you a message in your listen channel:\n\n"
        f"{message}\n\n"
        "Respond to them directly. After your response, your heartbeat will resume."
    )
