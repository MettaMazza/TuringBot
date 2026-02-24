"""
Sandbox — Restricts what hot-loaded extension code can do.

Prevents extensions from:
- Importing dangerous modules (os, subprocess, shutil, socket, etc.)
- Accessing the filesystem outside extensions/
- Making network calls
- Running system commands
- Modifying sys.modules or builtins

Extensions run inside a restricted global scope where dangerous
builtins and imports are blocked.
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Set

logger = logging.getLogger("turingbot.sandbox")

# ── Blocked Modules ──────────────────────────────────────────────────
# These modules (and sub-modules) are blocked from import inside extensions.
BLOCKED_MODULES: frozenset = frozenset({
    # System / process control
    "os",
    "sys",
    "subprocess",
    "shutil",
    "signal",
    "ctypes",
    "multiprocessing",
    "threading",

    # File system (beyond what we provide)
    "pathlib",
    "glob",
    "tempfile",
    "io",

    # Network
    "socket",
    "http",
    "urllib",
    "requests",
    "aiohttp",
    "httpx",
    "ftplib",
    "smtplib",
    "xmlrpc",

    # Code execution
    "code",
    "codeop",
    "compile",
    "compileall",
    "importlib",
    "runpy",
    "ast",

    # Dangerous introspection
    "inspect",
    "gc",
    "weakref",
    "traceback",

    # Pickle (arbitrary code execution)
    "pickle",
    "shelve",
    "marshal",
})

# ── Blocked Builtins ─────────────────────────────────────────────────
BLOCKED_BUILTINS: frozenset = frozenset({
    "exec",
    "eval",
    "compile",
    "__import__",
    "open",
    "breakpoint",
    "exit",
    "quit",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "memoryview",
})

# ── Safe Modules (allowlist) ─────────────────────────────────────────
# Only these modules can be imported inside sandboxed extensions.
ALLOWED_MODULES: frozenset = frozenset({
    "math",
    "random",
    "string",
    "re",
    "json",
    "datetime",
    "time",
    "collections",
    "itertools",
    "functools",
    "operator",
    "decimal",
    "fractions",
    "statistics",
    "hashlib",
    "hmac",
    "base64",
    "copy",
    "enum",
    "dataclasses",
    "typing",
    "abc",
    "textwrap",
    "unicodedata",
    "bisect",
    "heapq",
    "array",
    "struct",
    "pprint",
})


def _is_module_allowed(module_name: str) -> bool:
    """Check if a module is allowed for import inside the sandbox."""
    # Check exact match against allowlist
    if module_name in ALLOWED_MODULES:
        return True
    # Check if it's a sub-module of an allowed module (e.g., collections.abc)
    for allowed in ALLOWED_MODULES:
        if module_name.startswith(allowed + "."):
            return True
    # Check against blocklist
    root_module = module_name.split(".")[0]
    if root_module in BLOCKED_MODULES:
        return False
    # Default: block anything not explicitly allowed
    return False


def _make_safe_import(project_root: Path):
    """Create a restricted __import__ function for the sandbox."""
    real_import = builtins.__import__

    def safe_import(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if not _is_module_allowed(name):
            raise ImportError(
                f"🔒 SANDBOX: Import of '{name}' is blocked. "
                f"Allowed modules: {sorted(ALLOWED_MODULES)}"
            )
        return real_import(name, *args, **kwargs)

    return safe_import


def _make_safe_builtins() -> Dict[str, Any]:
    """Create a restricted builtins dict for the sandbox."""
    safe = {}
    for name in dir(builtins):
        if name in BLOCKED_BUILTINS:
            continue
        safe[name] = getattr(builtins, name)
    # Override print to be a no-op that returns the string (no I/O)
    safe["print"] = lambda *args, **kwargs: str(args)
    return safe


def create_sandbox_globals(
    project_root: Path,
    module_name: str,
) -> Dict[str, Any]:
    """
    Create a restricted globals dict for executing extension code.

    The sandbox:
    - Blocks dangerous imports (os, subprocess, socket, etc.)
    - Blocks dangerous builtins (exec, eval, open, etc.)
    - Only allows a curated set of safe modules
    """
    safe_builtins = _make_safe_builtins()
    safe_builtins["__import__"] = _make_safe_import(project_root)

    sandbox_globals: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "__name__": module_name,
        "__doc__": None,
        "__package__": module_name.rsplit(".", 1)[0] if "." in module_name else None,
    }

    return sandbox_globals


def sandboxed_exec(
    code: str,
    project_root: Path,
    module_name: str = "__sandbox__",
) -> Dict[str, Any]:
    """
    Execute code in a sandboxed environment.

    Returns the resulting namespace (variables defined by the code).
    Raises ImportError if the code tries to import blocked modules.
    Raises any exceptions the code itself raises.
    """
    sandbox_globals = create_sandbox_globals(project_root, module_name)
    exec(code, sandbox_globals)  # noqa: S102 — intentional sandboxed exec
    return sandbox_globals
