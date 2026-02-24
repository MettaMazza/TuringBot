"""
CodebaseRW — Self-modifying codebase read/write with core protection.

The model can read any file in the project but can only write to
the `extensions/` directory. Core files are protected from modification.
Hot-loading support lets the model execute newly created Python modules.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

from config import CORE_FILES, EXTENSIONS_DIR, PROJECT_ROOT
from sandbox import sandboxed_exec, create_sandbox_globals


class CoreProtectionError(Exception):
    """Raised when an attempt is made to modify a core-protected file."""
    pass


class CodebaseRW:
    """Read/write interface for the project codebase with core protection."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        extensions_dir: Optional[Path] = None,
        core_files: Optional[frozenset] = None,
    ) -> None:
        self.project_root = project_root or PROJECT_ROOT
        self.extensions_dir = extensions_dir or EXTENSIONS_DIR
        self.core_files = core_files if core_files is not None else CORE_FILES
        # Ensure extensions directory exists
        self.extensions_dir.mkdir(parents=True, exist_ok=True)

    def _is_core_protected(self, rel_path: str) -> bool:
        """Check if a relative path refers to a core-protected file."""
        # Normalize the path
        normalized = Path(rel_path).name
        # Also check if the path tries to escape extensions/
        parts = Path(rel_path).parts
        if normalized in self.core_files:
            return True
        # Any path that doesn't start with extensions/ is protected
        if not parts or parts[0] != "extensions":
            return True
        return False

    def read_file(self, rel_path: str) -> str:
        """
        Read any file in the project by relative path.
        Returns the file contents as a string.
        Raises FileNotFoundError if the file doesn't exist.
        """
        target = (self.project_root / rel_path).resolve()
        # Security: ensure the path is within project root
        if not str(target).startswith(str(self.project_root)):
            raise PermissionError(
                f"Path '{rel_path}' resolves outside the project root"
            )
        if not target.exists():
            raise FileNotFoundError(f"File not found: {rel_path}")
        if not target.is_file():
            raise IsADirectoryError(f"Path is a directory, not a file: {rel_path}")
        return target.read_text(encoding="utf-8")

    def write_file(self, rel_path: str, content: str) -> str:
        """
        Write content to a file. Only paths under `extensions/` are allowed.
        Creates parent directories as needed.
        Returns a confirmation message.
        Raises CoreProtectionError for protected paths.
        """
        if self._is_core_protected(rel_path):
            raise CoreProtectionError(
                f"Cannot write to '{rel_path}': this is a core-protected file. "
                f"You can only write to the 'extensions/' directory. "
                f"Protected files: {sorted(self.core_files)}"
            )

        target = (self.project_root / rel_path).resolve()
        # Security: ensure within project root
        if not str(target).startswith(str(self.project_root)):
            raise PermissionError(
                f"Path '{rel_path}' resolves outside the project root"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {rel_path}"

    def delete_file(self, rel_path: str) -> str:
        """
        Delete a file. Only files under `extensions/` can be deleted.
        Raises CoreProtectionError for protected paths.
        """
        if self._is_core_protected(rel_path):
            raise CoreProtectionError(
                f"Cannot delete '{rel_path}': core-protected file."
            )
        target = (self.project_root / rel_path).resolve()
        if not str(target).startswith(str(self.project_root)):
            raise PermissionError(
                f"Path '{rel_path}' resolves outside the project root"
            )
        if not target.exists():
            raise FileNotFoundError(f"File not found: {rel_path}")
        target.unlink()
        return f"Deleted {rel_path}"

    def list_files(self) -> List[Dict[str, str]]:
        """
        List all files in the project tree (excluding hidden files/dirs,
        __pycache__, and .env).
        Returns a list of dicts with 'path', 'type' (core/extension/other), and 'size'.
        """
        results: List[Dict[str, str]] = []
        for path in sorted(self.project_root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.project_root)
            rel_str = str(rel)
            # Skip hidden files, __pycache__, .env
            if any(part.startswith(".") for part in rel.parts):
                continue
            if "__pycache__" in rel.parts:
                continue

            file_type = "core" if rel.name in self.core_files else "extension" if rel_str.startswith("extensions") else "other"
            results.append({
                "path": rel_str,
                "type": file_type,
                "size": str(path.stat().st_size),
            })
        return results

    def hot_load(self, rel_path: str) -> str:
        """
        Hot-load (import/reload) a Python module from extensions/.
        The module runs in a SANDBOXED environment that blocks:
        - Dangerous imports (os, subprocess, socket, etc.)
        - Dangerous builtins (exec, eval, open, etc.)
        - File system and network access
        Returns a summary of the loaded module's namespace.
        """
        if not rel_path.startswith("extensions/"):
            raise CoreProtectionError(
                f"Can only hot-load modules from extensions/, got '{rel_path}'"
            )
        if not rel_path.endswith(".py"):
            raise ValueError(f"Can only hot-load .py files, got '{rel_path}'")

        target = (self.project_root / rel_path).resolve()
        if not target.exists():
            raise FileNotFoundError(f"Module not found: {rel_path}")

        # Build module name: extensions/foo/bar.py → extensions.foo.bar
        module_name = rel_path.replace("/", ".").removesuffix(".py")

        # Read the source code
        source = target.read_text(encoding="utf-8")

        # Execute in sandbox
        namespace = sandboxed_exec(source, self.project_root, module_name)

        # Create a module object from the sandbox namespace
        module = type(sys)(module_name)
        for key, value in namespace.items():
            if not key.startswith("__"):
                setattr(module, key, value)
        sys.modules[module_name] = module

        # Report what was defined
        exports = [k for k in namespace if not k.startswith("__")]
        return (
            f"Loaded module '{module_name}' in sandbox. "
            f"Defined: {exports if exports else '(nothing)'}"
        )
