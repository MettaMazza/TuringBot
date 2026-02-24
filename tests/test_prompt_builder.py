"""
Comprehensive tests for prompt_builder.
Covers: build_system_prompt, build_heartbeat_prompt, build_interrupt_prompt,
codebase introspection, file collection, and error handling.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from prompt_builder import (
    build_system_prompt,
    build_heartbeat_prompt,
    build_interrupt_prompt,
    _read_file_safe,
    _collect_source_files,
)
from config import CORE_FILES


class TestReadFileSafe:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("hello world")
        assert _read_file_safe(f) == "hello world"

    def test_returns_error_for_missing_file(self, tmp_path):
        f = tmp_path / "nonexistent.py"
        result = _read_file_safe(f)
        assert "Error reading file" in result


class TestCollectSourceFiles:
    def test_collects_root_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("# a")
        (tmp_path / "b.py").write_text("# b")
        (tmp_path / "data.txt").write_text("not python")

        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()

        files = _collect_source_files(tmp_path, ext_dir)
        names = [f.name for f in files]
        assert "a.py" in names
        assert "b.py" in names
        assert "data.txt" not in names

    def test_collects_extension_files(self, tmp_path):
        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()
        (ext_dir / "ext.py").write_text("# ext")

        files = _collect_source_files(tmp_path, ext_dir)
        names = [f.name for f in files]
        assert "ext.py" in names

    def test_collects_nested_extensions(self, tmp_path):
        ext_dir = tmp_path / "extensions"
        sub = ext_dir / "subdir"
        sub.mkdir(parents=True)
        (sub / "deep.py").write_text("# deep")

        files = _collect_source_files(tmp_path, ext_dir)
        names = [f.name for f in files]
        assert "deep.py" in names

    def test_handles_missing_extensions_dir(self, tmp_path):
        ext_dir = tmp_path / "nonexistent"
        (tmp_path / "main.py").write_text("# main")
        files = _collect_source_files(tmp_path, ext_dir)
        assert len(files) >= 1

    def test_sorted_output(self, tmp_path):
        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()
        (tmp_path / "z.py").write_text("")
        (tmp_path / "a.py").write_text("")

        files = _collect_source_files(tmp_path, ext_dir)
        names = [f.name for f in files]
        assert names == sorted(names)


class TestBuildSystemPrompt:
    def test_contains_identity(self, tmp_path):
        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()
        (tmp_path / "main.py").write_text("# entry point")

        prompt = build_system_prompt(tmp_path, ext_dir, CORE_FILES)
        assert "TURING MACHINE" in prompt
        assert "autonomous" in prompt.lower()

    def test_contains_codebase(self, tmp_path):
        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()
        (tmp_path / "main.py").write_text("print('hello')")

        prompt = build_system_prompt(tmp_path, ext_dir, CORE_FILES)
        assert "print('hello')" in prompt
        assert "main.py" in prompt

    def test_marks_core_protected(self, tmp_path):
        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()
        (tmp_path / "main.py").write_text("# core")

        prompt = build_system_prompt(tmp_path, ext_dir, CORE_FILES)
        assert "CORE-PROTECTED" in prompt

    def test_marks_extensions_mutable(self, tmp_path):
        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()
        (ext_dir / "ext.py").write_text("# mutable")

        prompt = build_system_prompt(tmp_path, ext_dir, CORE_FILES)
        assert "MUTABLE" in prompt

    def test_contains_tape_docs(self, tmp_path):
        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()

        prompt = build_system_prompt(tmp_path, ext_dir, CORE_FILES)
        assert "tape_read" in prompt
        assert "tape_write" in prompt
        assert "tape_move" in prompt
        assert "tape_jump" in prompt
        assert "tape_scan" in prompt

    def test_contains_code_rw_docs(self, tmp_path):
        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()

        prompt = build_system_prompt(tmp_path, ext_dir, CORE_FILES)
        assert "code_read" in prompt
        assert "code_write" in prompt
        assert "code_exec" in prompt

    def test_contains_heartbeat_docs(self, tmp_path):
        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()

        prompt = build_system_prompt(tmp_path, ext_dir, CORE_FILES)
        assert "heartbeat" in prompt.lower()

    def test_empty_project_shows_no_files(self, tmp_path):
        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()
        # No .py files at all
        prompt = build_system_prompt(tmp_path, ext_dir, CORE_FILES)
        assert "no source files found" in prompt

    def test_contains_rules(self, tmp_path):
        ext_dir = tmp_path / "extensions"
        ext_dir.mkdir()

        prompt = build_system_prompt(tmp_path, ext_dir, CORE_FILES)
        assert "RULES" in prompt
        assert "extensions/" in prompt


class TestBuildHeartbeatPrompt:
    def test_returns_nonempty_string(self):
        result = build_heartbeat_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mentions_heartbeat(self):
        result = build_heartbeat_prompt()
        assert "heartbeat" in result.lower()

    def test_encourages_autonomy(self):
        result = build_heartbeat_prompt()
        assert "tool" in result.lower() or "think" in result.lower()


class TestBuildInterruptPrompt:
    def test_contains_username(self):
        result = build_interrupt_prompt("Alice", "Hello!")
        assert "Alice" in result

    def test_contains_message(self):
        result = build_interrupt_prompt("Bob", "What is 2+2?")
        assert "What is 2+2?" in result

    def test_mentions_interrupt(self):
        result = build_interrupt_prompt("X", "Y")
        assert "INTERRUPT" in result

    def test_mentions_resume(self):
        result = build_interrupt_prompt("X", "Y")
        assert "resume" in result.lower()
