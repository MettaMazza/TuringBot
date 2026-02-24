"""
Shared test fixtures for TuringBot test suite.
"""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project structure for testing."""
    # Create extensions dir
    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir()
    (ext_dir / "__init__.py").write_text("# extensions init")

    # Create a dummy core file
    (tmp_path / "main.py").write_text("# main entry point")
    (tmp_path / "config.py").write_text("# config")

    return tmp_path


@pytest.fixture
def tape():
    """Fresh TuringTape3D instance."""
    from turing_tape import TuringTape3D
    return TuringTape3D()


@pytest.fixture
def codebase(tmp_project):
    """CodebaseRW pointed at a temporary project directory."""
    from codebase_rw import CodebaseRW
    from config import CORE_FILES
    return CodebaseRW(
        project_root=tmp_project,
        extensions_dir=tmp_project / "extensions",
        core_files=CORE_FILES,
    )


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for brain tests."""
    client = MagicMock()
    client.chat = AsyncMock()
    return client


@pytest.fixture
def brain(tape, codebase, mock_ollama_client):
    """Brain instance with mocked Ollama client."""
    from brain import Brain
    b = Brain(
        tape=tape,
        codebase=codebase,
        model="test-model",
        host="http://localhost:11434",
    )
    b._client = mock_ollama_client
    return b
