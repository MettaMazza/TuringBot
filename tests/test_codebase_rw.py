"""
Comprehensive tests for CodebaseRW.
Covers: read, write, delete, list, hot_load, core protection,
path traversal prevention, edge cases, and error handling.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from codebase_rw import CodebaseRW, CoreProtectionError
from config import CORE_FILES


class TestCodebaseRWInit:
    def test_creates_extensions_dir(self, tmp_path):
        ext = tmp_path / "new_extensions"
        assert not ext.exists()
        codebase = CodebaseRW(
            project_root=tmp_path,
            extensions_dir=ext,
            core_files=CORE_FILES,
        )
        assert ext.exists()

    def test_uses_provided_paths(self, tmp_project):
        codebase = CodebaseRW(
            project_root=tmp_project,
            extensions_dir=tmp_project / "extensions",
            core_files=frozenset({"custom.py"}),
        )
        assert codebase.project_root == tmp_project
        assert codebase.core_files == frozenset({"custom.py"})


class TestCodebaseRWCoreProtection:
    def test_is_core_protected_true_for_core(self, codebase):
        assert codebase._is_core_protected("main.py") is True
        assert codebase._is_core_protected("config.py") is True
        assert codebase._is_core_protected("brain.py") is True

    def test_is_core_protected_true_for_non_extension(self, codebase):
        assert codebase._is_core_protected("some_other_file.py") is True

    def test_is_core_protected_false_for_extension(self, codebase):
        assert codebase._is_core_protected("extensions/my_module.py") is False

    def test_is_core_protected_nested_extension(self, codebase):
        assert codebase._is_core_protected("extensions/sub/deep.py") is False


class TestCodebaseRWRead:
    def test_read_existing_file(self, codebase, tmp_project):
        assert "main entry point" in codebase.read_file("main.py")

    def test_read_nonexistent_raises(self, codebase):
        with pytest.raises(FileNotFoundError):
            codebase.read_file("nonexistent.py")

    def test_read_directory_raises(self, codebase):
        with pytest.raises(IsADirectoryError):
            codebase.read_file("extensions")

    def test_read_path_traversal_blocked(self, codebase):
        with pytest.raises(PermissionError, match="outside"):
            codebase.read_file("../../etc/passwd")

    def test_read_extension_file(self, codebase, tmp_project):
        (tmp_project / "extensions" / "test_ext.py").write_text("# test extension")
        content = codebase.read_file("extensions/test_ext.py")
        assert "test extension" in content


class TestCodebaseRWWrite:
    def test_write_to_extensions(self, codebase, tmp_project):
        result = codebase.write_file("extensions/new.py", "print('hello')")
        assert "Successfully wrote" in result
        assert (tmp_project / "extensions" / "new.py").read_text() == "print('hello')"

    def test_write_creates_subdirectories(self, codebase, tmp_project):
        codebase.write_file("extensions/sub/deep/mod.py", "# deep")
        assert (tmp_project / "extensions" / "sub" / "deep" / "mod.py").exists()

    def test_write_to_core_raises(self, codebase):
        with pytest.raises(CoreProtectionError, match="core-protected"):
            codebase.write_file("main.py", "# hacked")

    def test_write_to_non_extension_raises(self, codebase):
        with pytest.raises(CoreProtectionError):
            codebase.write_file("random_file.py", "# not allowed")

    def test_write_path_traversal_blocked(self, codebase):
        with pytest.raises((CoreProtectionError, PermissionError)):
            codebase.write_file("../../evil.py", "# evil")

    def test_write_overwrites_existing(self, codebase, tmp_project):
        codebase.write_file("extensions/file.py", "v1")
        codebase.write_file("extensions/file.py", "v2")
        assert (tmp_project / "extensions" / "file.py").read_text() == "v2"


class TestCodebaseRWDelete:
    def test_delete_extension_file(self, codebase, tmp_project):
        fpath = tmp_project / "extensions" / "todelete.py"
        fpath.write_text("# delete me")
        result = codebase.delete_file("extensions/todelete.py")
        assert "Deleted" in result
        assert not fpath.exists()

    def test_delete_core_raises(self, codebase):
        with pytest.raises(CoreProtectionError):
            codebase.delete_file("main.py")

    def test_delete_nonexistent_raises(self, codebase):
        with pytest.raises(FileNotFoundError):
            codebase.delete_file("extensions/nonexistent.py")

    def test_delete_path_traversal_blocked(self, codebase):
        with pytest.raises((CoreProtectionError, PermissionError)):
            codebase.delete_file("../../evil.py")


class TestCodebaseRWList:
    def test_list_files_structure(self, codebase, tmp_project):
        files = codebase.list_files()
        assert isinstance(files, list)
        assert all(isinstance(f, dict) for f in files)
        paths = [f["path"] for f in files]
        assert "main.py" in paths
        assert "config.py" in paths

    def test_list_files_types(self, codebase, tmp_project):
        (tmp_project / "extensions" / "ext.py").write_text("# ext")
        files = codebase.list_files()
        file_map = {f["path"]: f for f in files}
        assert file_map["main.py"]["type"] == "core"
        assert file_map["extensions/ext.py"]["type"] == "extension"

    def test_list_files_excludes_hidden(self, codebase, tmp_project):
        (tmp_project / ".hidden").write_text("secret")
        files = codebase.list_files()
        paths = [f["path"] for f in files]
        assert ".hidden" not in paths

    def test_list_files_excludes_pycache(self, codebase, tmp_project):
        pycache = tmp_project / "__pycache__"
        pycache.mkdir()
        (pycache / "cache.pyc").write_text("bytecode")
        files = codebase.list_files()
        paths = [f["path"] for f in files]
        assert not any("__pycache__" in p for p in paths)

    def test_list_files_has_size(self, codebase, tmp_project):
        files = codebase.list_files()
        for f in files:
            assert "size" in f
            assert int(f["size"]) >= 0


class TestCodebaseRWHotLoad:
    def test_hot_load_new_module(self, codebase, tmp_project):
        module_path = "extensions/hello.py"
        (tmp_project / "extensions" / "hello.py").write_text(
            "GREETING = 'Hello from extension!'\n"
        )
        result = codebase.hot_load(module_path)
        assert "Loaded" in result
        assert "extensions.hello" in result
        # Clean up
        if "extensions.hello" in sys.modules:
            del sys.modules["extensions.hello"]

    def test_hot_load_reload(self, codebase, tmp_project):
        module_path = "extensions/reloadme.py"
        (tmp_project / "extensions" / "reloadme.py").write_text("VAL = 1\n")

        result = codebase.hot_load(module_path)
        assert "Loaded" in result

        # Update and reload — should still work
        (tmp_project / "extensions" / "reloadme.py").write_text("VAL = 2\n")
        result = codebase.hot_load(module_path)
        assert "Loaded" in result
        assert "VAL" in result
        # Clean up
        sys.modules.pop("extensions.reloadme", None)

    def test_hot_load_sandbox_blocks_os(self, codebase, tmp_project):
        """Verify that hot-loaded extensions can't import os."""
        (tmp_project / "extensions" / "evil.py").write_text("import os\n")
        with pytest.raises(ImportError, match="SANDBOX"):
            codebase.hot_load("extensions/evil.py")


class TestCodebaseRWPathTraversalEdgeCases:
    """Cover path traversal within extensions/ that escapes project root."""

    def test_write_symlink_escape(self, tmp_path):
        """Cover line 88: write_file path resolves outside project root."""
        # Create a project structure where the extensions path resolves outside
        project = tmp_path / "project"
        project.mkdir()
        ext = project / "extensions"
        ext.mkdir()

        codebase = CodebaseRW(
            project_root=project,
            extensions_dir=ext,
            core_files=CORE_FILES,
        )

        # Mock resolve() to return a path outside project root
        with patch.object(Path, "resolve", return_value=Path("/outside/evil.py")):
            with pytest.raises(PermissionError, match="outside"):
                codebase.write_file("extensions/evil.py", "# evil")

    def test_delete_symlink_escape(self, tmp_path):
        """Cover line 107: delete_file path resolves outside project root."""
        project = tmp_path / "project"
        project.mkdir()
        ext = project / "extensions"
        ext.mkdir()

        codebase = CodebaseRW(
            project_root=project,
            extensions_dir=ext,
            core_files=CORE_FILES,
        )

        with patch.object(Path, "resolve", return_value=Path("/outside/evil.py")):
            with pytest.raises(PermissionError, match="outside"):
                codebase.delete_file("extensions/evil.py")
