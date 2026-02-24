"""
Comprehensive tests for Sandbox module.
Covers: module allowlist/blocklist, restricted builtins, sandboxed execution,
safe imports, blocked imports, and edge cases.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from sandbox import (
    ALLOWED_MODULES,
    BLOCKED_BUILTINS,
    BLOCKED_MODULES,
    _is_module_allowed,
    _make_safe_builtins,
    _make_safe_import,
    create_sandbox_globals,
    sandboxed_exec,
)


class TestModuleAllowlist:
    @pytest.mark.parametrize("module", [
        "math", "random", "json", "re", "datetime", "collections",
        "itertools", "functools", "string", "hashlib", "copy",
    ])
    def test_allowed_modules_pass(self, module):
        assert _is_module_allowed(module) is True

    def test_allowed_submodule(self):
        assert _is_module_allowed("collections.abc") is True

    @pytest.mark.parametrize("module", [
        "os", "subprocess", "shutil", "socket", "sys",
        "ctypes", "pickle", "multiprocessing", "signal",
    ])
    def test_blocked_modules_fail(self, module):
        assert _is_module_allowed(module) is False

    def test_blocked_submodule(self):
        assert _is_module_allowed("os.path") is False
        assert _is_module_allowed("subprocess.run") is False

    def test_unknown_module_blocked(self):
        # Anything not in the allowlist is blocked by default
        assert _is_module_allowed("some_unknown_thing") is False

    def test_blocked_modules_is_frozenset(self):
        assert isinstance(BLOCKED_MODULES, frozenset)

    def test_allowed_modules_is_frozenset(self):
        assert isinstance(ALLOWED_MODULES, frozenset)


class TestSafeBuiltins:
    def test_safe_builtins_excludes_dangerous(self):
        safe = _make_safe_builtins()
        for blocked in BLOCKED_BUILTINS:
            assert blocked not in safe, f"{blocked} should be blocked"

    def test_safe_builtins_includes_basic(self):
        safe = _make_safe_builtins()
        # Basic safe builtins should be present
        assert "print" in safe
        assert "len" in safe
        assert "range" in safe
        assert "int" in safe
        assert "str" in safe
        assert "list" in safe
        assert "dict" in safe
        assert "bool" in safe
        assert "type" in safe
        assert "isinstance" in safe
        assert "zip" in safe
        assert "map" in safe
        assert "filter" in safe
        assert "sorted" in safe
        assert "sum" in safe
        assert "min" in safe
        assert "max" in safe
        assert "abs" in safe
        assert "enumerate" in safe

    def test_safe_print_returns_string(self):
        safe = _make_safe_builtins()
        result = safe["print"]("hello", "world")
        assert isinstance(result, str)

    def test_blocked_builtins_is_frozenset(self):
        assert isinstance(BLOCKED_BUILTINS, frozenset)


class TestSafeImport:
    def test_allows_math(self, tmp_path):
        safe_import = _make_safe_import(tmp_path)
        mod = safe_import("math")
        assert hasattr(mod, "sqrt")

    def test_blocks_os(self, tmp_path):
        safe_import = _make_safe_import(tmp_path)
        with pytest.raises(ImportError, match="SANDBOX"):
            safe_import("os")

    def test_blocks_subprocess(self, tmp_path):
        safe_import = _make_safe_import(tmp_path)
        with pytest.raises(ImportError, match="SANDBOX"):
            safe_import("subprocess")

    def test_blocks_socket(self, tmp_path):
        safe_import = _make_safe_import(tmp_path)
        with pytest.raises(ImportError, match="SANDBOX"):
            safe_import("socket")

    def test_error_message_includes_allowlist(self, tmp_path):
        safe_import = _make_safe_import(tmp_path)
        with pytest.raises(ImportError, match="Allowed modules"):
            safe_import("os")


class TestCreateSandboxGlobals:
    def test_has_builtins(self, tmp_path):
        g = create_sandbox_globals(tmp_path, "test_module")
        assert "__builtins__" in g
        assert isinstance(g["__builtins__"], dict)

    def test_has_safe_import(self, tmp_path):
        g = create_sandbox_globals(tmp_path, "test_module")
        assert "__import__" in g["__builtins__"]

    def test_has_module_name(self, tmp_path):
        g = create_sandbox_globals(tmp_path, "my.module")
        assert g["__name__"] == "my.module"

    def test_has_package(self, tmp_path):
        g = create_sandbox_globals(tmp_path, "extensions.foo")
        assert g["__package__"] == "extensions"

    def test_no_package_for_root(self, tmp_path):
        g = create_sandbox_globals(tmp_path, "root_module")
        assert g["__package__"] is None

    def test_blocks_exec_in_builtins(self, tmp_path):
        g = create_sandbox_globals(tmp_path, "test")
        assert "exec" not in g["__builtins__"]

    def test_blocks_eval_in_builtins(self, tmp_path):
        g = create_sandbox_globals(tmp_path, "test")
        assert "eval" not in g["__builtins__"]

    def test_blocks_open_in_builtins(self, tmp_path):
        g = create_sandbox_globals(tmp_path, "test")
        assert "open" not in g["__builtins__"]


class TestSandboxedExec:
    def test_simple_code(self, tmp_path):
        ns = sandboxed_exec("x = 1 + 1", tmp_path)
        assert ns["x"] == 2

    def test_safe_import_works(self, tmp_path):
        ns = sandboxed_exec("import math\nresult = math.sqrt(4)", tmp_path)
        assert ns["result"] == 2.0

    def test_function_definition(self, tmp_path):
        code = "def add(a, b):\n    return a + b\nresult = add(3, 4)"
        ns = sandboxed_exec(code, tmp_path)
        assert ns["result"] == 7

    def test_class_definition(self, tmp_path):
        code = "class Foo:\n    val = 42\nresult = Foo.val"
        ns = sandboxed_exec(code, tmp_path)
        assert ns["result"] == 42

    def test_blocked_import_os_raises(self, tmp_path):
        with pytest.raises(ImportError, match="SANDBOX"):
            sandboxed_exec("import os", tmp_path)

    def test_blocked_import_subprocess_raises(self, tmp_path):
        with pytest.raises(ImportError, match="SANDBOX"):
            sandboxed_exec("import subprocess", tmp_path)

    def test_blocked_import_socket_raises(self, tmp_path):
        with pytest.raises(ImportError, match="SANDBOX"):
            sandboxed_exec("import socket", tmp_path)

    def test_blocked_import_shutil_raises(self, tmp_path):
        with pytest.raises(ImportError, match="SANDBOX"):
            sandboxed_exec("import shutil", tmp_path)

    def test_no_open_builtin(self, tmp_path):
        with pytest.raises(NameError):
            sandboxed_exec("f = open('/etc/passwd')", tmp_path)

    def test_no_eval_builtin(self, tmp_path):
        with pytest.raises(NameError):
            sandboxed_exec("eval('1+1')", tmp_path)

    def test_no_exec_builtin(self, tmp_path):
        with pytest.raises(NameError):
            sandboxed_exec("exec('x=1')", tmp_path)

    def test_runtime_error_propagates(self, tmp_path):
        with pytest.raises(ZeroDivisionError):
            sandboxed_exec("x = 1 / 0", tmp_path)

    def test_custom_module_name(self, tmp_path):
        ns = sandboxed_exec("name = __name__", tmp_path, "my.module")
        assert ns["name"] == "my.module"

    def test_safe_builtins_available(self, tmp_path):
        code = "result = len([1, 2, 3]) + sum([4, 5]) + max(1, 2)"
        ns = sandboxed_exec(code, tmp_path)
        assert ns["result"] == 3 + 9 + 2

    def test_list_comprehension_works(self, tmp_path):
        code = "result = [x**2 for x in range(5)]"
        ns = sandboxed_exec(code, tmp_path)
        assert ns["result"] == [0, 1, 4, 9, 16]

    def test_json_import_works(self, tmp_path):
        code = "import json\nresult = json.dumps({'a': 1})"
        ns = sandboxed_exec(code, tmp_path)
        assert ns["result"] == '{"a": 1}'

    def test_re_import_works(self, tmp_path):
        code = "import re\nresult = bool(re.match(r'\\d+', '123'))"
        ns = sandboxed_exec(code, tmp_path)
        assert ns["result"] is True
