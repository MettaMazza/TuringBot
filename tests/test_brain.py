"""
Comprehensive tests for Brain.
Covers: tool dispatch (all 11 tools), error handling, chat flow,
history management, tool chaining, and edge cases.
"""

import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from brain import Brain, TOOL_DEFINITIONS
from turing_tape import TuringTape3D
from codebase_rw import CodebaseRW, CoreProtectionError


class TestToolDefinitions:
    def test_all_tools_defined(self):
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        expected = {
            "tape_read", "tape_write", "tape_move", "tape_jump",
            "tape_scan", "tape_status",
            "code_read", "code_write", "code_delete", "code_list", "code_exec",
        }
        assert names == expected

    def test_all_tools_have_description(self):
        for tool in TOOL_DEFINITIONS:
            assert "description" in tool["function"]
            assert len(tool["function"]["description"]) > 0

    def test_all_tools_have_parameters(self):
        for tool in TOOL_DEFINITIONS:
            assert "parameters" in tool["function"]


class TestBrainInit:
    def test_default_values(self, brain):
        assert brain.model == "test-model"
        assert brain.history == []
        assert brain.max_history > 0

    def test_custom_values(self, tape, codebase):
        b = Brain(tape=tape, codebase=codebase, model="custom", max_history=10)
        assert b.model == "custom"
        assert b.max_history == 10


class TestBrainToolDispatch:
    """Test _dispatch_tool for every tool."""

    def test_tape_read_empty(self, brain):
        result = brain._dispatch_tool("tape_read", {})
        assert "empty" in result.lower()

    def test_tape_read_with_value(self, brain):
        brain.tape.write("hi")
        result = brain._dispatch_tool("tape_read", {})
        assert "'hi'" in result

    def test_tape_write(self, brain):
        result = brain._dispatch_tool("tape_write", {"value": "data"})
        assert "Wrote" in result
        assert brain.tape.read() == "data"

    def test_tape_move(self, brain):
        result = brain._dispatch_tool("tape_move", {"direction": "+x"})
        assert "Moved" in result
        assert brain.tape.head == (1, 0, 0)

    def test_tape_move_invalid(self, brain):
        result = brain._dispatch_tool("tape_move", {"direction": "invalid"})
        assert "ERROR" in result

    def test_tape_jump(self, brain):
        result = brain._dispatch_tool("tape_jump", {"x": 5, "y": 10, "z": -3})
        assert "Jumped" in result
        assert brain.tape.head == (5, 10, -3)

    def test_tape_scan_empty(self, brain):
        result = brain._dispatch_tool("tape_scan", {})
        assert "No written cells" in result

    def test_tape_scan_with_data(self, brain):
        brain.tape.write("val")
        result = brain._dispatch_tool("tape_scan", {"radius": 1})
        assert "0,0,0" in result

    def test_tape_scan_default_radius(self, brain):
        brain.tape.write("val")
        result = brain._dispatch_tool("tape_scan", {})
        assert "val" in result

    def test_tape_status(self, brain):
        result = brain._dispatch_tool("tape_status", {})
        assert "Head" in result

    def test_code_read(self, brain, tmp_project):
        result = brain._dispatch_tool("code_read", {"path": "main.py"})
        assert "main entry point" in result

    def test_code_read_missing(self, brain):
        result = brain._dispatch_tool("code_read", {"path": "nonexistent.py"})
        assert "FILE NOT FOUND" in result

    def test_code_write(self, brain, tmp_project):
        result = brain._dispatch_tool("code_write", {
            "path": "extensions/new.py",
            "content": "# new code",
        })
        assert "Successfully wrote" in result

    def test_code_write_protected(self, brain):
        result = brain._dispatch_tool("code_write", {
            "path": "main.py",
            "content": "# hacked",
        })
        assert "PROTECTION ERROR" in result

    def test_code_delete(self, brain, tmp_project):
        (tmp_project / "extensions" / "deleteme.py").write_text("bye")
        result = brain._dispatch_tool("code_delete", {"path": "extensions/deleteme.py"})
        assert "Deleted" in result

    def test_code_delete_protected(self, brain):
        result = brain._dispatch_tool("code_delete", {"path": "main.py"})
        assert "PROTECTION ERROR" in result

    def test_code_list(self, brain):
        result = brain._dispatch_tool("code_list", {})
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_code_exec(self, brain, tmp_project):
        (tmp_project / "extensions" / "exec_test.py").write_text("VAL = 42\n")
        result = brain._dispatch_tool("code_exec", {"path": "extensions/exec_test.py"})
        assert "Loaded" in result or "module" in result.lower()
        # Clean up sys.modules
        import sys
        sys.modules.pop("extensions.exec_test", None)

    def test_code_exec_protected(self, brain):
        result = brain._dispatch_tool("code_exec", {"path": "main.py"})
        assert "PROTECTION ERROR" in result

    def test_unknown_tool(self, brain):
        result = brain._dispatch_tool("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_generic_exception_handling(self, brain):
        # Force a TypeError by passing wrong args to tape_write
        result = brain._dispatch_tool("tape_write", {"value": 42})
        assert "ERROR" in result


class TestBrainHistory:
    def test_trim_history_respects_limit(self, brain):
        brain.max_history = 5
        brain.history = [{"role": "system", "content": "sys"}]
        for i in range(20):
            brain.history.append({"role": "user", "content": f"msg {i}"})
        brain._trim_history()
        assert len(brain.history) <= 5
        assert brain.history[0]["role"] == "system"

    def test_trim_history_under_limit(self, brain):
        brain.max_history = 50
        brain.history = [{"role": "user", "content": "hi"}]
        brain._trim_history()
        assert len(brain.history) == 1

    def test_clear_history_keeps_system(self, brain):
        brain.history = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        brain.clear_history()
        assert len(brain.history) == 1
        assert brain.history[0]["role"] == "system"

    def test_clear_history_empty(self, brain):
        brain.clear_history()
        assert brain.history == []


class TestBrainThink:
    @pytest.mark.asyncio
    async def test_simple_response(self, brain, mock_ollama_client):
        """Test a simple text response with no tool calls."""
        mock_msg = MagicMock()
        mock_msg.content = "Hello, I am thinking."
        mock_msg.tool_calls = None
        mock_msg.model_dump.return_value = {
            "role": "assistant",
            "content": "Hello, I am thinking.",
        }

        mock_response = MagicMock()
        mock_response.message = mock_msg
        mock_ollama_client.chat.return_value = mock_response

        text, tool_log = await brain.think("System prompt", "User message")
        assert text == "Hello, I am thinking."
        assert tool_log == []
        mock_ollama_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_response_with_tool_call(self, brain, mock_ollama_client):
        """Test response that includes a tool call followed by final text."""
        # First call: model wants to use tape_read
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "tape_read"
        mock_tool_call.function.arguments = {}

        mock_msg1 = MagicMock()
        mock_msg1.content = ""
        mock_msg1.tool_calls = [mock_tool_call]
        mock_msg1.model_dump.return_value = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "tape_read", "arguments": {}}}],
        }

        # Second call: model returns final text
        mock_msg2 = MagicMock()
        mock_msg2.content = "The cell is empty."
        mock_msg2.tool_calls = None
        mock_msg2.model_dump.return_value = {
            "role": "assistant",
            "content": "The cell is empty.",
        }

        mock_resp1 = MagicMock()
        mock_resp1.message = mock_msg1
        mock_resp2 = MagicMock()
        mock_resp2.message = mock_msg2

        mock_ollama_client.chat.side_effect = [mock_resp1, mock_resp2]

        text, tool_log = await brain.think("System", "Read the tape")
        assert text == "The cell is empty."
        assert len(tool_log) == 1
        assert tool_log[0]["tool"] == "tape_read"

    @pytest.mark.asyncio
    async def test_system_prompt_updated(self, brain, mock_ollama_client):
        """Test that system prompt is set/updated on each call."""
        mock_msg = MagicMock()
        mock_msg.content = "ok"
        mock_msg.tool_calls = None
        mock_msg.model_dump.return_value = {"role": "assistant", "content": "ok"}
        mock_resp = MagicMock()
        mock_resp.message = mock_msg
        mock_ollama_client.chat.return_value = mock_resp

        await brain.think("First prompt", "msg1")
        assert brain.history[0]["content"] == "First prompt"

        await brain.think("Second prompt", "msg2")
        assert brain.history[0]["content"] == "Second prompt"

    @pytest.mark.asyncio
    async def test_max_tool_rounds(self, brain, mock_ollama_client):
        """Test that max tool rounds limits infinite tool-call loops."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "tape_status"
        mock_tool_call.function.arguments = {}

        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.tool_calls = [mock_tool_call]
        mock_msg.model_dump.return_value = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "tape_status", "arguments": {}}}],
        }
        mock_resp = MagicMock()
        mock_resp.message = mock_msg
        mock_ollama_client.chat.return_value = mock_resp

        text, tool_log = await brain.think("System", "Loop forever")
        assert "max tool-call depth" in text
        assert len(tool_log) == 10  # Default max rounds

    @pytest.mark.asyncio
    async def test_empty_response(self, brain, mock_ollama_client):
        """Test handling of empty content from model."""
        mock_msg = MagicMock()
        mock_msg.content = ""
        mock_msg.tool_calls = None
        mock_msg.model_dump.return_value = {"role": "assistant", "content": ""}
        mock_resp = MagicMock()
        mock_resp.message = mock_msg
        mock_ollama_client.chat.return_value = mock_resp

        text, tool_log = await brain.think("System", "msg")
        assert text == ""

    @pytest.mark.asyncio
    async def test_none_content_response(self, brain, mock_ollama_client):
        """Test handling of None content from model."""
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_msg.tool_calls = None
        mock_msg.model_dump.return_value = {"role": "assistant", "content": None}
        mock_resp = MagicMock()
        mock_resp.message = mock_msg
        mock_ollama_client.chat.return_value = mock_resp

        text, tool_log = await brain.think("System", "msg")
        assert text == ""

    @pytest.mark.asyncio
    async def test_tool_call_with_no_arguments(self, brain, mock_ollama_client):
        """Test tool call where arguments is None."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "tape_read"
        mock_tool_call.function.arguments = None

        mock_msg1 = MagicMock()
        mock_msg1.content = ""
        mock_msg1.tool_calls = [mock_tool_call]
        mock_msg1.model_dump.return_value = {
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "tape_read", "arguments": None}}],
        }

        mock_msg2 = MagicMock()
        mock_msg2.content = "Done"
        mock_msg2.tool_calls = None
        mock_msg2.model_dump.return_value = {"role": "assistant", "content": "Done"}

        mock_resp1 = MagicMock()
        mock_resp1.message = mock_msg1
        mock_resp2 = MagicMock()
        mock_resp2.message = mock_msg2
        mock_ollama_client.chat.side_effect = [mock_resp1, mock_resp2]

        text, tool_log = await brain.think("System", "msg")
        assert text == "Done"
        assert len(tool_log) == 1
