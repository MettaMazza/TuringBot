"""
Brain — Ollama interface with tool-calling support.

Wraps the Ollama chat API, maintains conversation history, and defines
the tool schema that lets the model interact with its tape and codebase.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import ollama

from config import MAX_HISTORY_MESSAGES, OLLAMA_HOST, OLLAMA_MODEL
from turing_tape import TuringTape3D
from codebase_rw import CodebaseRW, CoreProtectionError

logger = logging.getLogger("turingbot.brain")

# ── Tool Definitions (Ollama function-calling schema) ────────────────

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "tape_read",
            "description": "Read the value at the current tape head position",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tape_write",
            "description": "Write a string value at the current tape head position",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "The value to write to the current cell",
                    }
                },
                "required": ["value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tape_move",
            "description": "Move the tape head one step in a direction: +x, -x, +y, -y, +z, -z (or up/down/left/right/forward/backward)",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "Direction to move: +x, -x, +y, -y, +z, -z, up, down, left, right, forward, backward",
                    }
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tape_jump",
            "description": "Jump the tape head to an arbitrary (x, y, z) coordinate",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate"},
                    "y": {"type": "integer", "description": "Y coordinate"},
                    "z": {"type": "integer", "description": "Z coordinate"},
                },
                "required": ["x", "y", "z"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tape_scan",
            "description": "Read all non-empty cells within a radius of the tape head",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius": {
                        "type": "integer",
                        "description": "Scan radius (default 1)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tape_status",
            "description": "Get the current tape head position and statistics",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_read",
            "description": "Read a file from the codebase by relative path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file, e.g. 'brain.py' or 'extensions/my_module.py'",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_write",
            "description": "Write/create a file in the extensions/ directory. Core files are protected.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path under extensions/, e.g. 'extensions/my_module.py'",
                    },
                    "content": {
                        "type": "string",
                        "description": "The file content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_delete",
            "description": "Delete a file from the extensions/ directory. Core files are protected.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to delete, e.g. 'extensions/old_module.py'",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_list",
            "description": "List all files in the codebase with their types (core/extension) and sizes",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_exec",
            "description": "Hot-load and execute a Python module from extensions/",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the module, e.g. 'extensions/my_module.py'",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


class Brain:
    """
    Ollama-powered brain with tool calling.

    Manages conversation history and dispatches tool calls to the tape
    and codebase subsystems.
    """

    def __init__(
        self,
        tape: TuringTape3D,
        codebase: CodebaseRW,
        model: Optional[str] = None,
        host: Optional[str] = None,
        max_history: Optional[int] = None,
    ) -> None:
        self.tape = tape
        self.codebase = codebase
        self.model = model or OLLAMA_MODEL
        self.host = host or OLLAMA_HOST
        self.max_history = max_history or MAX_HISTORY_MESSAGES
        self.history: List[Dict[str, Any]] = []
        self._client = ollama.AsyncClient(host=self.host)

    # ── Tool Dispatch ────────────────────────────────────────────────

    def _dispatch_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool call and return the result as a string."""
        try:
            if name == "tape_read":
                value = self.tape.read()
                return f"Cell value: '{value}'" if value else "Cell is empty"

            elif name == "tape_write":
                return self.tape.write(args["value"])

            elif name == "tape_move":
                return self.tape.move(args["direction"])

            elif name == "tape_jump":
                return self.tape.jump(args["x"], args["y"], args["z"])

            elif name == "tape_scan":
                radius = args.get("radius", 1)
                cells = self.tape.scan_neighborhood(radius)
                if not cells:
                    return "No written cells in scan radius"
                return json.dumps(cells, indent=2)

            elif name == "tape_status":
                return self.tape.status()

            elif name == "code_read":
                return self.codebase.read_file(args["path"])

            elif name == "code_write":
                return self.codebase.write_file(args["path"], args["content"])

            elif name == "code_delete":
                return self.codebase.delete_file(args["path"])

            elif name == "code_list":
                files = self.codebase.list_files()
                return json.dumps(files, indent=2)

            elif name == "code_exec":
                return self.codebase.hot_load(args["path"])

            else:
                return f"Unknown tool: {name}"

        except (CoreProtectionError, PermissionError) as e:
            return f"🔒 PROTECTION ERROR: {e}"
        except FileNotFoundError as e:
            return f"❌ FILE NOT FOUND: {e}"
        except Exception as e:
            return f"⚠️ ERROR: {type(e).__name__}: {e}"

    # ── Chat ─────────────────────────────────────────────────────────

    def _trim_history(self) -> None:
        """Keep history within the max length, preserving the system message."""
        if len(self.history) > self.max_history:
            # Always keep the first message (system prompt)
            self.history = self.history[:1] + self.history[-(self.max_history - 1):]

    async def think(
        self,
        system_prompt: str,
        user_message: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Send a message to the model and process any tool calls.

        Returns:
            (response_text, tool_log) where tool_log is a list of
            {tool, args, result} dicts for transparency logging.
        """
        # Set/update system prompt as first message
        if not self.history or self.history[0].get("role") != "system":
            self.history.insert(0, {"role": "system", "content": system_prompt})
        else:
            self.history[0]["content"] = system_prompt

        # Add user message
        self.history.append({"role": "user", "content": user_message})
        self._trim_history()

        tool_log: List[Dict[str, Any]] = []
        max_tool_rounds = 10  # Safety limit on chained tool calls

        for _ in range(max_tool_rounds):
            response = await self._client.chat(
                model=self.model,
                messages=self.history,
                tools=TOOL_DEFINITIONS,
            )

            msg = response.message
            # Add assistant response to history
            self.history.append(msg.model_dump())

            # Check for tool calls
            if not msg.tool_calls:
                return msg.content or "", tool_log

            # Process each tool call
            for tool_call in msg.tool_calls:
                fn = tool_call.function
                tool_name = fn.name
                tool_args = fn.arguments if fn.arguments else {}
                logger.info(f"Tool call: {tool_name}({tool_args})")
                result = self._dispatch_tool(tool_name, tool_args)
                logger.info(f"Tool result: {result[:200]}")
                tool_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result,
                })
                # Feed result back as a tool message
                self.history.append({
                    "role": "tool",
                    "content": result,
                })

        # If we exhaust tool rounds, return what we have
        return "(max tool-call depth reached)", tool_log

    def clear_history(self) -> None:
        """Clear conversation history (except system prompt)."""
        if self.history and self.history[0].get("role") == "system":
            self.history = [self.history[0]]
        else:
            self.history = []
