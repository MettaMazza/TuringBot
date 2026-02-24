"""
TuringBot Configuration
Loads environment variables and defines project-wide constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Discord ──────────────────────────────────────────────────────────
DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
LISTEN_CHANNEL_ID: int = int(os.getenv("LISTEN_CHANNEL_ID", "0"))
THOUGHTS_CHANNEL_ID: int = int(os.getenv("THOUGHTS_CHANNEL_ID", "0"))

# ── Ollama ───────────────────────────────────────────────────────────
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:32b")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
EXTENSIONS_DIR: Path = PROJECT_ROOT / "extensions"
TAPE_STATE_FILE: Path = PROJECT_ROOT / "tape_state.json"

# ── Core Protection ─────────────────────────────────────────────────
# These files cannot be modified by the model's code_write tool.
CORE_FILES: frozenset = frozenset({
    "main.py",
    "config.py",
    "bot.py",
    "brain.py",
    "heartbeat.py",
    "turing_tape.py",
    "codebase_rw.py",
    "prompt_builder.py",
    "sandbox.py",
    "requirements.txt",
})

# ── Heartbeat ────────────────────────────────────────────────────────
HEARTBEAT_INTERVAL: float = float(os.getenv("HEARTBEAT_INTERVAL", "10.0"))

# ── Conversation History ─────────────────────────────────────────────
MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "50"))
