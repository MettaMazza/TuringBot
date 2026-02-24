#!/usr/bin/env python3
"""
TuringBot — Entry point.

An autonomous Ollama Discord bot with a 3D Turing tape,
continuous heartbeat, and self-modifying codebase capabilities.
"""

import logging
import sys

from config import OLLAMA_MODEL, TAPE_STATE_FILE
from turing_tape import TuringTape3D
from codebase_rw import CodebaseRW
from brain import Brain
from bot import TuringBot

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("turingbot")


def main() -> None:
    """Initialize all subsystems and start the bot."""
    logger.info("=" * 60)
    logger.info("  TuringBot — Emergent Computation Experiment")
    logger.info("=" * 60)

    # 1. Load or create the 3D Turing tape
    tape = TuringTape3D.load(TAPE_STATE_FILE)
    logger.info(f"Tape loaded: {tape.status()}")

    # 2. Initialize codebase R/W
    codebase = CodebaseRW()
    logger.info(f"Codebase R/W initialized. Extensions dir: {codebase.extensions_dir}")

    # 3. Initialize the brain
    brain = Brain(tape=tape, codebase=codebase)
    logger.info(f"Brain initialized with model: {OLLAMA_MODEL}")

    # 4. Create and run the Discord bot
    bot = TuringBot(tape=tape, brain=brain)
    logger.info("Starting Discord bot...")
    bot.run_bot()


if __name__ == "__main__":
    main()
