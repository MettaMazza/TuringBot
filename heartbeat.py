"""
Heartbeat — Autonomous cognition loop.

Runs as an asyncio task inside the Discord bot. On each tick, it prompts
the brain to think autonomously. When a user message arrives, the heartbeat
pauses via an asyncio.Event, processes the interrupt, and resumes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from config import HEARTBEAT_INTERVAL

logger = logging.getLogger("turingbot.heartbeat")


class Heartbeat:
    """
    Autonomous cognition loop that calls the brain on a regular interval.

    The heartbeat can be paused (for user interrupts) and resumed.
    A callback is invoked with each thought/tool-log for streaming to Discord.
    """

    def __init__(
        self,
        think_fn: Callable[..., Coroutine],
        thought_callback: Callable[[str, List[Dict[str, Any]]], Coroutine],
        system_prompt_fn: Callable[[], str],
        heartbeat_prompt_fn: Callable[[], str],
        save_tape_fn: Callable[[], None],
        interval: Optional[float] = None,
    ) -> None:
        """
        Args:
            think_fn: async function(system_prompt, user_message) -> (text, tool_log)
            thought_callback: async function(text, tool_log) to stream thoughts
            system_prompt_fn: callable that returns the current system prompt
            heartbeat_prompt_fn: callable that returns the heartbeat tick prompt
            save_tape_fn: callable to persist tape state after each tick
            interval: seconds between heartbeat ticks
        """
        self.think_fn = think_fn
        self.thought_callback = thought_callback
        self.system_prompt_fn = system_prompt_fn
        self.heartbeat_prompt_fn = heartbeat_prompt_fn
        self.save_tape_fn = save_tape_fn
        self.interval = interval if interval is not None else HEARTBEAT_INTERVAL

        self._running = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Start unpaused
        self._task: Optional[asyncio.Task] = None
        self._tick_count = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def pause(self) -> None:
        """Pause the heartbeat (e.g., during user message processing)."""
        logger.info("Heartbeat paused")
        self._pause_event.clear()

    def resume(self) -> None:
        """Resume the heartbeat after a pause."""
        logger.info("Heartbeat resumed")
        self._pause_event.set()

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> asyncio.Task:
        """Start the heartbeat loop as an asyncio task."""
        if self._task and not self._task.done():
            raise RuntimeError("Heartbeat is already running")
        self._running = True
        self._task = asyncio.ensure_future(self._loop())
        return self._task

    def stop(self) -> None:
        """Stop the heartbeat loop."""
        self._running = False
        self._pause_event.set()  # Unblock if paused so loop can exit
        if self._task and not self._task.done():
            self._task.cancel()

    async def _loop(self) -> None:
        """Main heartbeat loop."""
        logger.info("Heartbeat started")
        try:
            while self._running:
                # Wait if paused
                await self._pause_event.wait()
                if not self._running:
                    break

                try:
                    self._tick_count += 1
                    logger.info(f"Heartbeat tick #{self._tick_count}")

                    system_prompt = self.system_prompt_fn()
                    heartbeat_prompt = self.heartbeat_prompt_fn()

                    text, tool_log = await self.think_fn(
                        system_prompt, heartbeat_prompt
                    )

                    await self.thought_callback(text, tool_log)

                    # Persist tape state after each tick
                    self.save_tape_fn()

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception(f"Heartbeat tick error: {e}")

                # Sleep between ticks (interruptible)
                if self._running:
                    await asyncio.sleep(self.interval)

        except asyncio.CancelledError:
            logger.info("Heartbeat cancelled")
        finally:
            self._running = False
            logger.info("Heartbeat stopped")
