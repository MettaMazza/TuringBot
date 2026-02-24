"""
TuringBot — Discord bot shell.

Connects to Discord, listens for messages in the listen channel,
streams thoughts to the thoughts channel, and runs the heartbeat.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import discord

from brain import Brain
from config import (
    DISCORD_TOKEN,
    LISTEN_CHANNEL_ID,
    THOUGHTS_CHANNEL_ID,
    TAPE_STATE_FILE,
)
from heartbeat import Heartbeat
from prompt_builder import (
    build_heartbeat_prompt,
    build_interrupt_prompt,
    build_system_prompt,
)
from turing_tape import TuringTape3D

logger = logging.getLogger("turingbot.bot")

DISCORD_MSG_LIMIT = 2000


def _chunk_message(text: str, limit: int = DISCORD_MSG_LIMIT) -> list[str]:
    """Split a long message into chunks that fit within Discord's limit."""
    if not text:
        return []
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at a newline
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


class TuringBot(discord.Client):
    """Discord client that runs the autonomous TuringBot."""

    def __init__(
        self,
        tape: TuringTape3D,
        brain: Brain,
        listen_channel_id: Optional[int] = None,
        thoughts_channel_id: Optional[int] = None,
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)

        self.tape = tape
        self.brain = brain
        self.listen_channel_id = listen_channel_id or LISTEN_CHANNEL_ID
        self.thoughts_channel_id = thoughts_channel_id or THOUGHTS_CHANNEL_ID
        self._token = token or DISCORD_TOKEN
        self.heartbeat_loop: Optional[Heartbeat] = None
        self._processing_message = False

    # ── Discord Events ───────────────────────────────────────────────

    async def on_ready(self) -> None:
        logger.info(f"Connected as {self.user} (ID: {self.user.id})")

        # Verify channels exist
        listen_ch = self.get_channel(self.listen_channel_id)
        thoughts_ch = self.get_channel(self.thoughts_channel_id)
        if not listen_ch:
            logger.error(f"Listen channel {self.listen_channel_id} not found!")
        if not thoughts_ch:
            logger.error(f"Thoughts channel {self.thoughts_channel_id} not found!")

        # Start heartbeat
        self.heartbeat_loop = Heartbeat(
            think_fn=self.brain.think,
            thought_callback=self._stream_thoughts,
            system_prompt_fn=build_system_prompt,
            heartbeat_prompt_fn=build_heartbeat_prompt,
            save_tape_fn=lambda: self.tape.save(TAPE_STATE_FILE),
        )
        self.heartbeat_loop.start()

        # Announce startup
        if thoughts_ch:
            await thoughts_ch.send(
                f"🧠 **TuringBot online.** Heartbeat started. "
                f"Tape: {self.tape.status()}"
            )

    async def on_message(self, message: discord.Message) -> None:
        # Ignore own messages
        if message.author == self.user:
            return

        # Only respond in the listen channel
        if message.channel.id != self.listen_channel_id:
            return

        # Ignore embed-only messages (link previews, bot cards, etc.)
        if not message.content and message.embeds:
            return

        logger.info(f"Message from {message.author}: {message.content}")

        # Pause heartbeat during response
        if self.heartbeat_loop and self.heartbeat_loop.is_running:
            self.heartbeat_loop.pause()

        self._processing_message = True
        try:
            # Build interrupt prompt
            system_prompt = build_system_prompt()
            interrupt_prompt = build_interrupt_prompt(
                username=str(message.author),
                message=message.content,
            )

            # Think and respond
            async with message.channel.typing():
                response_text, tool_log = await self.brain.think(
                    system_prompt, interrupt_prompt
                )

            # Stream tool usage to thoughts channel
            await self._stream_thoughts(
                f"[Responding to {message.author}]\n{response_text}",
                tool_log,
            )

            # Reply to the user (native Discord reply tags them automatically)
            if response_text:
                for i, chunk in enumerate(_chunk_message(response_text)):
                    if i == 0:
                        await message.reply(chunk)
                    else:
                        await message.channel.send(chunk)
            else:
                await message.reply("*(no response generated)*")

            # Save tape after interaction
            self.tape.save(TAPE_STATE_FILE)

        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            await message.reply(f"⚠️ Error: {e}")
        finally:
            self._processing_message = False
            # Resume heartbeat
            if self.heartbeat_loop and self.heartbeat_loop.is_running:
                self.heartbeat_loop.resume()

    # ── Thought Streaming ────────────────────────────────────────────

    async def _stream_thoughts(
        self, text: str, tool_log: List[Dict[str, Any]]
    ) -> None:
        """Send thoughts and tool usage to the thoughts channel."""
        thoughts_ch = self.get_channel(self.thoughts_channel_id)
        if not thoughts_ch:
            return

        # Format tool log
        if tool_log:
            tool_text = "**🔧 Tool Calls:**\n"
            for entry in tool_log:
                tool_text += f"• `{entry['tool']}({entry['args']})`\n"
                result_preview = str(entry["result"])[:300]
                tool_text += f"  → {result_preview}\n"
            for chunk in _chunk_message(tool_text):
                await thoughts_ch.send(chunk)

        # Send thought text
        if text:
            for chunk in _chunk_message(f"💭 {text}"):
                await thoughts_ch.send(chunk)

    # ── Run ──────────────────────────────────────────────────────────

    def run_bot(self) -> None:
        """Start the bot (blocking)."""
        if not self._token:
            raise ValueError(
                "No Discord token set! Add DISCORD_TOKEN to your .env file."
            )
        self.run(self._token)
