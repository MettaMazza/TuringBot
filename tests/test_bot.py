"""
Comprehensive tests for TuringBot (Discord bot shell).
Covers: message chunking, event handling, thought streaming,
interrupt/resume flow, and error handling.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import List

import pytest

from bot import TuringBot, _chunk_message


class TestChunkMessage:
    def test_empty_string(self):
        assert _chunk_message("") == []

    def test_short_message(self):
        result = _chunk_message("hello")
        assert result == ["hello"]

    def test_exactly_at_limit(self):
        msg = "x" * 2000
        result = _chunk_message(msg, limit=2000)
        assert result == [msg]

    def test_over_limit_splits(self):
        msg = "x" * 3000
        result = _chunk_message(msg, limit=2000)
        assert len(result) == 2
        assert len(result[0]) <= 2000
        assert "".join(result) == msg

    def test_splits_at_newline(self):
        msg = "a" * 1000 + "\n" + "b" * 1500
        result = _chunk_message(msg, limit=2000)
        assert len(result) == 2
        assert result[0] == "a" * 1000
        assert result[1] == "b" * 1500

    def test_no_newline_splits_at_limit(self):
        msg = "x" * 5000
        result = _chunk_message(msg, limit=2000)
        assert all(len(c) <= 2000 for c in result)
        assert "".join(result) == msg

    def test_many_chunks(self):
        msg = ("line\n") * 2000
        result = _chunk_message(msg, limit=100)
        assert all(len(c) <= 100 for c in result)


class TestTuringBotInit:
    def test_init_attributes(self):
        tape = MagicMock()
        brain = MagicMock()
        with patch("bot.discord.Client.__init__", return_value=None):
            bot = TuringBot.__new__(TuringBot)
            bot.tape = tape
            bot.brain = brain
            bot.listen_channel_id = 123
            bot.thoughts_channel_id = 456
            bot._token = "tok"
            bot.heartbeat_loop = None
            bot._processing_message = False

        assert bot.listen_channel_id == 123
        assert bot.thoughts_channel_id == 456


class TestTuringBotOnReady:
    @pytest.mark.asyncio
    async def test_on_ready_starts_heartbeat(self):
        tape = MagicMock()
        tape.status.return_value = "Head: (0,0,0)"
        brain = MagicMock()

        with patch("bot.discord.Client.__init__", return_value=None):
            bot = TuringBot.__new__(TuringBot)
            bot.tape = tape
            bot.brain = brain
            bot.listen_channel_id = 111
            bot.thoughts_channel_id = 222
            bot._token = "tok"
            bot.heartbeat_loop = None
            bot._processing_message = False

        mock_user = MagicMock()
        mock_user.id = 9999
        type(bot).user = PropertyMock(return_value=mock_user)

        # Mock channels
        mock_thoughts_ch = AsyncMock()
        mock_listen_ch = MagicMock()

        def get_channel(cid):
            if cid == 222:
                return mock_thoughts_ch
            if cid == 111:
                return mock_listen_ch
            return None

        bot.get_channel = get_channel

        with patch("bot.Heartbeat") as MockHB:
            mock_hb_instance = MagicMock()
            mock_hb_instance.start.return_value = MagicMock()
            MockHB.return_value = mock_hb_instance

            await bot.on_ready()

            MockHB.assert_called_once()
            mock_hb_instance.start.assert_called_once()
            mock_thoughts_ch.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_ready_missing_channels(self):
        """Cover the logging paths when channels are not found."""
        tape = MagicMock()
        tape.status.return_value = "Head: (0,0,0)"
        brain = MagicMock()

        with patch("bot.discord.Client.__init__", return_value=None):
            bot = TuringBot.__new__(TuringBot)
            bot.tape = tape
            bot.brain = brain
            bot.listen_channel_id = 111
            bot.thoughts_channel_id = 222
            bot._token = "tok"
            bot.heartbeat_loop = None
            bot._processing_message = False

        mock_user = MagicMock()
        mock_user.id = 9999
        type(bot).user = PropertyMock(return_value=mock_user)

        # Return None for all channels
        bot.get_channel = lambda cid: None

        with patch("bot.Heartbeat") as MockHB:
            mock_hb_instance = MagicMock()
            mock_hb_instance.start.return_value = MagicMock()
            MockHB.return_value = mock_hb_instance

            await bot.on_ready()

            # Heartbeat should still start even if channels are missing
            mock_hb_instance.start.assert_called_once()


class TestTuringBotConstructor:
    def test_init_via_constructor(self):
        """Cover the __init__ method directly (lines 66-76)."""
        tape = MagicMock()
        brain = MagicMock()
        with patch("bot.discord.Client.__init__", return_value=None):
            bot = TuringBot(
                tape=tape,
                brain=brain,
                listen_channel_id=123,
                thoughts_channel_id=456,
                token="test_token",
            )
        assert bot.tape is tape
        assert bot.brain is brain
        assert bot.listen_channel_id == 123
        assert bot.thoughts_channel_id == 456
        assert bot._token == "test_token"
        assert bot.heartbeat_loop is None
        assert bot._processing_message is False


class TestTuringBotOnMessage:
    @pytest.fixture
    def bot_fixture(self):
        tape = MagicMock()
        tape.save = MagicMock()
        brain = MagicMock()
        brain.think = AsyncMock(return_value=("Hello human!", []))

        with patch("bot.discord.Client.__init__", return_value=None):
            bot = TuringBot.__new__(TuringBot)
            bot.tape = tape
            bot.brain = brain
            bot.listen_channel_id = 111
            bot.thoughts_channel_id = 222
            bot._token = "tok"
            bot.heartbeat_loop = MagicMock()
            bot.heartbeat_loop.is_running = True
            bot._processing_message = False

        mock_user = MagicMock()
        mock_user.id = 9999
        type(bot).user = PropertyMock(return_value=mock_user)

        mock_thoughts_ch = AsyncMock()
        def get_channel(cid):
            if cid == 222:
                return mock_thoughts_ch
            return None
        bot.get_channel = get_channel

        return bot, tape, brain, mock_thoughts_ch

    @pytest.mark.asyncio
    async def test_ignores_own_messages(self, bot_fixture):
        bot, tape, brain, _ = bot_fixture
        msg = MagicMock()
        msg.author = bot.user
        msg.channel.id = 111

        await bot.on_message(msg)
        brain.think.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_other_channels(self, bot_fixture):
        bot, tape, brain, _ = bot_fixture
        msg = MagicMock()
        msg.author = MagicMock()
        msg.channel.id = 999

        await bot.on_message(msg)
        brain.think.assert_not_called()

    def _make_msg(self, content="hi", author_name="User"):
        """Helper to create a properly mocked Discord message."""
        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.__str__ = lambda s: author_name
        msg.content = content
        msg.reply = AsyncMock()

        # Build channel mock with proper typing() context manager
        channel = MagicMock()
        channel.id = 111
        channel.send = AsyncMock()

        # typing() must return an object with __aenter__/__aexit__
        typing_cm = MagicMock()
        typing_cm.__aenter__ = AsyncMock(return_value=None)
        typing_cm.__aexit__ = AsyncMock(return_value=False)
        channel.typing.return_value = typing_cm

        msg.channel = channel
        return msg

    @pytest.mark.asyncio
    async def test_responds_in_listen_channel(self, bot_fixture):
        bot, tape, brain, _ = bot_fixture
        msg = self._make_msg("Hello bot", "TestUser")

        with patch("bot.build_system_prompt", return_value="sys"):
            with patch("bot.build_interrupt_prompt", return_value="interrupt"):
                await bot.on_message(msg)

        brain.think.assert_called_once()
        msg.reply.assert_called_with("Hello human!")

    @pytest.mark.asyncio
    async def test_pauses_and_resumes_heartbeat(self, bot_fixture):
        bot, tape, brain, _ = bot_fixture
        msg = self._make_msg()

        with patch("bot.build_system_prompt", return_value="sys"):
            with patch("bot.build_interrupt_prompt", return_value="int"):
                await bot.on_message(msg)

        bot.heartbeat_loop.pause.assert_called_once()
        bot.heartbeat_loop.resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_saves_tape_after_message(self, bot_fixture):
        bot, tape, brain, _ = bot_fixture
        msg = self._make_msg()

        with patch("bot.build_system_prompt", return_value="sys"):
            with patch("bot.build_interrupt_prompt", return_value="int"):
                await bot.on_message(msg)

        tape.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_response_sends_fallback(self, bot_fixture):
        bot, tape, brain, _ = bot_fixture
        brain.think = AsyncMock(return_value=("", []))
        msg = self._make_msg()

        with patch("bot.build_system_prompt", return_value="sys"):
            with patch("bot.build_interrupt_prompt", return_value="int"):
                await bot.on_message(msg)

        msg.reply.assert_called_with("*(no response generated)*")

    @pytest.mark.asyncio
    async def test_exception_sends_error_message(self, bot_fixture):
        bot, tape, brain, _ = bot_fixture
        brain.think = AsyncMock(side_effect=RuntimeError("oops"))
        msg = self._make_msg()

        with patch("bot.build_system_prompt", return_value="sys"):
            with patch("bot.build_interrupt_prompt", return_value="int"):
                await bot.on_message(msg)

        error_call = msg.reply.call_args
        assert "Error" in str(error_call)

    @pytest.mark.asyncio
    async def test_resumes_heartbeat_even_on_error(self, bot_fixture):
        bot, tape, brain, _ = bot_fixture
        brain.think = AsyncMock(side_effect=RuntimeError("oops"))
        msg = self._make_msg()

        with patch("bot.build_system_prompt", return_value="sys"):
            with patch("bot.build_interrupt_prompt", return_value="int"):
                await bot.on_message(msg)

        bot.heartbeat_loop.resume.assert_called_once()


class TestTuringBotStreamThoughts:
    @pytest.mark.asyncio
    async def test_streams_tool_log(self):
        with patch("bot.discord.Client.__init__", return_value=None):
            bot = TuringBot.__new__(TuringBot)
            bot.thoughts_channel_id = 222

        mock_ch = AsyncMock()
        bot.get_channel = lambda cid: mock_ch if cid == 222 else None

        tool_log = [{"tool": "tape_read", "args": {}, "result": "empty"}]
        await bot._stream_thoughts("thinking...", tool_log)

        # Should have sent tool log and thought text
        assert mock_ch.send.call_count >= 2

    @pytest.mark.asyncio
    async def test_streams_text_only(self):
        with patch("bot.discord.Client.__init__", return_value=None):
            bot = TuringBot.__new__(TuringBot)
            bot.thoughts_channel_id = 222

        mock_ch = AsyncMock()
        bot.get_channel = lambda cid: mock_ch if cid == 222 else None

        await bot._stream_thoughts("just thinking", [])
        mock_ch.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_channel_no_error(self):
        with patch("bot.discord.Client.__init__", return_value=None):
            bot = TuringBot.__new__(TuringBot)
            bot.thoughts_channel_id = 222

        bot.get_channel = lambda cid: None

        # Should not raise
        await bot._stream_thoughts("test", [])

    @pytest.mark.asyncio
    async def test_empty_text_no_send(self):
        with patch("bot.discord.Client.__init__", return_value=None):
            bot = TuringBot.__new__(TuringBot)
            bot.thoughts_channel_id = 222

        mock_ch = AsyncMock()
        bot.get_channel = lambda cid: mock_ch if cid == 222 else None

        await bot._stream_thoughts("", [])
        mock_ch.send.assert_not_called()


class TestTuringBotRunBot:
    def test_run_bot_no_token_raises(self):
        with patch("bot.discord.Client.__init__", return_value=None):
            bot = TuringBot.__new__(TuringBot)
            bot._token = ""

        with pytest.raises(ValueError, match="No Discord token"):
            bot.run_bot()

    def test_run_bot_calls_run(self):
        with patch("bot.discord.Client.__init__", return_value=None):
            bot = TuringBot.__new__(TuringBot)
            bot._token = "test_token"
            bot.run = MagicMock()

        bot.run_bot()
        bot.run.assert_called_once_with("test_token")
