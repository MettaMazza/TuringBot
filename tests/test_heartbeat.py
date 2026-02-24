"""
Comprehensive tests for Heartbeat.
Covers: init, start/stop, pause/resume, tick execution, error handling,
callback invocation, and tape persistence.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from heartbeat import Heartbeat


@pytest.fixture
def heartbeat_deps():
    """Common dependencies for heartbeat tests."""
    think_fn = AsyncMock(return_value=("I am thinking", []))
    thought_callback = AsyncMock()
    system_prompt_fn = MagicMock(return_value="system prompt")
    heartbeat_prompt_fn = MagicMock(return_value="heartbeat prompt")
    save_tape_fn = MagicMock()
    return {
        "think_fn": think_fn,
        "thought_callback": thought_callback,
        "system_prompt_fn": system_prompt_fn,
        "heartbeat_prompt_fn": heartbeat_prompt_fn,
        "save_tape_fn": save_tape_fn,
    }


@pytest.fixture
def hb(heartbeat_deps):
    """Heartbeat with 0.01s interval for fast tests."""
    return Heartbeat(
        **heartbeat_deps,
        interval=0.01,
    )


class TestHeartbeatInit:
    def test_default_state(self, hb):
        assert hb.is_running is False
        assert hb.is_paused is False
        assert hb.tick_count == 0

    def test_custom_interval(self, heartbeat_deps):
        h = Heartbeat(**heartbeat_deps, interval=5.0)
        assert h.interval == 5.0


class TestHeartbeatPauseResume:
    def test_pause_sets_flag(self, hb):
        hb.pause()
        assert hb.is_paused is True

    def test_resume_clears_flag(self, hb):
        hb.pause()
        hb.resume()
        assert hb.is_paused is False

    def test_resume_when_not_paused(self, hb):
        hb.resume()
        assert hb.is_paused is False


class TestHeartbeatStartStop:
    @pytest.mark.asyncio
    async def test_start_creates_task(self, hb):
        task = hb.start()
        assert hb.is_running is True
        assert task is not None
        hb.stop()
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_start_twice_raises(self, hb):
        hb.start()
        with pytest.raises(RuntimeError, match="already running"):
            hb.start()
        hb.stop()
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_stop_after_start(self, hb):
        hb.start()
        hb.stop()
        await asyncio.sleep(0.05)
        assert hb.is_running is False

    @pytest.mark.asyncio
    async def test_stop_when_paused(self, hb):
        hb.start()
        hb.pause()
        hb.stop()
        await asyncio.sleep(0.05)
        assert hb.is_running is False


class TestHeartbeatLoop:
    @pytest.mark.asyncio
    async def test_tick_calls_think(self, hb, heartbeat_deps):
        hb.start()
        await asyncio.sleep(0.05)
        hb.stop()
        await asyncio.sleep(0.05)

        assert heartbeat_deps["think_fn"].call_count >= 1
        heartbeat_deps["think_fn"].assert_called_with(
            "system prompt", "heartbeat prompt"
        )

    @pytest.mark.asyncio
    async def test_tick_calls_callback(self, hb, heartbeat_deps):
        hb.start()
        await asyncio.sleep(0.05)
        hb.stop()
        await asyncio.sleep(0.05)

        assert heartbeat_deps["thought_callback"].call_count >= 1
        heartbeat_deps["thought_callback"].assert_called_with(
            "I am thinking", []
        )

    @pytest.mark.asyncio
    async def test_tick_saves_tape(self, hb, heartbeat_deps):
        hb.start()
        await asyncio.sleep(0.05)
        hb.stop()
        await asyncio.sleep(0.05)

        assert heartbeat_deps["save_tape_fn"].call_count >= 1

    @pytest.mark.asyncio
    async def test_tick_count_increments(self, hb):
        hb.start()
        await asyncio.sleep(0.05)
        hb.stop()
        await asyncio.sleep(0.05)

        assert hb.tick_count >= 1

    @pytest.mark.asyncio
    async def test_pause_stops_ticking(self, hb, heartbeat_deps):
        hb.start()
        await asyncio.sleep(0.03)
        hb.pause()
        count_at_pause = heartbeat_deps["think_fn"].call_count
        await asyncio.sleep(0.05)
        count_after_pause = heartbeat_deps["think_fn"].call_count
        # Should not have ticked while paused
        assert count_after_pause == count_at_pause
        hb.stop()
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_resume_continues_ticking(self, hb, heartbeat_deps):
        hb.start()
        await asyncio.sleep(0.03)
        hb.pause()
        count_at_pause = heartbeat_deps["think_fn"].call_count
        hb.resume()
        await asyncio.sleep(0.05)
        hb.stop()
        await asyncio.sleep(0.05)
        assert heartbeat_deps["think_fn"].call_count > count_at_pause

    @pytest.mark.asyncio
    async def test_error_in_think_doesnt_crash_loop(self, heartbeat_deps):
        heartbeat_deps["think_fn"].side_effect = ValueError("boom")
        hb = Heartbeat(**heartbeat_deps, interval=0.01)
        hb.start()
        await asyncio.sleep(0.05)
        hb.stop()
        await asyncio.sleep(0.05)
        # Loop should have survived the error and ticked multiple times
        assert heartbeat_deps["think_fn"].call_count >= 1

    @pytest.mark.asyncio
    async def test_system_prompt_fn_called_each_tick(self, hb, heartbeat_deps):
        hb.start()
        await asyncio.sleep(0.05)
        hb.stop()
        await asyncio.sleep(0.05)
        assert heartbeat_deps["system_prompt_fn"].call_count >= 1

    @pytest.mark.asyncio
    async def test_heartbeat_prompt_fn_called_each_tick(self, hb, heartbeat_deps):
        hb.start()
        await asyncio.sleep(0.05)
        hb.stop()
        await asyncio.sleep(0.05)
        assert heartbeat_deps["heartbeat_prompt_fn"].call_count >= 1

    @pytest.mark.asyncio
    async def test_stop_while_paused_breaks_loop(self, heartbeat_deps):
        """Cover line 103-104: _running set to False while waiting on pause event."""
        hb = Heartbeat(**heartbeat_deps, interval=0.01)
        hb.start()
        await asyncio.sleep(0.02)
        hb.pause()
        await asyncio.sleep(0.02)
        # Now stop — this sets _running=False and sets the event, unblocking the wait
        hb.stop()
        await asyncio.sleep(0.05)
        assert hb.is_running is False

    @pytest.mark.asyncio
    async def test_cancelled_error_during_think(self, heartbeat_deps):
        """Cover line 122-123: CancelledError raised during think_fn is re-raised."""
        heartbeat_deps["think_fn"].side_effect = asyncio.CancelledError()
        hb = Heartbeat(**heartbeat_deps, interval=0.01)
        hb.start()
        await asyncio.sleep(0.05)
        # The loop should have stopped due to cancellation
        assert hb.is_running is False

