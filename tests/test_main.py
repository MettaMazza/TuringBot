"""
Comprehensive tests for main.py entry point.
Covers: main() function wiring and startup sequence.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestMain:
    @patch("main.TuringBot")
    @patch("main.Brain")
    @patch("main.CodebaseRW")
    @patch("main.TuringTape3D")
    def test_main_wires_subsystems(self, MockTape, MockCodebase, MockBrain, MockBot):
        """Test that main() creates all subsystems and starts the bot."""
        mock_tape_instance = MagicMock()
        mock_tape_instance.status.return_value = "Head: (0,0,0)"
        MockTape.load.return_value = mock_tape_instance

        mock_codebase_instance = MagicMock()
        mock_codebase_instance.extensions_dir = "/fake/extensions"
        MockCodebase.return_value = mock_codebase_instance

        mock_brain_instance = MagicMock()
        MockBrain.return_value = mock_brain_instance

        mock_bot_instance = MagicMock()
        MockBot.return_value = mock_bot_instance

        from main import main
        main()

        # Verify subsystems were created
        MockTape.load.assert_called_once()
        MockCodebase.assert_called_once()
        MockBrain.assert_called_once_with(
            tape=mock_tape_instance,
            codebase=mock_codebase_instance,
        )
        MockBot.assert_called_once_with(
            tape=mock_tape_instance,
            brain=mock_brain_instance,
        )
        mock_bot_instance.run_bot.assert_called_once()

    @patch("main.TuringBot")
    @patch("main.Brain")
    @patch("main.CodebaseRW")
    @patch("main.TuringTape3D")
    def test_main_loads_tape_from_config_path(self, MockTape, MockCodebase, MockBrain, MockBot):
        """Test that main loads tape from the configured path."""
        mock_tape = MagicMock()
        mock_tape.status.return_value = "ok"
        MockTape.load.return_value = mock_tape
        MockCodebase.return_value = MagicMock(extensions_dir="/e")
        MockBrain.return_value = MagicMock()
        MockBot.return_value = MagicMock()

        from main import main, TAPE_STATE_FILE
        main()

        MockTape.load.assert_called_once_with(TAPE_STATE_FILE)
