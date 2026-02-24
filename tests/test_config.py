"""
Tests for config.py module.
Covers: all config values, type correctness, and defaults.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestConfigDefaults:
    def test_discord_token_default(self):
        with patch("dotenv.load_dotenv"):
            with patch.dict(os.environ, {}, clear=True):
                import importlib
                import config
                importlib.reload(config)
                assert config.DISCORD_TOKEN == ""

    def test_channel_ids_default(self):
        with patch("dotenv.load_dotenv"):
            with patch.dict(os.environ, {}, clear=True):
                import importlib
                import config
                importlib.reload(config)
                assert config.LISTEN_CHANNEL_ID == 0
                assert config.THOUGHTS_CHANNEL_ID == 0

    def test_ollama_model_default(self):
        with patch("dotenv.load_dotenv"):
            with patch.dict(os.environ, {}, clear=True):
                import importlib
                import config
                importlib.reload(config)
                assert config.OLLAMA_MODEL == "qwen3:32b"

    def test_ollama_host_default(self):
        with patch("dotenv.load_dotenv"):
            with patch.dict(os.environ, {}, clear=True):
                import importlib
                import config
                importlib.reload(config)
                assert config.OLLAMA_HOST == "http://localhost:11434"

    def test_heartbeat_interval_default(self):
        with patch("dotenv.load_dotenv"):
            with patch.dict(os.environ, {}, clear=True):
                import importlib
                import config
                importlib.reload(config)
                assert config.HEARTBEAT_INTERVAL == 10.0

    def test_max_history_default(self):
        with patch("dotenv.load_dotenv"):
            with patch.dict(os.environ, {}, clear=True):
                import importlib
                import config
                importlib.reload(config)
                assert config.MAX_HISTORY_MESSAGES == 50


class TestConfigFromEnv:
    def test_reads_env_variables(self):
        env = {
            "DISCORD_TOKEN": "test_token_123",
            "LISTEN_CHANNEL_ID": "12345",
            "THOUGHTS_CHANNEL_ID": "67890",
            "OLLAMA_MODEL": "custom-model",
            "OLLAMA_HOST": "http://custom:1234",
            "HEARTBEAT_INTERVAL": "5.5",
            "MAX_HISTORY_MESSAGES": "100",
        }
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import config
            importlib.reload(config)
            assert config.DISCORD_TOKEN == "test_token_123"
            assert config.LISTEN_CHANNEL_ID == 12345
            assert config.THOUGHTS_CHANNEL_ID == 67890
            assert config.OLLAMA_MODEL == "custom-model"
            assert config.OLLAMA_HOST == "http://custom:1234"
            assert config.HEARTBEAT_INTERVAL == 5.5
            assert config.MAX_HISTORY_MESSAGES == 100


class TestConfigPaths:
    def test_project_root_is_path(self):
        import config
        assert isinstance(config.PROJECT_ROOT, Path)

    def test_extensions_dir_is_path(self):
        import config
        assert isinstance(config.EXTENSIONS_DIR, Path)

    def test_tape_state_file_is_path(self):
        import config
        assert isinstance(config.TAPE_STATE_FILE, Path)

    def test_extensions_dir_under_project_root(self):
        import config
        assert str(config.EXTENSIONS_DIR).startswith(str(config.PROJECT_ROOT))


class TestConfigCoreFiles:
    def test_core_files_is_frozenset(self):
        import config
        assert isinstance(config.CORE_FILES, frozenset)

    def test_core_files_contains_essentials(self):
        import config
        for name in ["main.py", "config.py", "bot.py", "brain.py",
                      "heartbeat.py", "turing_tape.py", "codebase_rw.py",
                      "prompt_builder.py", "sandbox.py", "requirements.txt"]:
            assert name in config.CORE_FILES

    def test_core_files_immutable(self):
        import config
        with pytest.raises(AttributeError):
            config.CORE_FILES.add("hack.py")
