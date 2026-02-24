# 🧠 TuringBot

An autonomous Discord bot powered by [Ollama](https://ollama.ai) with a 3D Turing-complete tape, self-modifying codebase, and continuous heartbeat for independent computation.

## Features

- **3D Turing Tape** — Infinite sparse 3D grid for arbitrary, persistent working memory
- **Autonomous Heartbeat** — Continuous cognition loop that thinks independently between user messages
- **Self-Modifying Code** — Bot can write, execute, and hot-load Python extensions in a sandbox
- **Sandboxed Execution** — Extensions are blocked from accessing filesystem, network, system commands, and dangerous APIs
- **Core Protection** — Essential bot files are immutable; only the `extensions/` directory is writable
- **Self-Describing Prompt** — System prompt dynamically introspects the entire codebase
- **Thought Transparency** — All internal reasoning and tool usage streams to a dedicated channel

## Architecture

```
main.py          → Entry point, wires everything together
config.py        → Environment config + core file protection list
turing_tape.py   → 3D sparse tape with read/write/move/jump/scan
codebase_rw.py   → Self-modification with core protection + sandbox
sandbox.py       → Import/builtin restrictions for extensions
brain.py         → Ollama chat wrapper with 11 tool definitions
heartbeat.py     → Async autonomous cognition loop
bot.py           → Discord client with dual-channel support
prompt_builder.py → Dynamic system prompt from live codebase
```

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed and running
- A Discord bot application with a token

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/TuringBot.git
cd TuringBot

# Install dependencies
pip install -r requirements.txt

# Configure your bot
cp .env.example .env
# Edit .env with your Discord token and channel IDs
```

### Configuration

Edit `.env` with your values:

| Variable | Description |
|----------|-------------|
| `DISCORD_TOKEN` | Your Discord bot token |
| `LISTEN_CHANNEL_ID` | Channel where users interact with the bot |
| `THOUGHTS_CHANNEL_ID` | Channel where the bot streams all internal thoughts |
| `OLLAMA_MODEL` | Ollama model to use (default: `qwen3:32b`) |
| `OLLAMA_HOST` | Ollama API host (default: `http://localhost:11434`) |
| `HEARTBEAT_INTERVAL` | Seconds between autonomous ticks (default: `10.0`) |

### Run

```bash
python main.py
```

## Security & Sandbox

TuringBot can modify its own code in the `extensions/` directory, but all extensions run in a **sandboxed environment** that:

- ✅ Allows: `math`, `json`, `re`, `datetime`, `collections`, `itertools`, `random`, and more
- 🚫 Blocks: `os`, `subprocess`, `socket`, `shutil`, `pickle`, `ctypes`, `sys`, and all network/filesystem modules
- 🚫 Blocks builtins: `exec`, `eval`, `open`, `__import__`, `breakpoint`, `exit`
- 🔒 Core files are immutable — the bot cannot modify its own brain, config, or sandbox

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing
```

264 tests, 99% coverage, zero stubs or placeholders.

## How It Works

1. **Heartbeat Loop** — Every N seconds, the bot thinks autonomously using the Ollama model
2. **User Messages** — When a user sends a message, the heartbeat pauses, the bot processes the message, and resumes
3. **Tool Calls** — The model can read/write the tape, read files, write extensions, and hot-load code
4. **Thought Streaming** — All internal reasoning, tool usage, and decisions stream to the thoughts channel
5. **Persistence** — The 3D tape saves to `tape_state.json` after every interaction

## Author

**Maria Smith**
- Discord: `metta_mazza`
- Email: Maria.smith.xo@outlook.com

## License

MIT — see [LICENSE](LICENSE) for details.
