# LocalAI Platform

A complete local AI inference platform. Double-click to run — no terminal, no commands.

## Quick Start

**Windows:**
```
install.ps1    # one-time setup
start.bat      # launch
```

**Linux/Mac:**
```bash
./install.sh   # one-time setup
./start.sh     # launch
```

Then open: http://localhost:3000/api/localai/model-manager

## What It Does

- **Model Manager UI** — scan for GGUF models, load/eject with one click
- **Auto-optimization** — detects your GPU/RAM and computes optimal llama-server flags
- **MoE speed boost** — automatically applies `-ot ".ffn_.*_exps.=CPU"` for 4× speed on MoE models (Qwen, DeepSeek, etc.)
- **Chat interface** — full Open WebUI chat with tool calling
- **Agent tools** — 19 tools for file ops, web search, system management

## Architecture

```
User → http://localhost:3000 (Open WebUI)
         ↓
  /api/localai/* (our custom FastAPI routes)
         ↓
  llama-server on :8001 (managed subprocess)
```

## Configuration

Edit `config/settings.json`:
```json
{
  "llama_server_path": "/path/to/llama-server",
  "model_dirs": ["/path/to/models"],
  "port": 3000,
  "llama_port": 8001
}
```

## Build Windows .exe

```powershell
pip install pyinstaller
.\build_exe.ps1
```

Output: `dist/LocalAI/LocalAI.exe`

## Key llama-server Flags Applied Automatically

| Flag | Effect | When |
|------|--------|------|
| `-ot ".ffn_.*_exps.=CPU"` | MoE expert offload → 4× speed | MoE models |
| `--jinja` | Tool calling support | Always |
| `--flash-attn` | Flash attention | NVIDIA GPUs |
| `--cache-type-k q8_0` | KV cache compression | Always |
| `-ngl 999` | Max GPU layers | Always |

## File Structure

```
localai-platform/
├── start.py                    ← Main launcher
├── start.bat / start.sh        ← Platform launchers
├── extensions/
│   ├── model_manager/
│   │   ├── hardware.py         ← GPU/RAM detection
│   │   ├── gguf_inspector.py   ← Model metadata reader
│   │   ├── optimizer.py        ← Flag calculator
│   │   ├── server_process.py   ← llama-server manager
│   │   ├── model_scanner.py    ← GGUF file scanner
│   │   └── routes.py           ← FastAPI endpoints
│   └── tools/
│       └── tools.py            ← 19 agent tools
├── frontend/
│   ├── index.html              ← Model Manager UI
│   ├── styles.css              ← Dark theme
│   └── app.js                  ← Frontend logic
└── config/
    ├── settings.json           ← Runtime config
    └── system_prompt.txt       ← AI system prompt
```

## License

MIT
