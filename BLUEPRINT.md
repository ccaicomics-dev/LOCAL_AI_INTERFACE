# LocalAI Platform — Full Build Blueprint v2
> A spec document for Claude Code to build a complete local AI platform.
> Double-click LocalAI.exe or start.bat — browser opens, everything runs. No terminal ever.

---

## 1. Core Architecture Change From v1

**v1 (old):** User runs `launcher.py` → starts llama-server → starts Open WebUI separately.

**v2 (this spec):** Open WebUI IS the platform. Everything lives inside it.
The user double-clicks **LocalAI.exe**. A browser opens automatically.
They see the Model Manager, click **Load** — llama-server starts in the
background. Click **Eject** — it stops. A tray icon lets them quit.
No terminal. No commands. No setup after install.

```
User opens http://localhost:3000
         ↓
┌────────────────────────────────────────────────────┐
│              LocalAI Platform UI                   │
│         (Open WebUI — heavily customized)          │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  🧠 Model Manager Panel (NEW - built by us)  │  │
│  │                                              │  │
│  │  [Scan Models]  Found 3 models:              │  │
│  │                                              │  │
│  │  ● Qwen3.5-122B-A10B-Q4  73GB               │  │
│  │    Status: ⚫ Not loaded                     │  │
│  │    [  Load ▶  ]  [Settings ⚙]               │  │
│  │                                              │  │
│  │  ● Qwen3.5-35B-A3B-Q4   22GB               │  │
│  │    Status: 🟢 Running  20.3 t/s             │  │
│  │    Context: 65536/131072 tokens used         │  │
│  │    [  Eject ⏏  ]  [Settings ⚙]              │  │
│  │                                              │  │
│  │  ● DeepSeek-R1-32B-Q6   24GB               │  │
│  │    Status: ⚫ Not loaded                     │  │
│  │    [  Load ▶  ]  [Settings ⚙]               │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  [Chat]  [Documents]  [Model Manager]  [Tools]    │
└────────────────────────────────────────────────────┘
         ↓  when Load is clicked
┌────────────────────────────────────────────────────┐
│  llama-server (subprocess, hidden from user)       │
│  Auto-computed flags based on hardware + model     │
│  --jinja (tool calling) -ot (MoE speed)           │
└────────────────────────────────────────────────────┘
```

---

## 2. The Three Source Projects

### 2.1 Open WebUI
- **Repo:** https://github.com/open-webui/open-webui
- **License:** MIT
- **Role:** The entire frontend + API layer. We extend it, not replace it.
- **Key extension points:**
  - **Tools** — Python functions the AI can call as agent tools
  - **Custom routes** — we add FastAPI routes to Open WebUI's backend
  - **Frontend** — SvelteKit, we add new Svelte components/pages
- **Install:** `pip install open-webui`
- **Start:** `python -m open_webui serve --port 3000`
- **Key env vars:**
  ```
  WEBUI_AUTH=False
  OPENAI_API_BASE_URL=http://localhost:8001/v1
  OPENAI_API_KEY=none
  DATA_DIR=./config/data
  WEBUI_NAME=LocalAI Platform
  PORT=3000
  ```
- **What Open WebUI already provides (no custom code needed):**

  | Endpoint | What it does |
  |---|---|
  | `POST /api/chat/completions` | OpenAI-compatible chat, streaming, tool calling ✅ |
  | `GET  /api/models` | List all configured models ✅ |
  | `GET  /v1/models` | Same, OpenAI-format ✅ |
  | `POST /api/v1/files/` | Upload documents for RAG ✅ |
  | `POST /api/v1/knowledge/` | Group docs into knowledge collections ✅ |
  | `POST /api/v1/messages` | Anthropic-compatible endpoint ✅ |

- **What Open WebUI does NOT have (we must build these):**

  | Endpoint | What it does |
  |---|---|
  | `POST /api/localai/load` | Start llama-server with a model ❌ build it |
  | `POST /api/localai/eject` | Stop llama-server, free VRAM ❌ build it |
  | `GET  /api/localai/status` | Running? Speed? VRAM used? ❌ build it |
  | `GET  /api/localai/models` | Scan disk for GGUF files ❌ build it |
  | `GET  /api/localai/hardware` | GPU/RAM/CPU info ❌ build it |
  | `POST /api/localai/optimize` | Compute optimal flags ❌ build it |

  Note: Open WebUI assumes the backend (llama-server) is already running.
  It has no concept of starting or stopping it. That is the gap we fill.

### 2.2 llama.cpp (llama-server)
- **Repo:** https://github.com/ggml-org/llama.cpp
- **License:** MIT
- **Role:** Inference engine, managed as a background subprocess by us
- **Binary:** `llama-server.exe` (Windows) / `llama-server` (Linux/Mac)
- **Download:** https://github.com/ggml-org/llama.cpp/releases
  - RTX 5090 (Blackwell SM120): use `llama-*-bin-win-cuda-cu12.8-x64.zip`
- **Critical flags:**

  | Flag | Effect | When |
  |---|---|---|
  | `--model path` | GGUF file (first shard if split) | Always |
  | `-ngl 999` | Max GPU layers | Always |
  | `--jinja` | OpenAI-format tool calling ✅ | Always |
  | `-ot ".ffn_.*_exps.=CPU"` | MoE expert offload → 4x speed | MoE models |
  | `--ctx-size N` | Context window | Computed from VRAM |
  | `--threads N` | Physical CPU cores | Always |
  | `-b 4096 -ub 4096` | Batch size for MoE hybrid | MoE models |
  | `--flash-attn` | Flash attention | NVIDIA only |
  | `--cache-type-k q8_0` | KV cache compression | Always |
  | `--cache-type-v q8_0` | KV cache compression | Always |
  | `--host 0.0.0.0` | Listen on all interfaces | Always |
  | `--port 8001` | Port | Always |

- **Endpoints used:**
  ```
  GET  http://localhost:8001/health
  GET  http://localhost:8001/v1/models
  POST http://localhost:8001/v1/chat/completions
  ```

- **Why MoE offload matters:**
  ```
  Qwen 122B-A10B = 73GB total
    ~18GB = attention layers (need GPU speed) → VRAM
    ~55GB = dormant expert weights → RAM
  
  Without -ot: LM Studio 5 t/s (VRAM saturated)
  With    -ot: Our platform 20 t/s (experts in RAM, attention on GPU)
  
  This one flag is the entire reason for the 4x speed difference.
  ```

### 2.3 Claw-Code
- **Repo:** https://github.com/ultraworkers/claw-code
- **Parity:** https://github.com/ultraworkers/claw-code-parity
- **License:** Open source
- **Role:** Tool concepts. We reimplement them as Open WebUI Python Tools.
- **Full tool list from claw-code:**
  ```
  BashTool, FileReadTool, FileWriteTool, FileEditTool,
  GlobTool, GrepTool, WebSearchTool, WebFetchTool,
  AgentTool, TodoWriteTool, AskUserQuestionTool,
  ScheduleCronTool, LSPTool, ToolSearchTool,
  SkillTool, MCPTool, ConfigTool, NotebookEditTool
  ```

---

## 3. Complete API Surface for Your App

This is the full set of endpoints your Python app (or any client) can call
once the platform is running. Clearly separated into what Open WebUI gives
you for free vs what we build on top.

### 3a. Already Built Into Open WebUI ✅

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/api",
    api_key="none"   # auth disabled for personal use
)

# List all available models
models = client.models.list()

# Chat completion with tool calling
response = client.chat.completions.create(
    model="Qwen3.5-122B-A10B",
    messages=[{"role": "user", "content": "Hello"}],
    tools=your_tools,
    tool_choice="auto",
    stream=True      # streaming works too
)

# Upload a document for RAG
import requests
r = requests.post(
    "http://localhost:3000/api/v1/files/",
    files={"file": open("doc.pdf", "rb")}
)
file_id = r.json()["id"]

# Chat with that document in context
response = client.chat.completions.create(
    model="Qwen3.5-122B-A10B",
    messages=[{"role": "user", "content": "Summarize this doc"}],
    extra_body={"files": [{"type": "file", "id": file_id}]}
)
```

### 3b. Custom Routes We Build ❌ → ✅

```python
import requests

BASE = "http://localhost:3000"

# Scan disk for available GGUF models
models = requests.get(f"{BASE}/api/localai/models").json()
# Returns: [{ name, path, size_gb, is_moe, fits_in_vram, estimated_tps }]

# Load a model (SSE stream of progress events)
with requests.post(
    f"{BASE}/api/localai/load",
    json={"model_path": "C:/models/Qwen3.5-122B.gguf"},
    stream=True
) as r:
    for line in r.iter_lines():
        event = json.loads(line)
        # {"step": "inspecting",  "message": "Reading model..."}
        # {"step": "optimizing",  "message": "Computing flags..."}
        # {"step": "launching",   "message": "Starting server..."}
        # {"step": "loading",     "message": "Loading weights...", "progress": 65}
        # {"step": "ready",       "message": "Done!", "speed_tps": 20.3}
        # {"step": "error",       "message": "CUDA OOM"}
        print(event)

# Eject current model
result = requests.post(f"{BASE}/api/localai/eject").json()
# Returns: { success: true, freed_vram_gb: 28.4 }

# Get server status
status = requests.get(f"{BASE}/api/localai/status").json()
# Returns: { running: true, model_name, speed_tps, vram_used_mb,
#            context_used, context_max, uptime_seconds }

# Get hardware info
hw = requests.get(f"{BASE}/api/localai/hardware").json()
# Returns: { gpu_name, gpu_vram_mb, gpu_vram_free_mb, ram_mb, cpu_cores }

# Compute optimal flags without loading
config = requests.post(
    f"{BASE}/api/localai/optimize",
    json={"model_path": "C:/models/Qwen3.5-122B.gguf"}
).json()
# Returns: { flags, ctx_size, threads, is_moe, command_preview }
```

### 3c. Summary Table

| Endpoint | Source | Purpose |
|---|---|---|
| `POST /api/chat/completions` | Open WebUI ✅ | Chat, streaming, tool calling |
| `GET  /api/models` | Open WebUI ✅ | List configured models |
| `GET  /v1/models` | Open WebUI ✅ | OpenAI-format model list |
| `POST /api/v1/files/` | Open WebUI ✅ | Upload docs for RAG |
| `POST /api/v1/knowledge/` | Open WebUI ✅ | Knowledge collections |
| `POST /api/localai/load` | **We build** ❌ | Start llama-server + model |
| `POST /api/localai/eject` | **We build** ❌ | Stop server, free VRAM |
| `GET  /api/localai/status` | **We build** ❌ | Running status + speed |
| `GET  /api/localai/models` | **We build** ❌ | Scan disk for GGUFs |
| `GET  /api/localai/hardware` | **We build** ❌ | GPU/RAM/CPU info |
| `POST /api/localai/optimize` | **We build** ❌ | Compute optimal flags |

---

## 4. Full File Structure

```
localai-platform/
│
├── LocalAI.exe                       ← Windows double-click launcher (built from start.py)
├── start.bat                         ← fallback if .exe not available
├── start.py                          ← source for the .exe (PyInstaller)
│
├── build_exe.ps1                     ← builds LocalAI.exe from start.py (PyInstaller)
├── install.ps1                       ← Windows one-click setup (deps + build .exe)
├── install.sh                        ← Linux/Mac installer
├── requirements.txt
│
├── extensions/
│   ├── model_manager/
│   │   ├── __init__.py
│   │   ├── routes.py                 ← FastAPI routes (/api/localai/*)
│   │   ├── hardware.py               ← GPU/RAM/CPU detection
│   │   ├── gguf_inspector.py         ← GGUF metadata reader
│   │   ├── optimizer.py              ← optimal flags calculator
│   │   ├── server_process.py         ← llama-server lifecycle manager
│   │   └── model_scanner.py          ← find GGUFs on disk
│   │
│   └── tools/
│       └── tools.py                  ← all 19 Open WebUI agent tools
│
├── frontend/
│   ├── ModelManager.svelte           ← full model manager page
│   ├── ModelCard.svelte              ← per-model card component
│   ├── LoadProgress.svelte           ← loading overlay with progress
│   ├── HardwareStats.svelte          ← GPU/RAM bar charts
│   └── SettingsPanel.svelte          ← per-model settings + command preview
│
└── config/
    ├── settings.json                 ← auto-generated on first run
    ├── system_prompt.txt             ← AI behavior
    └── data/                         ← Open WebUI database + uploads
```

---

## 5. Launch Files — How the User Starts the Platform

### 4a. LocalAI.exe (Primary — Windows double-click)
Built using PyInstaller from start.py. This is the main deliverable.

```
User double-clicks LocalAI.exe
         ↓
Small system tray icon appears (🧠)
         ↓
Browser opens automatically → http://localhost:3000/model-manager
         ↓
Platform is running. User never sees a terminal.
```

PyInstaller build command (in build_exe.ps1):
```powershell
pyinstaller start.py `
  --onefile `
  --windowed `
  --name "LocalAI" `
  --icon "assets/icon.ico" `
  --add-data "config;config" `
  --add-data "extensions;extensions" `
  --add-data "frontend;frontend" `
  --hidden-import open_webui `
  --hidden-import psutil `
  --hidden-import gguf
```

The --windowed flag means NO terminal window appears.
A small tray icon lets the user quit cleanly.

### 4b. start.bat (Fallback — if .exe not built yet)
```batch
@echo off
cd /d "%~dp0"
start "" /B pythonw start.py
timeout /t 4 /nobreak >nul
start http://localhost:3000/model-manager
```
Uses `pythonw` instead of `python` — runs Python without showing a
terminal window. Same effect as the .exe but requires Python installed.

### 4c. start.py — Source Code Spec
```python
# The source that gets compiled into LocalAI.exe
#
# What it does:
# 1. Show system tray icon with "LocalAI Platform — Starting..." tooltip
# 2. Find llama-server binary (check Unsloth folder, PATH)
#    If not found: show GUI dialog (tkinter) asking user to locate it
#    Save path to config/settings.json
# 3. Set all required env vars for Open WebUI
# 4. Mount our FastAPI routes onto Open WebUI's FastAPI app
# 5. Start Open WebUI in a background thread
# 6. Wait until http://localhost:3000 responds
# 7. Open browser: webbrowser.open("http://localhost:3000/model-manager")
# 8. Update tray icon tooltip: "LocalAI Platform — Running"
# 9. Tray icon right-click menu:
#      ● Open Interface
#      ● Eject Model
#      ● View Logs
#      ─────────────
#      ● Quit
# 10. On Quit: kill llama-server, stop Open WebUI, exit cleanly
#
# System tray using: pystray + Pillow
# No terminal ever shown to the user.
#
# If first run (no settings.json):
#   Show a one-time setup wizard dialog (tkinter):
#   "Welcome to LocalAI Platform"
#   "Please locate your llama-server.exe"
#   [Browse...] button → file picker
#   "Add model folders" → folder picker (can add multiple)
#   [Finish] → saves settings.json, starts platform
```

---

## 6. Model Manager Routes Spec

```python
# extensions/model_manager/routes.py

GET  /api/localai/models
# Scan disk for GGUFs, return list:
# [{ name, path, size_gb, is_moe, is_split, shard_count,
#    fits_in_vram, fits_with_ram_offload, estimated_speed_tps }]

GET  /api/localai/status
# { running: bool, model_name, model_path, uptime_seconds,
#   tokens_per_second, vram_used_mb, context_used, context_max }

POST /api/localai/load
# Body: { model_path, override_flags (optional) }
# Returns: SSE stream of progress events:
#   {"step": "inspecting",  "message": "Reading model..."}
#   {"step": "optimizing",  "message": "Computing flags..."}
#   {"step": "launching",   "message": "Starting server..."}
#   {"step": "loading",     "message": "Loading weights...", "progress": 65}
#   {"step": "ready",       "message": "Done!", "speed_tps": 20.3}
#   {"step": "error",       "message": "CUDA OOM — try smaller model"}

POST /api/localai/eject
# Kill llama-server, return: { success, freed_vram_gb }

GET  /api/localai/hardware
# { gpu_name, gpu_vram_mb, gpu_vram_free_mb, ram_mb, cpu_cores }

POST /api/localai/optimize
# Body: { model_path }
# Return: { flags, ctx_size, threads, is_moe, command_preview }

GET  /api/localai/log
# SSE stream of llama-server stdout/stderr (last 100 lines + live)

GET  /api/localai/settings
PUT  /api/localai/settings
# Read/write config/settings.json
# { llama_server_path, model_dirs[], default_ctx_size, ... }
```

---

## 7. server_process.py Spec

```python
class LlamaServerProcess:
    process: subprocess.Popen | None
    loaded_model: str | None
    start_time: float | None
    flags_used: dict

    def start(model_path, flags) -> Generator[dict, None, None]:
        """SSE generator — yields progress events during load."""
        # 1. Eject if already running
        # 2. Inspect model → gguf_inspector.py
        # 3. Detect hardware → hardware.py
        # 4. Compute flags → optimizer.py
        # 5. Build command
        # 6. subprocess.Popen(cmd, stdout=log_file, stderr=STDOUT)
        # 7. Poll GET /health every 2s, yield progress events
        # 8. On ready: yield {"step": "ready", "speed_tps": estimate}
        # 9. On timeout (120s): yield {"step": "error", ...}

    def stop() -> dict:
        """Terminate llama-server, return freed VRAM estimate."""
        # proc.terminate() → wait 10s → proc.kill() if still alive
        # Return { freed_vram_gb }

    @property
    def is_running() -> bool:
        return process is not None and process.poll() is None

    def get_metrics() -> dict:
        """Parse llama-server /metrics for live tokens/sec."""
        # GET http://localhost:8001/metrics (prometheus format)
        # Parse: llamacpp:tokens_per_second
```

---

## 8. optimizer.py Spec

```python
def compute_optimal_flags(hw: dict, model: dict) -> dict:
    """
    hw fields: gpu_vram_mb, ram_mb, cpu_physical_cores, has_nvidia
    model fields: is_moe, size_gb, expert_count, layer_count, context_length

    Always:
      n-gpu-layers: 999
      cache-type-k: q8_0
      cache-type-v: q8_0
      host: 0.0.0.0
      port: 8001
      jinja: True        ← CRITICAL

    If NVIDIA:
      flash-attn: True

    If MoE:
      override-tensor: ".ffn_.*_exps.=CPU"
      batch-size: 4096
      ubatch-size: 4096

    If Dense:
      batch-size: 2048
      ubatch-size: 512

    Threads:
      min(hw.cpu_physical_cores, 32)

    Context (VRAM math):
      non_moe_vram_gb = model.size_gb * 0.25
      remaining_mb = hw.gpu_vram_mb - (non_moe_vram_gb * 1024) - 1500
      ctx = clamp((remaining_mb / 2000) * 8192, min=8192, max=131072)
      round to nearest power of 2
    """
```

---

## 9. Model Manager UI Spec

### ModelManager.svelte — Main Page
```
Layout:
  Top: Hardware bar (GPU name, VRAM used/total, RAM used/total)
  Middle: Model list (scrollable)
  Bottom: Status bar (server status, port, API endpoint)

Hardware bar always visible at top:
  GPU: RTX 5090  [████████░░] 24.1 / 32 GB VRAM
  RAM: 192 GB    [███░░░░░░░] 65 / 192 GB
```

### ModelCard.svelte — Per Model
```
States:
  NOT LOADED:
    Name + size + type badges (MoE/Dense, quantization)
    "Fits: ✅ VRAM" or "Fits: ✅ VRAM+RAM" or "⚠️ Too large"
    [ Load ▶ ]  [ ⚙ Settings ]

  LOADING:
    Progress bar with step description
    [ Cancel ]

  LOADED (current model):
    Speed: XX.X t/s
    Context: XXXXX / XXXXXX tokens [progress bar]
    VRAM: XX.X / 32 GB [progress bar]
    [ ⏏ Eject ]  [ ⚙ Settings ]
```

### LoadProgress.svelte — Loading Overlay
```
Full-screen overlay when loading:
  Model name
  Step icons with checkmarks:
    ✅ Hardware detected
    ✅ Model inspected (MoE, 128 experts)
    ✅ Optimal flags computed
    ✅ Server launched
    🔄 Loading weights... [animated]
  Progress bar (estimated %)
  Elapsed time
  [ Cancel ]
```

### SettingsPanel.svelte — Per-Model Settings
```
Slide-in panel:
  Context Size:    [65536 ▼]   (shows "auto" if not overridden)
  CPU Threads:     [16    ▼]
  GPU Layers:      [999   ▼]
  MoE Offload:     [ON  ●   ]   (disabled toggle if not MoE)
  Flash Attention: [ON  ●   ]
  KV Quantization: [Q8_0  ▼]

  Terminal-style command preview (monospace, dark bg):
  ┌─────────────────────────────────────────────────┐
  │ $ llama-server \                                │
  │     --model Qwen3.5-122B... \                  │
  │     -ngl 999 \                                  │
  │     -ot ".ffn_.*_exps.=CPU" \                  │
  │     --ctx-size 65536 \                          │
  │     --threads 16 \                              │
  │     --flash-attn \                              │
  │     --jinja                                     │
  └─────────────────────────────────────────────────┘

  [Reset to Auto]  [Save]  [Load with These Settings]
```

---

## 10. How to Mount Routes Into Open WebUI

```python
# start.py approach:
import uvicorn
from open_webui.main import app
from extensions.model_manager.routes import router

# Mount our routes before serving
app.include_router(router, prefix="/api/localai")

# Set env vars
import os
os.environ["WEBUI_AUTH"] = "False"
os.environ["OPENAI_API_BASE_URL"] = "http://localhost:8001/v1"
os.environ["OPENAI_API_KEY"] = "none"
os.environ["DATA_DIR"] = "./config/data"
os.environ["WEBUI_NAME"] = "LocalAI Platform"

# Start
uvicorn.run(app, host="0.0.0.0", port=3000)
```

---

## 11. Frontend Integration Strategy

Open WebUI uses SvelteKit. Two options for adding the Model Manager page:

**Option A — Custom HTML page injected via FastAPI route (simpler):**
```python
@router.get("/model-manager", response_class=HTMLResponse)
async def model_manager_page():
    # Serve a standalone HTML page with our Svelte components compiled
    # This page talks to /api/localai/* endpoints
    # Doesn't require modifying Open WebUI's source
    return HTMLResponse(content=open("frontend/dist/index.html").read())
```

**Option B — Fork Open WebUI and add Svelte pages directly (more integrated):**
- Fork Open WebUI repo
- Add `ModelManager.svelte` to `src/lib/components/`
- Add route to `src/routes/`
- Add nav link to sidebar
- More work but feels native to the UI

**Recommended: Start with Option A. Upgrade to B if needed.**

---

## 12. tools.py — Complete Tool List

```python
"""
title: LocalAI Platform Tools
author: LocalAI Platform
description: Full claw-code agent toolset for local AI
version: 2.0.0
"""

class Tools:
    # 1.  execute_command(command, timeout_seconds)
    # 2.  read_file(filepath, max_lines)
    # 3.  write_file(filepath, content)
    # 4.  edit_file(filepath, old_text, new_text)       patch-style
    # 5.  list_directory(path, pattern, recursive)
    # 6.  grep_search(pattern, path, file_pattern)       search inside files
    # 7.  get_hardware_profile()
    # 8.  inspect_gguf_model(filepath)
    # 9.  generate_optimal_config(model_path)
    # 10. check_llama_server(port)
    # 11. find_models_on_system(search_dirs)
    # 12. web_search(query, num_results)                 DuckDuckGo, no key
    # 13. web_fetch(url, timeout_seconds)
    # 14. todo_write(action, task, status)               task planner
    # 15. ask_user_question(question, options)           clarify before acting
    # 16. schedule_task(action, command, interval_min)   background jobs
    # 17. analyze_code(filepath, analysis_type)          LSP-style overview
    # 18. manage_process(action, process_name, port)     process manager
    # 19. run_python(code, timeout_seconds)              execute Python inline
```

---

## 13. System Prompt

```
You are LocalAI Platform's self-optimizing AI assistant with full
access to the local machine via tools.

CORE BEHAVIORS:
- Always start multi-step tasks with todo_write to make a visible plan.
- Call ask_user_question before any destructive action.
- Use edit_file for changes to existing files, write_file for new files.
- Use grep_search to find code before reading entire files.
- Use analyze_code to understand file structure before editing.

SELF-OPTIMIZATION:
When asked to configure, optimize, or set up a model:
1. get_hardware_profile()
2. find_models_on_system()
3. ask_user_question() if needed
4. inspect_gguf_model(path)
5. generate_optimal_config(path)
6. Explain settings and why they were chosen

KEY KNOWLEDGE:
- MoE models need -ot ".ffn_.*_exps.=CPU" for 4x speed boost
- --jinja is required for tool calling to work
- --flash-attn always on for NVIDIA
- Context = computed from remaining VRAM after model layers load
- Threads = physical CPU cores only (not logical/hyperthreaded)

PLATFORM PORTS:
- llama-server: http://localhost:8001
- Open WebUI:   http://localhost:3000
- Your app API: http://localhost:3000/api  (no auth needed)
```

---

## 14. Build Priority for Claude Code

**Phase 1 — Backend (make model load/eject work):**
1. `extensions/model_manager/hardware.py`
2. `extensions/model_manager/gguf_inspector.py`
3. `extensions/model_manager/optimizer.py`
4. `extensions/model_manager/server_process.py`
5. `extensions/model_manager/model_scanner.py`
6. `extensions/model_manager/routes.py`
7. `start.py` + `start.bat` + `build_exe.ps1`

**Phase 2 — Frontend (the Model Manager UI):**
8. `frontend/ModelManager.svelte` — main page
9. `frontend/ModelCard.svelte` — per model card
10. `frontend/LoadProgress.svelte` — loading overlay
11. `frontend/HardwareStats.svelte` — GPU/RAM bars
12. `frontend/SettingsPanel.svelte` — settings + command preview

**Phase 3 — Agent Tools:**
13. `extensions/tools/tools.py` — all 19 tools
14. Auto-install tools on startup

**Phase 4 — Installers & Polish:**
15. `install.ps1`, `install.sh`, build and test `LocalAI.exe`
16. Unit tests for hardware detection and optimizer
17. Auto-open browser on start
18. Custom branding / dark theme

---

## 15. UI Design Direction

**Feel:** Professional GPU workstation tool — like NVIDIA's control panel
but for AI models. Not a chat app with a sidebar.

**Theme:** Dark. Deep charcoal background (#1a1a1a), not pure black.

**Typography:**
- UI labels: clean sans-serif (e.g. IBM Plex Sans)
- Stats and numbers: monospace (e.g. JetBrains Mono, IBM Plex Mono)
- The command preview MUST be monospace — it looks like a real terminal

**Colors:**
- Loaded / healthy: green (#22c55e)
- Not loaded: muted grey (#6b7280)
- Loading: amber (#f59e0b) with animation
- Error: red (#ef4444)
- VRAM bar: gradient blue → purple as it fills
- Accent: electric blue (#3b82f6)

**Cards:** Subtle border, slight elevation shadow. Not flat.

**Animations:**
- Load progress bar: smooth fill animation
- Status dot: pulsing glow when loading
- Card transitions: subtle slide when state changes
- VRAM bar: animated fill when model loads

**Layout:** Left sidebar nav (same as Open WebUI) + main content area.
Model Manager is a full-page view, not a modal.

---

## 16. Nice-To-Have Features

- Download models from HuggingFace URL directly in the UI
- Speed benchmark button (100-token test, display t/s)
- Side-by-side model comparison (load two at once if VRAM allows)
- GPU temperature in hardware stats bar
- Auto-restart watchdog if llama-server crashes
- Windows system tray icon
- Multiple model folder support with folder picker
- Launch profiles (save named flag presets)
- Export command as `.sh` / `.bat` script
- Log viewer panel showing live llama-server output

---

## 17. Licenses

| Project | License | Notes |
|---|---|---|
| Open WebUI | MIT | Full use, no restrictions |
| llama.cpp | MIT | Full use, no restrictions |
| claw-code | Open source | Concepts reimplemented in Python |
| This platform | MIT recommended | Give back to community |
