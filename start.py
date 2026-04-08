"""
LocalAI Platform — Main Launcher
Starts Open WebUI with custom model management routes, system tray, and auto-opens browser.

Usage:
    python start.py          # Run in development
    pythonw start.py         # Run without terminal window (Windows)
    LocalAI.exe              # Compiled via PyInstaller
"""
import json
import os
import platform
import sys
import threading
import time
import webbrowser
from pathlib import Path


# ─── Repo root ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_CONFIG_PATH = _ROOT / "config" / "settings.json"


# ─── Settings ─────────────────────────────────────────────────────

def _load_settings() -> dict:
    defaults = {
        "llama_server_path": None,
        "model_dirs": [],
        "default_ctx_size": None,
        "port": 3000,
        "llama_port": 8001,
        "auto_open_browser": True,
    }
    if _CONFIG_PATH.exists():
        try:
            saved = json.loads(_CONFIG_PATH.read_text())
            defaults.update(saved)
        except Exception:
            pass
    return defaults


def _save_settings(settings: dict):
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(settings, indent=2))


# ─── First-run setup wizard ───────────────────────────────────────

def _run_setup_wizard() -> dict:
    """Show a tkinter setup wizard on first run. Returns updated settings."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox

        root = tk.Tk()
        root.title("LocalAI Platform — Setup")
        root.geometry("540x400")
        root.configure(bg="#1a1a1a")
        root.resizable(False, False)

        settings = {"llama_server_path": None, "model_dirs": []}

        def style_label(parent, text, fg="#e8e8e8", font_size=12):
            return tk.Label(parent, text=text, bg="#1a1a1a", fg=fg,
                            font=("Segoe UI", font_size))

        def style_button(parent, text, command, bg="#3b82f6"):
            return tk.Button(parent, text=text, command=command,
                             bg=bg, fg="white", relief="flat",
                             font=("Segoe UI", 10), padx=12, pady=6,
                             cursor="hand2")

        frame = tk.Frame(root, bg="#1a1a1a", padx=24, pady=20)
        frame.pack(fill="both", expand=True)

        style_label(frame, "Welcome to LocalAI Platform", font_size=16).pack(anchor="w", pady=(0, 4))
        style_label(frame, "Let's set up your configuration.", fg="#9ca3af", font_size=11).pack(anchor="w", pady=(0, 20))

        # llama-server path
        style_label(frame, "1. Locate llama-server binary", font_size=11).pack(anchor="w")
        llama_var = tk.StringVar()
        llama_frame = tk.Frame(frame, bg="#1a1a1a")
        llama_frame.pack(fill="x", pady=(4, 12))
        llama_entry = tk.Entry(llama_frame, textvariable=llama_var, bg="#2f2f2f", fg="#e8e8e8",
                               insertbackground="white", relief="flat", font=("Consolas", 10))
        llama_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))

        def browse_llama():
            ext = ".exe" if platform.system() == "Windows" else ""
            fname = filedialog.askopenfilename(
                title="Select llama-server binary",
                filetypes=[("llama-server", f"llama-server{ext}"), ("All files", "*")]
            )
            if fname:
                llama_var.set(fname)

        style_button(llama_frame, "Browse...", browse_llama).pack(side="right")

        # Try common default paths
        _defaults = []
        if platform.system() == "Windows":
            _defaults = [
                r"C:\Users\{}\AppData\Local\unsloth\llama-server.exe".format(os.getenv("USERNAME", "user")),
                r"C:\llama.cpp\llama-server.exe",
            ]
        for d in _defaults:
            if Path(d).exists():
                llama_var.set(d)
                break

        # Model dirs
        style_label(frame, "2. Add model directories", font_size=11).pack(anchor="w")
        style_label(frame, "(folders containing .gguf files)", fg="#6b7280", font_size=9).pack(anchor="w")

        dirs_listbox = tk.Listbox(frame, bg="#2f2f2f", fg="#e8e8e8", relief="flat",
                                  height=4, font=("Consolas", 10), selectmode="single")
        dirs_listbox.pack(fill="x", pady=(4, 6))

        def add_dir():
            d = filedialog.askdirectory(title="Select model directory")
            if d and d not in dirs_listbox.get(0, "end"):
                dirs_listbox.insert("end", d)

        def remove_dir():
            sel = dirs_listbox.curselection()
            if sel:
                dirs_listbox.delete(sel[0])

        dir_btns = tk.Frame(frame, bg="#1a1a1a")
        dir_btns.pack(anchor="w", pady=(0, 20))
        style_button(dir_btns, "+ Add Folder", add_dir).pack(side="left", padx=(0, 8))
        style_button(dir_btns, "- Remove", remove_dir, bg="#7f1d1d").pack(side="left")

        def finish():
            settings["llama_server_path"] = llama_var.get().strip() or None
            settings["model_dirs"] = list(dirs_listbox.get(0, "end"))
            root.destroy()

        style_button(frame, "Start LocalAI Platform →", finish, bg="#16a34a").pack(
            side="bottom", pady=10
        )

        root.mainloop()
        return settings

    except ImportError:
        # tkinter not available — return empty settings and continue
        print("[LocalAI] tkinter not available, skipping setup wizard.")
        return {}


# ─── System tray ──────────────────────────────────────────────────

def _run_tray():
    """Start the system tray icon (pystray). Runs in its own thread."""
    try:
        from PIL import Image, ImageDraw
        import pystray

        # Create a simple brain-icon image (32×32, dark bg with blue circle)
        img = Image.new("RGB", (32, 32), color="#1a1a1a")
        draw = ImageDraw.Draw(img)
        draw.ellipse([4, 4, 28, 28], fill="#3b82f6")
        draw.text((10, 8), "AI", fill="white")

        from extensions.model_manager import server_process as sp

        def on_open(icon, item):
            webbrowser.open("http://localhost:3000/api/localai/model-manager")

        def on_eject(icon, item):
            sp.get_server().stop()

        def on_quit(icon, item):
            sp.get_server().stop()
            icon.stop()
            os._exit(0)

        menu = pystray.Menu(
            pystray.MenuItem("Open Interface", on_open, default=True),
            pystray.MenuItem("Eject Model", on_eject),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", on_quit),
        )
        icon = pystray.Icon("LocalAI Platform", img, "LocalAI Platform", menu)
        icon.run()
    except ImportError:
        # pystray/Pillow not available — skip tray (Linux dev mode)
        pass
    except Exception as e:
        print(f"[LocalAI] Tray icon error: {e}")


# ─── Wait for server ready ────────────────────────────────────────

def _wait_for_server(port: int, timeout: int = 30) -> bool:
    """Poll localhost until Open WebUI responds or timeout."""
    import urllib.request
    url = f"http://localhost:{port}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


# ─── Register tools with Open WebUI ──────────────────────────────

def _register_tools(port: int):
    """Register the LocalAI tools file with Open WebUI via API."""
    try:
        import urllib.request
        tools_path = _ROOT / "extensions" / "tools" / "tools.py"
        if not tools_path.exists():
            return

        tools_content = tools_path.read_text(encoding="utf-8")
        payload = json.dumps({
            "id": "localai-platform-tools",
            "name": "LocalAI Platform Tools",
            "content": tools_content,
            "meta": {
                "description": "Full agent toolset: file ops, system, web, model management.",
                "version": "2.0.0",
            },
        }).encode()

        req = urllib.request.Request(
            f"http://localhost:{port}/api/v1/tools/create",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        print("[LocalAI] Tools registered with Open WebUI.")
    except Exception as e:
        # Non-fatal — tools can be added manually via UI
        print(f"[LocalAI] Tool registration skipped: {e}")


# ─── Main entry point ─────────────────────────────────────────────

def main():
    # 1. Load or create settings
    settings = _load_settings()
    is_first_run = not _CONFIG_PATH.exists()

    if is_first_run:
        print("[LocalAI] First run detected — launching setup wizard...")
        wizard_settings = _run_setup_wizard()
        settings.update(wizard_settings)
        _save_settings(settings)

    port = settings.get("port", 3000)

    # 2. CRITICAL: Set env vars BEFORE importing open_webui
    #    (Open WebUI reads these at module import time)
    os.environ.setdefault("WEBUI_AUTH", "False")
    os.environ.setdefault("OPENAI_API_BASE_URLS", f"http://localhost:{settings.get('llama_port', 8001)}/v1")
    os.environ.setdefault("OPENAI_API_KEYS", "none")
    os.environ.setdefault("WEBUI_NAME", "LocalAI Platform")
    os.environ.setdefault("PORT", str(port))
    data_dir = str(_ROOT / "config" / "data")
    os.environ.setdefault("DATA_DIR", data_dir)
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Add repo root to Python path so our extensions are importable
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    # 3. Import Open WebUI app (must be after env vars)
    print("[LocalAI] Loading Open WebUI...")
    try:
        from open_webui.main import app
    except ImportError:
        print("[LocalAI] ERROR: open-webui is not installed.")
        print("  Run: pip install open-webui")
        sys.exit(1)

    # 4. Mount our custom routes
    from extensions.model_manager.routes import router
    app.include_router(router, prefix="/api/localai")
    print("[LocalAI] Model Manager routes mounted at /api/localai/*")

    # 5. Start system tray in background thread
    tray_thread = threading.Thread(target=_run_tray, daemon=True)
    tray_thread.start()

    # 6. Open browser once server is ready (in background)
    def _open_browser():
        if _wait_for_server(port, timeout=60):
            url = f"http://localhost:{port}/api/localai/model-manager"
            print(f"[LocalAI] Opening browser: {url}")
            webbrowser.open(url)
            # Register tools after server is up
            _register_tools(port)
        else:
            print("[LocalAI] WARNING: Server did not start in time.")

    if settings.get("auto_open_browser", True):
        browser_thread = threading.Thread(target=_open_browser, daemon=True)
        browser_thread.start()

    # 7. Start uvicorn (blocking — main thread)
    print(f"[LocalAI] Starting server on http://localhost:{port}")
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    except KeyboardInterrupt:
        print("\n[LocalAI] Shutting down...")
        from extensions.model_manager import server_process as sp
        sp.get_server().stop()


if __name__ == "__main__":
    main()
