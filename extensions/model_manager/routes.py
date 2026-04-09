"""
FastAPI routes for LocalAI Platform model management.
Mounted at /api/localai/* in start.py.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import hardware as hw_module
from . import gguf_inspector
from . import optimizer as opt_module
from . import model_scanner
from . import server_process as sp_module
from . import tool_validator
from . import auto_optimizer

router = APIRouter()

# Path to config/settings.json (relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SETTINGS_PATH = _REPO_ROOT / "config" / "settings.json"
_FRONTEND_PATH = _REPO_ROOT / "frontend"


# ------------------------------------------------------------------
# Settings helpers
# ------------------------------------------------------------------

def _load_settings() -> dict:
    """Load settings from config/settings.json, creating defaults if missing."""
    defaults = {
        "llama_server_path": None,
        "model_dirs": [],
        "default_ctx_size": None,
        "port": 3000,
        "llama_port": 8001,
        "auto_open_browser": True,
    }
    if _SETTINGS_PATH.exists():
        try:
            with open(_SETTINGS_PATH, "r") as f:
                saved = json.load(f)
            defaults.update(saved)
        except Exception:
            pass
    return defaults


def _save_settings(settings: dict) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


# ------------------------------------------------------------------
# SSE streaming helper
# ------------------------------------------------------------------

async def _event_stream(generator: AsyncGenerator) -> AsyncGenerator[bytes, None]:
    """Wrap an async generator of dicts into SSE byte stream."""
    async for event in generator:
        data = json.dumps(event)
        yield f"data: {data}\n\n".encode()
    yield b"data: {\"step\": \"done\"}\n\n"


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class LoadRequest(BaseModel):
    model_path: str
    override_flags: Optional[dict] = None


class OptimizeRequest(BaseModel):
    model_path: str
    overrides: Optional[dict] = None


class EjectResponse(BaseModel):
    success: bool
    freed_vram_gb: float = 0.0


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.get("/hardware")
async def get_hardware():
    """Detect and return GPU/RAM/CPU info."""
    try:
        return hw_module.detect_hardware()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models():
    """Scan configured model directories for GGUF files."""
    settings = _load_settings()
    model_dirs = settings.get("model_dirs", [])
    if not model_dirs:
        return []
    try:
        hw = hw_module.detect_hardware()
        models = model_scanner.scan_models(model_dirs, hw)
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status():
    """Return current llama-server status and live metrics."""
    server = sp_module.get_server()
    return server.get_status()


@router.post("/load")
async def load_model(req: LoadRequest):
    """
    Start llama-server with the specified model.
    Returns an SSE stream of progress events.
    """
    settings = _load_settings()
    llama_binary = settings.get("llama_server_path")

    if not llama_binary:
        # Try to find in PATH
        import shutil
        found = shutil.which("llama-server")
        if not found and sys.platform == "win32":
            found = shutil.which("llama-server.exe")
        if found:
            llama_binary = found
        else:
            async def error_stream():
                yield {"step": "error", "message": "llama-server binary not configured. Please set llama_server_path in Settings."}
            return StreamingResponse(
                _event_stream(error_stream()),
                media_type="text/event-stream",
            )

    model_path = req.model_path
    overrides = req.override_flags or {}

    try:
        hw = hw_module.detect_hardware()
        model_info = gguf_inspector.inspect_model(model_path)
        model_info["path"] = model_path
        opt = opt_module.compute_optimal_flags(hw, model_info, overrides)
        flags = opt["flags"]
    except Exception as e:
        async def error_stream():
            yield {"step": "error", "message": f"Failed to compute flags: {e}"}
        return StreamingResponse(
            _event_stream(error_stream()),
            media_type="text/event-stream",
        )

    server = sp_module.get_server()

    async def load_generator():
        async for event in server.start(
            model_path=model_path,
            llama_server_binary=llama_binary,
            flags=flags,
            model_name=model_info.get("name", ""),
        ):
            yield event

    return StreamingResponse(
        _event_stream(load_generator()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/eject")
async def eject_model():
    """Stop llama-server and free VRAM."""
    server = sp_module.get_server()
    result = server.stop()
    return result


@router.post("/optimize")
async def optimize_flags(req: OptimizeRequest):
    """
    Compute optimal llama-server flags for a model without loading it.
    Useful for previewing the command in the SettingsPanel.
    """
    try:
        hw = hw_module.detect_hardware()
        model_info = gguf_inspector.inspect_model(req.model_path)
        model_info["path"] = req.model_path
        result = opt_module.compute_optimal_flags(hw, model_info, req.overrides or {})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/log")
async def stream_log(lines: int = 100):
    """
    Stream llama-server log output as SSE.
    Returns last N buffered lines first, then live updates.
    """
    server = sp_module.get_server()

    async def log_generator():
        # Send buffered lines first
        for line in server.get_log_lines(lines):
            yield {"line": line}
        # Then poll for new lines (simple approach: re-read buffer tail)
        seen = server.get_log_lines(lines)
        while server.is_running:
            await asyncio.sleep(1)
            new_lines = server.get_log_lines(lines)
            if len(new_lines) > len(seen):
                for line in new_lines[len(seen):]:
                    yield {"line": line}
            seen = new_lines

    return StreamingResponse(
        _event_stream(log_generator()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/settings")
async def get_settings():
    """Return current settings."""
    return _load_settings()


@router.put("/settings")
async def update_settings(request: Request):
    """Update settings (partial update — merges with existing)."""
    body = await request.json()
    settings = _load_settings()
    settings.update(body)
    try:
        _save_settings(settings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {e}")
    return settings


# ------------------------------------------------------------------
# Tool validation endpoints
# ------------------------------------------------------------------

@router.post("/validate-tool-call")
async def validate_tool_call(request: Request):
    """
    Validate a tool call before execution. Also recovers XML leaks.

    Body: { "name": "tool_name", "arguments": {...} }
    Or:   { "raw_text": "model output with <tool_call> tags" }
    """
    body = await request.json()

    # If raw_text is provided, try to extract tool calls from XML leaks
    if "raw_text" in body:
        cleaned, extracted = tool_validator.clean_response(body["raw_text"])
        validated = []
        errors = []
        for tc in extracted:
            try:
                v = tool_validator.validate_tool_call(tc)
                validated.append(v)
            except tool_validator.ToolCallValidationError as e:
                errors.append({"tool": tc.get("name"), "error": str(e)})
        return {
            "cleaned_text": cleaned,
            "tool_calls": validated,
            "errors": errors,
            "xml_leak_detected": tool_validator.is_xml_leak(body["raw_text"]),
        }

    # Direct tool call validation
    try:
        result = tool_validator.validate_tool_call(body)
        return {"valid": True, "tool_call": result}
    except tool_validator.ToolCallValidationError as e:
        return {"valid": False, "error": str(e)}


@router.get("/tool-stats")
async def tool_stats():
    """Return tool call validation stats for debugging."""
    return tool_validator.get_validation_stats()


# ------------------------------------------------------------------
# Auto-optimization endpoints
# ------------------------------------------------------------------

class AutoOptimizeRequest(BaseModel):
    model_path: Optional[str] = None
    max_iterations: int = 10


@router.post("/auto-optimize")
async def run_auto_optimize(req: AutoOptimizeRequest):
    """
    Run the autoresearch-style self-optimization loop.
    Benchmarks current settings, tries variations, keeps improvements.
    Returns SSE stream of progress events.

    Requires a model to already be loaded.
    """
    server = sp_module.get_server()
    if not server.is_running:
        async def error_stream():
            yield {"step": "error", "message": "No model loaded. Load a model first, then run auto-optimize."}
        return StreamingResponse(
            _event_stream(error_stream()),
            media_type="text/event-stream",
        )

    model_path = req.model_path or server.loaded_model_path
    model_name = server.loaded_model_name or "Unknown"

    settings = _load_settings()
    llama_binary = settings.get("llama_server_path")
    if not llama_binary:
        import shutil
        llama_binary = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if not llama_binary:
        async def error_stream():
            yield {"step": "error", "message": "llama-server binary not configured."}
        return StreamingResponse(
            _event_stream(error_stream()),
            media_type="text/event-stream",
        )

    hw = hw_module.detect_hardware()
    model_info = gguf_inspector.inspect_model(model_path)
    current_flags = server._flags_used or {}

    async def optimize_stream():
        async for event in auto_optimizer.auto_optimize(
            model_path=model_path,
            model_name=model_name,
            current_flags=current_flags,
            llama_server_binary=llama_binary,
            hw=hw,
            is_moe=model_info.get("is_moe", False),
            max_iterations=req.max_iterations,
        ):
            yield event

    return StreamingResponse(
        _event_stream(optimize_stream()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/auto-optimize/history")
async def optimization_history():
    """Return past optimization results."""
    return auto_optimizer.get_optimization_history()


@router.post("/benchmark")
async def run_benchmark(request: Request):
    """
    Run a quick benchmark on the currently loaded model.
    Returns tokens/sec and timing info.
    """
    server = sp_module.get_server()
    if not server.is_running:
        raise HTTPException(status_code=400, detail="No model loaded")

    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    max_tokens = body.get("max_tokens", 100)

    result = auto_optimizer.run_benchmark(max_tokens=max_tokens)
    return {
        "tokens_per_second": result.tokens_per_second,
        "total_tokens": result.total_tokens,
        "duration_seconds": result.duration_seconds,
        "error": result.error,
        "success": result.success,
    }


# ------------------------------------------------------------------
# Frontend serving (Model Manager UI)
# ------------------------------------------------------------------

@router.get("/model-manager", response_class=HTMLResponse)
async def model_manager_page():
    """Serve the standalone Model Manager UI."""
    html_path = _FRONTEND_PATH / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h1>Model Manager UI not found</h1>"
            "<p>Run the frontend build or check the frontend/ directory.</p>",
            status_code=404,
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@router.get("/frontend/{filename:path}")
async def frontend_static(filename: str):
    """Serve static frontend assets (CSS, JS, fonts)."""
    file_path = _FRONTEND_PATH / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Basic content-type detection
    ext = file_path.suffix.lower()
    content_types = {
        ".css": "text/css",
        ".js": "application/javascript",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".png": "image/png",
        ".svg": "image/svg+xml",
    }
    content_type = content_types.get(ext, "application/octet-stream")

    return StreamingResponse(
        open(file_path, "rb"),
        media_type=content_type,
    )
