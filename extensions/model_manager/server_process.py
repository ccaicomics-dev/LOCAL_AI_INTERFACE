"""
llama-server subprocess lifecycle manager for LocalAI Platform.
Manages starting, stopping, health checking, and metrics collection.
"""
import asyncio
import collections
import json
import platform
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import AsyncGenerator, Optional

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _requests = None
    _REQUESTS_AVAILABLE = False


LLAMA_HOST = "http://127.0.0.1"
LLAMA_PORT = 8001
HEALTH_TIMEOUT = 120  # seconds to wait for server ready
HEALTH_POLL_INTERVAL = 2  # seconds between health checks
LOG_BUFFER_SIZE = 1000  # lines to keep in memory


class LlamaServerProcess:
    """
    Singleton manager for a llama-server subprocess.
    Only one model can be loaded at a time.
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._loaded_model_path: Optional[str] = None
        self._loaded_model_name: Optional[str] = None
        self._start_time: Optional[float] = None
        self._flags_used: dict = {}
        self._log_buffer: collections.deque = collections.deque(maxlen=LOG_BUFFER_SIZE)
        self._log_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def loaded_model_path(self) -> Optional[str]:
        return self._loaded_model_path if self.is_running else None

    @property
    def loaded_model_name(self) -> Optional[str]:
        return self._loaded_model_name if self.is_running else None

    # ------------------------------------------------------------------
    # Start (SSE async generator)
    # ------------------------------------------------------------------

    async def start(
        self,
        model_path: str,
        llama_server_binary: str,
        flags: dict,
        model_name: str = "",
    ) -> AsyncGenerator[dict, None]:
        """
        Start llama-server with the given model and flags.
        Yields SSE-style progress event dicts throughout the process.

        Args:
            model_path: Absolute path to the .gguf file (first shard for split)
            llama_server_binary: Path to the llama-server executable
            flags: Flags dict from optimizer.compute_optimal_flags()["flags"]
            model_name: Human-readable model name for display
        """
        # Eject any running model first
        if self.is_running:
            yield {"step": "ejecting", "message": "Stopping previous model..."}
            self.stop()

        yield {"step": "inspecting", "message": f"Preparing {model_name or Path(model_path).stem}..."}
        await asyncio.sleep(0)

        # Validate binary exists
        binary_path = Path(llama_server_binary)
        if not binary_path.exists():
            yield {"step": "error", "message": f"llama-server not found: {llama_server_binary}"}
            return

        yield {"step": "launching", "message": "Starting llama-server..."}
        await asyncio.sleep(0)

        # Build command from flags
        cmd = [str(binary_path)]
        for key, value in flags.items():
            cmd.append(key)
            if value is not None:
                cmd.append(str(value))

        # Override model path to ensure it's set
        if "--model" not in flags:
            cmd.extend(["--model", model_path])

        self._flags_used = flags

        # Platform-specific subprocess flags
        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        if platform.system() == "Windows":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            with self._lock:
                self._process = subprocess.Popen(cmd, **popen_kwargs)
                self._loaded_model_path = model_path
                self._loaded_model_name = model_name or Path(model_path).stem
                self._start_time = time.time()
                self._log_buffer.clear()

            # Start background log reader thread
            self._log_thread = threading.Thread(
                target=self._read_logs, daemon=True
            )
            self._log_thread.start()

        except Exception as e:
            yield {"step": "error", "message": f"Failed to launch: {e}"}
            return

        # Poll health endpoint until ready or timeout
        yield {"step": "loading", "message": "Loading model weights...", "progress": 0}

        deadline = time.time() + HEALTH_TIMEOUT
        last_progress = 0

        while time.time() < deadline:
            await asyncio.sleep(HEALTH_POLL_INTERVAL)

            if not self.is_running:
                # Process died — get last log lines for error context
                last_logs = "\n".join(list(self._log_buffer)[-5:])
                yield {"step": "error", "message": f"Server exited unexpectedly.\n{last_logs}"}
                return

            health = self._check_health()
            if health == "ok":
                # Server is ready
                speed = self._estimate_speed()
                yield {
                    "step": "ready",
                    "message": "Model loaded successfully!",
                    "speed_tps": speed,
                    "model_name": self._loaded_model_name,
                }
                return
            elif health == "loading":
                # Estimate progress from elapsed time
                elapsed = time.time() - self._start_time
                progress = min(int((elapsed / HEALTH_TIMEOUT) * 90), 90)
                if progress > last_progress:
                    last_progress = progress
                    yield {
                        "step": "loading",
                        "message": "Loading model weights...",
                        "progress": progress,
                    }
            # else: not yet responding, keep polling

        # Timeout
        yield {"step": "error", "message": f"Timeout after {HEALTH_TIMEOUT}s waiting for server."}
        self.stop()

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop(self) -> dict:
        """
        Terminate llama-server process.
        Returns dict with freed VRAM estimate.
        """
        with self._lock:
            if self._process is None:
                return {"success": True, "freed_vram_gb": 0.0}

            freed_vram_gb = 0.0
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)

                # Estimate freed VRAM (rough)
                # We don't have exact VRAM info here; caller can re-check hardware
                freed_vram_gb = 0.0

            except Exception:
                pass
            finally:
                self._process = None
                self._loaded_model_path = None
                self._loaded_model_name = None
                self._start_time = None

        return {"success": True, "freed_vram_gb": freed_vram_gb}

    # ------------------------------------------------------------------
    # Status & metrics
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current server status and metrics."""
        if not self.is_running:
            return {
                "running": False,
                "model_name": None,
                "model_path": None,
                "uptime_seconds": 0,
                "tokens_per_second": 0.0,
                "vram_used_mb": 0,
                "context_used": 0,
                "context_max": 0,
            }

        uptime = int(time.time() - self._start_time) if self._start_time else 0
        metrics = self._get_metrics()

        return {
            "running": True,
            "model_name": self._loaded_model_name,
            "model_path": self._loaded_model_path,
            "uptime_seconds": uptime,
            "tokens_per_second": metrics.get("tokens_per_second", 0.0),
            "vram_used_mb": metrics.get("vram_used_mb", 0),
            "context_used": metrics.get("context_used", 0),
            "context_max": metrics.get("context_max", 0),
        }

    def get_log_lines(self, last_n: int = 100) -> list:
        """Return the last N lines from the llama-server log buffer."""
        return list(self._log_buffer)[-last_n:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_health(self) -> str:
        """
        Check llama-server health endpoint.
        Returns: "ok", "loading", or "unavailable"
        """
        if not _REQUESTS_AVAILABLE:
            return "unavailable"
        try:
            resp = _requests.get(
                f"{LLAMA_HOST}:{LLAMA_PORT}/health", timeout=3
            )
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "")
                if status == "ok":
                    return "ok"
                elif "loading" in status.lower():
                    return "loading"
                return "ok"  # 200 with unknown status — assume ok
            elif resp.status_code == 503:
                return "loading"
            return "unavailable"
        except Exception:
            return "unavailable"

    def _get_metrics(self) -> dict:
        """Parse llama-server /metrics endpoint (Prometheus format)."""
        if not _REQUESTS_AVAILABLE:
            return {}
        try:
            resp = _requests.get(
                f"{LLAMA_HOST}:{LLAMA_PORT}/metrics", timeout=3
            )
            if resp.status_code != 200:
                return {}
            metrics = {}
            for line in resp.text.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                # Parse: metric_name{labels} value
                parts = line.split()
                if len(parts) < 2:
                    continue
                name = parts[0].split("{")[0]
                try:
                    value = float(parts[-1])
                except ValueError:
                    continue
                if "tokens_per_second" in name:
                    metrics["tokens_per_second"] = round(value, 1)
                elif "kv_cache_usage" in name:
                    # kv_cache_usage_ratio is 0.0–1.0
                    metrics["kv_cache_ratio"] = value
            return metrics
        except Exception:
            return {}

    def _estimate_speed(self) -> float:
        """Get tokens/sec from metrics, or return 0."""
        metrics = self._get_metrics()
        return metrics.get("tokens_per_second", 0.0)

    def _read_logs(self):
        """Background thread: read llama-server stdout/stderr into log buffer."""
        if self._process is None or self._process.stdout is None:
            return
        try:
            for line in self._process.stdout:
                self._log_buffer.append(line.rstrip())
        except Exception:
            pass


# Module-level singleton
_server = LlamaServerProcess()


def get_server() -> LlamaServerProcess:
    """Get the global LlamaServerProcess singleton."""
    return _server
