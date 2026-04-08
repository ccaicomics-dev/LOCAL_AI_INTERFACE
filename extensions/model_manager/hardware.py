"""
Hardware detection module for LocalAI Platform.
Detects GPU (NVIDIA via pynvml), RAM, and CPU info cross-platform.
"""
import platform
import subprocess
import sys
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


def _get_cpu_name() -> str:
    """Get CPU name cross-platform."""
    system = platform.system()
    try:
        if system == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name", "/value"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if line.startswith("Name="):
                    return line.split("=", 1)[1].strip()
        elif system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def _detect_nvidia_pynvml() -> Optional[dict]:
    """Try to detect NVIDIA GPU via pynvml. Returns None if unavailable."""
    if not _PYNVML_AVAILABLE:
        return None
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return None
        # Use first GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "gpu_name": name,
            "gpu_vram_mb": mem_info.total // (1024 * 1024),
            "gpu_vram_free_mb": mem_info.free // (1024 * 1024),
            "has_nvidia": True,
        }
    except Exception:
        return None


def _detect_nvidia_smi() -> Optional[dict]:
    """Fallback: parse nvidia-smi output."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None
        return {
            "gpu_name": parts[0],
            "gpu_vram_mb": int(parts[1]),
            "gpu_vram_free_mb": int(parts[2]),
            "has_nvidia": True,
        }
    except Exception:
        return None


def detect_hardware() -> dict:
    """
    Detect hardware info for the current system.

    Returns:
        dict with keys:
            gpu_name (str | None): GPU display name
            gpu_vram_mb (int): Total VRAM in MB (0 if no GPU)
            gpu_vram_free_mb (int): Free VRAM in MB
            has_nvidia (bool): Whether an NVIDIA GPU was found
            ram_mb (int): Total system RAM in MB
            ram_free_mb (int): Free system RAM in MB
            cpu_cores (int): Physical CPU core count
            cpu_name (str): CPU display name
    """
    # GPU detection — try pynvml first, then nvidia-smi, then no-GPU
    gpu_info = _detect_nvidia_pynvml() or _detect_nvidia_smi() or {
        "gpu_name": None,
        "gpu_vram_mb": 0,
        "gpu_vram_free_mb": 0,
        "has_nvidia": False,
    }

    # RAM
    if psutil is not None:
        vm = psutil.virtual_memory()
        ram_mb = vm.total // (1024 * 1024)
        ram_free_mb = vm.available // (1024 * 1024)
        cpu_cores = psutil.cpu_count(logical=False) or 1
    else:
        ram_mb = 0
        ram_free_mb = 0
        cpu_cores = 1

    return {
        **gpu_info,
        "ram_mb": ram_mb,
        "ram_free_mb": ram_free_mb,
        "cpu_cores": cpu_cores,
        "cpu_name": _get_cpu_name(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(detect_hardware(), indent=2))
