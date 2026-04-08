"""
GGUF model file scanner for LocalAI Platform.
Scans configured directories for .gguf files and returns enriched model info.
"""
import os
import re
import time
from pathlib import Path
from typing import Optional

from . import gguf_inspector
from . import hardware as hw_module
from . import optimizer as opt_module


# Regex for split shard filenames: model-00001-of-00005.gguf
_SPLIT_RE = re.compile(r"^(.+)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

# Simple file-level scan cache: path -> (mtime, result)
_cache: dict = {}


def _is_first_shard(name: str) -> bool:
    """Return True if the filename is the first shard (-00001-of-NNNNN.gguf)."""
    m = _SPLIT_RE.match(name)
    return m is not None and int(m.group(2)) == 1


def _is_any_shard(name: str) -> bool:
    """Return True if the filename looks like any shard of a split model."""
    return _SPLIT_RE.match(name) is not None


def scan_models(model_dirs: list, hw: Optional[dict] = None) -> list:
    """
    Scan directories for GGUF model files and return enriched model info.

    Args:
        model_dirs: List of directory paths to scan recursively.
        hw: Optional hardware info dict. If None, detect_hardware() is called.

    Returns:
        List of dicts, one per model (split models count as one), sorted by name.
        Each dict extends gguf_inspector.inspect_model() output with:
            fits_in_vram (bool)
            fits_with_ram (bool)
            estimated_speed_tps (float | None)
    """
    if hw is None:
        try:
            hw = hw_module.detect_hardware()
        except Exception:
            hw = {
                "gpu_vram_mb": 0,
                "gpu_vram_free_mb": 0,
                "has_nvidia": False,
                "ram_mb": 0,
                "cpu_cores": 4,
            }

    found_paths: list = []

    for dir_path in model_dirs:
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            continue
        for root, _, files in os.walk(dir_path):
            for fname in files:
                if not fname.lower().endswith(".gguf"):
                    continue
                full = str(Path(root) / fname)

                # For split models, only include the first shard
                if _is_any_shard(fname):
                    if _is_first_shard(fname):
                        found_paths.append(full)
                    # Skip non-first shards entirely
                else:
                    found_paths.append(full)

    results = []
    for path in found_paths:
        info = _inspect_cached(path)
        info = _enrich_with_hw(info, hw)
        results.append(info)

    results.sort(key=lambda m: m.get("name", "").lower())
    return results


def _inspect_cached(path: str) -> dict:
    """Return cached inspection result if file hasn't changed, else re-inspect."""
    global _cache
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0

    if path in _cache:
        cached_mtime, cached_result = _cache[path]
        if cached_mtime == mtime:
            return dict(cached_result)

    result = gguf_inspector.inspect_model(path)
    result["path"] = path
    _cache[path] = (mtime, result)
    return dict(result)


def _enrich_with_hw(model: dict, hw: dict) -> dict:
    """Add fit/speed estimates based on hardware info."""
    gpu_vram_mb = hw.get("gpu_vram_mb", 0)
    gpu_vram_free_mb = hw.get("gpu_vram_free_mb", 0)
    ram_mb = hw.get("ram_mb", 0)

    size_mb = model.get("size_gb", 0) * 1024
    is_moe = model.get("is_moe", False)

    if is_moe:
        vram_needed_mb = size_mb * 0.25
    else:
        vram_needed_mb = size_mb

    fits_in_vram = gpu_vram_free_mb > 0 and vram_needed_mb <= gpu_vram_free_mb
    fits_with_ram = (
        gpu_vram_free_mb + ram_mb * 0.8
    ) >= size_mb if (gpu_vram_free_mb > 0 or ram_mb > 0) else False

    # Rough speed estimate (t/s) — very approximate
    estimated_speed_tps = None
    if fits_in_vram and gpu_vram_mb > 0:
        # Dense: ~10-20 t/s per 24GB VRAM; MoE with offload: similar
        estimated_speed_tps = round(12.0 * (gpu_vram_mb / 24576), 1)
    elif fits_with_ram:
        estimated_speed_tps = 2.0  # CPU-heavy, slow

    model["fits_in_vram"] = fits_in_vram
    model["fits_with_ram"] = fits_with_ram
    model["estimated_speed_tps"] = estimated_speed_tps
    return model


def clear_cache():
    """Clear the inspection cache (call when model files may have changed)."""
    global _cache
    _cache = {}


if __name__ == "__main__":
    import json
    import sys
    dirs = sys.argv[1:] if len(sys.argv) > 1 else ["."]
    models = scan_models(dirs)
    for m in models:
        print(f"{m['name']:50s}  {m['size_gb']:6.1f} GB  "
              f"{'MoE' if m['is_moe'] else 'Dense':5s}  "
              f"{'✓ VRAM' if m['fits_in_vram'] else '✗'}")
