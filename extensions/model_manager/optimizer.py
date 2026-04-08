"""
Optimal llama-server flag calculator for LocalAI Platform.
Given hardware info and model metadata, computes the best flags for inference.
"""
import math
from typing import Optional


def _next_power_of_two(n: int) -> int:
    """Return the largest power of 2 <= n."""
    if n <= 0:
        return 8192
    return 2 ** int(math.log2(n))


def _clamp(value: int, min_val: int, max_val: int) -> int:
    return max(min_val, min(max_val, value))


def compute_context_size(gpu_vram_mb: int, model_size_gb: float, is_moe: bool) -> int:
    """
    Compute optimal context size based on remaining VRAM after model loads.

    For MoE models, only ~25% of model size goes to VRAM (attention layers).
    For dense models, the full model size is in VRAM.
    """
    if is_moe:
        # Only attention/non-expert layers go to VRAM
        vram_used_by_model_mb = (model_size_gb * 0.25) * 1024
    else:
        vram_used_by_model_mb = model_size_gb * 1024

    safety_margin_mb = 1500
    remaining_mb = gpu_vram_mb - vram_used_by_model_mb - safety_margin_mb

    if remaining_mb <= 0:
        return 8192

    # ~2000 MB per 8192 tokens of context (q8_0 KV cache)
    raw_ctx = int((remaining_mb / 2000) * 8192)
    ctx = _clamp(raw_ctx, 8192, 131072)
    return _next_power_of_two(ctx)


def compute_optimal_flags(hw: dict, model: dict, overrides: Optional[dict] = None) -> dict:
    """
    Compute optimal llama-server command flags for the given hardware and model.

    Args:
        hw: Hardware info dict from hardware.detect_hardware()
        model: Model info dict from gguf_inspector.inspect_model()
        overrides: Optional dict of user overrides (ctx_size, threads, gpu_layers, etc.)

    Returns:
        dict with keys:
            flags (dict): All flags as key-value pairs
            command_args (list[str]): Ready-to-use argument list for subprocess
            command_preview (str): Human-readable shell command
            ctx_size (int): Context window size
            threads (int): CPU thread count
            is_moe (bool): Whether MoE offload was applied
            fits_in_vram (bool): Whether model fits in GPU VRAM
            fits_with_ram (bool): Whether model fits with RAM offload
            estimated_vram_gb (float): Estimated VRAM usage in GB
    """
    overrides = overrides or {}

    gpu_vram_mb = hw.get("gpu_vram_mb", 0)
    gpu_vram_free_mb = hw.get("gpu_vram_free_mb", 0)
    has_nvidia = hw.get("has_nvidia", False)
    cpu_cores = hw.get("cpu_cores", 4)
    ram_mb = hw.get("ram_mb", 0)

    is_moe = model.get("is_moe", False)
    model_size_gb = model.get("size_gb", 0.0)
    model_path = model.get("path", "")

    # --- Compute derived values ---
    threads = overrides.get("threads", min(cpu_cores, 32))
    gpu_layers = overrides.get("gpu_layers", 999)

    if gpu_vram_mb > 0:
        ctx_size = overrides.get("ctx_size") or compute_context_size(
            gpu_vram_mb, model_size_gb, is_moe
        )
    else:
        ctx_size = overrides.get("ctx_size") or 8192

    # Fit estimates
    model_size_mb = model_size_gb * 1024
    if is_moe:
        vram_needed_mb = model_size_mb * 0.25
    else:
        vram_needed_mb = model_size_mb
    fits_in_vram = vram_needed_mb <= gpu_vram_free_mb
    fits_with_ram = model_size_mb <= (gpu_vram_free_mb + ram_mb * 0.8)

    # --- Build flags dict ---
    flags = {
        "--model": model_path,
        "-ngl": str(gpu_layers),
        "--ctx-size": str(ctx_size),
        "--threads": str(threads),
        "--cache-type-k": "q8_0",
        "--cache-type-v": "q8_0",
        "--host": "0.0.0.0",
        "--port": "8001",
        "--jinja": None,  # boolean flag, no value
    }

    if has_nvidia or overrides.get("flash_attn"):
        flags["--flash-attn"] = None

    if is_moe:
        flags["-ot"] = ".ffn_.*_exps.=CPU"
        flags["-b"] = "4096"
        flags["-ub"] = "4096"
    else:
        flags["-b"] = "2048"
        flags["-ub"] = "512"

    # Apply any remaining user overrides
    if "kv_quant" in overrides:
        flags["--cache-type-k"] = overrides["kv_quant"]
        flags["--cache-type-v"] = overrides["kv_quant"]

    # --- Build command args list ---
    command_args = ["llama-server"]
    for key, value in flags.items():
        command_args.append(key)
        if value is not None:
            command_args.append(value)

    # --- Build human-readable preview ---
    lines = ["llama-server \\"]
    items = list(flags.items())
    for i, (key, value) in enumerate(items):
        suffix = " \\" if i < len(items) - 1 else ""
        if value is None:
            lines.append(f"    {key}{suffix}")
        else:
            # Quote values that contain spaces or special chars
            val_str = f'"{value}"' if " " in value or "=" in value else value
            lines.append(f"    {key} {val_str}{suffix}")
    command_preview = "\n".join(lines)

    return {
        "flags": flags,
        "command_args": command_args,
        "command_preview": command_preview,
        "ctx_size": ctx_size,
        "threads": threads,
        "gpu_layers": gpu_layers,
        "is_moe": is_moe,
        "fits_in_vram": fits_in_vram,
        "fits_with_ram": fits_with_ram,
        "estimated_vram_gb": round(vram_needed_mb / 1024, 1),
    }


if __name__ == "__main__":
    import json
    # Example: RTX 5090 with a 122B MoE model
    hw = {
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_vram_mb": 32768,
        "gpu_vram_free_mb": 30000,
        "has_nvidia": True,
        "ram_mb": 196608,
        "cpu_cores": 24,
    }
    model = {
        "name": "Qwen3.5-122B-A10B-Q4_K_M",
        "path": "/models/Qwen3.5-122B-A10B-Q4_K_M.gguf",
        "size_gb": 73.0,
        "is_moe": True,
        "expert_count": 128,
    }
    result = compute_optimal_flags(hw, model)
    print(result["command_preview"])
    print(f"\nContext: {result['ctx_size']:,} tokens")
    print(f"Threads: {result['threads']}")
    print(f"Fits in VRAM: {result['fits_in_vram']}")
