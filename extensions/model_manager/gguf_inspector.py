"""
GGUF model metadata inspector for LocalAI Platform.
Reads model info from GGUF files without loading weights into memory.
"""
import os
import re
from pathlib import Path
from typing import Optional

try:
    from gguf import GGUFReader
    _GGUF_AVAILABLE = True
except ImportError:
    _GGUF_AVAILABLE = False


# Regex to detect split GGUF shards: model-00001-of-00005.gguf
_SPLIT_PATTERN = re.compile(r"^(.+)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)

# Known GGUF quantization type IDs → human-readable names
_QUANT_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K", 19: "Q8_K", 20: "IQ2_XXS", 21: "IQ2_XS",
    24: "IQ3_XXS", 26: "IQ1_S", 27: "IQ4_NL", 28: "IQ3_S",
    29: "IQ3_M", 30: "IQ2_S", 31: "IQ2_M", 32: "IQ4_XS",
    33: "IQ1_M", 34: "BF16",
}


def _read_field_str(reader, key: str) -> Optional[str]:
    """Read a string field from a GGUFReader."""
    try:
        field = reader.get_field(key)
        if field is None:
            return None
        # String parts are stored as byte arrays in the last part
        value = field.parts[-1]
        if hasattr(value, "tobytes"):
            return value.tobytes().decode("utf-8", errors="replace")
        return str(value)
    except Exception:
        return None


def _read_field_int(reader, key: str) -> Optional[int]:
    """Read an integer field from a GGUFReader."""
    try:
        field = reader.get_field(key)
        if field is None:
            return None
        return int(field.parts[-1][0])
    except Exception:
        return None


def _find_split_shards(path: str) -> list:
    """
    Given the first shard path, return all shard paths in order.
    Returns [path] if not a split model.
    """
    p = Path(path)
    m = _SPLIT_PATTERN.match(p.name)
    if not m:
        return [path]
    prefix = m.group(1)
    total = int(m.group(3))
    parent = p.parent
    shards = []
    for i in range(1, total + 1):
        shard = parent / f"{prefix}-{i:05d}-of-{total:05d}.gguf"
        if shard.exists():
            shards.append(str(shard))
    return shards if shards else [path]


def inspect_model(path: str) -> dict:
    """
    Read GGUF metadata from a model file (or first shard of a split model).

    Args:
        path: Absolute path to the .gguf file.

    Returns:
        dict with keys:
            name (str): Model name from metadata or filename
            architecture (str): e.g. "llama", "qwen2moe"
            context_length (int | None): Native context window
            block_count (int | None): Number of transformer layers
            embedding_length (int | None): Hidden dimension size
            expert_count (int | None): MoE expert count (None for dense)
            expert_used_count (int | None): Active experts per token
            is_moe (bool): True if mixture-of-experts model
            quantization (str): e.g. "Q4_K_M"
            is_split (bool): True if model is split across multiple files
            shard_count (int): Number of shards (1 if not split)
            shard_paths (list[str]): All shard paths
            size_bytes (int): Total size across all shards
            size_gb (float): Total size in GB
    """
    path = str(Path(path).resolve())

    # Detect split model shards
    shard_paths = _find_split_shards(path)
    is_split = len(shard_paths) > 1
    shard_count = len(shard_paths)

    # Total size across all shards
    size_bytes = sum(
        os.path.getsize(s) for s in shard_paths if os.path.exists(s)
    )
    size_gb = round(size_bytes / (1024 ** 3), 2)

    # Read metadata from first shard only
    if not _GGUF_AVAILABLE:
        return {
            "name": Path(path).stem,
            "architecture": "unknown",
            "context_length": None,
            "block_count": None,
            "embedding_length": None,
            "expert_count": None,
            "expert_used_count": None,
            "is_moe": False,
            "quantization": "unknown",
            "is_split": is_split,
            "shard_count": shard_count,
            "shard_paths": shard_paths,
            "size_bytes": size_bytes,
            "size_gb": size_gb,
        }

    try:
        reader = GGUFReader(path, mode="r")
    except Exception as e:
        return {
            "name": Path(path).stem,
            "architecture": "unknown",
            "context_length": None,
            "block_count": None,
            "embedding_length": None,
            "expert_count": None,
            "expert_used_count": None,
            "is_moe": False,
            "quantization": "unknown",
            "is_split": is_split,
            "shard_count": shard_count,
            "shard_paths": shard_paths,
            "size_bytes": size_bytes,
            "size_gb": size_gb,
            "error": str(e),
        }

    arch = _read_field_str(reader, "general.architecture") or "unknown"
    name = _read_field_str(reader, "general.name") or Path(path).stem

    # Architecture-specific fields use {arch} prefix
    context_length = _read_field_int(reader, f"{arch}.context_length")
    block_count = _read_field_int(reader, f"{arch}.block_count")
    embedding_length = _read_field_int(reader, f"{arch}.embedding_length")
    expert_count = _read_field_int(reader, f"{arch}.expert_count")
    expert_used_count = _read_field_int(reader, f"{arch}.expert_used_count")

    # MoE detection: expert_count > 1
    is_moe = expert_count is not None and expert_count > 1

    # Quantization type
    file_type = _read_field_int(reader, "general.file_type")
    quantization = _QUANT_NAMES.get(file_type, f"type_{file_type}") if file_type is not None else "unknown"

    return {
        "name": name,
        "architecture": arch,
        "context_length": context_length,
        "block_count": block_count,
        "embedding_length": embedding_length,
        "expert_count": expert_count,
        "expert_used_count": expert_used_count,
        "is_moe": is_moe,
        "quantization": quantization,
        "is_split": is_split,
        "shard_count": shard_count,
        "shard_paths": shard_paths,
        "size_bytes": size_bytes,
        "size_gb": size_gb,
    }


if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print("Usage: python gguf_inspector.py <model.gguf>")
        sys.exit(1)
    print(json.dumps(inspect_model(sys.argv[1]), indent=2))
