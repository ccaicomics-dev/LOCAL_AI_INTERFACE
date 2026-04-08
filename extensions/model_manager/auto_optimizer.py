"""
Auto-Optimizer for LocalAI Platform.

Inspired by Karpathy's autoresearch pattern: an autonomous loop that
benchmarks the current inference config, proposes variations, measures
improvement, and keeps what works.

The AI optimizes its own inference engine while running on it.

Flow:
  1. Baseline benchmark (current flags → tokens/sec)
  2. Propose a flag variation (batch size, KV quant, context, threads, etc.)
  3. Eject model → reload with new flags
  4. Benchmark again
  5. If faster → keep; if slower → revert
  6. Log results → repeat for N iterations
  7. Return the best configuration found

Metric: tokens/sec from llama-server /metrics endpoint
"""
import copy
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    import requests as _req
except ImportError:
    _req = None

logger = logging.getLogger("localai.auto_optimizer")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_PATH = _REPO_ROOT / "config" / "optimization_results.json"

# Benchmark prompt — designed to exercise the model reasonably
_BENCHMARK_PROMPT = (
    "Explain the key differences between transformers and recurrent neural "
    "networks, covering attention mechanisms, parallelization, and memory "
    "efficiency. Be thorough and detailed."
)

LLAMA_BASE = "http://127.0.0.1"


# ─── Tuning dimensions ───────────────────────────────────────────

# Each dimension: (flag_key, list of values to try)
# Ordered by expected impact — biggest gains first
TUNING_DIMENSIONS = [
    # Batch sizes — big impact on throughput
    ("-b", ["512", "1024", "2048", "4096", "8192"]),
    ("-ub", ["128", "256", "512", "1024", "2048", "4096"]),
    # KV cache quantization — trades quality for speed/VRAM
    ("--cache-type-k", ["q4_0", "q8_0", "f16"]),
    ("--cache-type-v", ["q4_0", "q8_0", "f16"]),
    # Thread count — hardware-dependent sweet spot
    ("--threads", None),  # Generated dynamically based on CPU cores
    # Context size — less context = faster
    ("--ctx-size", None),  # Generated dynamically based on VRAM
]

# MoE-specific dimensions
MOE_DIMENSIONS = [
    # Expert offload pattern variants
    ("-ot", [
        ".ffn_.*_exps.=CPU",           # All experts to CPU (default)
        ".ffn_.*_exps.weight=CPU",      # Only expert weights to CPU
    ]),
    # MoE benefits from larger batches
    ("-b", ["2048", "4096", "8192", "16384"]),
    ("-ub", ["1024", "2048", "4096", "8192"]),
]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0
    total_tokens: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.tokens_per_second > 0


@dataclass
class ExperimentResult:
    """Result of a single optimization experiment."""
    iteration: int
    dimension: str
    old_value: str
    new_value: str
    baseline_tps: float
    result_tps: float
    improvement_pct: float
    kept: bool
    flags_snapshot: dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class OptimizationReport:
    """Full report from an optimization run."""
    model_name: str
    model_path: str
    total_iterations: int
    improvements_found: int
    baseline_tps: float
    final_tps: float
    total_improvement_pct: float
    best_flags: dict
    command_preview: str
    experiments: list = field(default_factory=list)
    duration_seconds: float = 0.0


def _generate_thread_values(cpu_cores: int) -> list:
    """Generate thread count candidates based on CPU cores."""
    candidates = set()
    for divisor in [1, 0.75, 0.5, 0.25]:
        val = max(1, int(cpu_cores * divisor))
        candidates.add(str(min(val, 64)))
    # Also try common values
    for v in [4, 8, 12, 16, 24, 32]:
        if v <= cpu_cores:
            candidates.add(str(v))
    return sorted(candidates, key=int)


def _generate_ctx_values(current_ctx: int) -> list:
    """Generate context size candidates around the current value."""
    candidates = set()
    # Try halving and doubling
    for mult in [0.25, 0.5, 1.0, 2.0]:
        val = int(current_ctx * mult)
        # Round to nearest power of 2
        import math
        if val > 0:
            val = 2 ** round(math.log2(val))
            val = max(2048, min(val, 131072))
            candidates.add(str(val))
    return sorted(candidates, key=int)


def run_benchmark(port: int = 8001, max_tokens: int = 100) -> BenchmarkResult:
    """
    Run a benchmark by generating tokens and measuring speed.

    Args:
        port: llama-server port.
        max_tokens: Number of tokens to generate.

    Returns:
        BenchmarkResult with speed metrics.
    """
    if _req is None:
        return BenchmarkResult(error="requests library not available")

    url = f"{LLAMA_BASE}:{port}/v1/chat/completions"

    # Check server health first
    try:
        health = _req.get(f"{LLAMA_BASE}:{port}/health", timeout=5)
        if health.status_code != 200:
            return BenchmarkResult(error=f"Server not healthy: {health.status_code}")
    except Exception as e:
        return BenchmarkResult(error=f"Server unreachable: {e}")

    # Run inference benchmark
    start = time.time()
    try:
        resp = _req.post(
            url,
            json={
                "model": "benchmark",
                "messages": [{"role": "user", "content": _BENCHMARK_PROMPT}],
                "max_tokens": max_tokens,
                "temperature": 0.1,  # Low temp for consistency
                "stream": False,
            },
            timeout=120,
        )
        duration = time.time() - start

        if resp.status_code != 200:
            return BenchmarkResult(error=f"Inference failed: {resp.status_code}")

        data = resp.json()
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        if completion_tokens == 0:
            return BenchmarkResult(error="No tokens generated")

        tps = completion_tokens / duration if duration > 0 else 0

        # Also check /metrics for more accurate server-side TPS
        try:
            metrics_resp = _req.get(f"{LLAMA_BASE}:{port}/metrics", timeout=3)
            if metrics_resp.status_code == 200:
                for line in metrics_resp.text.splitlines():
                    if "tokens_per_second" in line and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) >= 2:
                            server_tps = float(parts[-1])
                            if server_tps > 0:
                                tps = server_tps  # Prefer server-side metric
        except Exception:
            pass

        return BenchmarkResult(
            tokens_per_second=round(tps, 2),
            total_tokens=completion_tokens,
            duration_seconds=round(duration, 2),
        )

    except Exception as e:
        return BenchmarkResult(error=f"Benchmark error: {e}")


async def auto_optimize(
    model_path: str,
    model_name: str,
    current_flags: dict,
    llama_server_binary: str,
    hw: dict,
    is_moe: bool = False,
    max_iterations: int = 10,
    port: int = 8001,
):
    """
    Autoresearch-style optimization loop.
    Yields progress events (SSE-compatible) as it runs.

    Args:
        model_path: Path to the GGUF model file.
        model_name: Human-readable name.
        current_flags: Starting flags dict from optimizer.py.
        llama_server_binary: Path to llama-server binary.
        hw: Hardware info dict.
        is_moe: Whether this is a MoE model.
        max_iterations: Maximum experiments to run.
        port: llama-server port.

    Yields:
        Progress event dicts with step/message/data fields.
    """
    from . import server_process as sp_module

    server = sp_module.get_server()
    best_flags = copy.deepcopy(current_flags)
    experiments = []
    improvements = 0
    start_time = time.time()

    yield {
        "step": "starting",
        "message": f"Starting auto-optimization for {model_name}",
        "iterations": max_iterations,
    }

    # ── Step 1: Baseline benchmark ──────────────────────────────
    yield {"step": "baseline", "message": "Running baseline benchmark..."}

    baseline = run_benchmark(port=port, max_tokens=100)
    if not baseline.success:
        yield {"step": "error", "message": f"Baseline benchmark failed: {baseline.error}"}
        return

    baseline_tps = baseline.tokens_per_second
    best_tps = baseline_tps

    yield {
        "step": "baseline_done",
        "message": f"Baseline: {baseline_tps} t/s",
        "tps": baseline_tps,
    }

    # ── Step 2: Build experiment queue ──────────────────────────
    dimensions = list(TUNING_DIMENSIONS)
    if is_moe:
        dimensions.extend(MOE_DIMENSIONS)

    # Fill in dynamic values
    resolved_dims = []
    for flag_key, values in dimensions:
        if values is None:
            if flag_key == "--threads":
                values = _generate_thread_values(hw.get("cpu_cores", 8))
            elif flag_key == "--ctx-size":
                current_ctx = int(best_flags.get("--ctx-size", "8192"))
                values = _generate_ctx_values(current_ctx)
            else:
                continue
        resolved_dims.append((flag_key, values))

    # ── Step 3: Optimization loop ───────────────────────────────
    iteration = 0
    for flag_key, candidates in resolved_dims:
        if iteration >= max_iterations:
            break

        current_value = best_flags.get(flag_key, "")

        for new_value in candidates:
            if iteration >= max_iterations:
                break
            if str(new_value) == str(current_value):
                continue  # Skip current value

            iteration += 1

            yield {
                "step": "experiment",
                "message": f"[{iteration}/{max_iterations}] Trying {flag_key}={new_value} (was {current_value})",
                "iteration": iteration,
                "dimension": flag_key,
                "old_value": str(current_value),
                "new_value": str(new_value),
            }

            # Apply new flag
            test_flags = copy.deepcopy(best_flags)
            if new_value is None:
                test_flags.pop(flag_key, None)
            else:
                test_flags[flag_key] = str(new_value)

            # Eject and reload with new flags
            server.stop()
            import asyncio
            await asyncio.sleep(2)  # Brief pause for VRAM to free

            # Reload model
            load_success = False
            async for event in server.start(
                model_path=model_path,
                llama_server_binary=llama_server_binary,
                flags=test_flags,
                model_name=model_name,
            ):
                if event.get("step") == "ready":
                    load_success = True
                elif event.get("step") == "error":
                    yield {
                        "step": "experiment_error",
                        "message": f"Failed to load with {flag_key}={new_value}: {event.get('message')}",
                        "iteration": iteration,
                    }
                    break

            if not load_success:
                # Revert to best known flags
                async for event in server.start(
                    model_path=model_path,
                    llama_server_binary=llama_server_binary,
                    flags=best_flags,
                    model_name=model_name,
                ):
                    if event.get("step") in ("ready", "error"):
                        break

                experiments.append(ExperimentResult(
                    iteration=iteration,
                    dimension=flag_key,
                    old_value=str(current_value),
                    new_value=str(new_value),
                    baseline_tps=best_tps,
                    result_tps=0,
                    improvement_pct=0,
                    kept=False,
                    error="Failed to load model",
                ))
                continue

            # Wait for model to settle
            await asyncio.sleep(3)

            # Benchmark with new flags
            result = run_benchmark(port=port, max_tokens=100)

            if not result.success:
                yield {
                    "step": "experiment_error",
                    "message": f"Benchmark failed: {result.error}",
                    "iteration": iteration,
                }
                # Revert
                server.stop()
                await asyncio.sleep(2)
                async for event in server.start(
                    model_path=model_path,
                    llama_server_binary=llama_server_binary,
                    flags=best_flags,
                    model_name=model_name,
                ):
                    if event.get("step") in ("ready", "error"):
                        break

                experiments.append(ExperimentResult(
                    iteration=iteration,
                    dimension=flag_key,
                    old_value=str(current_value),
                    new_value=str(new_value),
                    baseline_tps=best_tps,
                    result_tps=0,
                    improvement_pct=0,
                    kept=False,
                    error=result.error,
                ))
                continue

            new_tps = result.tokens_per_second
            improvement_pct = round(((new_tps - best_tps) / best_tps) * 100, 2) if best_tps > 0 else 0

            kept = new_tps > best_tps * 1.01  # Require >1% improvement to keep

            exp = ExperimentResult(
                iteration=iteration,
                dimension=flag_key,
                old_value=str(current_value),
                new_value=str(new_value),
                baseline_tps=best_tps,
                result_tps=new_tps,
                improvement_pct=improvement_pct,
                kept=kept,
                flags_snapshot=copy.deepcopy(test_flags) if kept else {},
            )
            experiments.append(exp)

            if kept:
                best_flags = copy.deepcopy(test_flags)
                best_tps = new_tps
                current_value = new_value
                improvements += 1
                yield {
                    "step": "improvement",
                    "message": f"KEPT {flag_key}={new_value}: {new_tps} t/s (+{improvement_pct}%)",
                    "iteration": iteration,
                    "tps": new_tps,
                    "improvement_pct": improvement_pct,
                }
            else:
                yield {
                    "step": "reverted",
                    "message": f"Reverted {flag_key}={new_value}: {new_tps} t/s ({improvement_pct:+.1f}%)",
                    "iteration": iteration,
                    "tps": new_tps,
                    "improvement_pct": improvement_pct,
                }
                # Revert to best flags
                server.stop()
                await asyncio.sleep(2)
                async for event in server.start(
                    model_path=model_path,
                    llama_server_binary=llama_server_binary,
                    flags=best_flags,
                    model_name=model_name,
                ):
                    if event.get("step") in ("ready", "error"):
                        break

    # ── Step 4: Final report ────────────────────────────────────
    total_improvement = round(((best_tps - baseline_tps) / baseline_tps) * 100, 2) if baseline_tps > 0 else 0
    duration = round(time.time() - start_time, 1)

    # Build command preview from best flags
    lines = ["llama-server \\"]
    items = list(best_flags.items())
    for i, (key, value) in enumerate(items):
        suffix = " \\" if i < len(items) - 1 else ""
        if value is None:
            lines.append(f"    {key}{suffix}")
        else:
            val_str = f'"{value}"' if " " in value or "=" in value else value
            lines.append(f"    {key} {val_str}{suffix}")
    command_preview = "\n".join(lines)

    report = OptimizationReport(
        model_name=model_name,
        model_path=model_path,
        total_iterations=iteration,
        improvements_found=improvements,
        baseline_tps=baseline_tps,
        final_tps=best_tps,
        total_improvement_pct=total_improvement,
        best_flags=best_flags,
        command_preview=command_preview,
        experiments=[asdict(e) for e in experiments],
        duration_seconds=duration,
    )

    # Save results to disk
    _save_results(report)

    yield {
        "step": "complete",
        "message": (
            f"Optimization complete! {improvements} improvements found.\n"
            f"Baseline: {baseline_tps} t/s → Final: {best_tps} t/s "
            f"(+{total_improvement}%)\n"
            f"Ran {iteration} experiments in {duration}s"
        ),
        "report": asdict(report),
    }


def _save_results(report: OptimizationReport):
    """Save optimization results to config/optimization_results.json."""
    try:
        results = []
        if _RESULTS_PATH.exists():
            results = json.loads(_RESULTS_PATH.read_text())
        results.append({
            "timestamp": time.time(),
            "model": report.model_name,
            "baseline_tps": report.baseline_tps,
            "final_tps": report.final_tps,
            "improvement_pct": report.total_improvement_pct,
            "improvements_found": report.improvements_found,
            "iterations": report.total_iterations,
            "duration_s": report.duration_seconds,
            "best_flags": report.best_flags,
            "command_preview": report.command_preview,
        })
        _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RESULTS_PATH.write_text(json.dumps(results, indent=2))
    except Exception as e:
        logger.error(f"Failed to save optimization results: {e}")


def get_optimization_history() -> list:
    """Return all past optimization results."""
    if _RESULTS_PATH.exists():
        try:
            return json.loads(_RESULTS_PATH.read_text())
        except Exception:
            pass
    return []
