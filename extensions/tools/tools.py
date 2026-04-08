"""
title: LocalAI Platform Tools
author: LocalAI Platform
description: Full agent toolset for local AI — file ops, system, web, model management.
version: 2.0.0
"""
import ast
import io
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Optional

# ─── Optional imports (graceful fallback) ────────────────────────
try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# Path to the LocalAI Platform repo root (parent of extensions/)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Tools:
    """LocalAI Platform agent toolset — all tools available to the AI assistant."""

    # ─── 1. execute_command ──────────────────────────────────────
    def execute_command(self, command: str, timeout_seconds: int = 30) -> str:
        """
        Execute a shell command and return its output.

        Args:
            command: Shell command to run (passed to /bin/sh or cmd.exe).
            timeout_seconds: Maximum seconds to wait (default 30).

        Returns:
            Combined stdout/stderr output, or error message.
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout_seconds}s"
        except Exception as e:
            return f"Error: {e}"

    # ─── 2. read_file ────────────────────────────────────────────
    def read_file(self, filepath: str, max_lines: int = 500) -> str:
        """
        Read the contents of a file.

        Args:
            filepath: Absolute or relative path to the file.
            max_lines: Maximum number of lines to return (default 500).

        Returns:
            File contents as a string (truncated if too long).
        """
        try:
            p = Path(filepath)
            if not p.exists():
                return f"Error: file not found: {filepath}"
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            if len(lines) > max_lines:
                truncated = len(lines) - max_lines
                lines = lines[:max_lines]
                lines.append(f"\n... ({truncated} more lines truncated)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error reading file: {e}"

    # ─── 3. write_file ───────────────────────────────────────────
    def write_file(self, filepath: str, content: str) -> str:
        """
        Write content to a file (creates or overwrites).

        Args:
            filepath: Absolute or relative path to the file.
            content: Content to write.

        Returns:
            Success message or error.
        """
        try:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} bytes to {filepath}"
        except Exception as e:
            return f"Error writing file: {e}"

    # ─── 4. edit_file ────────────────────────────────────────────
    def edit_file(self, filepath: str, old_text: str, new_text: str) -> str:
        """
        Replace exact text in a file (patch-style edit).

        Args:
            filepath: Path to the file to edit.
            old_text: Exact text to find and replace.
            new_text: Replacement text.

        Returns:
            Success message or error.
        """
        try:
            p = Path(filepath)
            if not p.exists():
                return f"Error: file not found: {filepath}"
            content = p.read_text(encoding="utf-8")
            if old_text not in content:
                return f"Error: old_text not found in {filepath}"
            count = content.count(old_text)
            new_content = content.replace(old_text, new_text, 1)
            p.write_text(new_content, encoding="utf-8")
            return f"Replaced 1 occurrence (of {count} total) in {filepath}"
        except Exception as e:
            return f"Error editing file: {e}"

    # ─── 5. list_directory ───────────────────────────────────────
    def list_directory(
        self,
        path: str = ".",
        pattern: str = "*",
        recursive: bool = False,
    ) -> str:
        """
        List files and directories at a path, optionally with a glob pattern.

        Args:
            path: Directory path to list.
            pattern: Glob pattern to filter (default "*").
            recursive: If true, search recursively (default false).

        Returns:
            Newline-separated list of matching paths.
        """
        try:
            p = Path(path)
            if not p.is_dir():
                return f"Error: not a directory: {path}"
            if recursive:
                matches = sorted(p.rglob(pattern))
            else:
                matches = sorted(p.glob(pattern))
            if not matches:
                return "(no matches)"
            lines = []
            for m in matches[:500]:
                rel = m.relative_to(p)
                suffix = "/" if m.is_dir() else ""
                lines.append(str(rel) + suffix)
            if len(matches) > 500:
                lines.append(f"... ({len(matches) - 500} more)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing directory: {e}"

    # ─── 6. grep_search ──────────────────────────────────────────
    def grep_search(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        case_sensitive: bool = True,
    ) -> str:
        """
        Search for a regex pattern inside files.

        Args:
            pattern: Regular expression to search for.
            path: Directory or file to search in.
            file_pattern: Glob pattern to filter files (e.g. "*.py").
            case_sensitive: If false, search is case-insensitive.

        Returns:
            Matching lines in 'file:line: content' format.
        """
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
            p = Path(path)
            files = [p] if p.is_file() else list(p.rglob(file_pattern))
            results = []
            for f in files[:200]:
                if not f.is_file():
                    continue
                try:
                    for i, line in enumerate(
                        f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
                    ):
                        if regex.search(line):
                            results.append(f"{f}:{i}: {line}")
                            if len(results) >= 200:
                                results.append("... (limit 200 matches)")
                                return "\n".join(results)
                except Exception:
                    continue
            return "\n".join(results) if results else "(no matches)"
        except Exception as e:
            return f"Error in grep_search: {e}"

    # ─── 7. get_hardware_profile ─────────────────────────────────
    def get_hardware_profile(self) -> str:
        """
        Get GPU, RAM, and CPU info for the current machine.

        Returns:
            JSON string with hardware details.
        """
        try:
            sys.path.insert(0, str(_REPO_ROOT))
            from extensions.model_manager import hardware
            hw = hardware.detect_hardware()
            return json.dumps(hw, indent=2)
        except Exception as e:
            return f"Error detecting hardware: {e}"

    # ─── 8. inspect_gguf_model ───────────────────────────────────
    def inspect_gguf_model(self, filepath: str) -> str:
        """
        Read metadata from a GGUF model file (architecture, context, MoE, etc).

        Args:
            filepath: Path to the .gguf file.

        Returns:
            JSON string with model metadata.
        """
        try:
            sys.path.insert(0, str(_REPO_ROOT))
            from extensions.model_manager import gguf_inspector
            info = gguf_inspector.inspect_model(filepath)
            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error inspecting model: {e}"

    # ─── 9. generate_optimal_config ──────────────────────────────
    def generate_optimal_config(self, model_path: str) -> str:
        """
        Generate the optimal llama-server command flags for a model based on current hardware.

        Args:
            model_path: Path to the .gguf model file.

        Returns:
            JSON with flags and the command preview string.
        """
        try:
            sys.path.insert(0, str(_REPO_ROOT))
            from extensions.model_manager import hardware, gguf_inspector, optimizer
            hw = hardware.detect_hardware()
            model = gguf_inspector.inspect_model(model_path)
            model["path"] = model_path
            result = optimizer.compute_optimal_flags(hw, model)
            return json.dumps({
                "command_preview": result["command_preview"],
                "ctx_size": result["ctx_size"],
                "threads": result["threads"],
                "is_moe": result["is_moe"],
                "fits_in_vram": result["fits_in_vram"],
            }, indent=2)
        except Exception as e:
            return f"Error computing config: {e}"

    # ─── 10. check_llama_server ──────────────────────────────────
    def check_llama_server(self, port: int = 8001) -> str:
        """
        Check whether llama-server is running and return its status.

        Args:
            port: Port to check (default 8001).

        Returns:
            JSON with running status and model info.
        """
        if not _HAS_REQUESTS:
            return '{"error": "requests library not available"}'
        try:
            resp = _req.get(f"http://127.0.0.1:{port}/health", timeout=3)
            health = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            return json.dumps({"running": True, "status": health, "port": port}, indent=2)
        except Exception:
            return json.dumps({"running": False, "port": port}, indent=2)

    # ─── 11. find_models_on_system ───────────────────────────────
    def find_models_on_system(self, search_dirs: Optional[list] = None) -> str:
        """
        Find all GGUF model files on the system.

        Args:
            search_dirs: List of directories to search. Defaults to configured model_dirs.

        Returns:
            JSON list of found model paths with sizes.
        """
        try:
            sys.path.insert(0, str(_REPO_ROOT))
            from extensions.model_manager import model_scanner, hardware

            if not search_dirs:
                settings_path = _REPO_ROOT / "config" / "settings.json"
                if settings_path.exists():
                    settings = json.loads(settings_path.read_text())
                    search_dirs = settings.get("model_dirs", [])
                else:
                    search_dirs = []

            if not search_dirs:
                return json.dumps({"error": "No model directories configured. Add them in Settings."})

            hw = hardware.detect_hardware()
            models = model_scanner.scan_models(search_dirs, hw)
            summary = [
                {
                    "name": m["name"],
                    "path": m["path"],
                    "size_gb": m["size_gb"],
                    "is_moe": m["is_moe"],
                    "fits_in_vram": m["fits_in_vram"],
                }
                for m in models
            ]
            return json.dumps(summary, indent=2)
        except Exception as e:
            return f"Error finding models: {e}"

    # ─── 12. web_search ──────────────────────────────────────────
    def web_search(self, query: str, num_results: int = 5) -> str:
        """
        Search the web using DuckDuckGo (no API key required).

        Args:
            query: Search query string.
            num_results: Number of results to return (default 5).

        Returns:
            JSON list of results with title, url, snippet.
        """
        if not _HAS_REQUESTS:
            return '{"error": "requests library not available"}'
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; LocalAI/2.0)"}
            resp = _req.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=headers,
                timeout=10,
            )
            # Parse results from HTML with simple regex
            results = []
            title_re = re.compile(r'<a[^>]+class="result__a"[^>]*>(.+?)</a>', re.DOTALL)
            url_re = re.compile(r'<a[^>]+class="result__url"[^>]*>([^<]+)</a>')
            snippet_re = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.+?)</a>', re.DOTALL)

            titles = title_re.findall(resp.text)
            urls = url_re.findall(resp.text)
            snippets = snippet_re.findall(resp.text)

            for i in range(min(num_results, len(titles))):
                title = re.sub(r'<[^>]+>', '', titles[i]).strip()
                url = urls[i].strip() if i < len(urls) else ""
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
                results.append({"title": title, "url": url, "snippet": snippet})

            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error in web_search: {e}"

    # ─── 13. web_fetch ───────────────────────────────────────────
    def web_fetch(self, url: str, timeout_seconds: int = 15) -> str:
        """
        Fetch the text content of a web page.

        Args:
            url: URL to fetch.
            timeout_seconds: Request timeout (default 15).

        Returns:
            Plain text content of the page (HTML stripped, max 8000 chars).
        """
        if not _HAS_REQUESTS:
            return "Error: requests library not available"
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; LocalAI/2.0)"}
            resp = _req.get(url, headers=headers, timeout=timeout_seconds)
            # Strip HTML tags
            text = re.sub(r'<style[^>]*>.*?</style>', '', resp.text, flags=re.DOTALL)
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 8000:
                text = text[:8000] + "\n... (truncated)"
            return text
        except Exception as e:
            return f"Error fetching {url}: {e}"

    # ─── 14. todo_write ──────────────────────────────────────────
    def todo_write(self, action: str, task: str, status: str = "pending") -> str:
        """
        Manage a simple task list stored in config/todos.json.

        Args:
            action: One of 'add', 'update', 'list', 'clear'.
            task: Task description (used for add/update).
            status: One of 'pending', 'in_progress', 'done' (for add/update).

        Returns:
            Updated task list as a formatted string.
        """
        todos_path = _REPO_ROOT / "config" / "todos.json"
        todos_path.parent.mkdir(parents=True, exist_ok=True)

        todos = []
        if todos_path.exists():
            try:
                todos = json.loads(todos_path.read_text())
            except Exception:
                todos = []

        if action == "list":
            if not todos:
                return "No tasks."
            return "\n".join(
                f"[{t['status'].upper()[:1]}] {t['task']}" for t in todos
            )

        elif action == "add":
            todos.append({"task": task, "status": status})
            todos_path.write_text(json.dumps(todos, indent=2))
            return f"Added task: {task}"

        elif action == "update":
            for t in todos:
                if t["task"] == task:
                    t["status"] = status
                    todos_path.write_text(json.dumps(todos, indent=2))
                    return f"Updated '{task}' → {status}"
            return f"Task not found: {task}"

        elif action == "clear":
            todos_path.write_text("[]")
            return "Cleared all tasks."

        else:
            return f"Unknown action: {action}. Use: add, update, list, clear"

    # ─── 15. ask_user_question ───────────────────────────────────
    def ask_user_question(self, question: str, options: Optional[list] = None) -> str:
        """
        Signal that the AI needs clarification before proceeding.
        Returns the question formatted for display to the user.

        Args:
            question: The question to ask the user.
            options: Optional list of suggested answer options.

        Returns:
            Formatted question string (the AI should pause and show this to the user).
        """
        if options:
            opts = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(options))
            return f"QUESTION FOR USER:\n{question}\n\nOptions:\n{opts}"
        return f"QUESTION FOR USER:\n{question}"

    # ─── 16. schedule_task ───────────────────────────────────────
    def schedule_task(
        self, action: str, command: str = "", interval_min: int = 5
    ) -> str:
        """
        Schedule a recurring background task.

        Args:
            action: 'start', 'stop', or 'list'.
            command: Shell command to run on each interval.
            interval_min: Minutes between runs (default 5).

        Returns:
            Status message.
        """
        tasks_path = _REPO_ROOT / "config" / "scheduled_tasks.json"
        tasks_path.parent.mkdir(parents=True, exist_ok=True)

        tasks = []
        if tasks_path.exists():
            try:
                tasks = json.loads(tasks_path.read_text())
            except Exception:
                tasks = []

        if action == "list":
            if not tasks:
                return "No scheduled tasks."
            return "\n".join(
                f"[{t.get('id')}] every {t['interval_min']}m: {t['command']}" for t in tasks
            )

        elif action == "start":
            task_id = len(tasks) + 1
            tasks.append({"id": task_id, "command": command, "interval_min": interval_min})
            tasks_path.write_text(json.dumps(tasks, indent=2))

            def runner():
                while True:
                    time.sleep(interval_min * 60)
                    try:
                        subprocess.run(command, shell=True, timeout=60)
                    except Exception:
                        pass

            t = threading.Thread(target=runner, daemon=True)
            t.start()
            return f"Scheduled task #{task_id}: '{command}' every {interval_min} minutes"

        elif action == "stop":
            # Remove from config (won't stop running thread in this session)
            tasks = [t for t in tasks if str(t.get("id")) != str(command)]
            tasks_path.write_text(json.dumps(tasks, indent=2))
            return f"Removed task {command} from schedule."

        else:
            return f"Unknown action: {action}"

    # ─── 17. analyze_code ────────────────────────────────────────
    def analyze_code(self, filepath: str, analysis_type: str = "overview") -> str:
        """
        Analyze a Python file's structure (functions, classes, imports).

        Args:
            filepath: Path to the Python source file.
            analysis_type: 'overview', 'functions', 'classes', or 'imports'.

        Returns:
            Analysis result as formatted text.
        """
        try:
            p = Path(filepath)
            if not p.exists():
                return f"Error: file not found: {filepath}"
            source = p.read_text(encoding="utf-8")
            tree = ast.parse(source)

            results = []
            if analysis_type in ("overview", "imports"):
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        mod = node.module or ""
                        for alias in node.names:
                            imports.append(f"{mod}.{alias.name}")
                results.append(f"Imports ({len(imports)}):\n  " + "\n  ".join(imports[:30]))

            if analysis_type in ("overview", "classes"):
                classes = [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                ]
                results.append(f"Classes ({len(classes)}):\n  " + "\n  ".join(classes))

            if analysis_type in ("overview", "functions"):
                functions = [
                    f"{node.name}({', '.join(a.arg for a in node.args.args)})"
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                results.append(
                    f"Functions ({len(functions)}):\n  " +
                    "\n  ".join(functions[:50])
                )

            lines = source.count("\n")
            results.append(f"Lines of code: {lines}")
            return "\n\n".join(results)
        except SyntaxError as e:
            return f"Syntax error in {filepath}: {e}"
        except Exception as e:
            return f"Error analyzing code: {e}"

    # ─── 18. manage_process ──────────────────────────────────────
    def manage_process(
        self, action: str, process_name: str = "", port: int = 0
    ) -> str:
        """
        List, check, or kill system processes.

        Args:
            action: 'list', 'check', or 'kill'.
            process_name: Process name to filter/kill.
            port: Port to find owning process (for 'check').

        Returns:
            Process info or status message.
        """
        if not _HAS_PSUTIL:
            return "Error: psutil not available"
        try:
            if action == "list":
                procs = []
                for p in _psutil.process_iter(["pid", "name", "status", "memory_mb"]):
                    if not process_name or process_name.lower() in p.info["name"].lower():
                        procs.append(
                            f"PID {p.info['pid']:6d}  {p.info['status']:10s}  {p.info['name']}"
                        )
                return "\n".join(procs[:50]) or "No matching processes"

            elif action == "check":
                if port:
                    for conn in _psutil.net_connections():
                        if conn.laddr.port == port and conn.status == "LISTEN":
                            try:
                                p = _psutil.Process(conn.pid)
                                return json.dumps({
                                    "port": port,
                                    "pid": conn.pid,
                                    "name": p.name(),
                                    "status": "listening",
                                })
                            except Exception:
                                pass
                    return json.dumps({"port": port, "status": "not listening"})
                return "Specify port or process_name"

            elif action == "kill":
                killed = []
                for p in _psutil.process_iter(["pid", "name"]):
                    if process_name.lower() in p.info["name"].lower():
                        p.terminate()
                        killed.append(f"{p.info['name']} (PID {p.info['pid']})")
                if killed:
                    return "Terminated: " + ", ".join(killed)
                return f"No processes matching '{process_name}'"
            else:
                return f"Unknown action: {action}. Use: list, check, kill"
        except Exception as e:
            return f"Error in manage_process: {e}"

    # ─── 19. run_python ──────────────────────────────────────────
    def run_python(self, code: str, timeout_seconds: int = 30) -> str:
        """
        Execute a Python code snippet and return its output.
        WARNING: Executes arbitrary code — use with caution.

        Args:
            code: Python code to execute.
            timeout_seconds: Execution timeout (default 30).

        Returns:
            stdout output of the code, or error traceback.
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        result = {"output": "", "error": ""}

        def _exec():
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(compile(code, "<tool:run_python>", "exec"), {})  # noqa: S102
            except Exception as e:
                import traceback
                result["error"] = traceback.format_exc()

        thread = threading.Thread(target=_exec, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)

        if thread.is_alive():
            return f"Error: execution timed out after {timeout_seconds}s"

        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue() or result["error"]

        if err:
            return f"{out}\n[stderr/error]\n{err}".strip()
        return out.strip() or "(no output)"

    # ─── 20. benchmark_model ─────────────────────────────────────
    def benchmark_model(self, max_tokens: int = 100, port: int = 8001) -> str:
        """
        Run a speed benchmark on the currently loaded model.
        Generates tokens and measures tokens/sec.

        Args:
            max_tokens: Number of tokens to generate (default 100).
            port: llama-server port (default 8001).

        Returns:
            JSON with tokens_per_second, total_tokens, duration.
        """
        try:
            sys.path.insert(0, str(_REPO_ROOT))
            from extensions.model_manager.auto_optimizer import run_benchmark
            result = run_benchmark(port=port, max_tokens=max_tokens)
            return json.dumps({
                "tokens_per_second": result.tokens_per_second,
                "total_tokens": result.total_tokens,
                "duration_seconds": result.duration_seconds,
                "success": result.success,
                "error": result.error,
            }, indent=2)
        except Exception as e:
            return f"Error running benchmark: {e}"

    # ─── 21. get_optimization_results ───────────────────────────
    def get_optimization_results(self) -> str:
        """
        Read past auto-optimization results.
        The auto-optimizer runs from the UI (Model Manager > Optimize button),
        not from chat, because it requires restarting llama-server.
        Use this tool to check what optimizations were found.

        Returns:
            JSON with past optimization runs showing baseline vs final speed,
            improvements found, and the optimal command for each model.
        """
        try:
            sys.path.insert(0, str(_REPO_ROOT))
            from extensions.model_manager.auto_optimizer import get_optimization_history
            history = get_optimization_history()
            if not history:
                return (
                    "No optimization history found.\n"
                    "To run the auto-optimizer: open the Model Manager UI, "
                    "load a model, and click the 'Optimize' button. "
                    "It will benchmark different settings and find the fastest config."
                )
            # Format the most recent result nicely
            latest = history[-1]
            summary = (
                f"Last Optimization Run:\n"
                f"  Model: {latest.get('model', '?')}\n"
                f"  Baseline: {latest.get('baseline_tps', 0)} t/s\n"
                f"  Final:    {latest.get('final_tps', 0)} t/s\n"
                f"  Improvement: +{latest.get('improvement_pct', 0)}%\n"
                f"  Experiments: {latest.get('iterations', 0)}\n"
                f"  Improvements kept: {latest.get('improvements_found', 0)}\n"
                f"\nOptimal command:\n{latest.get('command_preview', '')}\n"
                f"\nTotal past runs: {len(history)}"
            )
            return summary
        except Exception as e:
            return f"Error reading optimization results: {e}"
