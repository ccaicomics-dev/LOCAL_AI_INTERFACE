"""
Tool Call Validator for LocalAI Platform.

Intercepts and validates tool calls from the LLM before execution.
Catches XML leaks, malformed JSON, and common model mistakes.
Can be used as middleware in the Open WebUI pipeline or standalone.
"""
import json
import logging
import re
import time
from typing import Optional

logger = logging.getLogger("localai.tool_validator")

# ─── Common XML leak patterns from various models ─────────────────
_XML_TOOL_CALL_PATTERNS = [
    # Qwen-style: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    re.compile(
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
        re.DOTALL
    ),
    # Qwen alternate: <|tool_call|>...<|/tool_call|>
    re.compile(
        r'<\|tool_call\|>\s*(\{.*?\})\s*<\|/tool_call\|>',
        re.DOTALL
    ),
    # Generic XML function call
    re.compile(
        r'<function_call>\s*(\{.*?\})\s*</function_call>',
        re.DOTALL
    ),
    # DeepSeek-style: <|tool▁call|>...<|tool▁call▁end|>
    re.compile(
        r'<\|tool▁call\|>\s*(\{.*?\})\s*<\|tool▁call▁end\|>',
        re.DOTALL
    ),
    # Mistral-style: [TOOL_CALLS] [{"name": ..., "arguments": ...}]
    re.compile(
        r'\[TOOL_CALLS\]\s*(\[.*?\])',
        re.DOTALL
    ),
    # Loose JSON in backticks (common hallucination)
    re.compile(
        r'```(?:json)?\s*(\{[^`]*?"name"\s*:.*?\})\s*```',
        re.DOTALL
    ),
]

# Known tool names from our tools.py
_KNOWN_TOOLS = {
    "execute_command", "read_file", "write_file", "edit_file",
    "list_directory", "grep_search", "get_hardware_profile",
    "inspect_gguf_model", "generate_optimal_config", "check_llama_server",
    "find_models_on_system", "web_search", "web_fetch", "todo_write",
    "ask_user_question", "schedule_task", "analyze_code",
    "manage_process", "run_python", "benchmark_model",
    "get_optimization_results",
}

# Tools that need confirmation before execution (destructive)
_DANGEROUS_TOOLS = {
    "execute_command",
    "write_file",
    "edit_file",
    "manage_process",
    "run_python",
    "schedule_task",
}


class ToolCallValidationError(Exception):
    """Raised when a tool call fails validation."""
    pass


class ToolCallLog:
    """Simple in-memory log of tool calls for debugging."""

    def __init__(self, max_entries: int = 200):
        self._entries = []
        self._max = max_entries

    def log(self, entry: dict):
        entry["timestamp"] = time.time()
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]

    def get_recent(self, n: int = 20) -> list:
        return self._entries[-n:]

    def clear(self):
        self._entries = []


# Module-level log instance
call_log = ToolCallLog()


def extract_tool_calls_from_xml(text: str) -> list:
    """
    Attempt to extract tool calls from raw model output that contains
    XML-style tool call tags (the "XML leak" problem).

    Args:
        text: Raw model output text that may contain XML tool calls.

    Returns:
        List of dicts with 'name' and 'arguments' keys, or empty list.
    """
    extracted = []

    for pattern in _XML_TOOL_CALL_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            try:
                parsed = json.loads(match)
                # Handle both single object and array
                if isinstance(parsed, list):
                    for item in parsed:
                        tc = _normalize_tool_call(item)
                        if tc:
                            extracted.append(tc)
                elif isinstance(parsed, dict):
                    tc = _normalize_tool_call(parsed)
                    if tc:
                        extracted.append(tc)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                fixed = _attempt_json_repair(match)
                if fixed:
                    tc = _normalize_tool_call(fixed)
                    if tc:
                        extracted.append(tc)

    return extracted


def _normalize_tool_call(data: dict) -> Optional[dict]:
    """
    Normalize a tool call dict to a consistent format.
    Models output tool calls in various formats — this standardizes them.
    """
    if not isinstance(data, dict):
        return None

    # Extract name — models use different keys
    name = (
        data.get("name") or
        data.get("function", {}).get("name") or
        data.get("tool_name") or
        data.get("action")
    )
    if not name:
        return None

    # Extract arguments
    arguments = (
        data.get("arguments") or
        data.get("parameters") or
        data.get("function", {}).get("arguments") or
        data.get("args") or
        {}
    )

    # Arguments might be a JSON string instead of dict
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}

    return {"name": str(name), "arguments": arguments}


def _attempt_json_repair(text: str) -> Optional[dict]:
    """Try to fix common JSON formatting issues from model output."""
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Fix single quotes → double quotes
    text = text.replace("'", '"')
    # Fix unquoted keys
    text = re.sub(r'(\{|,)\s*(\w+)\s*:', r'\1 "\2":', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_tool_call(tool_call: dict) -> dict:
    """
    Validate a tool call before execution.

    Args:
        tool_call: Dict with 'name' and 'arguments'.

    Returns:
        Validated and potentially cleaned tool call dict.

    Raises:
        ToolCallValidationError: If the tool call is invalid.
    """
    name = tool_call.get("name", "")
    arguments = tool_call.get("arguments", {})

    # 1. Check tool exists
    if name not in _KNOWN_TOOLS:
        # Try fuzzy matching (model might hallucinate similar names)
        close = _find_closest_tool(name)
        if close:
            logger.warning(f"Tool '{name}' not found, did you mean '{close}'?")
            raise ToolCallValidationError(
                f"Unknown tool '{name}'. Did you mean '{close}'? "
                f"Available tools: {', '.join(sorted(_KNOWN_TOOLS))}"
            )
        raise ToolCallValidationError(
            f"Unknown tool '{name}'. "
            f"Available tools: {', '.join(sorted(_KNOWN_TOOLS))}"
        )

    # 2. Validate arguments is a dict
    if not isinstance(arguments, dict):
        raise ToolCallValidationError(
            f"Tool '{name}' arguments must be a dict, got {type(arguments).__name__}"
        )

    # 3. Tool-specific validation
    _validate_tool_args(name, arguments)

    # 4. Log the call
    call_log.log({
        "name": name,
        "arguments": arguments,
        "status": "validated",
    })

    return {"name": name, "arguments": arguments}


def _validate_tool_args(name: str, args: dict):
    """Tool-specific argument validation."""

    if name == "execute_command":
        cmd = args.get("command", "")
        if not cmd:
            raise ToolCallValidationError("execute_command requires 'command' argument")
        # Block obviously dangerous commands
        dangerous_patterns = [
            r'\brm\s+-rf\s+/',  # rm -rf /
            r'\bmkfs\b',         # format disk
            r'\bdd\s+if=',       # disk destroyer
            r'>\s*/dev/sd',      # overwrite disk
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, cmd):
                raise ToolCallValidationError(
                    f"Blocked potentially destructive command: {cmd}"
                )

    elif name == "write_file":
        if not args.get("filepath"):
            raise ToolCallValidationError("write_file requires 'filepath' argument")
        if not args.get("content") and args.get("content") != "":
            raise ToolCallValidationError("write_file requires 'content' argument")

    elif name == "edit_file":
        if not args.get("filepath"):
            raise ToolCallValidationError("edit_file requires 'filepath' argument")
        if not args.get("old_text"):
            raise ToolCallValidationError("edit_file requires 'old_text' argument")

    elif name == "read_file":
        if not args.get("filepath"):
            raise ToolCallValidationError("read_file requires 'filepath' argument")

    elif name == "grep_search":
        if not args.get("pattern"):
            raise ToolCallValidationError("grep_search requires 'pattern' argument")

    elif name == "web_search":
        if not args.get("query"):
            raise ToolCallValidationError("web_search requires 'query' argument")

    elif name == "web_fetch":
        if not args.get("url"):
            raise ToolCallValidationError("web_fetch requires 'url' argument")

    elif name == "manage_process":
        action = args.get("action", "")
        if action not in ("list", "check", "kill"):
            raise ToolCallValidationError(
                f"manage_process action must be 'list', 'check', or 'kill', got '{action}'"
            )

    elif name == "todo_write":
        action = args.get("action", "")
        if action not in ("add", "update", "list", "clear"):
            raise ToolCallValidationError(
                f"todo_write action must be 'add', 'update', 'list', or 'clear', got '{action}'"
            )


def _find_closest_tool(name: str) -> Optional[str]:
    """Find the closest matching tool name (simple edit distance)."""
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    # Direct substring match
    for tool in _KNOWN_TOOLS:
        if name_lower in tool or tool in name_lower:
            return tool

    # Simple character overlap scoring
    best_score = 0
    best_match = None
    for tool in _KNOWN_TOOLS:
        common = set(name_lower) & set(tool)
        score = len(common) / max(len(name_lower), len(tool))
        if score > best_score and score > 0.5:
            best_score = score
            best_match = tool

    return best_match


def is_xml_leak(text: str) -> bool:
    """
    Quick check: does this text look like it contains XML tool call tags
    that should have been parsed by llama-server but weren't?
    """
    leak_indicators = [
        "<tool_call>",
        "<|tool_call|>",
        "<function_call>",
        "<|tool▁call|>",
        "[TOOL_CALLS]",
    ]
    return any(indicator in text for indicator in leak_indicators)


def clean_response(text: str) -> tuple:
    """
    Process a model response: detect XML leaks, extract tool calls if found,
    and return cleaned text.

    Args:
        text: Raw model response text.

    Returns:
        Tuple of (cleaned_text, extracted_tool_calls).
        If no XML leaks found, returns (original_text, []).
    """
    if not is_xml_leak(text):
        return text, []

    logger.warning("XML tool call leak detected in model output — attempting recovery")

    tool_calls = extract_tool_calls_from_xml(text)

    if tool_calls:
        # Remove the XML tags from the text to get the "clean" response
        cleaned = text
        for pattern in _XML_TOOL_CALL_PATTERNS:
            cleaned = pattern.sub('', cleaned)
        cleaned = cleaned.strip()

        call_log.log({
            "event": "xml_leak_recovered",
            "tool_calls_found": len(tool_calls),
            "tools": [tc["name"] for tc in tool_calls],
        })

        return cleaned, tool_calls
    else:
        call_log.log({
            "event": "xml_leak_unrecoverable",
            "raw_text": text[:500],
        })
        return text, []


def get_validation_stats() -> dict:
    """Return stats about recent tool calls for debugging."""
    recent = call_log.get_recent(100)
    total = len(recent)
    xml_leaks = sum(1 for e in recent if e.get("event") == "xml_leak_recovered")
    failures = sum(1 for e in recent if e.get("status") == "error")
    tools_used = {}
    for e in recent:
        name = e.get("name")
        if name:
            tools_used[name] = tools_used.get(name, 0) + 1

    return {
        "total_calls": total,
        "xml_leaks_recovered": xml_leaks,
        "validation_failures": failures,
        "tools_used": tools_used,
    }
