"""Shared tool-call formatting — unified emoji map and compact line renderer.

Provides a single source of truth for tool-call display across all providers
(Claude, Pi, Codex, Gemini).  Every provider should build its tool-use summary
by calling ``format_tool_line``; direct emoji/bold/backtick assembly in
individual providers is replaced by this module.
"""

from __future__ import annotations

import re
import json
import os

from .expandable_quote import format_expandable_quote

# ---------------------------------------------------------------------------
# Emoji map
# ---------------------------------------------------------------------------

TOOL_EMOJI: dict[str, str] = {
    # Claude canonical names
    "Bash": "\U0001f4bb",
    "Read": "\U0001f4d6",
    "Write": "\U0001f4dd",
    "Edit": "✏️",
    "MultiEdit": "✏️",
    "NotebookEdit": "✏️",
    "Grep": "\U0001f50e",
    "Glob": "\U0001f4c2",
    "LS": "\U0001f4c2",
    "Task": "\U0001f916",
    "TaskCreate": "\U0001f4cb",
    "TaskUpdate": "\U0001f4cb",
    "TaskList": "\U0001f4cb",
    "TodoWrite": "\U0001f4cb",
    "TodoRead": "\U0001f4cb",
    "ExitPlanMode": "\U0001f4cb",
    "WebFetch": "\U0001f310",
    "WebSearch": "\U0001f50e",
    "AskUserQuestion": "❓",
    "Skill": "\U0001f4da",
    # Pi lowercase aliases
    "bash": "\U0001f4bb",
    "read": "\U0001f4d6",
    "write": "\U0001f4dd",
    "edit": "✏️",
    "grep": "\U0001f50e",
    "glob": "\U0001f4c2",
    "find": "\U0001f4c2",
    "ls": "\U0001f4c2",
    "list": "\U0001f4c2",
    "webfetch": "\U0001f310",
    "web_fetch": "\U0001f310",
    "websearch": "\U0001f50e",
    "web_search": "\U0001f50e",
    # Codex native names
    "exec_command": "\U0001f4bb",
    "shell": "\U0001f4bb",
    "terminal": "\U0001f4bb",
    "run": "\U0001f4bb",
    "apply_patch": "✏️",
    "write_file": "\U0001f4dd",
    "read_file": "\U0001f4d6",
    "view": "\U0001f4d6",
    "search_files": "\U0001f50e",
    "ripgrep": "\U0001f50e",
    "search": "\U0001f50e",
    "fetch": "\U0001f310",
    # MCP bare tool names (used after prefix stripping)
    "ask_question": "❓",
}

_FALLBACK_EMOJI = "\U0001f527"

_MCP_PREFIX_RE = re.compile(r"^mcp__[^_]+__(.+)$")
_WHITESPACE_RE = re.compile(r"\s+")

_MAX_COMPACT_ARG = 50
_MAX_CCBOT_ARG = 200
_CCBOT_NAMES = {name.lower(): name for name in TOOL_EMOJI if name[0].isupper()} | {
    "exec_command": "Bash",
    "shell": "Bash",
    "run_shell_command": "Bash",
    "read_file": "Read",
    "write_file": "Write",
    "apply_patch": "Edit",
    "exec": "Exec",
    "functions.exec": "Exec",
}
_MARKDOWN_LITERAL_RE = re.compile(r"([\\`*_{}\[\]<>()#+.!|~$&-])")


def use_ccbot_tool_style() -> bool:
    """Whether tool messages should use the terminal-style ccbot layout."""
    return os.getenv("CCGRAM_TOOL_STYLE", "compact").strip().lower() == "ccbot"


def format_ccbot_tool_result(text: str, tool_name: str | None = None) -> str:
    """Render the ccbot result label and expandable output body."""
    if not text:
        return ""
    name = _CCBOT_NAMES.get((tool_name or "").lower(), tool_name)
    line_count = text.count("\n") + 1
    if name == "Read":
        return f"  ⎿  Read {line_count} lines"
    if name == "Write":
        return f"  ⎿  Wrote {line_count} lines"
    nonempty_lines = sum(bool(line.strip()) for line in text.split("\n"))
    search_results = text.count("\n\n") + 1
    stats = {
        "Bash": f"Output {line_count} lines",
        "Exec": f"Output {line_count} lines",
        "Grep": f"Found {nonempty_lines} matches",
        "Glob": f"Found {nonempty_lines} files",
        "Task": f"Agent output {line_count} lines",
        "WebFetch": f"Fetched {len(text)} characters",
        "WebSearch": f"{search_results} search results",
    }.get(name or "")
    body = format_expandable_quote(text)
    return f"  ⎿  {stats}\n{body}" if stats else body


# Case-folded index over every TOOL_EMOJI key, so genuine case-insensitive
# lookup works (e.g. "TASKCREATE" → "TaskCreate"), not just round-trips on
# entries that happen to have a lowercase alias.
_TOOL_EMOJI_LOWER: dict[str, str] = {k.lower(): v for k, v in TOOL_EMOJI.items()}


def tool_emoji(name: str) -> str:
    """Return the display emoji for a tool name.

    Exact match first; then case-insensitive over the full key set; for
    ``mcp__server__tool`` names strip the prefix and retry.  Falls back to
    wrench (🔧).  Never returns an empty string.
    """
    if name in TOOL_EMOJI:
        return TOOL_EMOJI[name]

    lower = name.lower()
    if lower in _TOOL_EMOJI_LOWER:
        return _TOOL_EMOJI_LOWER[lower]

    mcp_match = _MCP_PREFIX_RE.match(name)
    if mcp_match:
        bare = mcp_match.group(1)
        if bare in TOOL_EMOJI:
            return TOOL_EMOJI[bare]
        bare_lower = bare.lower()
        if bare_lower in _TOOL_EMOJI_LOWER:
            return _TOOL_EMOJI_LOWER[bare_lower]

    return _FALLBACK_EMOJI


def compact_arg(text: str, cap: int = _MAX_COMPACT_ARG) -> str:
    """Collapse whitespace, replace backticks, and trim to *cap* characters.

    Collapses all whitespace runs (including newlines) to a single space,
    strips leading/trailing whitespace, replaces backtick characters with
    single quotes, and trims the result to *cap* characters appending "…"
    if the original was longer.
    """
    collapsed = _WHITESPACE_RE.sub(" ", text).strip()
    cleaned = collapsed.replace("`", "'")
    if len(cleaned) > cap:
        return cleaned[:cap] + "…"
    return cleaned


def format_tool_line(name: str, summary: str) -> str:
    """Build a compact one-line tool-call display string.

    Returns ``{emoji} **{name}**: `{summary}` `` when *summary* is non-empty
    after applying ``compact_arg``, otherwise ``{emoji} **{name}**``.
    The action word is bold (`**read**`), the argument is inline monospace
    (`` `src/foo.py` ``).  Names are lowercased for visual quietness.
    """
    if use_ccbot_tool_style():
        display_name = _CCBOT_NAMES.get(name.lower(), name)
        trimmed = summary[:_MAX_CCBOT_ARG]
        if len(summary) > _MAX_CCBOT_ARG:
            trimmed += "…"
        # Character references stay literal through both the math preprocessor
        # and Markdown. Backslash escapes such as \( and \[ trigger LaTeX.
        literal = _MARKDOWN_LITERAL_RE.sub(lambda m: f"&#{ord(m[0])};", trimmed)
        return f"**{display_name}**({literal})" if literal else f"**{display_name}**"
    emoji = tool_emoji(name)
    display_name = name.lower()
    trimmed = compact_arg(summary)
    if trimmed:
        return f"{emoji} **{display_name}**: `{trimmed}`"
    return f"{emoji} **{display_name}**"


def format_tool_details(summary: str, inputs: object) -> str:
    """Retain tool arguments below the compact heading for verbose delivery."""
    if use_ccbot_tool_style() or not inputs:
        return summary
    if isinstance(inputs, str):
        details = inputs
    elif isinstance(inputs, dict):
        command = inputs.get("cmd") or inputs.get("command")
        if isinstance(command, str):
            rest = {k: v for k, v in inputs.items() if k not in {"cmd", "command"}}
            details = command
            if rest:
                details += "\n\n" + json.dumps(rest, ensure_ascii=False, indent=2)
        else:
            details = json.dumps(inputs, ensure_ascii=False, indent=2)
    else:
        details = str(inputs)
    return summary + "\n" + format_expandable_quote(details)
