"""Codex CLI provider — OpenAI's terminal agent behind AgentProvider protocol.

Codex CLI uses a similar tmux-based launch model but differs in hook mechanism
(no SessionStart hook) and resume syntax (``resume`` subcommand, not a flag).

Transcript format: JSONL with entries ``{timestamp, type, payload}``.
Entry types: ``session_meta``, ``response_item``, ``input_item``, ``event_msg``,
``turn_context``.

Modern Codex ``response_item`` payloads use typed shapes:
  - ``type=message`` with ``role`` + content blocks
  - ``type=function_call`` with ``name``, ``arguments``, ``call_id``
  - ``type=function_call_output`` with ``call_id``, ``output``
"""

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, cast

from ccgram.expandable_quote import format_expandable_quote
from ccgram.providers.codex_format import format_codex_interactive_prompt
from ccgram.providers._jsonl import JsonlProvider
from ccgram.tool_format import (
    format_ccbot_tool_result,
    format_tool_details,
    format_tool_line,
    use_ccbot_tool_style,
)
from ccgram.providers.base import (
    RESUME_ID_RE,
    AgentMessage,
    MessageRole,
    ProviderCapabilities,
    ResumableSession,
    SessionStartEvent,
    StatusUpdate,
)
from ccgram.terminal_parser import extract_interactive_content

# Codex CLI known slash commands. Refreshed from
# github.com/openai/codex/blob/main/codex-rs/tui/tooltips.txt.
_CODEX_BUILTINS: dict[str, str] = {
    "/clear": "Clear conversation history",
    "/compact": "Summarize history and free up context",
    "/copy": "Copy last response as Markdown",
    "/feedback": "Send logs to maintainers",
    "/fork": "Branch chat into a new thread",
    "/init": "Create AGENTS.md project guidance",
    "/mcp": "List configured MCP tools",
    "/mention": "Attach files to conversation",
    "/mode": "Switch approval mode (suggest/auto-edit/full-auto)",
    "/model": "Switch model and reasoning effort",
    "/permissions": "Manage confirmation rules",
    "/personality": "Customize Codex communication style",
    "/plan": "Enter plan mode",
    "/rename": "Rename current thread",
    "/review": "Review current changes",
    "/side": "Start a side conversation in a temporary fork",
    "/skills": "List available skills",
    "/statusline": "Configure status line items",
    "/status": "Show model, approvals, token usage",
}

_TOOL_NAME_ALIASES: dict[str, str] = {
    "request_user_input": "AskUserQuestion",
    "apply_patch": "Edit",
}

# Minimum line count to trigger stats + expandable quote for tool results.
_TOOL_RESULT_QUOTE_THRESHOLD = 3
_EXEC_COMMAND_LITERAL_RE = re.compile(
    r'\btools\.exec_command\(\s*\{\s*(?:cmd|"cmd")\s*:\s*("(?:\\.|[^"\\])*")'
)


def build_trusted_codex_launch_command(command: str, cwd: str) -> str:
    """Trust the selected directory for this launch without editing user config.

    Use an inline TOML table so dots in directory names cannot be interpreted
    as config-key separators. Shell-quote the complete override as one argument.
    """
    path = json.dumps(str(Path(cwd).expanduser().resolve()), ensure_ascii=False)
    override = f'projects={{{path}={{trust_level="trusted"}}}}'
    return f"{command} -c {shlex.quote(override)}"


def _format_codex_tool_result(raw_tool_name: str, output_text: str) -> str:
    """Format a Codex tool result with stats summary and expandable quote.

    Mirrors Claude's ``TranscriptParser._format_tool_result_text`` behaviour:
    shell/exec output gets ``N lines`` + expandable quote; short outputs stay inline.
    """
    if not output_text:
        return "Done"
    if use_ccbot_tool_style() and raw_tool_name != "apply_patch":
        return format_ccbot_tool_result(output_text, raw_tool_name)
    line_count = output_text.count("\n") + 1

    if raw_tool_name in ("exec_command", "shell"):
        stats = f"  \u23bf  {line_count} lines"
        return stats + "\n" + format_expandable_quote(output_text)

    if raw_tool_name == "apply_patch":
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError, TypeError:
            parsed = None
        if isinstance(parsed, dict):
            result_text = parsed.get("output", "") or parsed.get("result", "")
            if isinstance(result_text, str) and result_text:
                return result_text
        return output_text

    if line_count > _TOOL_RESULT_QUOTE_THRESHOLD:
        stats = f"  \u23bf  {line_count} lines"
        return stats + "\n" + format_expandable_quote(output_text)

    return output_text


def _canonical_tool_name(name: str) -> str:
    """Map provider-native tool names to ccgram canonical names."""
    return _TOOL_NAME_ALIASES.get(name, name)


def _extract_text_blocks(content: Any) -> str:
    """Extract visible text from Codex message content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("output_text", "input_text"):
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts)


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Parse tool arguments from stringified JSON or dict."""
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _first_nonempty_string(data: dict[str, Any]) -> str:
    """Return first non-empty string value in a dict."""
    for value in data.values():
        if isinstance(value, str) and value:
            return value
    return ""


def _summarize_exec_command(args: dict[str, Any]) -> str:
    """Extract a concise command preview from exec_command/shell args."""
    cmd = args.get("cmd")
    if isinstance(cmd, str) and cmd:
        return cmd

    command = args.get("command")
    if isinstance(command, list):
        joined = " ".join(part for part in command if isinstance(part, str))
        if joined:
            return joined
    if isinstance(command, str) and command:
        return command
    return ""


def _format_tool_use_text(raw_tool_name: str, args: dict[str, Any]) -> str:
    """Build display text for a Codex tool_use item."""
    tool_name = _canonical_tool_name(raw_tool_name)
    summary = _summarize_tool_use(raw_tool_name, tool_name, args)
    return format_tool_details(format_tool_line(tool_name, summary), args)


def _summarize_tool_use(
    raw_tool_name: str,
    tool_name: str,
    args: dict[str, Any],
) -> str:
    """Create a short tool-use summary string."""
    if tool_name == "AskUserQuestion":
        return _summarize_question(args)
    if raw_tool_name in ("exec_command", "shell"):
        return _summarize_exec_command(args)
    if raw_tool_name == "write_stdin":
        chars = args.get("chars")
        return chars if isinstance(chars, str) else ""
    if raw_tool_name == "update_plan":
        plan = args.get("plan")
        if isinstance(plan, list):
            return f"{len(plan)} step(s)"
        return ""
    return _first_nonempty_string(args)


def _summarize_question(args: dict[str, Any]) -> str:
    """Extract the first question prompt for request_user_input."""
    questions = args.get("questions")
    if not (isinstance(questions, list) and questions):
        return ""
    first = questions[0]
    if not isinstance(first, dict):
        return ""
    question = first.get("question")
    return question if isinstance(question, str) else ""


def _extract_tool_output_text(output: Any) -> str:
    """Unwrap legacy strings and modern Codex content-block tool results."""
    if isinstance(output, str):
        try:
            decoded = json.loads(output)
        except json.JSONDecodeError, ValueError:
            decoded = None
        if isinstance(decoded, (dict, list)):
            return _extract_tool_output_text(decoded)
        marker = "\nOutput:\n"
        if marker in output:
            return output.split(marker, 1)[1].strip()
        return output.strip()
    if isinstance(output, list):
        return "\n".join(
            text for block in output if (text := _extract_tool_output_text(block))
        )
    if isinstance(output, dict):
        return _extract_tool_output_mapping(output)
    return "" if output is None else json.dumps(output, ensure_ascii=False)


def _extract_tool_output_mapping(output: dict[str, Any]) -> str:
    """Read text blocks, shell results, and composed tool-call envelopes."""
    if output.get("type") in ("input_text", "output_text", "text"):
        return _extract_tool_output_text(output.get("text", ""))
    if output.get("type") in ("input_image", "image", "image_url"):
        return "[Image output]"
    if "output" in output:
        # A shell envelope's output is already stdout. Do not reinterpret file
        # contents containing JSON or an "Output:" heading as another wrapper.
        inner = output["output"]
        text = (
            inner.strip()
            if isinstance(inner, str)
            else _extract_tool_output_text(inner)
        )
        if not text and output.get("session_id") and output.get("exit_code") is None:
            text = "Process still running."
        elif not text and output.get("exit_code"):
            text = f"Process exited with code {output['exit_code']}."
        return text
    if output.get("status") == "fulfilled" and "value" in output:
        return _extract_tool_output_text(output["value"])
    if output.get("status") == "rejected" and "reason" in output:
        return "Error: " + _extract_tool_output_text(output["reason"])
    if isinstance(output.get("content"), list):
        return _extract_tool_output_text(output["content"])
    return json.dumps(output, ensure_ascii=False)


def _single_wrapped_shell_command(input_text: str) -> str:
    """Display a simple exec wrapper as Bash(command), without evaluating JS."""
    if len(re.findall(r"\btools\.", input_text)) != 1:
        return ""
    match = _EXEC_COMMAND_LITERAL_RE.search(input_text)
    if match is None:
        return ""
    try:
        return json.loads(match[1])
    except json.JSONDecodeError, ValueError:
        return ""


def _format_request_user_input_result(output_text: str) -> str:
    """Summarize request_user_input answers."""
    if not output_text:
        return output_text
    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError:
        return output_text
    if not isinstance(parsed, dict):
        return output_text

    answers = parsed.get("answers")
    if not isinstance(answers, dict):
        return output_text

    selected: list[str] = []
    for answer in answers.values():
        if not isinstance(answer, dict):
            continue
        items = answer.get("answers")
        if isinstance(items, list):
            selected.extend(item for item in items if isinstance(item, str) and item)

    if selected:
        return "Selected: " + ", ".join(selected)
    return output_text


def _parse_custom_tool_call(
    payload: dict[str, Any],
    pending: dict[str, Any],
) -> tuple[list[AgentMessage], dict[str, Any]]:
    """Parse a custom_tool_call payload (e.g. apply_patch)."""
    raw_name_value = payload.get("name", "unknown")
    raw_name = (
        raw_name_value if isinstance(raw_name_value, str) else str(raw_name_value)
    )
    tool_name = _canonical_tool_name(raw_name)
    call_id = payload.get("call_id", "")

    # For apply_patch, summarize by counting file updates in the input string.
    input_text = payload.get("input", "")
    summary = ""
    if raw_name in ("exec", "functions.exec") and isinstance(input_text, str):
        summary = _single_wrapped_shell_command(input_text)
        if summary:
            tool_name = "Bash"
    if raw_name == "apply_patch" and isinstance(input_text, str):
        file_count = input_text.count("*** Update File:")
        file_count += input_text.count("*** Add File:")
        file_count += input_text.count("*** Delete File:")
        if file_count:
            summary = f"{file_count} file(s)"
    if not summary and isinstance(input_text, str) and input_text:
        # Pre-slice large apply_patch payloads so compact_arg's whitespace
        # regex never scans tens of KB — the final cap is 80 chars anyway.
        summary = input_text[:512]

    text = format_tool_details(format_tool_line(tool_name, summary), input_text)
    if isinstance(call_id, str) and call_id:
        pending[call_id] = (
            (raw_name, tool_name, text)
            if use_ccbot_tool_style()
            else (raw_name, tool_name)
        )
    return (
        [
            AgentMessage(
                text=text,
                role="assistant",
                content_type="tool_use",
                tool_use_id=call_id or None,
                tool_name=tool_name,
            )
        ],
        pending,
    )


def _pending_summary(call_id: Any, pending: dict[str, Any]) -> str:
    """Read the optional ccbot heading before consuming a pending tool."""
    value = pending.get(call_id) if isinstance(call_id, str) else None
    match value:
        case (_, _, str(heading)):
            return heading
    return ""


def _resolve_pending(
    call_id: Any,
    pending: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Pop a pending tool entry and return (raw_name, tool_name).

    Handles both current tuple format and legacy string values.
    """
    resolved = (
        pending.pop(call_id, None) if isinstance(call_id, str) and call_id else None
    )
    if isinstance(resolved, tuple):
        return resolved[0], resolved[1]
    if isinstance(resolved, str):
        return resolved, resolved
    return None, None


def _parse_custom_tool_call_output(
    payload: dict[str, Any],
    pending: dict[str, Any],
) -> tuple[list[AgentMessage], dict[str, Any]]:
    """Parse a custom_tool_call_output payload."""
    call_id = payload.get("call_id", "")
    heading = _pending_summary(call_id, pending)
    raw_name, tool_name = _resolve_pending(call_id, pending)

    output_text = _extract_tool_output_text(payload.get("output", ""))

    if raw_name and output_text:
        output_text = _format_codex_tool_result(raw_name, output_text)
    if not output_text:
        output_text = "Done"
    if heading:
        output_text = heading + "\n" + output_text

    return (
        [
            AgentMessage(
                text=output_text,
                role="assistant",
                content_type="tool_result",
                tool_use_id=call_id if isinstance(call_id, str) else None,
                tool_name=tool_name,
            )
        ],
        pending,
    )


def _parse_codex_response_item(
    payload: dict[str, Any],
    pending: dict[str, Any],
) -> tuple[list[AgentMessage], dict[str, Any]]:
    """Parse a Codex response_item payload."""
    payload_type = payload.get("type", "")
    if payload_type == "function_call":
        return _parse_function_call(payload, pending)
    if payload_type == "function_call_output":
        return _parse_function_call_output(payload, pending)
    if payload_type == "custom_tool_call":
        return _parse_custom_tool_call(payload, pending)
    if payload_type == "custom_tool_call_output":
        return _parse_custom_tool_call_output(payload, pending)
    return _parse_response_message(payload, pending)


def _parse_function_call(
    payload: dict[str, Any],
    pending: dict[str, Any],
) -> tuple[list[AgentMessage], dict[str, Any]]:
    """Parse a function_call payload into a tool_use AgentMessage."""
    raw_name_value = payload.get("name", "unknown")
    raw_name = (
        raw_name_value if isinstance(raw_name_value, str) else str(raw_name_value)
    )
    tool_name = _canonical_tool_name(raw_name)
    call_id = payload.get("call_id", "")
    if isinstance(call_id, str) and call_id:
        pending[call_id] = (raw_name, tool_name)
    args = _parse_tool_arguments(payload.get("arguments", {}))
    text = _format_tool_use_text(raw_name, args)
    if use_ccbot_tool_style() and isinstance(call_id, str) and call_id:
        pending[call_id] = (raw_name, tool_name, text)
    return (
        [
            AgentMessage(
                text=text,
                role="assistant",
                content_type="tool_use",
                tool_use_id=call_id or None,
                tool_name=tool_name,
            )
        ],
        pending,
    )


def _parse_function_call_output(
    payload: dict[str, Any],
    pending: dict[str, Any],
) -> tuple[list[AgentMessage], dict[str, Any]]:
    """Parse a function_call_output payload into a tool_result AgentMessage."""
    call_id = payload.get("call_id", "")
    heading = _pending_summary(call_id, pending)
    raw_name, tool_name = _resolve_pending(call_id, pending)

    output_text = _extract_tool_output_text(payload.get("output", ""))
    if tool_name == "AskUserQuestion":
        output_text = _format_request_user_input_result(output_text)
    elif raw_name and output_text:
        output_text = _format_codex_tool_result(raw_name, output_text)
    if not output_text:
        output_text = "Done"
    if heading:
        output_text = heading + "\n" + output_text

    return (
        [
            AgentMessage(
                text=output_text,
                role="assistant",
                content_type="tool_result",
                tool_use_id=call_id if isinstance(call_id, str) else None,
                tool_name=tool_name,
            )
        ],
        pending,
    )


def _parse_response_message(
    payload: dict[str, Any],
    pending: dict[str, Any],
) -> tuple[list[AgentMessage], dict[str, Any]]:
    """Parse message payloads (assistant/user text)."""
    role = payload.get("role", "")
    if role not in ("user", "assistant"):
        return [], pending
    text = _extract_text_blocks(payload.get("content", ""))
    if not text:
        return [], pending
    phase = payload.get("phase")
    return (
        [
            AgentMessage(
                text=text,
                role=cast(MessageRole, role),
                content_type="text",
                phase=phase if isinstance(phase, str) and phase else None,
            )
        ],
        pending,
    )


def _parse_event_message(
    payload: dict[str, Any],
    pending: dict[str, Any],
) -> tuple[list[AgentMessage], dict[str, Any]]:
    """Parse Codex event_msg payloads that carry assistant-visible text."""
    payload_type = payload.get("type", "")
    if payload_type == "agent_message":
        text = payload.get("message", "")
        if not isinstance(text, str) or not text:
            return [], pending
        return (
            [
                AgentMessage(
                    text=text,
                    role="assistant",
                    content_type="text",
                    is_complete=False,
                )
            ],
            pending,
        )
    if payload_type == "task_complete":
        text = payload.get("last_agent_message", "")
        if not isinstance(text, str) or not text:
            return [], pending
        return (
            [
                AgentMessage(
                    text=text,
                    role="assistant",
                    content_type="text",
                    phase="final_answer",
                )
            ],
            pending,
        )
    return (
        [],
        pending,
    )


def _parse_input_item(
    payload: dict[str, Any],
    pending: dict[str, Any],
) -> tuple[list[AgentMessage], dict[str, Any]]:
    """Parse Codex input_item payloads."""
    if payload.get("role", "") != "user":
        return [], pending
    content = payload.get("content", "")
    if not isinstance(content, str) or not content:
        return [], pending
    return ([AgentMessage(text=content, role="user", content_type="text")], pending)


def _append_unique_messages(
    dest: list[AgentMessage],
    candidates: list[AgentMessage],
    last_signature: tuple[str, str, str] | None,
) -> tuple[str, str, str] | None:
    """Append messages while dropping exact adjacent duplicates."""
    current = last_signature
    for message in candidates:
        signature = (message.role, message.content_type, message.text)
        if signature == current:
            if dest and _prefer_duplicate_message(dest[-1], message):
                dest[-1] = message
            continue
        dest.append(message)
        current = signature
    return current


def _prefer_duplicate_message(previous: AgentMessage, candidate: AgentMessage) -> bool:
    """Keep the duplicate carrying richer metadata, such as final_answer."""
    if previous.phase != candidate.phase:
        return previous.phase is None and candidate.phase is not None
    if previous.tool_use_id != candidate.tool_use_id:
        return previous.tool_use_id is None and candidate.tool_use_id is not None
    if previous.tool_name != candidate.tool_name:
        return previous.tool_name is None and candidate.tool_name is not None
    return False


# Transcripts older than this are considered stale and skipped during discovery.
# Prevents matching a finished session when a new Codex window opens in the same cwd.
_TRANSCRIPT_MAX_AGE_SECS = 120.0


def _collect_codex_sessions(sessions_dir: Path) -> list[tuple[float, Path]]:
    """Collect all JSONL files under sessions_dir, sorted newest-first by mtime."""
    result: list[tuple[float, Path]] = []
    for fpath in sessions_dir.rglob("*.jsonl"):
        try:
            result.append((fpath.stat().st_mtime, fpath))
        except OSError:
            continue
    result.sort(reverse=True)
    return result


def _read_codex_session_meta(fpath: Path) -> dict[str, Any] | None:
    """Read the session_meta payload from the first line of a Codex JSONL file."""
    try:
        with open(fpath, encoding="utf-8") as f:
            first_line = f.readline()
    except OSError, UnicodeError:
        return None
    if not first_line:
        return None
    try:
        data = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("type") != "session_meta":
        return None
    payload = data.get("payload")
    return payload if isinstance(payload, dict) else None


def _is_primary_codex_session(meta: dict[str, Any]) -> bool:
    """Whether session metadata represents a top-level Codex CLI session.

    Hookless discovery should bind a tmux window to the main user-visible
    conversation, not transient guardian/subagent transcripts created for
    approvals or delegated work, nor non-interactive ``codex exec`` runs that
    happen to share a cwd with a real interactive session. All of those write
    JSONL transcripts under the same ``~/.codex/sessions/`` tree and can be
    newer than the main transcript, so they must be filtered out here.

    Filtering precedence:

    * ``originator == "codex_exec"`` -> non-interactive ``codex exec``; never
      bind regardless of ``source``. Some ``codex exec`` runs report
      ``source = "exec"`` and others report no ``source`` field at all, so the
      originator check must happen before the ``source`` branches below.
    * Non-dict ``source`` (string such as ``"cli"`` / ``"exec"``, or missing) ->
      treated as a primary session. This covers ``codex_cli_rs`` interactive
      sessions and Codex Desktop variants that legitimately set
      ``source = "exec"`` but whose ``originator`` is not ``codex_exec``.
    * Dict ``source`` -> filter out anything carrying a ``subagent`` key
      (guardian / delegated work).
    """
    originator = meta.get("originator")
    if isinstance(originator, str) and originator == "codex_exec":
        return False
    source = meta.get("source")
    if not isinstance(source, dict):
        return True
    return "subagent" not in source


def _read_codex_session_summary(path: Path) -> str:
    """Use the first human prompt as the picker label, skipping injected context."""
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                payload = entry.get("payload")
                if not isinstance(payload, dict):
                    continue
                text = ""
                if entry.get("type") == "event_msg":
                    if payload.get("type") == "user_message":
                        text = payload.get("message", "")
                elif (
                    entry.get("type") in {"response_item", "input_item"}
                    and payload.get("role") == "user"
                ):
                    text = _extract_text_blocks(payload.get("content"))
                if (
                    isinstance(text, str)
                    and text.strip()
                    and not text.lstrip().startswith(
                        (
                            "# AGENTS.md instructions",
                            "<environment_context",
                            "<permissions",
                            "<user_instructions",
                        )
                    )
                ):
                    return text.strip()
    except OSError, UnicodeError:
        pass
    return ""


class CodexProvider(JsonlProvider):
    """AgentProvider implementation for OpenAI Codex CLI."""

    _CAPS = ProviderCapabilities(
        name="codex",
        launch_command="codex",
        supports_hook=True,
        hook_install_managed_by_ccgram=True,
        supports_resume=True,
        supports_resume_picker=True,
        supports_continue=True,
        supports_structured_transcript=True,
        builtin_commands=tuple(_CODEX_BUILTINS.keys()),
        supports_user_command_discovery=True,
        supports_status_snapshot=True,
        # Pickers per tooltips.txt: each opens a modal selector users
        # drive with arrow keys + Enter.
        tui_picker_commands=frozenset(
            {"model", "permissions", "skills", "statusline", "personality"}
        ),
    )

    _BUILTINS = _CODEX_BUILTINS

    def make_launch_args(
        self,
        resume_id: str | None = None,
        use_continue: bool = False,
    ) -> str:
        """Build Codex CLI args for launching or resuming a session.

        Resume uses ``resume <id>`` subcommand syntax.
        Continue uses ``resume --last`` to pick up the most recent session.
        """
        if resume_id:
            if not RESUME_ID_RE.match(resume_id):
                raise ValueError(f"Invalid resume_id: {resume_id!r}")
            return f"resume {resume_id}"
        if use_continue:
            return "resume --last"
        return ""

    def discover_resumable_sessions(
        self,
        *,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> list[ResumableSession]:
        """Enumerate saved interactive chats, including old and idle sessions."""
        if limit is not None and limit <= 0:
            return []
        try:
            resolved_cwd = str(Path(cwd).expanduser().resolve()) if cwd else None
            codex_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
            files = _collect_codex_sessions(codex_home.expanduser() / "sessions")
        except OSError, ValueError:
            return []
        sessions: list[ResumableSession] = []
        seen_ids: set[str] = set()
        for mtime, path in files:
            meta = _read_codex_session_meta(path)
            if meta is None or not _is_primary_codex_session(meta):
                continue
            session_id, file_cwd = meta.get("id"), meta.get("cwd")
            if (
                not isinstance(session_id, str)
                or not RESUME_ID_RE.fullmatch(session_id)
                or session_id in seen_ids
                or not isinstance(file_cwd, str)
                or not Path(file_cwd).is_absolute()
            ):
                continue
            try:
                session_cwd = str(Path(file_cwd).resolve())
            except OSError, ValueError:
                continue
            if resolved_cwd is not None and session_cwd != resolved_cwd:
                continue
            seen_ids.add(session_id)
            sessions.append(
                ResumableSession(
                    session_id=session_id,
                    summary=_read_codex_session_summary(path) or session_id[:12],
                    cwd=session_cwd,
                    provider_name="codex",
                    mtime=mtime,
                    transcript_path=str(path),
                )
            )
            if limit is not None and len(sessions) >= limit:
                break
        return sessions

    # ── Codex-specific transcript parsing ─────────────────────────────

    def parse_transcript_entries(
        self,
        entries: list[dict[str, Any]],
        pending_tools: dict[str, Any],
        cwd: str | None = None,  # noqa: ARG002
    ) -> tuple[list[AgentMessage], dict[str, Any]]:
        """Parse Codex JSONL entries into AgentMessages."""
        messages: list[AgentMessage] = []
        pending = dict(pending_tools)
        last_signature: tuple[str, str, str] | None = None

        for entry in entries:
            entry_type = entry.get("type", "")
            payload = entry.get("payload", {})
            if not isinstance(payload, dict):
                continue

            if entry_type == "response_item":
                parsed, pending = _parse_codex_response_item(payload, pending)
                last_signature = _append_unique_messages(
                    messages, parsed, last_signature
                )
                continue

            if entry_type == "event_msg":
                parsed, pending = _parse_event_message(payload, pending)
                last_signature = _append_unique_messages(
                    messages, parsed, last_signature
                )
                continue

            if entry_type == "input_item":
                parsed, pending = _parse_input_item(payload, pending)
                last_signature = _append_unique_messages(
                    messages, parsed, last_signature
                )

        return messages, pending

    def is_user_transcript_entry(self, entry: dict[str, Any]) -> bool:
        """Check if this Codex entry is a human turn."""
        entry_type = entry.get("type", "")
        payload = entry.get("payload", {})
        if not isinstance(payload, dict):
            return False
        if entry_type == "response_item" and payload.get("role") == "user":
            # Skip system/developer messages that look like user
            content = payload.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "input_text":
                        text = block.get("text", "")
                        if text.startswith(("<permissions", "<environment_context")):
                            return False
            return True
        return entry_type == "input_item" and payload.get("role") == "user"

    def parse_history_entry(self, entry: dict[str, Any]) -> AgentMessage | None:
        """Parse a single Codex transcript entry for history display."""
        entry_type = entry.get("type", "")
        payload = entry.get("payload", {})
        if not isinstance(payload, dict):
            return None

        if entry_type == "response_item":
            role = payload.get("role", "")
            if role not in ("user", "assistant"):
                return None
            content = payload.get("content", "")
            text = _extract_text_blocks(content)
            if not text:
                return None
            return AgentMessage(
                text=text,
                role=cast(MessageRole, role),
                content_type="text",
            )
        if entry_type == "input_item" and payload.get("role") == "user":
            content = payload.get("content", "")
            text = content if isinstance(content, str) else ""
            if not text:
                return None
            return AgentMessage(text=text, role="user", content_type="text")

        return None

    def parse_terminal_status(
        self,
        pane_text: str,
        *,
        pane_title: str = "",  # noqa: ARG002
    ) -> StatusUpdate | None:
        """Parse Codex pane content for interactive prompts."""
        interactive = extract_interactive_content(pane_text)
        if interactive:
            formatted = format_codex_interactive_prompt(
                interactive.content, interactive.name
            )
            return StatusUpdate(
                raw_text=formatted,
                display_label=interactive.name,
                is_interactive=True,
                ui_type=interactive.name,
            )
        return None

    def discover_transcript(
        self,
        cwd: str,
        window_key: str,
        *,
        max_age: float | None = None,
    ) -> SessionStartEvent | None:
        """Scan ~/.codex/sessions/ for the most recent transcript matching cwd.

        Codex transcript path: ~/.codex/sessions/YYYY/MM/DD/<name>-<ts>-<uuid>.jsonl
        First line: {"type": "session_meta", "payload": {"id": "<uuid>", "cwd": "..."}}

        Args:
            max_age: Maximum transcript age in seconds. ``None`` uses the
                default ``_TRANSCRIPT_MAX_AGE_SECS`` (120s). Pass ``0`` or
                negative to disable the age check entirely.
        """
        sessions_dir = Path.home() / ".codex" / "sessions"
        if not sessions_dir.is_dir():
            return None

        # Lazy: only used inside the polling branch below
        import time

        age_limit = _TRANSCRIPT_MAX_AGE_SECS if max_age is None else max_age

        jsonl_files = _collect_codex_sessions(sessions_dir)
        now = time.time()
        resolved_cwd = str(Path(cwd).resolve())
        for mtime, fpath in jsonl_files[:20]:
            if age_limit > 0 and now - mtime > age_limit:
                break  # sorted newest-first; remaining are all older
            meta = _read_codex_session_meta(fpath)
            if not meta:
                continue
            if not _is_primary_codex_session(meta):
                continue
            file_cwd = meta.get("cwd", "")
            if file_cwd and str(Path(file_cwd).resolve()) == resolved_cwd:
                session_id = meta.get("id", "")
                if session_id:
                    return SessionStartEvent(
                        session_id=session_id,
                        cwd=file_cwd,
                        transcript_path=str(fpath),
                        window_key=window_key,
                    )
        return None

    # ── Status snapshot (Codex-specific) ─────────────────────────────

    def build_status_snapshot(
        self,
        transcript_path: str,
        *,
        display_name: str = "",
        session_id: str = "",
        cwd: str = "",
    ) -> str | None:
        # Lazy: codex_status pulls transcript parsers; defer until polling tick
        from ccgram.providers.codex_status import build_codex_status_snapshot

        return build_codex_status_snapshot(
            transcript_path,
            display_name=display_name,
            session_id=session_id,
            cwd=cwd,
        )

    def has_output_since(self, transcript_path: str, offset: int) -> bool:
        # Lazy: codex_status pulls transcript parsers; defer until polling tick
        from ccgram.providers.codex_status import has_codex_assistant_output_since

        return has_codex_assistant_output_since(transcript_path, offset)
