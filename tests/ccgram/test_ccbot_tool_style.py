"""Render terminal-style calls and keep their results in the same message."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ccgram.entity_formatting import convert_to_entities
from ccgram.expandable_quote import EXPANDABLE_QUOTE_START
from ccgram.providers.codex import CodexProvider
from ccgram.tool_format import format_tool_details, format_tool_line
from ccgram.transcript_parser import TranscriptParser


@pytest.fixture(autouse=True)
def ccbot_style(monkeypatch):
    monkeypatch.setenv("CCGRAM_TOOL_STYLE", "ccbot")


COMMAND = "S=/tmp/scratchpad\ncd /home/tazhu/m3-compare\npython3 $S/rd.py 1422 1426"
OUTPUT = "\n".join(f"{line}|source line" for line in range(1422, 1427))


def _parse_pair(provider):
    if provider == "claude":
        call = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "id": "call-1",
                        "input": {"command": COMMAND, "description": "Read source"},
                    }
                ]
            },
        }
        result = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": OUTPUT,
                    }
                ]
            },
        }
        pending = {}
        started, pending = TranscriptParser.parse_entries([call], pending_tools=pending)
        finished, _ = TranscriptParser.parse_entries([result], pending_tools=pending)
        return started[0], finished[0]
    parser = CodexProvider()
    call = {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "exec_command",
            "call_id": "call-1",
            "arguments": json.dumps({"cmd": COMMAND, "yield_time_ms": 1000}),
        },
    }
    result = {
        "type": "response_item",
        "payload": {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "Wall time: 0.1\nProcess exited with code 0\nOutput:\n" + OUTPUT,
        },
    }
    started, pending = parser.parse_transcript_entries([call], {})
    finished, pending = parser.parse_transcript_entries([result], pending)
    assert not pending
    return started[0], finished[0]


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_multiline_command_and_five_output_lines_survive_pairing(provider):
    started, finished = _parse_pair(provider)
    start_plain, _ = convert_to_entities(started.text)
    result_plain, entities = convert_to_entities(finished.text)
    assert start_plain.strip() == f"Bash({COMMAND})"
    assert result_plain.startswith(start_plain.strip())
    assert "⎿  Output 5 lines\n" + OUTPUT in result_plain
    assert OUTPUT in result_plain
    assert "yield_time_ms" not in result_plain
    assert "Read source" not in result_plain
    assert any(entity.type == "expandable_blockquote" for entity in entities)
    assert started.tool_use_id == finished.tool_use_id == "call-1"


def test_command_markdown_is_displayed_literally():
    command = "echo `date` && cat a_b.py | head -n 5\nprintf '<tag> **literal**'"
    heading = format_tool_line("Bash", command)
    assert convert_to_entities(heading)[0].strip() == f"Bash({command})"
    assert format_tool_details(heading, {"command": command}) == heading
    assert EXPANDABLE_QUOTE_START not in heading


@pytest.mark.parametrize(
    "code",
    [
        'text(await tools.write_stdin({session_id:5392,chars:"",yield_time_ms:1000}));',
        'const results = await Promise.allSettled([tools.exec_command({cmd:"echo $S"})]);',
    ],
)
def test_exec_heading_does_not_turn_javascript_into_math(code):
    heading = format_tool_line("exec", code)
    assert convert_to_entities(heading)[0].strip() == f"Exec({code})"


@pytest.mark.parametrize("call_type", ["custom_tool_call", "function_call"])
def test_content_block_results_include_actual_file_contents(call_type):
    parser = CodexProvider()
    command = "sed -n '1,24p' CONTRIBUTING.md"
    code = "text(await tools.exec_command({cmd:" + json.dumps(command) + "}));"
    call = {
        "type": "response_item",
        "payload": {
            "type": call_type,
            "name": "exec",
            "call_id": "wrapped-read",
            "input": code,
            "arguments": json.dumps({"code": code}),
        },
    }
    contents = "# Contributing to CCGram\n\n## Before You Write Code"
    result = {
        "type": "response_item",
        "payload": {
            "type": call_type + "_output",
            "call_id": "wrapped-read",
            "output": [
                {
                    "type": "input_text",
                    "text": "Script completed\nWall time 0.1 seconds\nOutput:\n",
                },
                {
                    "type": "input_text",
                    "text": json.dumps(
                        {
                            "chunk_id": "example",
                            "exit_code": 0,
                            "output": contents,
                        }
                    ),
                },
            ],
        },
    }
    started, pending = parser.parse_transcript_entries([call], {})
    finished, pending = parser.parse_transcript_entries([result], pending)
    plain, _ = convert_to_entities(finished[0].text)
    assert len(finished) == 1 and not pending
    assert "Contributing to CCGram" in plain
    assert "Before You Write Code" in plain
    assert "Output 3 lines" in plain
    assert "Done" not in plain
    assert "chunk_id" not in plain
    if call_type == "custom_tool_call":
        assert convert_to_entities(started[0].text)[0].strip() == f"Bash({command})"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            [
                {"type": "input_text", "text": "first"},
                {"type": "input_text", "text": "second"},
            ],
            "first\nsecond",
        ),
        (
            {"status": "fulfilled", "value": {"output": "file contents"}},
            "file contents",
        ),
        (
            {"status": "rejected", "reason": "permission denied"},
            "Error: permission denied",
        ),
        ({"output": "", "session_id": 123}, "Process still running."),
        ({"output": "", "exit_code": 1}, "Process exited with code 1."),
        (
            [{"type": "input_image", "image_url": "data:image/png;base64,not-text"}],
            "[Image output]",
        ),
        ({"unexpected": "preserve this"}, '{"unexpected": "preserve this"}'),
    ],
)
def test_structured_outputs_are_not_replaced_with_done(value, expected):
    from ccgram.providers.codex import _extract_tool_output_text

    assert _extract_tool_output_text(value) == expected


def test_file_content_is_not_reinterpreted_as_a_tool_envelope():
    from ccgram.providers.codex import _extract_tool_output_text

    contents = 'Example\nOutput:\n{"output": "this is file content"}'
    payload = [{"type": "input_text", "text": json.dumps({"output": contents})}]
    assert _extract_tool_output_text(payload) == contents


@pytest.mark.parametrize("provider", ["claude", "codex"])
async def test_result_edits_original_message_in_verbose_mode(provider, monkeypatch):
    from ccgram.handlers.messaging_pipeline import message_queue as mq
    from ccgram.handlers.messaging_pipeline.message_task import ContentTask
    from ccgram.telegram_client import FakeTelegramClient

    started, finished = _parse_pair(provider)
    client = FakeTelegramClient()
    send = AsyncMock(return_value=SimpleNamespace(message_id=123))
    edit = AsyncMock(return_value=True)
    monkeypatch.setattr(mq, "rate_limit_send_message", send)
    monkeypatch.setattr(mq, "edit_with_fallback", edit)
    monkeypatch.setattr(mq, "clear_status_message", AsyncMock())
    monkeypatch.setattr(mq, "get_batch_mode", lambda _: "verbose")
    monkeypatch.setattr(mq, "_tool_msg_ids", {})
    for message in (started, finished):
        task = ContentTask(
            window_id="@0",
            parts=(message.text,),
            content_type=message.content_type,
            tool_use_id=message.tool_use_id,
            tool_name=message.tool_name,
            chat_id=-1001,
            thread_id=42,
        )
        await mq._process_content_task(client, 12345, task)

    send.assert_awaited_once()
    edit.assert_awaited_once_with(client, -1001, 123, finished.text)
    assert not mq._tool_msg_ids
