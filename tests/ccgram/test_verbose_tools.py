"""Verbose tool delivery retains calls and results so Telegram can stand alone."""

from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Message

from ccgram.handlers.messaging_pipeline.message_queue import (
    _process_content_task,
    DeliveryOutcome,
)
from ccgram.handlers.messaging_pipeline.message_task import ContentTask
from ccgram.providers.codex import CodexProvider
from ccgram.telegram_client import FakeTelegramClient
from ccgram.transcript_parser import TranscriptParser
from ccgram.window_state_store import DEFAULT_BATCH_MODE


def test_verbose_is_the_default():
    assert DEFAULT_BATCH_MODE == "verbose"


def test_codex_keeps_full_script_arguments_and_output():
    script = "const result = await tools.exec_command({cmd: 'git status --short'});\ntext(result);"
    entries = [
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "name": "functions.exec",
                "call_id": "call-1",
                "input": script,
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call_output",
                "call_id": "call-1",
                "output": "M src/example.py",
            },
        },
    ]
    messages, _ = CodexProvider().parse_transcript_entries(entries, {})
    assert script in messages[0].text
    assert messages[0].content_type == "tool_use"
    assert "M src/example.py" in messages[1].text
    assert messages[1].content_type == "tool_result"


def test_claude_keeps_command_details_without_suppressing_result():
    command = "git status --short\npython -m pytest tests/test_example.py -v"
    entries = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "id": "call-1",
                        "input": {"command": command},
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": "12 passed",
                    }
                ]
            },
        },
    ]
    messages, _ = TranscriptParser.parse_entries(entries, {})
    assert command in messages[0].text
    assert "12 passed" in messages[1].text


async def test_result_does_not_replace_the_command_message_in_verbose_mode():
    message = MagicMock(spec=Message, message_id=10)
    client = FakeTelegramClient(returns={"send_message": message})
    call = ContentTask(
        "@1",
        ("Run the tests",),
        content_type="tool_use",
        tool_use_id="call-1",
        tool_name="exec_command",
        chat_id=42,
        thread_id=7,
    )
    result = ContentTask(
        "@1",
        ("12 passed",),
        content_type="tool_result",
        tool_use_id="call-1",
        tool_name="exec_command",
        chat_id=42,
        thread_id=7,
    )
    with (
        patch(
            "ccgram.handlers.messaging_pipeline.message_sender.rate_limit_send",
            new_callable=AsyncMock,
        ),
        patch(
            "ccgram.handlers.messaging_pipeline.message_queue.get_batch_mode",
            return_value="verbose",
        ),
    ):
        assert await _process_content_task(client, 1, call) is DeliveryOutcome.DELIVERED
        assert (
            await _process_content_task(client, 1, result) is DeliveryOutcome.DELIVERED
        )
    assert [c.method for c in client.calls] == ["send_message", "send_message"]
    assert "Run the tests" in client.calls[0].kwargs["text"]
    assert "12 passed" in client.calls[1].kwargs["text"]
    assert "exec_command result" in client.calls[1].kwargs["text"]
