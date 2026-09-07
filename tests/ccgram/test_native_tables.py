"""Native Telegram tables preserve their place and use the rich-message API."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Message
from telegram.error import BadRequest, RetryAfter

from ccgram.handlers.messaging_pipeline.message_queue import (
    _can_merge_tasks,
    _process_content_task,
    DeliveryOutcome,
)
from ccgram.handlers.messaging_pipeline.message_sender import (
    rate_limit_send_rich_message,
)
from ccgram.handlers.messaging_pipeline.message_task import ContentTask
from ccgram.handlers.response_builder import build_response_parts
from ccgram.markdown_tables import has_markdown_table, split_markdown_tables
from ccgram.telegram_client import FakeTelegramClient, PTBTelegramClient

TABLE = "| 商品 | 数量 |\n| --- | ---: |\n| 苹果 | 2 |\n| 香蕉 | 3 |"
_SENDER = "ccgram.handlers.messaging_pipeline.message_sender"
_QUEUE = "ccgram.handlers.messaging_pipeline.message_queue"


def test_table_stays_between_surrounding_prose():
    text = "购买清单：\n\n" + TABLE + "\n\n共五个。"
    segments = split_markdown_tables(text)
    assert "".join(text for text, _ in segments) == text
    assert [is_table for _, is_table in segments] == [False, True, False]
    assert build_response_parts(text, True) == ["购买清单：", TABLE, "共五个。"]


@pytest.mark.parametrize("fence", ["```", "~~~", "````"])
def test_tables_inside_code_blocks_remain_code(fence):
    text = f"{fence}markdown\n{TABLE}\n{fence}"
    assert not has_markdown_table(text)
    assert split_markdown_tables(text) == [(text, False)]


def test_large_table_is_not_split_before_native_send():
    table = "| ID | Value |\n| --- | --- |\n" + "\n".join(
        f"| {i} | {'x' * 30} |" for i in range(200)
    )
    assert build_response_parts(table, True) == [table]


def test_table_is_not_merged_into_entity_formatted_text():
    table = ContentTask("@1", (TABLE,), chat_id=42, thread_id=7)
    text = ContentTask("@1", ("Hello",), chat_id=42, thread_id=7)
    assert not _can_merge_tasks(table, text)
    assert not _can_merge_tasks(text, table)


async def test_adapter_calls_send_rich_message_with_topic_and_message_return_type():
    bot = MagicMock()
    message = MagicMock(spec=Message)
    bot.do_api_request = AsyncMock(return_value=message)
    result = await PTBTelegramClient(bot).send_rich_message(
        42, TABLE, message_thread_id=7
    )
    assert result is message
    bot.do_api_request.assert_awaited_once_with(
        "sendRichMessage",
        api_kwargs={
            "chat_id": 42,
            "rich_message": {"markdown": TABLE},
            "message_thread_id": 7,
        },
        return_type=Message,
    )


async def test_queue_sends_prose_table_prose_in_order():
    message = MagicMock(spec=Message, message_id=10)
    client = FakeTelegramClient(
        returns={"send_message": message, "send_rich_message": message}
    )
    task = ContentTask(
        "@1",
        tuple(build_response_parts("Before\n\n" + TABLE + "\n\nAfter", True)),
        chat_id=42,
        thread_id=7,
    )
    with patch(f"{_SENDER}.rate_limit_send", new_callable=AsyncMock):
        outcome = await _process_content_task(client, 1, task)
    assert outcome is DeliveryOutcome.DELIVERED
    assert [call.method for call in client.calls] == [
        "send_message",
        "send_rich_message",
        "send_message",
    ]
    assert all(call.kwargs["message_thread_id"] == 7 for call in client.calls)
    assert client.calls[1].kwargs["markdown"] == TABLE


async def test_unsupported_rich_api_falls_back_to_normal_text():
    message = MagicMock(spec=Message, message_id=10)
    client = FakeTelegramClient(returns={"send_message": message})
    client.set_side_effect("send_rich_message", [BadRequest("Method not found")])
    with patch(f"{_SENDER}.rate_limit_send", new_callable=AsyncMock):
        sent = await rate_limit_send_rich_message(
            client, 42, TABLE, message_thread_id=7
        )
    assert sent is message
    assert [call.method for call in client.calls] == [
        "send_rich_message",
        "send_message",
    ]
    assert "苹果" in client.calls[1].kwargs["text"]


async def test_flood_control_is_retried_without_sending_a_duplicate_fallback():
    client = FakeTelegramClient()
    client.set_side_effect("send_rich_message", [RetryAfter(1)])
    with (
        patch(f"{_SENDER}.rate_limit_send", new_callable=AsyncMock),
        pytest.raises(RetryAfter),
    ):
        await rate_limit_send_rich_message(client, 42, TABLE, message_thread_id=7)
    assert [call.method for call in client.calls] == ["send_rich_message"]
