"""Provider/mode callbacks preserve the chosen launch policy into target creation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.callback_data import CB_MODE_SELECT, CB_PROV_SELECT
from ccgram.handlers.topics.directory_browser import BROWSE_PATH_KEY
from ccgram.handlers.topics.provider_mode_callbacks import (
    _handle_mode_select,
    _handle_provider_select,
)
from ccgram.handlers.topics.topic_creation_draft import (
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
)


def _callback_context() -> tuple[AsyncMock, MagicMock, MagicMock]:
    query = AsyncMock()
    query.answer = AsyncMock()
    message = MagicMock()
    message.message_thread_id = 42
    update = MagicMock()
    update.message = None
    update.callback_query.message = message
    context = MagicMock()
    context.user_data = {BROWSE_PATH_KEY: "/workspace/project", PENDING_THREAD_ID: 42}
    return query, update, context


async def test_normal_provider_callback_launches_normal_mode_without_picker() -> None:
    query, update, context = _callback_context()
    with (
        patch(
            "ccgram.handlers.topics.provider_mode_callbacks.provider_registry"
        ) as registry,
        patch("ccgram.providers.has_yolo_mode", return_value=False),
        patch(
            "ccgram.handlers.topics.provider_mode_callbacks.launch_window",
            new_callable=AsyncMock,
        ) as launch,
        patch("ccgram.handlers.topics.provider_mode_callbacks.clear_browse_state"),
    ):
        registry.is_valid.return_value = True
        await _handle_provider_select(
            query, 99, f"{CB_PROV_SELECT}shell", update, context
        )

    launch.assert_awaited_once()
    call = launch.await_args
    assert call is not None
    request = call.args[2]
    assert request.provider_name == "shell"
    assert request.mode == "normal"
    assert request.cwd == "/workspace/project"


async def test_yolo_mode_callback_launches_selected_provider_and_yolo_mode() -> None:
    query, update, context = _callback_context()
    with (
        patch(
            "ccgram.handlers.topics.provider_mode_callbacks.provider_registry"
        ) as registry,
        patch(
            "ccgram.handlers.topics.provider_mode_callbacks.launch_window",
            new_callable=AsyncMock,
        ) as launch,
        patch("ccgram.handlers.topics.provider_mode_callbacks.clear_browse_state"),
    ):
        registry.is_valid.return_value = True
        await _handle_mode_select(
            query, 99, f"{CB_MODE_SELECT}claude:yolo", update, context
        )

    launch.assert_awaited_once()
    call = launch.await_args
    assert call is not None
    request = call.args[2]
    assert request.provider_name == "claude"
    assert request.mode == "yolo"
    assert request.thread_id == 42


@pytest.mark.parametrize(
    ("provider", "default_mode", "expected_mode"),
    [
        ("codex", "yolo", "yolo"),
        ("claude", "normal", "normal"),
        ("shell", "yolo", "normal"),
    ],
)
async def test_default_mode_skips_picker_and_preserves_pending_text(
    provider, default_mode, expected_mode
) -> None:
    query, update, context = _callback_context()
    context.user_data[PENDING_THREAD_TEXT] = "continue my task"
    module = "ccgram.handlers.topics.provider_mode_callbacks."
    with (
        patch(f"{module}config.default_approval_mode", default_mode),
        patch(f"{module}thread_router") as router,
        patch(f"{module}offer_session_picker", AsyncMock(return_value=False)),
        patch(f"{module}launch_window", new_callable=AsyncMock) as launch,
        patch(f"{module}build_mode_picker") as picker,
    ):
        router.get_window_for_thread.return_value = None
        await _handle_provider_select(
            query, 99, f"{CB_PROV_SELECT}{provider}", update, context
        )

    picker.assert_not_called()
    launch.assert_awaited_once()
    request = launch.await_args.args[2]
    assert (request.provider_name, request.mode) == (provider, expected_mode)
    assert (request.cwd, request.thread_id) == ("/workspace/project", 42)
    assert request.pending_text == "continue my task"
