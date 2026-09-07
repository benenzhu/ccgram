"""Topic creation offers provider-scoped history and consumes a selection once."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.callback_data import CB_CREATE_SESSION
from ccgram.handlers.topics.directory_browser import (
    BROWSE_PATH_KEY,
    STATE_BROWSING_DIRECTORY,
    STATE_KEY,
    clear_browse_state,
)
from ccgram.handlers.topics.session_picker import (
    SessionSelection,
    handle_session_selection,
    offer_session_picker,
)
from ccgram.handlers.topics.topic_creation_draft import (
    PENDING_SESSION_SELECTION,
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
    PENDING_WORKSPACE_ID,
    PENDING_WORKTREE_BRANCH,
    PENDING_WORKTREE_PATH,
)
from ccgram.handlers.topics.window_launch_service import WindowLaunchRequest
from ccgram.providers.base import ResumableSession

_MODULE = "ccgram.handlers.topics.session_picker"


@pytest.fixture
def flow(tmp_path):
    request = WindowLaunchRequest(
        12345, 42, "claude", str(tmp_path), "yolo", "Fix this"
    )
    context = MagicMock()
    context.user_data = {
        BROWSE_PATH_KEY: str(tmp_path),
        STATE_KEY: STATE_BROWSING_DIRECTORY,
        PENDING_THREAD_ID: 42,
        PENDING_THREAD_TEXT: "Fix this",
        PENDING_WORKSPACE_ID: "workspace-1",
        PENDING_WORKTREE_PATH: str(tmp_path),
        PENDING_WORKTREE_BRANCH: "feature",
    }
    update = MagicMock()
    update.message = None
    update.effective_user.id = 12345
    update.callback_query = AsyncMock()
    update.callback_query.message.message_thread_id = 42
    return request, context, update


def _sessions(request, count=8):
    return [
        ResumableSession(
            session_id=f"aaaaaaaa-bbbb-cccc-dddd-{index:012d}",
            summary=f"Conversation {index}",
            cwd=request.cwd,
            provider_name=request.provider_name,
            mtime=100 - index,
        )
        for index in range(count)
    ]


def _select(context, request):
    selection = SessionSelection(request, _sessions(request))
    context.user_data[PENDING_SESSION_SELECTION] = selection
    return selection


async def _tap(update, context, selection, action):
    update.callback_query.data = f"{CB_CREATE_SESSION}{selection.token}:{action}"
    await handle_session_selection(update, context)


async def test_empty_history_leaves_fresh_launch_available(flow):
    request, context, update = flow
    with patch(f"{_MODULE}.provider_registry") as registry:
        registry.get.return_value.discover_resumable_sessions.return_value = []
        assert (
            await offer_session_picker(update.callback_query, context, request) is False
        )
    assert PENDING_SESSION_SELECTION not in context.user_data


async def test_pagination_reaches_older_sessions(flow):
    request, context, update = flow
    selection = _select(context, request)
    with patch(f"{_MODULE}.safe_edit", new_callable=AsyncMock) as edit:
        await _tap(update, context, selection, "page:1")
    rows = edit.call_args.kwargs["reply_markup"].inline_keyboard
    assert "Conversation 6" in rows[0][0].text
    assert rows[0][0].callback_data.endswith("pick:6")
    assert "Page 2/2" in edit.call_args.args[1]


@pytest.mark.parametrize("provider_name", ["claude", "codex"])
@pytest.mark.parametrize("action", ["pick:7", "fresh"])
async def test_pick_preserves_launch_context_and_double_taps_launch_once(
    flow, provider_name, action
):
    request, context, update = flow
    request = replace(request, provider_name=provider_name)
    selection = _select(context, request)
    with patch(f"{_MODULE}.launch_window", new_callable=AsyncMock) as launch:
        await _tap(update, context, selection, action)
        await _tap(update, context, selection, action)
    launch.assert_awaited_once()
    assert launch.await_args is not None
    actual = launch.await_args.args[2]
    assert actual.mode == "yolo"
    assert actual.pending_text == "Fix this"
    assert actual.cwd == request.cwd
    assert actual.resume_session == (
        selection.sessions[7] if action == "pick:7" else None
    )
    assert context.user_data[PENDING_WORKSPACE_ID] == "workspace-1"
    assert PENDING_SESSION_SELECTION not in context.user_data


@pytest.mark.parametrize("change", ["thread", "token", "reset", "cwd"])
async def test_stale_or_cross_topic_buttons_cannot_launch(flow, change):
    request, context, update = flow
    selection = _select(context, request)
    if change == "thread":
        update.callback_query.message.message_thread_id = 99
    elif change == "token":
        context.user_data[PENDING_SESSION_SELECTION] = SessionSelection(request)
    elif change == "reset":
        clear_browse_state(context.user_data)
    else:
        context.user_data[BROWSE_PATH_KEY] = "/another/project"
    with patch(f"{_MODULE}.launch_window", new_callable=AsyncMock) as launch:
        await _tap(update, context, selection, "pick:0")
    launch.assert_not_awaited()
    assert "expired" in update.callback_query.answer.call_args.args[0]


@pytest.mark.parametrize("action", ["pick:-1", "pick:99", "pick:bad", "unknown"])
async def test_invalid_selection_is_recoverable(flow, action):
    request, context, update = flow
    selection = _select(context, request)
    with patch(f"{_MODULE}.launch_window", new_callable=AsyncMock) as launch:
        await _tap(update, context, selection, action)
    launch.assert_not_awaited()
    assert context.user_data[PENDING_SESSION_SELECTION] is selection


async def test_deleted_transcript_cannot_be_resumed(flow):
    request, context, update = flow
    selection = _select(context, request)
    selection.sessions[0] = replace(
        selection.sessions[0], transcript_path=request.cwd + "/gone.jsonl"
    )
    with patch(f"{_MODULE}.launch_window", new_callable=AsyncMock) as launch:
        await _tap(update, context, selection, "pick:0")
    launch.assert_not_awaited()
    assert "no longer exists" in update.callback_query.answer.call_args.args[0]
