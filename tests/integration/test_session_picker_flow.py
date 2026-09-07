"""Real Telegram dispatch from provider selection through exact-session launch.

Only transport and terminal side effects are mocked. Provider history discovery,
callback registration, picker state, and window-launch orchestration are real.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.callback_data import CB_MODE_SELECT
from ccgram.handlers.topics.directory_browser import (
    BROWSE_PATH_KEY,
    STATE_BROWSING_DIRECTORY,
    STATE_KEY,
)
from ccgram.handlers.topics.topic_creation_draft import (
    PENDING_SESSION_SELECTION,
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
)
from ccgram.providers import registry as provider_registry

pytestmark = pytest.mark.integration

_MODE = "ccgram.handlers.topics.provider_mode_callbacks"
_PICKER = "ccgram.handlers.topics.session_picker"
_LAUNCH = "ccgram.handlers.topics.window_launch_service"


@pytest.mark.parametrize("provider_name", ["claude", "codex"])
async def test_new_topic_can_select_an_older_conversation(
    dispatch_app, make_callback_update, tmp_path, monkeypatch, provider_name
):
    project = tmp_path / "project"
    project.mkdir()
    history = tmp_path / "history"
    history.mkdir()
    if provider_name == "claude":
        monkeypatch.setattr("ccgram.config.config.claude_projects_path", history)
        session_dir = history / "encoded-project"
    else:
        monkeypatch.setenv("CODEX_HOME", str(history))
        session_dir = history / "sessions" / "2026" / "01" / "01"
    session_dir.mkdir(parents=True)
    session_ids = [f"aaaaaaaa-bbbb-cccc-dddd-{index:012d}" for index in (1, 2)]
    for index, sid in enumerate(session_ids):
        entry = (
            {
                "cwd": str(project),
                "type": "user",
                "message": {"content": f"Prompt {index}"},
            }
            if provider_name == "claude"
            else {
                "type": "session_meta",
                "payload": {"id": sid, "cwd": str(project), "source": "cli"},
            }
        )
        path = session_dir / f"{sid}.jsonl"
        path.write_text(json.dumps(entry) + "\n")
        os.utime(path, (index + 1, index + 1))

    state = dispatch_app.user_data[12345]
    state.update(
        {
            BROWSE_PATH_KEY: str(project),
            STATE_KEY: STATE_BROWSING_DIRECTORY,
            PENDING_THREAD_ID: 42,
            PENDING_THREAD_TEXT: "Continue the fix",
        }
    )
    router = MagicMock()
    router.get_window_for_thread.return_value = None
    router.resolve_chat_id.return_value = -100999
    mux = MagicMock()
    mux.capabilities.native_worktrees = False
    mux.capabilities.native_topic_targets = False
    mux.create_window = AsyncMock(return_value=(True, "created", "project", "@5"))
    mux.stamp_pane_title = AsyncMock()
    with (
        patch(f"{_MODE}.thread_router", router),
        patch(f"{_LAUNCH}.thread_router", router),
        patch(f"{_LAUNCH}.tmux_manager", mux),
        patch(f"{_LAUNCH}.session_manager"),
        patch(
            f"{_LAUNCH}.session_map_sync.wait_for_session_map_entry",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(f"{_LAUNCH}.topic_orchestration"),
        patch(f"{_LAUNCH}.user_preferences"),
        patch(f"{_LAUNCH}.safe_edit", new_callable=AsyncMock),
        patch(
            f"{_LAUNCH}.send_telegram_to_window",
            new_callable=AsyncMock,
            return_value=(True, "Sent"),
        ) as send,
        patch(f"{_PICKER}.safe_edit", new_callable=AsyncMock) as edit,
    ):
        await dispatch_app.process_update(
            make_callback_update(
                f"{CB_MODE_SELECT}{provider_name}:normal", bot=dispatch_app.bot
            )
        )
        mux.create_window.assert_not_awaited()
        selection = state[PENDING_SESSION_SELECTION]
        assert [session.session_id for session in selection.sessions] == list(
            reversed(session_ids)
        )
        keyboard = edit.call_args.kwargs["reply_markup"]
        await dispatch_app.process_update(
            make_callback_update(
                keyboard.inline_keyboard[1][0].callback_data, bot=dispatch_app.bot
            )
        )

    mux.create_window.assert_awaited_once_with(
        str(project),
        launch_command=f"{provider_name} {provider_registry.get(provider_name).make_launch_args(resume_id=session_ids[0])}",
        workspace_id=None,
    )
    router.bind_thread.assert_called_once()
    send.assert_awaited_once_with(12345, "@5", 42, "Continue the fix", -100999)
    assert PENDING_SESSION_SELECTION not in state
