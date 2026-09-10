"""Terminal-size buttons recapture the intended screenshot and enforce ownership."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccgram.handlers.callback_data import CB_SCREENSHOT_RESIZE
from ccgram.handlers.live.screenshot_callbacks import (
    _handle_resize,
    build_screenshot_keyboard,
    handle_screenshot_callback,
)
from ccgram.multiplexer.base import PaneDims

_MODULE = "ccgram.handlers.live.screenshot_callbacks."


@pytest.fixture
def resize_env():
    query = AsyncMock()
    query.message.chat.id = -1001
    with (
        patch(f"{_MODULE}tmux_manager") as mux,
        patch(f"{_MODULE}user_owns_window", return_value=True) as owns,
        patch(f"{_MODULE}text_to_image", AsyncMock(return_value=b"png")) as render,
        patch(f"{_MODULE}asyncio.sleep", AsyncMock()),
    ):
        mux.capabilities.supports_window_resize = True
        mux.window_dims = AsyncMock(return_value=PaneDims(80, 24))
        mux.resize_window = AsyncMock(return_value=True)
        mux.find_window_by_id = AsyncMock(return_value=MagicMock(window_id="@12"))
        mux.capture_pane = AsyncMock(return_value="new viewport\ncomplete status bar")
        mux.capture_pane_by_id = AsyncMock(return_value="specific pane")
        yield query, mux, owns, render


@pytest.mark.parametrize(
    ("action", "before", "expected"),
    [
        ("larger", PaneDims(80, 24), PaneDims(120, 34)),
        ("smaller", PaneDims(160, 45), PaneDims(120, 35)),
        ("reset", PaneDims(200, 55), PaneDims(160, 45)),
    ],
)
async def test_resize_updates_same_screenshot(resize_env, action, before, expected):
    query, mux, owns, render = resize_env
    mux.window_dims.return_value = before
    await handle_screenshot_callback(
        query, 12345, f"{CB_SCREENSHOT_RESIZE}{action}:@12", MagicMock(), MagicMock()
    )
    owns.assert_called_with(12345, "@12", -1001)
    mux.resize_window.assert_awaited_once_with(
        "@12", width=expected.width, height=expected.height
    )
    mux.capture_pane.assert_awaited_once_with("@12", with_ansi=True)
    render.assert_awaited_once_with("new viewport\ncomplete status bar", with_ansi=True)
    query.edit_message_media.assert_awaited_once()


async def test_resize_preserves_explicit_pane_target(resize_env):
    query, mux, _owns, render = resize_env
    await _handle_resize(query, 12345, f"{CB_SCREENSHOT_RESIZE}larger:@12|%5")
    mux.window_dims.assert_awaited_once_with("@12")
    mux.capture_pane_by_id.assert_awaited_once_with(
        "%5", with_ansi=True, window_id="@12"
    )
    mux.capture_pane.assert_not_awaited()
    render.assert_awaited_once_with("specific pane", with_ansi=True)


async def test_foreign_window_cannot_be_resized(resize_env):
    query, mux, owns, _render = resize_env
    owns.return_value = False
    await _handle_resize(query, 12345, f"{CB_SCREENSHOT_RESIZE}larger:@12")
    mux.window_dims.assert_not_awaited()
    mux.resize_window.assert_not_awaited()
    query.answer.assert_awaited_once_with("Not your session", show_alert=True)


@pytest.mark.parametrize("data", ["larger", "invalid:@12"])
async def test_invalid_resize_action_has_no_effect(resize_env, data):
    query, mux, _owns, _render = resize_env
    await _handle_resize(query, 12345, CB_SCREENSHOT_RESIZE + data)
    mux.resize_window.assert_not_awaited()
    query.edit_message_media.assert_not_awaited()


async def test_unsupported_backend_has_no_buttons_or_resize(resize_env):
    query, mux, _owns, _render = resize_env
    mux.capabilities.supports_window_resize = False
    keyboard = build_screenshot_keyboard("target")
    assert not any(
        str(b.callback_data).startswith(CB_SCREENSHOT_RESIZE)
        for row in keyboard.inline_keyboard
        for b in row
    )
    await _handle_resize(query, 12345, f"{CB_SCREENSHOT_RESIZE}larger:target")
    mux.window_dims.assert_not_awaited()
    mux.resize_window.assert_not_awaited()


async def test_failed_resize_does_not_claim_success(resize_env):
    query, mux, _owns, render = resize_env
    mux.resize_window.return_value = False
    await _handle_resize(query, 12345, f"{CB_SCREENSHOT_RESIZE}larger:@12")
    render.assert_not_awaited()
    query.edit_message_media.assert_not_awaited()
    query.answer.assert_awaited_once_with("Failed to resize terminal", show_alert=True)


@pytest.mark.parametrize(
    ("action", "dims"), [("larger", PaneDims(240, 80)), ("smaller", PaneDims(80, 24))]
)
async def test_resize_stops_at_viewport_limits(resize_env, action, dims):
    query, mux, _owns, _render = resize_env
    mux.window_dims.return_value = dims
    await _handle_resize(query, 12345, f"{CB_SCREENSHOT_RESIZE}{action}:@12")
    mux.resize_window.assert_not_awaited()
    query.edit_message_media.assert_not_awaited()


def test_resize_buttons_keep_target_and_fit_callback_limit(resize_env):
    keyboard = build_screenshot_keyboard("@12", pane_id="%5")
    buttons = [
        b
        for row in keyboard.inline_keyboard
        for b in row
        if str(b.callback_data).startswith(CB_SCREENSHOT_RESIZE)
    ]
    assert len(buttons) == 3
    assert all(str(b.callback_data).endswith("@12|%5") for b in buttons)
    assert all(len(str(b.callback_data).encode()) <= 64 for b in buttons)
