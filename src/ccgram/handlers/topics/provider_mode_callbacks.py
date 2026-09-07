"""Provider and mode selection callbacks for the topic-creation flow.

Handles CB_PROV_SELECT and CB_MODE_SELECT, then offers saved conversations
before creating the window when the provider supports history browsing.

Both ultimately call ``launch_window`` from ``window_launch_service``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from ...providers import registry as provider_registry
from ...thread_router import thread_router
from ..callback_data import CB_MODE_SELECT, CB_PROV_SELECT
from ..callback_helpers import get_thread_id
from ..messaging_pipeline.message_sender import safe_edit
from .directory_browser import (
    build_mode_picker,
    clear_browse_state,
    clear_worktree_state,
)
from .topic_creation_draft import (
    PENDING_THREAD_ID,
    PENDING_THREAD_TEXT,
    _required_selected_path,
)
from .window_launch_service import WindowLaunchRequest, launch_window
from .session_picker import offer_session_picker

if TYPE_CHECKING:
    from telegram import CallbackQuery, Update
    from telegram.ext import ContextTypes

logger = structlog.get_logger()

__all__ = [
    "_validate_provider_select",
    "_handle_provider_select",
    "_parse_mode_select",
    "_handle_mode_select",
]


async def _validate_provider_select(
    query: CallbackQuery,
    user_id: int,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    pending_thread_id: int | None,
) -> bool:
    """Validate provider select callback; returns True if request should proceed."""

    confirm_thread_id = get_thread_id(update)
    if pending_thread_id is not None and confirm_thread_id != pending_thread_id:
        # Discard the mismatched flow's worktree intent so its CREATING
        # re-entrancy flag cannot block a subsequent worktree confirm.
        clear_browse_state(context.user_data)
        clear_worktree_state(context.user_data)
        if context.user_data is not None:
            context.user_data.pop(PENDING_THREAD_ID, None)
            context.user_data.pop(PENDING_THREAD_TEXT, None)
        await query.answer("Stale browser (topic mismatch)", show_alert=True)
        return False

    await query.answer()

    # Guard against double-click: if thread already has a window, skip
    if pending_thread_id is not None:
        existing_wid = thread_router.get_window_for_thread(user_id, pending_thread_id)
        if existing_wid is not None:
            display = thread_router.get_display_name(existing_wid)
            logger.warning(
                "Thread %d already bound to window %s (%s), ignoring duplicate provider select",
                pending_thread_id,
                existing_wid,
                display,
            )
            await safe_edit(query, f"✅ Already bound to window {display}.")
            return False

    return True


async def _handle_provider_select(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_PROV_SELECT: select provider and show mode picker.

    Providers without a YOLO flag (e.g. shell) skip the mode picker
    and go directly to window creation with approval_mode="normal".
    """
    # Lazy: providers package heavy bootstrap
    from ccgram.providers import has_yolo_mode

    provider_name = data[len(CB_PROV_SELECT) :]
    if not provider_registry.is_valid(provider_name):
        await query.answer("Unknown provider", show_alert=True)
        return

    selected_path = _required_selected_path(context)
    if selected_path is None:
        await query.answer()
        await safe_edit(query, "❌ Selection expired. Tap Cancel and retry.")
        return
    pending_thread_id: int | None = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )

    if not await _validate_provider_select(
        query, user_id, update, context, pending_thread_id
    ):
        return

    if not has_yolo_mode(provider_name):
        request = WindowLaunchRequest(
            user_id=user_id,
            thread_id=pending_thread_id,
            provider_name=provider_name,
            cwd=selected_path,
            mode="normal",
            pending_text=(
                context.user_data.get(PENDING_THREAD_TEXT)
                if context.user_data
                else None
            ),
        )
        if await offer_session_picker(query, context, request):
            return
        clear_browse_state(context.user_data)
        await launch_window(query, context, request)
        return

    text, keyboard = build_mode_picker(selected_path, provider_name)
    await safe_edit(query, text, reply_markup=keyboard)


def _parse_mode_select(data: str) -> tuple[str, str] | None:
    """Parse mode callback data as (provider_name, approval_mode)."""
    raw = data[len(CB_MODE_SELECT) :]
    provider_name, sep, approval_mode = raw.partition(":")
    if not sep:
        return None
    return provider_name, approval_mode.lower()


async def _handle_mode_select(
    query: CallbackQuery,
    user_id: int,
    data: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle CB_MODE_SELECT: select launch mode and create tmux window."""
    parsed = _parse_mode_select(data)
    if parsed is None:
        await query.answer("Invalid mode", show_alert=True)
        return

    provider_name, approval_mode = parsed
    if not provider_registry.is_valid(provider_name):
        await query.answer("Unknown provider", show_alert=True)
        return
    if approval_mode not in ("normal", "yolo"):
        await query.answer("Unknown mode", show_alert=True)
        return

    selected_path = _required_selected_path(context)
    if selected_path is None:
        await query.answer()
        await safe_edit(query, "❌ Selection expired. Tap Cancel and retry.")
        return
    pending_thread_id: int | None = (
        context.user_data.get(PENDING_THREAD_ID) if context.user_data else None
    )

    if not await _validate_provider_select(
        query, user_id, update, context, pending_thread_id
    ):
        return

    request = WindowLaunchRequest(
        user_id=user_id,
        thread_id=pending_thread_id,
        provider_name=provider_name,
        cwd=selected_path,
        mode=approval_mode,
        pending_text=(
            context.user_data.get(PENDING_THREAD_TEXT) if context.user_data else None
        ),
    )
    if await offer_session_picker(query, context, request):
        return
    clear_browse_state(context.user_data)
    await launch_window(query, context, request)
