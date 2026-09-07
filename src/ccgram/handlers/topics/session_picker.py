"""Choose a saved conversation or start fresh during topic creation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from pathlib import Path
from secrets import token_hex
from typing import TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ...providers import registry as provider_registry
from ...providers.base import ResumableSession
from ..callback_data import CB_CREATE_SESSION, CB_DIR_CANCEL
from ..callback_registry import register
from ..messaging_pipeline.message_sender import safe_edit
from .directory_browser import clear_browse_state
from .topic_creation_draft import (
    PENDING_SESSION_SELECTION,
    PENDING_THREAD_ID,
    _browser_flow_stale,
    _required_selected_path,
)
from .window_launch_service import WindowLaunchRequest, launch_window

if TYPE_CHECKING:
    from telegram import CallbackQuery, Update
    from telegram.ext import ContextTypes

_SESSIONS_PER_PAGE = 6


@dataclass
class SessionSelection:
    request: WindowLaunchRequest
    sessions: list[ResumableSession] = field(default_factory=list)
    token: str = field(default_factory=lambda: token_hex(6))


async def offer_session_picker(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
    request: WindowLaunchRequest,
) -> bool:
    """Return True if history selection handled (or superseded) this launch."""
    provider = provider_registry.get(request.provider_name)
    if (
        not provider.capabilities.supports_resume_picker
        or context.user_data is None
        or request.thread_id is None
    ):
        return False

    selection = SessionSelection(request)
    context.user_data[PENDING_SESSION_SELECTION] = selection
    sessions = await asyncio.to_thread(
        provider.discover_resumable_sessions, cwd=request.cwd
    )
    # A reset or another topic can replace the flow while disk discovery runs.
    if (
        context.user_data.get(PENDING_SESSION_SELECTION) is not selection
        or context.user_data.get(PENDING_THREAD_ID) != request.thread_id
        or _required_selected_path(context) != request.cwd
    ):
        return True
    if not sessions:
        context.user_data.pop(PENDING_SESSION_SELECTION, None)
        return False
    selection.sessions = sessions
    await _render_picker(query, selection)
    return True


async def _render_picker(
    query: CallbackQuery, selection: SessionSelection, page: int = 0
) -> None:
    # Lazy: recovery package imports topic callbacks during handler registration.
    from ..recovery.resume_command import format_session_entry

    prefix = f"{CB_CREATE_SESSION}{selection.token}:"
    sessions = selection.sessions
    pages = (len(sessions) + _SESSIONS_PER_PAGE - 1) // _SESSIONS_PER_PAGE
    page = max(0, min(page, pages - 1))
    start = page * _SESSIONS_PER_PAGE
    rows = [
        [
            InlineKeyboardButton(
                format_session_entry(
                    summary=session.summary,
                    session_id=session.session_id,
                    mtime=session.mtime,
                    msg_count=session.msg_count,
                ),
                callback_data=f"{prefix}pick:{index}",
            )
        ]
        for index, session in enumerate(
            sessions[start : start + _SESSIONS_PER_PAGE], start=start
        )
    ]
    navigation = []
    if page > 0:
        navigation.append(
            InlineKeyboardButton("⬅ Previous", callback_data=f"{prefix}page:{page - 1}")
        )
    if page + 1 < pages:
        navigation.append(
            InlineKeyboardButton("Next ➡", callback_data=f"{prefix}page:{page + 1}")
        )
    if navigation:
        rows.append(navigation)
    rows.append(
        [
            InlineKeyboardButton("🆕 Start fresh", callback_data=f"{prefix}fresh"),
            InlineKeyboardButton("✖ Cancel", callback_data=CB_DIR_CANCEL),
        ]
    )
    request = selection.request
    await safe_edit(
        query,
        f"⏪ Resume a {request.provider_name} conversation?\n"
        f"📂 `{request.cwd}`\n\n"
        f"Select a past conversation or start fresh. Page {page + 1}/{pages}.",
        reply_markup=InlineKeyboardMarkup(rows),
    )


def _selected_session(
    selection: SessionSelection, action: str
) -> ResumableSession | None:
    if action == "fresh":
        return None
    if not action.startswith("pick:"):
        raise ValueError("Invalid selection")
    try:
        index = int(action.removeprefix("pick:"))
    except ValueError:
        raise ValueError("Invalid selection") from None
    if not 0 <= index < len(selection.sessions):
        raise ValueError("Session no longer in list")
    session = selection.sessions[index]
    if session.transcript_path and not Path(session.transcript_path).is_file():
        raise ValueError(
            "Session file no longer exists. Start fresh or reopen the picker."
        )
    return session


@register(CB_CREATE_SESSION)
async def handle_session_selection(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query, user = update.callback_query, update.effective_user
    if query is None or query.data is None or user is None:
        return
    selection = (
        context.user_data.get(PENDING_SESSION_SELECTION) if context.user_data else None
    )
    token, _, action = query.data.removeprefix(CB_CREATE_SESSION).partition(":")
    if (
        not isinstance(selection, SessionSelection)
        or selection.token != token
        or _browser_flow_stale(update, context)
        or _required_selected_path(context) != selection.request.cwd
    ):
        await query.answer(
            "Session picker expired. Reopen it in this topic.", show_alert=True
        )
        return
    if action.startswith("page:"):
        try:
            page = int(action.removeprefix("page:"))
        except ValueError:
            await query.answer("Invalid page", show_alert=True)
            return
        await query.answer()
        await _render_picker(query, selection, page)
        return
    request = selection.request
    try:
        session = _selected_session(selection, action)
    except (ValueError, OSError) as exc:
        await query.answer(str(exc), show_alert=True)
        return
    # Consume before the first await: double taps cannot create two windows.
    clear_browse_state(context.user_data)
    await query.answer()
    await launch_window(query, context, replace(request, resume_session=session))
