"""Saved Codex chats feed both the project picker and /resume."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ccgram.handlers.recovery.resume_command import scan_all_sessions
from ccgram.handlers.recovery.resume_picker import scan_sessions_for_cwd
from ccgram.providers.codex import CodexProvider


@pytest.fixture
def sessions_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    path = tmp_path / "codex" / "sessions" / "2026" / "01" / "02"
    path.mkdir(parents=True)
    return path


def _session(
    root: Path,
    session_id: str,
    cwd: str,
    *,
    mtime: float = 100.0,
    source: object = "cli",
    originator: str = "codex_cli_rs",
    entries: list[object] | None = None,
    filename: str | None = None,
) -> Path:
    path = root / (filename or f"rollout-{session_id}.jsonl")
    lines = [
        {
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "cwd": cwd,
                "source": source,
                "originator": originator,
            },
        },
        *(entries or []),
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in lines) + "\n")
    os.utime(path, (mtime, mtime))
    return path


def test_history_is_not_limited_to_recent_active_transcripts(sessions_dir, tmp_path):
    cwd = str(tmp_path / "project")
    old = _session(sessions_dir, "old-session", cwd, mtime=1)
    for index in range(25):
        _session(
            sessions_dir,
            f"other-{index}",
            str(tmp_path / "elsewhere"),
            mtime=100 + index,
        )
    latest = _session(sessions_dir, "latest-session", cwd, mtime=30)
    _session(sessions_dir, "child-session", cwd + "/child", mtime=50)
    before = old.read_bytes()

    sessions = CodexProvider().discover_resumable_sessions(cwd=cwd)

    assert [s.session_id for s in sessions] == ["latest-session", "old-session"]
    assert [s.transcript_path for s in sessions] == [str(latest), str(old)]
    assert old.read_bytes() == before
    assert [
        s.session_id
        for s in CodexProvider().discover_resumable_sessions(cwd=cwd, limit=1)
    ] == ["latest-session"]
    assert CodexProvider().discover_resumable_sessions(limit=0) == []


def test_filters_subagents_exec_and_duplicate_sessions(sessions_dir, tmp_path):
    cwd = str(tmp_path)
    _session(sessions_dir, "main", cwd, mtime=1)
    newest = _session(sessions_dir, "main", cwd, mtime=10, filename="duplicate.jsonl")
    _session(sessions_dir, "guardian", cwd, source={"subagent": "guardian_review"})
    _session(sessions_dir, "worker", cwd, source={"subagent": {"thread_spawn": {}}})
    _session(sessions_dir, "batch", cwd, originator="codex_exec")
    _session(sessions_dir, "desktop", cwd, source="exec", originator="codex_desktop")

    sessions = CodexProvider().discover_resumable_sessions()

    assert {s.session_id for s in sessions} == {"main", "desktop"}
    assert next(s for s in sessions if s.session_id == "main").transcript_path == str(
        newest
    )


@pytest.mark.parametrize("entry_type", ["event_msg", "response_item", "input_item"])
def test_summary_uses_human_prompt_after_injected_context(
    sessions_dir, tmp_path, entry_type
):
    prompt = "Fix the parser\nand add regression coverage"
    payload = (
        {"type": "user_message", "message": prompt}
        if entry_type == "event_msg"
        else {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
    )
    entries: list[object] = [
        {
            "type": "response_item",
            "payload": {
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
        for text in [
            "# AGENTS.md instructions for /repo",
            "<environment_context>cwd</environment_context>",
        ]
    ]
    entries += [None, {"payload": []}, {"type": entry_type, "payload": payload}]
    _session(sessions_dir, "main", str(tmp_path), entries=entries)

    assert CodexProvider().discover_resumable_sessions()[0].summary == prompt


def test_malformed_files_and_unsafe_metadata_do_not_break_enumeration(
    sessions_dir, tmp_path
):
    for index, raw in enumerate(
        [b"not-json", b"[]", b"null", b"\xff", b'{"type":"session_meta","payload":[]}']
    ):
        (sessions_dir / f"bad-{index}.jsonl").write_bytes(raw)
    _session(sessions_dir, "unsafe;id", str(tmp_path))
    _session(sessions_dir, "relative-cwd", "project")
    _session(sessions_dir, "main", str(tmp_path))

    assert [s.session_id for s in CodexProvider().discover_resumable_sessions()] == [
        "main"
    ]


def test_workspace_alias_and_both_existing_pickers_use_codex_history(
    sessions_dir, tmp_path
):
    project = tmp_path / "project"
    project.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(project, target_is_directory=True)
    _session(sessions_dir, "main", str(alias))

    sessions = scan_sessions_for_cwd(str(project), "codex")
    assert [s.session_id for s in sessions] == ["main"]
    assert [s.provider_name for s in scan_all_sessions("codex")] == ["codex"]


def test_absent_history_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing"))
    assert CodexProvider().discover_resumable_sessions() == []
