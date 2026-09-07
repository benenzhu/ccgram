"""Keep Markdown tables intact for Telegram's native rich-message renderer."""

import re

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")


def split_markdown_tables(text: str) -> list[tuple[str, bool]]:
    """Return ordered (Markdown, is_table) segments, preserving fenced code."""
    lines = text.splitlines(keepends=True)
    segments: list[tuple[str, bool]] = []
    start = index = 0
    fence = ""
    while index < len(lines):
        match = _FENCE.match(lines[index])
        if match:
            marker = match.group(1)
            if not fence:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = ""
            index += 1
            continue
        if (
            not fence
            and "|" in lines[index]
            and index + 1 < len(lines)
            and _SEPARATOR.fullmatch(lines[index + 1].strip())
        ):
            if index > start:
                segments.append(("".join(lines[start:index]), False))
            end = index + 2
            while end < len(lines) and "|" in lines[end] and lines[end].strip():
                if _FENCE.match(lines[end]):
                    break
                end += 1
            segments.append(("".join(lines[index:end]), True))
            start = index = end
            continue
        index += 1
    if start < len(lines):
        segments.append(("".join(lines[start:]), False))
    return segments or [(text, False)]


def has_markdown_table(text: str) -> bool:
    return any(is_table for _, is_table in split_markdown_tables(text))
