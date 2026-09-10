"""Bounds and size steps for remotely viewed terminal windows."""

from .base import PaneDims

DEFAULT_COLUMNS = 160
DEFAULT_ROWS = 45
MIN_COLUMNS = 80
MAX_COLUMNS = 240
MIN_ROWS = 24
MAX_ROWS = 80
_COLUMN_STEP = 40
_ROW_STEP = 10


def valid_viewport_size(width: int, height: int) -> bool:
    """Bound image size and keep a usable terminal viewport."""
    return MIN_COLUMNS <= width <= MAX_COLUMNS and MIN_ROWS <= height <= MAX_ROWS


def adjusted_viewport(current: PaneDims, action: str, default: PaneDims) -> PaneDims:
    """Calculate a larger, smaller, or default terminal viewport."""
    if action == "reset":
        return default
    if action not in ("larger", "smaller"):
        raise ValueError(f"Unknown viewport action: {action}")
    direction = 1 if action == "larger" else -1
    return PaneDims(
        width=max(
            MIN_COLUMNS, min(MAX_COLUMNS, current.width + direction * _COLUMN_STEP)
        ),
        height=max(MIN_ROWS, min(MAX_ROWS, current.height + direction * _ROW_STEP)),
    )
