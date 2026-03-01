"""Shared type-inspection helpers used across the post-processing pipeline."""
from __future__ import annotations

from typing import Any, get_args, get_origin


def is_str_type(tp: Any) -> bool:
    """True if *tp* is ``str`` or ``List[str]``."""
    if tp is str:
        return True
    if get_origin(tp) is list:
        args = get_args(tp)
        return bool(args and args[0] is str)
    return False
