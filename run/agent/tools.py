from __future__ import annotations

TOOL_NAMES = ("list_files", "read_file", "search")

def make_tools(files, names=TOOL_NAMES, extra_tools=()):
    from .runner import make_tools as _make
    return _make(files, list(names), extra_tools)
