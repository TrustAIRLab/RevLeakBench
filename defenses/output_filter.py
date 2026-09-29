from __future__ import annotations

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation._core import granularity

WITHHELD = "withheld"

def filter_reply(text: str, *, aliases=None, protect=(), family=None, rule="v5") -> dict:
    text = text or ""
    if not text.strip():
        return {"delivered": None, "status": WITHHELD, "reason": "empty input",
                "removed": [], "conflicts": []}
    spans = granularity.analyse(text, aliases, family=family, rule=rule)["spans"]
    if not spans:
        return {"delivered": text, "status": "delivered", "removed": [], "conflicts": []}

    offsets = sorted({m.span() for s in spans for m in re.finditer(re.escape(s), text)})
    merged = []
    for a, b in offsets:
        if merged and a < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    kept = text
    for a, b in reversed(merged):
        kept = kept[:a] + kept[b:]
    removed = [text[a:b] for a, b in merged]
    if not kept.strip():
        return {"delivered": None, "status": WITHHELD, "reason": "filtered to empty",
                "removed": removed, "conflicts": []}
    for p in protect or ():
        if p and p in text and p not in kept:
            return {"delivered": None, "status": WITHHELD,
                    "reason": "required content was removed: %.40s" % p,
                    "removed": removed, "conflicts": [s for s in removed if p in s]}
    return {"delivered": kept, "status": "delivered", "removed": removed, "conflicts": []}
