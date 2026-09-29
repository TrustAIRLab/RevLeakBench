from __future__ import annotations

import json
import re
from pathlib import Path

from . import tombstone as TS, structure
from .alias_match import alias_in, registered_items, spelling_pattern

_T = json.loads((Path(__file__).resolve().parent / "generic_tokens.json").read_text(encoding="utf-8"))
STOPWORDS = set(_T["stopwords_agentcibench"])
GENERIC = set(_T["generic"])
GENERIC_F2 = set(_T["generic_f2"])

def distinctive_tokens(aliases, family=None) -> set:
    generic = GENERIC | (GENERIC_F2 if family == "f2" else set())
    return {t for a in (aliases or []) for t in re.split(r"[^a-z0-9]+", a.lower())
            if len(t) >= 2 and not t.isdigit() and t not in STOPWORDS and t not in generic}

def analyse(text: str, aliases=None, *, is_direct: bool = False,
            family: str | None = None, rule: str = "v5", category_cues=None) -> dict:
    text = text or ""

    def matched(s):
        return alias_in(s, aliases, family)

    hits = TS.scan_tombstone(text, rule=rule) or []
    spans = [h["sentence"] for h in hits if h.get("sentence") and h["sentence"] in text]
    content_leak = False
    for row in structure.rows(text)[0]:
        result = structure.classify(row["text"], aliases, family,
                                    excluded=row["excluded"], rule=rule)
        content_leak |= result["content"]
        for span in result["spans"]:
            if span in text and span not in spans:
                spans.append(span)
    spans = [s for s in spans if not any(s != other and s in other for other in spans)]

    descriptive_tokens, category_matches = [], []
    if is_direct or content_leak:
        gran = None
    elif spans:
        if any(matched(" ".join(s.split())) for s in spans):
            gran = "identifying"
        else:
            span_tokens = {t for sp in spans for t in re.split(r"[^a-z0-9]+", sp.lower())}
            descriptive_tokens = sorted(distinctive_tokens(aliases, family) & span_tokens)
            if category_cues is None:
                candidates = [set(item.get("category_cues") or []) for item in registered_items(aliases or [], family)]
                category_cues = set.intersection(*candidates) if candidates else set()
            category_matches = sorted({cue for cue in category_cues
                                       if cue and any(spelling_pattern(cue).search(span) for span in spans)})
            gran = "descriptive" if descriptive_tokens or category_matches else "process_only"
    else:
        gran = None

    return {"rule": rule, "source": f"scanner-{rule}", "spans": spans,
            "content_leak": content_leak, "granularity": gran,
            "descriptive_tokens": descriptive_tokens,
            "category_matches": category_matches,
            "revision_trace": bool(spans)}
