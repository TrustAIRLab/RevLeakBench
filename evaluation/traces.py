from __future__ import annotations

import re

from ._core import body_split, tombstone
from urllib.error import URLError

SCANNER_RULE = "v5"
SPLIT_VERSION = body_split.VERSION
TIERS = ("preface", "body_opening", "body_middle", "body_closing", "afterword")

def scan(text: str, rule: str = SCANNER_RULE) -> dict:
    hits = tombstone.scan_tombstone(text or "", rule=rule)
    return {"source": f"scanner-{rule}", "hit": bool(hits), "n": len(hits), "hits": hits}

def split(text: str) -> dict:
    return body_split.split(text or "")

def position(text: str, quote: str, parts: dict | None = None, *, start=0) -> str:
    parts = parts or split(text)
    spans = parts.get("spans") or []
    loc = body_split.locate(text or "", quote or "", spans, start=start)
    if loc != "body":
        return loc
    i = (text or "").find(quote, start)
    if i < 0:
        match = re.search(r"\s+".join(re.escape(w) for w in quote.split()), text[start:])
        i = start + match.start()
    offset = sum(max(0, min(i, b) - a) for a, b in spans)
    f = offset / max(1, sum(b - a for a, b in spans))
    return "body_opening" if f < 0.15 else "body_closing" if f >= 0.85 else "body_middle"

def positions(text: str, quotes) -> dict:
    text = text or ""
    parts = split(text)
    per, seen = [], set()
    for quote in dict.fromkeys(q for q in quotes if q and q.strip()):
        matches = list(re.finditer(r"\s+".join(re.escape(w) for w in quote.split()), text))
        if not matches:
            per.append({"quote": quote, "start": None, "end": None, "position": "missing"})
        for match in matches:
            if match.span() in seen:
                continue
            seen.add(match.span())
            per.append({"quote": quote, "start": match.start(), "end": match.end(),
                        "position": position(text, match.group(), parts, start=match.start())})
    per.sort(key=lambda r: r["start"] if r["start"] is not None else len(text))
    counts = {k: sum(1 for row in per if row["position"] == k) for k in (*TIERS, "missing")}
    return {"source": f"split-{SPLIT_VERSION}", "parts": parts, "per_quote": per, "counts": counts}

def levels(text: str, aliases=None, *, family: str | None = None,
           is_direct: bool = False, rule: str = SCANNER_RULE, category_cues=None) -> dict:
    from ._core import granularity
    return granularity.analyse(text or "", aliases, is_direct=is_direct, family=family, rule=rule,
                               category_cues=category_cues)

def judge(text: str, api_key: str, body: str | None = None) -> dict:
    from ._core import cited_judge
    ctx = body if body is not None else (text or "")
    try:
        reply = cited_judge.ask(api_key, ctx)
    except (URLError, OSError, ValueError, RuntimeError) as exc:
        return {"source": "llm-cited-judge", "judge_reply": None,
                "status": "missing_judgment", "verdict": None, "hit": None, "error": type(exc).__name__}
    parsed = cited_judge.parse_and_verify(reply, ctx)
    if not parsed["valid"]:
        return {"source": "llm-cited-judge", "judge_reply": reply,
                "status": "missing_judgment", "verdict": None, "hit": None}
    verdict = "yes_verified" if parsed["final"] else ("yes_unverified" if parsed["raw"] else "no")
    resolved = verdict == "yes_verified" or (verdict == "no" and (not parsed["quote"] or parsed["verified"]))
    return {"source": "llm-cited-judge", "judge_reply": reply,
            "status": "judged" if resolved else "missing_judgment",
            "hit": (verdict == "yes_verified") if resolved else None,
            "verdict": verdict, "quote": parsed.get("quote"),
            "quote_verified": parsed.get("verified"), "raw_verdict": parsed.get("raw")}

def analyse(text: str, *, aliases=None, family: str | None = None, is_direct: bool = False,
            api_key: str | None = None, rule: str = SCANNER_RULE, category_cues=None) -> dict:
    sc = scan(text, rule=rule)
    lv = levels(text, aliases, family=family, is_direct=is_direct, rule=rule, category_cues=category_cues)
    quotes = lv["spans"] or [h["sentence"] for h in sc["hits"]]
    out = {"scanner": {k: lv[k] for k in ("revision_trace", "granularity", "spans")},
           "raw_scan": sc,
           "position": positions(text, quotes),
           "levels": lv,
           "llm": None}
    if api_key:
        out["llm"] = judge(text, api_key)
    return out
