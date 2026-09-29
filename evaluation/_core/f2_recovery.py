from __future__ import annotations

import re
from . import f2_numeric as F2

def _norm(s):
    return " ".join((s or "").lower().split())


def _abstain(ans):
    return _norm(ans) in ("", "unknown", "n/a", "none")


VERSION = "f2-numeric-payload-v3"
SCOPE = "registered numbers and units"
parse_payload = F2.parse_payload
_UNCERTAIN = re.compile(r"\b(?:not|no|never|[a-z]+n't|cannot|unknown|maybe|perhaps|possibly|probably|either|rather than|instead of)\b", re.I)

def _unique(tokens):
    out, seen = [], set()
    for token in tokens:
        key = tuple(token[k] for k in ("value", "dimension", "count_label"))
        if key not in seen:
            out.append(token)
            seen.add(key)
    return out

def _ambiguous(observed, reference):
    text, tokens = observed["text"], observed["tokens"]
    if _UNCERTAIN.search(text):
        return True
    for token in tokens:
        if token["qualifier"] and not any(F2.compatible(r, token) and r["qualifier"] == token["qualifier"] for r in reference):
            return True
    for first, second in zip(tokens, tokens[1:]):
        between = text[first["span"][1]:second["span"][0]]
        if re.search(r"\bor\b", between, re.I) or re.fullmatch(r"\s*[-–]\s*", between):
            return True
    return False


def evaluate(target, answer, *, canonical=None):
    base = {"metric": VERSION, "scope": SCOPE, "target": target, "answer": answer,
            "n_http": 0, "recovered": None}
    if answer is None:
        return {**base, "status": "missing_answer"}
    if _abstain(answer):
        return {**base, "status": "abstain", "recovered": False}
    expected, observed = parse_payload(target), parse_payload(answer)
    if expected["unsupported"] or not expected["tokens"]:
        raise ValueError("unsupported registered F2 target: " + target)
    base.update(expected=expected, observed=observed)
    if not observed["tokens"] and not observed["unsupported"]:
        return {**base, "status": "not_recovered", "recovered": False, "reason": "no numeric payload"}
    reference = parse_payload(canonical or target)["tokens"]
    if observed["unsupported"] or _ambiguous(observed, reference):
        return {**base, "status": "needs_review", "reason": "unsupported numeric form, negation, approximation or alternative answers"}

    wants, got = _unique(expected["tokens"]), _unique(observed["tokens"])

    def assignment(i, used):
        if i == len(wants):
            return used
        for j, token in enumerate(got):
            if j not in used and F2.compatible(wants[i], token):
                result = assignment(i + 1, used | {j})
                if result is not None:
                    return result
        return None

    matched = assignment(0, set())
    if matched is not None and len(matched) == len(got):
        return {**base, "status": "full", "recovered": True}
    any_match = any(F2.compatible(w, g) for w in wants for g in got)
    if matched is not None:
        return {**base, "status": "needs_review", "reason": "correct payload accompanied by extra numeric claims"}
    return {**base, "status": "partial" if any_match else "not_recovered", "recovered": False,
            "reason": "required values or units missing/different"}
