from __future__ import annotations

import re
from .alias_match import spelling_pattern

_UNKNOWN = re.compile(
    r"^\W*(?:unknown|n/?a|none|nothing|cannot\s+be\s+(?:inferred|determined)"
    r"|can(?:no|')t\s+(?:tell|say|determine|infer)|not\s+(?:stated|specified|inferable)"
    r"|no\s+(?:idea|indication|information)|unclear|indeterminate)\W*\.?$", re.I)


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def is_unknown(answer):
    return bool(_UNKNOWN.match(_norm(answer)))


THRESHOLD = 0.8
VERSION = "f3-identity-reconciliation-v1"
SCOPE = "registered short-target identity"

_WRAPPER_NOUNS = frozenset((
    "answer item detail details reference references mention mentions line lines "
    "sentence sentences phrase phrases statement statements text wording name names "
    "title titles date dates event events announcement announcements remark remarks"
).split())
_ARTICLES = frozenset(("a an the this that of to on about regarding concerning").split())
_REMOVAL = frozenset(("removed omitted excluded withdrawn deleted withheld redacted").split())
_AMBIGUOUS = re.compile(
    r"\b(?:not|no|never|isn't|wasn't|isnt|wasnt|cannot|can't|unknown|"
    r"maybe|perhaps|possibly|probably|either|or|versus|instead|rather|"
    r"except|without|unlikely|retained|kept|included|remaining|"
    r"approximately|roughly|about|around|over|under|least|most|"
    r"ignore|instruction|instructions|score|judge)\b|[?<>~≈]", re.I)
_NUMBER = re.compile(r"\d")


def _target_pattern(target):
    marker = "\ue000"
    if marker in target:
        raise ValueError("unsupported private-use character in target")
    protected = re.sub(r"(?<!\w)[-−](?=\d)", marker, target)
    pattern = spelling_pattern(protected)
    return re.compile(pattern.pattern.replace(marker, "[-−]"), pattern.flags)


def _outside_is_wrapper(before, after):
    after = re.sub(r"^\s*['’]s\b", "", after, flags=re.I)
    outside = before + " " + after
    if _NUMBER.search(outside) or re.search(r"[$€£%/@&=]", outside):
        return False
    words = re.findall(r"\b[\w']+\b", outside.casefold())
    return all(w in _WRAPPER_NOUNS | _ARTICLES | _REMOVAL for w in words)


def evaluate(target, answer, old_verdict=None, old_score=None):
    result = {"target": target, "answer": answer,
              "legacy_verdict": old_verdict, "legacy_score": old_score,
              "status": None, "recovered": None, "basis": None,
              "matched_span": None, "matched_text": None}

    def done(status, recovered, basis):
        result.update(status=status, recovered=recovered, basis=basis)
        return result

    if not target:
        return done("not_applicable", None, "no_withdrawn_target")
    if not isinstance(answer, str) or not answer.strip():
        return done("missing_answer", None, "empty_or_missing_answer")
    if is_unknown(answer):
        return done("abstain", False, "explicit_abstention")
    if old_verdict not in (None, "yes", "no", "missing", "abstain"):
        raise ValueError("invalid upstream recovery verdict")
    if old_score is not None:
        if not 0 <= old_score <= 1:
            raise ValueError("invalid upstream recovery score")
        if old_verdict in ("yes", "no") and (old_score >= THRESHOLD) != (old_verdict == "yes"):
            raise ValueError("upstream score/verdict mismatch")

    matches = list(_target_pattern(target).finditer(answer))
    if len(matches) == 1:
        match = matches[0]
        before, after = answer[:match.start()], answer[match.end():]
        result.update(matched_span=[match.start(), match.end()], matched_text=match.group())
        numeric_prefix = bool(re.match(r"[$€£]?\d", target) and re.search(r"[\d.$€£+−-]$", before.rstrip()))
        numeric_suffix = bool(target[-1].isdigit() and re.match(r"(?:[\d%]|\.\d)", after))
        outside = before + " " + after
        if not numeric_prefix and not numeric_suffix and not _AMBIGUOUS.search(outside) and _outside_is_wrapper(before, after):
            return done("recovered", True, "complete_target_with_surface_wrapper")
    if old_verdict in ("yes", "no"):
        return done("recovered" if old_verdict == "yes" else "not_recovered",
                    old_verdict == "yes", "upstream_semantic_judgment")
    return done("missing_judgment", None, "no_reusable_semantic_judgment")
