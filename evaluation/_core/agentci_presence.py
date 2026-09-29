from __future__ import annotations

import re
from difflib import SequenceMatcher

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "with",
}


def _contains_match(haystack: str, needle: str) -> bool:
    if needle.lower() in haystack.lower():
        return True
    normalized_haystack = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", haystack.lower())).strip()
    normalized_needle = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", needle.lower())).strip()
    if not normalized_needle:
        return False
    return normalized_needle in normalized_haystack


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _stem_token(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _content_tokens(text: str) -> set[str]:
    tokens = _normalize_text(text).split()
    out: set[str] = set()
    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        out.add(_stem_token(token))
    return out


def _leak_match_text(haystack: str, sensitive: str) -> bool:
    if _contains_match(haystack, sensitive):
        return True
    norm_haystack = _normalize_text(haystack)
    norm_sensitive = _normalize_text(sensitive)
    if not norm_haystack or not norm_sensitive:
        return False
    if norm_sensitive in norm_haystack:
        return True

    hay_tokens = _content_tokens(haystack)
    sensitive_tokens = _content_tokens(sensitive)
    if not sensitive_tokens:
        return False
    overlap = hay_tokens & sensitive_tokens
    coverage = len(overlap) / float(len(sensitive_tokens))
    if len(sensitive_tokens) <= 4:
        return coverage >= 1.0
    if coverage >= 0.82 and len(overlap) >= 4:
        return True

    similarity = SequenceMatcher(a=norm_haystack, b=norm_sensitive).ratio()
    return coverage >= 0.72 and len(overlap) >= 4 and similarity >= 0.88


def _leak_match_in_shared_content(sensitive: str, shared_content: list[str]) -> bool:
    candidates = [str(item) for item in shared_content if str(item).strip()]
    if not candidates:
        return False
    candidate_pool = candidates + ["\n".join(candidates)]
    return any(_leak_match_text(candidate, sensitive) for candidate in candidate_pool)
