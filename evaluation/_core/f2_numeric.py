from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_UNITS = (r"years?|months?|weeks?|days?|hours?|minutes?|seconds?|violations?|providers?|sessions?|"
          r"hires?|incidents?|tickets?|members?|stores?|projects?|forms?|documents?|vehicles?|"
          r"batteries|agents?|employees?|customers?|patients?|products?|points?|requests?")
_NUM = re.compile(r"(?<![\w.,$€£])[$€£]?" + _NUMBER +
                  r"(?:\s*(?:trillion|billion|million|thousand|[MKB])\b)?"
                  r"(?:\s*%|[\s-]+(?:" + _UNITS + r")\b)?(?!\w)", re.I)
_EXCLUDED_NUMBER_CONTEXT = re.compile(r"\b(?:[QH]\s*[1-4]|quarter\s*[1-4]|FSMA\s+\d+|ISO\s+\d+(?:-\d+)*)\b", re.I)
_UNIT_WORD = {"M": "million", "K": "thousand", "B": "billion"}
_UNIT_ABBR = {v: k for k, v in _UNIT_WORD.items()}
_SCALE = {"k": Decimal(1000), "thousand": Decimal(1000),
          "m": Decimal(1000000), "million": Decimal(1000000),
          "b": Decimal(1000000000), "billion": Decimal(1000000000),
          "trillion": Decimal(1000000000000)}
_TIME = {"year": ("calendar_month", 12), "month": ("calendar_month", 1),
         "week": ("second", 604800), "day": ("second", 86400),
         "hour": ("second", 3600), "minute": ("second", 60), "second": ("second", 1)}
_TIME_ALIASES = {"yr": "year", "yrs": "years", "mo": "month", "mos": "months",
                 "wk": "week", "wks": "weeks", "hr": "hour", "hrs": "hours", "h": "hours",
                 "min": "minute", "mins": "minutes", "sec": "second", "secs": "seconds", "s": "seconds"}
_WORD_NUMBER = re.compile(r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
                          r"eleven|twelve|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b", re.I)
_QUALIFIERS = {"roughly": "approximate", "about": "approximate", "approximately": "approximate",
               "~": "approximate", "≈": "approximate", "over": "greater", "more than": "greater", ">": "greater",
               "under": "less", "less than": "less", "<": "less", "at least": "minimum",
               "at most": "maximum", "up to": "maximum", "between": "range"}
_QUALIFIER = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(_QUALIFIERS, key=len, reverse=True)) + r")\s*$", re.I)

def normalize_text(text):
    text = str(text or "").replace("−", "-").replace("％", "%").replace("’", "'")
    text = re.sub(r"([$€£])\s+(?=[+-]?\d)", r"\1", text)
    text = re.sub(r"\bUS(?:D)?\s*\$\s*(?=[+-]?\d)", "$", text, flags=re.I)
    text = re.sub(r"(?<=\d)\s*(?:per\s*cent\b|percent\b|pct\b\.?)", "%", text, flags=re.I)
    text = re.sub(r"\b(?:USD|US dollars?)\s*(?=[+-]?\d)", "$", text, flags=re.I)
    text = re.sub(r"(?<=\d)[\s-]*(" + "|".join(sorted(_TIME_ALIASES, key=len, reverse=True)) + r")(?=\b|\d)\.?",
                  lambda m: " " + _TIME_ALIASES[m[1].lower()] + " ", text, flags=re.I)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*([kmb])\b", lambda m: m[1] + m[2].upper(), text, flags=re.I)
    text = re.sub(r"(?<![$€£\w.])(\d[\d,.]*(?:\s*(?:million|billion|thousand|trillion|[MKB]))?)\s*(?:(?:US )?dollars?|USD)\b",
                  lambda m: "$" + m[1], text, flags=re.I)
    return text

def compatible(want, got):
    return (want["value"] == got["value"] and want["dimension"] == got["dimension"]
            and (not want["count_label"] or want["count_label"] == got["count_label"]))

def parse_payload(text):
    text = normalize_text(text)
    audit = numeric_audit(text)
    allowed = set(audit["fragments"])
    tokens, consumed = [], set()
    for context in _EXCLUDED_NUMBER_CONTEXT.finditer(text):
        consumed.update(range(context.start(), context.end()))
    for m in _NUM.finditer(text):
        literal = m.group().strip()
        consumed.update(range(m.start(), m.end()))
        if literal not in allowed:
            continue
        part = re.fullmatch(r"([$€£]?)(" + _NUMBER +
                            r")(?:\s*(trillion|billion|million|thousand|[MKB]))?"
                            r"(?:\s*(%)|[\s-]+([A-Za-z]+))?", literal, re.I)
        if not part:
            return {"text": text, "tokens": tokens, "unsupported": True, "excluded": audit["excluded"]}
        currency, number, scale, percent, unit = part.groups()
        value = Decimal(number.replace(",", "")) * _SCALE.get((scale or "").lower(), Decimal(1))
        prefix = text[:m.start()]
        if prefix.endswith("-"):
            value = -value
            if len(prefix) > 1 and prefix[-2] in "$€£":
                currency = prefix[-2]
        unit = (unit or "").lower().rstrip("s")
        count_label = ""
        if currency:
            dimension = currency
        elif percent:
            dimension = "%"
        elif unit in _TIME:
            dimension, factor = _TIME[unit]
            value *= factor
        else:
            dimension, count_label = "count", unit
        qualifier = _QUALIFIER.search(prefix)
        tokens.append({"value": str(value.normalize()), "dimension": dimension,
                       "count_label": count_label, "literal": literal, "span": [m.start(), m.end()],
                       "number_end": m.start() + part.end(2),
                       "time_factor": _TIME[unit][1] if unit in _TIME and not currency and not percent else None,
                       "time_units": [unit] if unit in _TIME and not currency and not percent else [],
                       "qualifier": _QUALIFIERS[qualifier[1].lower()] if qualifier else None})
    combined = []
    for token in tokens:
        previous = combined[-1] if combined else None
        if (previous and previous["time_factor"] and token["time_factor"]
                and previous["dimension"] == token["dimension"]
                and token["time_units"][0] not in previous["time_units"]
                and Decimal(previous["value"]) >= 0 and Decimal(token["value"]) >= 0
                and token["qualifier"] is None
                and re.fullmatch(r"\s*(?:,\s*)?(?:and\s+)?", text[previous["span"][1]:token["span"][0]], re.I)):
            previous["value"] = str((Decimal(previous["value"]) + Decimal(token["value"])).normalize())
            previous["span"][1] = token["span"][1]
            previous["literal"] = text[previous["span"][0]:previous["span"][1]]
            previous["time_factor"] = token["time_factor"]
            previous["time_units"].extend(token["time_units"])
        else:
            combined.append(token)
    unsupported = any(ch.isdigit() and i not in consumed for i, ch in enumerate(text))
    unsupported |= bool(_WORD_NUMBER.search(text))
    return {"text": text, "tokens": combined, "unsupported": unsupported, "excluded": audit["excluded"]}


def numeric_fragments(text: str) -> list[str]:
    return numeric_audit(text)["fragments"]


def numeric_audit(text: str) -> dict[str, Any]:
    text = str(text or "")
    contexts = list(_EXCLUDED_NUMBER_CONTEXT.finditer(text))
    excluded = [{"text": m.group(), "reason": "calendar label or public standard identifier"} for m in contexts]
    fragments = []
    for m in _NUM.finditer(text):
        if any(x.start() <= m.start() < x.end() for x in contexts):
            continue
        value = m.group().strip()
        if re.fullmatch(r"(?:19|20)\d{2}|2100", value):
            excluded.append({"text": value, "reason": "calendar year"})
        elif value not in fragments:
            fragments.append(value)
    return {"fragments": fragments, "excluded": excluded}


def fragment_variants(fragment: str) -> list[str]:
    variants = [fragment]
    m = re.fullmatch(r"(\$[\d,.]+)\s*(trillion|billion|million|thousand|M|K|B)", fragment)
    if m:
        num, unit = m.groups()
        if unit in _UNIT_WORD:
            variants.append(f"{num} {_UNIT_WORD[unit]}")
        elif unit in _UNIT_ABBR:
            variants.append(f"{num}{_UNIT_ABBR[unit]}")
    for v in list(variants):
        if "," in v:
            variants.append(v.replace(",", ""))
    m = re.fullmatch(r"(" + _NUMBER + r")[\s-]+(years?|months?|weeks?|days?|hours?|minutes?|seconds?)", fragment)
    if m:
        num, unit = m.groups()
        stem = unit.rstrip("s")
        variants += [f"{num} {stem}", f"{num} {stem}s", f"{num}-{stem}"]
    return list(dict.fromkeys(variants))


def fragment_pattern(fragment: str) -> re.Pattern:
    body = re.escape(fragment).replace("\\ ", r"\s*")
    suffix = r"(?![\w%]|[.,]\d)" if re.fullmatch(_NUMBER, fragment) else r"(?!\w|[.,]\d)"
    return re.compile(r"(?<![\w.,$€£])" + body + suffix, re.I)


def count_fragment(fragment: str, text: str) -> int:
    return len(fragment_pattern(fragment).findall(text or ""))


def fragment_present(fragment: str, text: str) -> bool:
    return count_fragment(fragment, text) > 0


def item_numeric_coverage(item: dict[str, Any], text: str) -> dict[str, Any]:
    groups = item["numeric_groups"]
    observed = parse_payload(text)
    matched = []
    for group in groups:
        expected = parse_payload(group["literal"])["tokens"]
        if not expected:
            continue
        for token in observed["tokens"]:
            if not compatible(expected[0], token):
                continue
            if re.fullmatch(_NUMBER, group["literal"]) and group["literal"] not in group["variants"]:
                suffixes = [v[len(group["literal"]):].strip() for v in group["variants"] if v.startswith(group["literal"])]
                tail = observed["text"][token["number_end"]:]
                if not any(re.match(r"\s+" + r"\s+".join(re.escape(w) for w in s.split()) + r"\b", tail, re.I) for s in suffixes if s):
                    continue
            matched.append(group["literal"])
            break
    missing = [g["literal"] for g in groups if g["literal"] not in matched]
    return {"matched": matched, "missing": missing, "any_numeric_residue": bool(matched),
            "all_numeric_values_present": bool(groups) and not missing}
