import calendar
import json, re
from pathlib import Path

ALIAS_MIN_LEN = 4
ALIAS_MATCH_VERSION = "boundary-v2"
_SHORT_NAMES = json.loads((Path(__file__).resolve().parent / "f5_short_names.json").read_text(encoding="utf-8"))

_MONTHS = {name.lower(): i for i in range(1, 13)
           for name in (calendar.month_name[i], calendar.month_abbr[i])}
_DATE = re.compile(r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) +
                   r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+((?:19|20)\d{2}))?\b", re.I)
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

def spelling_pattern(value):
    chunks, pos = [], 0
    for m in _DATE.finditer(value):
        chunks.append(_words(value[pos:m.start()]))
        month, day, year = m.groups()
        names = "(?:" + calendar.month_name[_MONTHS[month.lower()]] + "|" + calendar.month_abbr[_MONTHS[month.lower()]] + r"\.?)"
        d = str(int(day)) + r"(?:st|nd|rd|th)?"
        date = r"(?:" + names + r"[\s-]+0?" + d + r"|0?" + d + r"[\s-]+" + names + r")"
        if year:
            date += r",?\s+" + year
        chunks.append(date)
        pos = m.end()
    chunks.append(_words(value[pos:]))
    return re.compile(r"(?<![\w$.,])" + "".join(chunks) + r"(?!\w)", re.I)
def _words(value):
    parts = re.split(r"(,?\s+and\s+|,\s*|[\s\-–—]+)", value)
    out = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r",?\s+and\s+|,\s*", part):
            out.append(r"(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s+)")
        elif re.fullmatch(r"[\s\-–—]+", part):
            out.append(r"[\s\-–—]+")
        else:
            out.append(re.escape(part).replace("'", "['’]"))
    return "".join(out)

def _literal(text):
    return re.compile(r"(?<!\w)" + r"\s+".join(re.escape(w) for w in text.split()) + r"(?!\w)", re.I)

def alias_pattern(alias):
    item = _index().get(alias.lower())
    if item:
        return item_pattern(item)
    return re.compile(r"(?<!\w)" + r"\s+".join(re.escape(s) for s in alias.split()) + r"(?!\w)", re.I)
def item_pattern(item):
    match = item.get("diagnostic_match") or {}
    context, values = match.get("context_any", []), match.get("values_any", [])
    groups = match.get("required_context_groups", [])
    checks = []
    for choices in ([context] if context else []) + groups:
        checks.append(r"(?=[^\n]*(?:" + "|".join(r"(?<!\w)" + re.escape(s) for s in choices) + "))")
    if values:
        checks.append(r"(?=[^\n]*(?:" + "|".join(r"(?<![+\-−])" + fragment_pattern(str(v)).pattern.replace(r"\-", "[-−]") for v in values) + "))")
    if not checks:
        return re.compile(r"\s+".join(re.escape(s) for s in item["canonical"].split()), re.I)
    return re.compile("".join(checks) + r"[^\n]+", re.I)
_INDEX_CACHE = None
_RECORDS_CACHE = None

def registered_items(aliases=None, family=None):
    global _RECORDS_CACHE
    if _RECORDS_CACHE is None:
        _RECORDS_CACHE = []
        root = Path(__file__).resolve().parents[2] / "data"
        for p in sorted(root.glob("*/*/task.json")):
            annotation = json.loads(p.read_text(encoding="utf-8"))["evaluation"]
            for item in annotation["evaluation_targets"].values():
                _RECORDS_CACHE.append({**item, "family": annotation["family"].split("_")[0]})
    values = {" ".join(a.lower().split()) for a in aliases or []}
    return [item for item in _RECORDS_CACHE if (family is None or item["family"] == family)
            and (aliases is None or values == {" ".join(a.lower().split()) for a in item["target_aliases"]})]

def _index():
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        items = {}
        for item in registered_items():
            for alias in [item["canonical"], *item["target_aliases"]]:
                items[alias.lower()] = item
        _INDEX_CACHE = items
    return _INDEX_CACHE

def fragment_pattern(fragment: str) -> re.Pattern:
    body = re.escape(fragment).replace("\\ ", r"\s*")
    suffix = r"(?![\w%]|[.,]\d)" if re.fullmatch(_NUMBER, fragment) else r"(?!\w|[.,]\d)"
    return re.compile(r"(?<![\w.,$€£])" + body + suffix, re.I)

def usable_aliases(aliases, family=None):
    trusted = set()
    if family == "f5":
        values = {a.lower() for a in (aliases or []) if a}
        trusted = {n for n, full in _SHORT_NAMES.items() if values & set(full)}
    return [a for a in (aliases or []) if a and
            (len(a.strip()) >= ALIAS_MIN_LEN or any(c.isdigit() for c in a) or a.lower() in trusted)]

def alias_re(alias, family=None):
    if family == "f3":
        return spelling_pattern(alias)
    if family == "f5":
        return _literal(alias)
    if family == "f4":
        return alias_pattern(alias)
    if family == "f2" and any(c.isdigit() for c in alias):
        return fragment_pattern(alias)
    body = r"\s+".join(re.escape(w) for w in alias.strip().split())
    return re.compile(r"(?<![\w$.,])" + body + r"(?!\w)", re.I)

def alias_in(text, aliases, family=None):
    t = text or ""
    return any(alias_re(a, family).search(t) for a in usable_aliases(aliases, family))
