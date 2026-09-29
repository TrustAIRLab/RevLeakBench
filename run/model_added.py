from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation._core.alias_match import alias_re, usable_aliases
from evaluation._core import redacbench, structure
from evaluation._core.tombstone import _sentences
from .conversation import generate, result

MAX_DRAFTS = 4
_TIME = re.compile(r"(?<![\w:])(?:\d{1,2}(?::\d{2})?\s*[AP]\.?M\.?(?!\w)|\d{1,2}:\d{2}(?!\w|:\d|\s*[AP]\.?M\b))", re.I)
_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
             "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
             "eighteenth", "nineteenth", "twentieth"]

_draft_rows = structure.rows

def _content_rows(draft, aliases, family):
    rows, groups = _draft_rows(draft)
    return [r for r in rows if not r["excluded"] and _content(r["text"], aliases, family)]

def _content(ln, aliases, family):
    return structure.classify(ln, aliases, family)["content"]

def content_line(draft: str, aliases, family=None):
    rows = _content_rows(draft, aliases, family)
    return (rows[0]["text"], rows[0]["position"]) if rows else (None, None)

def _clock_value(text):
    text = re.sub(r"[.\s]", "", text).upper()
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(AM|PM)?", text)
    if not match:
        return None
    hour, minute, meridiem = int(match[1]), int(match[2] or 0), match[3]
    if minute > 59 or (meridiem and not 1 <= hour <= 12) or (not meridiem and hour > 23):
        return None
    if meridiem:
        hour = hour % 12 + (12 if meridiem == "PM" else 0)
    return hour, minute

def _time_entries(line):
    entries = []
    for sent in _sentences(line):
        for clause in re.split(r";|\b(?:followed by|and then|then)\b", sent, flags=re.I):
            groups = []
            for match in _TIME.finditer(clause):
                if groups and re.fullmatch(r"\s*(?:[-–—]|to|until)\s*", clause[groups[-1][-1].end():match.start()], re.I):
                    groups[-1].append(match)
                else:
                    groups.append([match])
            if not groups:
                continue
            if len(groups) == 1:
                entries.append((_clock_value(groups[0][0].group()), clause, clause))
                continue
            before, after = clause[:groups[0][0].start()], clause[groups[-1][-1].end():]
            prefix = re.sub(r"^\s*\d+[.)]\s+", "", before)
            if re.search(r"\w", after) and (not re.search(r"\w", prefix) or prefix.rstrip().endswith(":")):
                for i, group in enumerate(groups):
                    end = groups[i + 1][0].start() if i + 1 < len(groups) else len(clause)
                    entries.append((_clock_value(group[0].group()), clause[group[0].start():end], clause))
            elif re.search(r"\w", prefix) and not re.search(r"\w", after):
                for i, group in enumerate(groups):
                    start = groups[i - 1][-1].end() if i else 0
                    entries.append((_clock_value(group[0].group()), clause[start:group[-1].end()], clause))
    return entries

def _time_binding(entry, clause, aliases, family):
    patterns = [alias_re(a, family) for a in sorted(usable_aliases(aliases, family), key=len, reverse=True)]
    outside = clause
    for pattern in patterns:
        outside = pattern.sub("", outside)
    if re.search(r"\b(?:not|no|never|[a-z]+n['’]t|cannot|either|or|maybe|perhaps|possibly|probably|except|instead)\b", outside, re.I):
        return False
    clocks = list(_TIME.finditer(entry))
    if not clocks:
        return False
    links = {"at", "from", "is", "on", "scheduled", "for", "starts", "starting", "begins", "beginning",
             "held", "set", "to", "take", "place", "the", "a", "an", "time", "block", "slot"}
    for pattern in patterns:
        for match in pattern.finditer(entry):
            if match.end() <= clocks[0].start():
                bridge = entry[match.end():clocks[0].start()]
            elif match.start() >= clocks[-1].end():
                bridge = entry[clocks[-1].end():match.start()]
            else:
                continue
            if set(re.findall(r"\w+", bridge.lower())) <= links:
                return True
    return False

def _scope_words(text):
    text = re.sub(r"\bto[ -]?do(?:s)?\b|\btasks?\b", "todo", text, flags=re.I)
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower())} - {"the", "a", "an", "item", "list"}

def eligible(draft: str, aliases, family=None) -> bool:
    return content_line(draft, aliases, family)[0] is not None

def removal_turn(task: dict, annotation: dict, slot: str, draft: str, *,
                 api_key=None, model=redacbench.DEFAULT_EVAL_MODEL):
    ev = (annotation.get("evaluation_targets") or {}).get(slot) or {}
    aliases = ev.get("target_aliases")
    family = (task.get("family") or "").split("_")[0]
    ref = (annotation.get("short_reference") or {}).get(slot)
    template = ((task.get("dialogue") or {}).get("case1")
                or (annotation.get("dialogue") or {}).get("case1"))

    all_rows, groups = _draft_rows(draft)
    if family == "f2":
        content = []
        for row in all_rows:
            if row["excluded"]:
                continue
            value = row["text"]
            for span in structure.classify(value)["spans"]:
                value = value.replace(span, " ")
            content.append(value)
        context = "\n".join(content)
        if not context.strip():
            return None, {"reason": "item is not present as content in the draft"}
        if not api_key:
            return None, {"reason": "F2 construction requires a proposition judgment"}
        proposition = ev["canonical"]
        try:
            verdicts = redacbench.check_propositions(api_key, context, [proposition], model=model)
        except redacbench.BridgeError:
            return None, {"reason": "F2 construction judgment is unresolved"}
        matches = [v for v in verdicts if isinstance(v, dict)
                   and str(v.get("proposition", "")).strip() == proposition.strip()] if isinstance(verdicts, list) else []
        if len(matches) != 1 or type(matches[0].get("is_true")) is not bool:
            return None, {"reason": "F2 construction judgment is unresolved"}
        if not matches[0]["is_true"]:
            return None, {"reason": "the draft does not contain the registered fact"}
        turns = task.get("case1_turns") or annotation.get("case1_turns") or {}
        return turns.get(slot), {"rule": "registered fact verified by checkPropositions",
                                 "line": context, "reference": ref}
    rows = [r for r in all_rows if not r["excluded"] and _content(r["text"], aliases, family)]
    if not rows:
        return None, {"reason": "item is not present as content in the draft"}
    line = rows[0]["text"]

    if annotation.get("case1_reference_kind") == "semantic" or task.get("case1_reference_kind") == "semantic":
        turns = task.get("case1_turns") or annotation.get("case1_turns") or {}
        return turns.get(slot), {"rule": "preselected semantic reference; item observed as draft content",
                                 "line": line, "reference": ref}
    if not (ref and template):
        return None, {"reason": "missing short_reference or the dialogue.case1 template"}

    m_time = _TIME.search(ref)
    if m_time:
        expected = _clock_value(m_time.group())
        entries = [(entry, clause) for row in all_rows if not row["excluded"] for clock, entry, clause in _time_entries(row["text"])
                   if clock is not None and clock == expected]
        if len(entries) == 1:
            entry, clause = entries[0]
            if _content(entry, aliases, family) and _time_binding(entry, clause, aliases, family):
                return template.format(ref=ref), {"rule": "time reference from spec", "line": entry, "reference": ref}
        return None, {"reason": "the target is not bound to the reference time",
                      "line": line, "reference": ref}

    m_ord = re.match(r"^the (\w+) (.+)$", ref)
    if not m_ord or m_ord.group(1) not in _ORDINALS:
        return None, {"reason": "cannot derive a positional reference from %r" % ref, "line": line}
    listed = [r for r in rows if r["position"] is not None and r["position"] <= len(_ORDINALS)]
    if not listed:
        return None, {"reason": "item appears in prose, not as a list item", "line": line}
    words = _scope_words(m_ord.group(2))
    groups = {key: group for key, group in groups.items() if not group["excluded"]}
    matching = {key for key, group in groups.items() if words & _scope_words(group["heading"])}
    if matching:
        roots = {key for key in matching if groups[key]["parent"] not in matching}
        listed = [r for r in listed if len(roots) == 1 and r["list"] in roots]
    else:
        parents = {g["parent"] for g in groups.values()}
        leaves = set(groups) - parents
        candidates = {r["list"] for r in listed}
        listed = [r for r in listed if not groups[r["list"]]["heading"]
                  and (len(groups) == 1 or len(candidates) == 1 and len(leaves) == 1)]
    if not listed or len({r["list"] for r in listed}) != 1:
        return None, {"reason": "the reference does not identify a unique list", "reference": ref}
    line, pos = listed[0]["text"], listed[0]["position"]
    new_ref = "the %s %s" % (_ORDINALS[pos - 1], m_ord.group(2))
    return template.format(ref=new_ref), {"rule": "position in the model's own list",
                                          "line": line, "position": pos, "reference": new_ref}

def build(task_id: str, send, track: str = "conversation", *, direct_reply,
          max_drafts: int = MAX_DRAFTS, api_key=None,
          model=redacbench.DEFAULT_EVAL_MODEL) -> dict:
    import conditions
    task = conditions.load_task(task_id)
    ann = conditions.load_annotation(task_id)
    if task["family"].split("_")[0] == "f2" and not api_key:
        raise ValueError("api_key is required for F2 draft qualification")
    if type(max_drafts) is not int or not 1 <= max_drafts <= MAX_DRAFTS:
        raise ValueError("max_drafts must be between 1 and 4")
    if not isinstance(direct_reply, dict):
        raise TypeError("direct_reply must be a generation result with completion metadata")
    for field, expected in (("task_id", task_id), ("track", track), ("condition", "direct")):
        if field in direct_reply and direct_reply[field] != expected:
            raise ValueError("direct reply has a different " + field)
    msgs = conditions.messages(task_id, "direct", track)
    drafts, cases = [], {}
    current = result(direct_reply)
    for sample in range(max_drafts):
        drafts.append({"sample": sample, "reused": sample == 0, **current})
        if current["status"] != "not_generated":
            for slot in ("A", "B"):
                if slot in cases:
                    continue
                options = {"api_key": api_key, "model": model} if task["family"].split("_")[0] == "f2" else {}
                turn, why = removal_turn(task, ann, slot, current["reply"], **options)
                if turn:
                    cases[slot] = {"task_id": task_id, "slot": slot, "track": track,
                                   "status": "constructed", "draft": current["reply"],
                                   "sample": sample, "removal_turn": turn, "reference_rule": why}
        if len(cases) == 2 or sample + 1 == max_drafts:
            break
        current = generate(msgs, send, request_budget=1)
    for slot in ("A", "B"):
        if slot not in cases:
            cases[slot] = {"task_id": task_id, "slot": slot, "track": track,
                           "status": "construction_failed", "reply": None}
    return {"task_id": task_id, "track": track, "drafts": drafts,
            "n_drafts": len(drafts), "cases": cases}

def replay(built: dict, send, track: str = None, *, request_budget=3) -> dict:
    import conditions
    track = track or built.get("track") or "conversation"
    if built.get("track") and track != built["track"]:
        raise ValueError("replay track %r does not match the track the draft was built on (%r)"
                         % (track, built["track"]))
    if built.get("status") != "constructed":
        return {**built, "track": track, "condition": "model_added_%s" % built["slot"],
                "construction_status": built.get("status"), "status": "not_generated",
                "reply": None, "n_requests": 0}
    first = conditions.messages(built["task_id"], "direct", track)[0]
    msgs = [first,
            {"role": "assistant", "content": built["draft"]},
            {"role": "user", "content": built["removal_turn"]}]
    return {"task_id": built["task_id"], "slot": built["slot"], "track": track,
            "condition": "model_added_%s" % built["slot"], "messages": msgs,
            "reference_rule": built["reference_rule"],
            **generate(msgs, send, request_budget=request_budget)}
