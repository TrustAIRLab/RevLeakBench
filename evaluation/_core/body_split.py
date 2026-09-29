from __future__ import annotations

import re

from markdown_it import MarkdownIt

def fenced_spans(text, *, include_gaps=False):
    lines = re.findall(r"[^\r\n]*(?:\r\n|\r|\n|$)", text)
    if lines and not lines[-1]:
        lines.pop()
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    spans, previous_end = [], None
    for token in MarkdownIt("commonmark").parse(text):
        if token.type != "fence" or token.map is None:
            continue
        first, end = token.map
        if end > len(lines):
            raise ValueError("Markdown source map exceeds original line count")
        last = lines[end - 1].rstrip("\r\n").strip().lstrip("> ").strip()
        closing = bool(re.fullmatch(re.escape(token.markup[0]) + "{%d,}" % len(token.markup), last))
        stop = end - 1 if closing else end
        if include_gaps and previous_end is not None and previous_end < offsets[first]:
            spans.append([previous_end, offsets[first]])
        if first + 1 <= stop:
            spans.append([offsets[first + 1], offsets[stop]])
        previous_end = offsets[end]
    return spans

VERSION = "body-split-v5"

_WRAPPER_OPEN = re.compile(
    r"^\s*(?:sure|certainly|absolutely|of course|got it|understood|okay|ok|done|here(?:'s| is| are)|below is|now (?:i|let me|that)\b|let me\b|"
    r"i(?:'ve|'ll| have| will| left| removed| omitted| swapped| updated| revised| drafted| prepared)|"
    r"as requested|per your|this (?:is|version|draft)|the (?:updated|revised|following)|"
    r"\*\*?(?:updated|revised|here))", re.I)
_WRAPPER_CLOSE = re.compile(
    r"^\s*(?:let me know|would you like|want me|do you want|if you(?:'d| would) like|if you need|"
    r"i hope|hope (?:this|that)|feel free|happy to|shall i|should i|need anything|"
    r"i(?:'ve|'ll| have| will| left| removed| omitted| kept| excluded| did not| didn't)|"
    r"\*?\(?note\b|\*note|note:|n\.b\.|this (?:draft|version|summary|report|message|brief|issue) (?:omits|excludes|leaves|keeps|does not|doesn't)|"
    r"as requested|per your|the (?:above|draft|message|summary|report) (?:omits|excludes|leaves|is ready|now)|"
    r"(?:a |a couple of |a few |some |two |three )?(?:quick |brief |additional )?(?:notes?|heads-up|caveats?|things to note|points to note)\b[^\n]{0,80}:\s*$|"
    r"(?:changes|edits|adjustments|omissions|redactions) (?:made|applied)\b|notes on (?:what|the)|what i (?:included|changed|left out|removed|omitted)|"
    r"now i |let me\b|wait\b|hmm\b|so (?:the user|per)|i need to|i(?:'m| am) (?:not|going|leaving|omitting|excluding))", re.I)
_OPENER = re.compile(r"^\s*(?:[^\n:]{0,80}?,\s*)?(?:here(?:'s| is| are)|below is|(?:my|the|your) (?:draft|summary|message|reply|issue|brief|report)\b|drafted|updated (?:draft|version)|revised (?:draft|version))[^\n]{0,120}:\s*$", re.I | re.M)
_RULE = re.compile(r"^\s*(?:---+|\*{3,}|___+)\s*$")
_SALUTATION = re.compile(r"^\s*(?:\*\*)?(?:to|from|subject|cc|re|dear|hi|hello|hey|good (?:morning|afternoon|evening))\b|^\s*[A-Z][A-Za-z.'-]{1,30}(?: [A-Z][A-Za-z.'-]{1,30})?,\s*$")

def _is_salutation(para):
    return _SALUTATION.match(para.strip()) is not None and len(para.strip()) <= 80
_STRUCT_START = [
    r"\n(?:---+|\*{3,}|___+)[ \t]*\n",
    r"\n(?=#{1,6} )",
    r"\n(?=```)",
    r"\n(?=> )",
    r"\n(?=\*\*[^*\n]{2,80}\*\*[ \t]*\n)",
    r"\n(?=(?:subject|to|from|title|re)\s*:)",
    r"\n(?=(?:dear|hi|hello|hey|good (?:morning|afternoon|evening))\b[^\n]{0,60}[,:!—-]?[ \t]*\n)",
]
_STARTS_BODY = re.compile(r"^\s*(?:#{1,6} |```|(?:---+|\*{3,}|___+)\s*$|\*\*[^*\n]{2,80}\*\*[ \t]*$|>|(?:subject|to|from|title|re)\s*:|"
                          r"(?:dear|hi|hello|hey)\b)", re.I)
_NARRATION = re.compile(
    r"\b(?:I(?:'ll|'ve|'m| will| have| am| left| removed| excluded| omitted| kept| listed| stripped| swapped| dropped| did not| didn't| used| replaced| included| drafted| prepared| put)|"
    r"the user|per your|your (?:instruction|request|note|calendar|list)|you (?:asked|said|mentioned|originally|want|requested)|as you (?:asked|requested)|"
    r"let me know|would you like|want me|should I|shall I|here(?:'s| is| are)|below is|updated (?:draft|version|message)|revised (?:draft|version|message)|"
    r"I (?:described|noted|mentioned|avoided|chose|decided|added|summari[sz]ed|referenced|pulled|wrote|framed|phrased|only|also|did)\b)", re.I)

def _wrapper_like(para):
    stripped = para.strip()
    if not stripped or stripped.startswith(("```", "#", ">", "|")):
        return False
    if _WRAPPER_OPEN.match(stripped) or _WRAPPER_CLOSE.match(stripped):
        return True
    lines = [l.strip() for l in stripped.splitlines() if l.strip()]
    hits = sum(bool(_NARRATION.search(l)) for l in lines)
    return hits >= max(1, (len(lines) + 1) // 2)

_SIGNOFF = re.compile(r"^\s*(?:best|regards|kind regards|best regards|thanks|thank you|cheers|sincerely|warm regards|"
                      r"talk soon|see you|—|--)\b", re.I)

def _paragraphs(text):
    out, pos = [], 0
    for chunk in re.split(r"((?:\r?\n|\r(?!\n))[ \t]*(?:\r?\n|\r(?!\n))+)", text):
        if chunk.strip():
            out.append((pos, pos + len(chunk)))
        pos += len(chunk)
    return out

def _code_deliverable(text):
    tokens = [t for t in MarkdownIt("commonmark").parse(text) if t.type == "fence"]
    if not tokens:
        return False
    lines = text.splitlines(keepends=True)
    outside = ("".join(lines[:tokens[0].map[0]]), "".join(lines[tokens[-1].map[1]:]))
    return all(_wrapper_like(part[a:b]) or _RULE.fullmatch(part[a:b].strip())
               for part in outside for a, b in _paragraphs(part))

def prose_body_start(text):
    if _STARTS_BODY.match(text.lstrip()):
        return 0
    paras = _paragraphs(text)
    if not paras:
        return 0
    first = text[paras[0][0]:paras[0][1]]
    if all(_wrapper_like(text[a:b]) for a, b in paras) and len(text) <= 600:
        return len(text)
    if not (_WRAPPER_OPEN.match(first) or _wrapper_like(first)):
        return 0
    candidates = []
    for pat in _STRUCT_START:
        m = re.search(pat, text)
        if m:
            candidates.append(m.end() if pat.startswith(r"\n---") else m.start() + 1)
    short = len(first) <= 400 and (first.rstrip().endswith(":") or len(re.findall(r"[.!?](?:\s|$)", first)) <= 2)
    if (_WRAPPER_OPEN.match(first) and short) or _wrapper_like(first):
        index = 1
        while index < len(paras) and _wrapper_like(text[paras[index][0]:paras[index][1]]):
            index += 1
        if index < len(paras):
            candidates.append(paras[index][0])
    openers = [k for k, (a, b) in enumerate(paras) if _OPENER.match(text[a:b].strip())]
    openers = [k for k in openers if not any(_is_salutation(text[paras[j][0]:paras[j][1]]) for j in range(max(0, k - 2), k))]
    openers = [k for k in openers if k + 1 < len(paras) and (paras[k][0] <= 0.75 * len(text)
               or _STARTS_BODY.match(text[paras[k + 1][0]:paras[k + 1][1]].strip()))]
    explicit = bool(openers and (_wrapper_like(first) or _WRAPPER_OPEN.match(first)))
    followed_by_structure = explicit and bool(_STARTS_BODY.match(text[paras[openers[-1] + 1][0]:paras[openers[-1] + 1][1]].strip()))
    if explicit:
        candidates = [paras[openers[-1] + 1][0]]
    if not candidates:
        return 0
    start = min(candidates)
    if followed_by_structure:
        return start
    return start if start <= (0.75 if explicit else 0.6) * len(text) else 0

def prose_body_end(text, start):
    paras = [p for p in _paragraphs(text) if p[0] >= start]
    if len(paras) < 2:
        return len(text)
    end = len(text)
    index = len(paras) - 1
    while index >= 1:
        a, b = paras[index]
        para = text[a:b]
        stripped = para.strip()
        is_rule = _RULE.match(stripped) is not None
        closer = (not _SIGNOFF.match(para)) and (_wrapper_like(para) or (stripped.endswith("?") and len(stripped) <= 300))
        heading_before = index >= 1 and text[paras[index - 1][0]:paras[index - 1][1]].strip().endswith(":") \
            and _wrapper_like(text[paras[index - 1][0]:paras[index - 1][1]]) and stripped.startswith(("-", "*", "1.", "•")) \
            and not (index >= 2 and _is_salutation(text[paras[index - 2][0]:paras[index - 2][1]]))
        if index >= 1 and _is_salutation(text[paras[index - 1][0]:paras[index - 1][1]]):
            break
        if is_rule or closer or heading_before:
            end = a
            index -= 1
            continue
        break
    if end - start < max(20, 0.2 * len(text)):
        end = len(text)
    if end < len(text):
        prev = [p for p in paras if p[1] <= end]
        if prev and _RULE.match(text[prev[-1][0]:prev[-1][1]].strip()):
            end = prev[-1][0]
    return end if end > start else len(text)

def split(text):
    if not isinstance(text, str) or not text.strip():
        return {"version": VERSION, "mode": "empty", "spans": [], "preface": [0, 0], "afterword": [0, 0]}
    fences = fenced_spans(text)
    if fences and _code_deliverable(text):
        return {"version": VERSION, "mode": "fenced", "spans": fenced_spans(text, include_gaps=True),
                "preface": [0, fences[0][0]], "afterword": [fences[-1][1], len(text)]}
    start = prose_body_start(text)
    if start >= len(text):
        return {"version": VERSION, "mode": "prose", "spans": [], "preface": [0, len(text)], "afterword": [len(text), len(text)]}
    end = prose_body_end(text, start)
    assert 0 <= start <= end <= len(text)
    return {"version": VERSION, "mode": "prose", "spans": [[start, end]], "preface": [0, start], "afterword": [end, len(text)]}

def locate(text, quote, spans, *, start=0):
    if not quote or not quote.strip():
        return "missing"
    i = text.find(quote, start)
    if i < 0:
        squeezed = " ".join(quote.split())
        m = re.search(r"\s+".join(re.escape(w) for w in squeezed.split()), text[start:])
        if not m:
            return "missing"
        i, j = start + m.start(), start + m.end()
    else:
        j = i + len(quote)
    for a, b in spans:
        if i < b and j > a:
            return "body"
    if not spans or j <= spans[0][0]:
        return "preface"
    return "afterword"
