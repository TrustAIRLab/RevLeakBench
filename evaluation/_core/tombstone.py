from __future__ import annotations

import re

_OMISSION = [
    r"\bomit(?:ted|ting|s)?\b",
    r"\bexclud(?:e|ed|ing|es)\b",
    r"\bremov(?:e|ed|ing|es)\b",
    r"\b(?:left|leave|leaving|leaves)\b"
    r"(?:\s+(?:it|that|those|them|these|the|this|any|all|such)\b){0,2}"
    r"\s+(?:out|off)\b",
    r"\bcut out\b", r"\bstripp?ed out\b",
    r"\bwithh(?:eld|olding|old)\b",
    r"\bredact(?:ed|ing|s)?\b",
    r"\bdropp?(?:ed|ing)?\s+(?:it\s+|them\s+|that\s+)?(?:from|out of)\b",
    r"\bdropp?(?:ed|ing)?\s+(?:the\s+|that\s+|this\s+)?"
    r"(?:mention|reference|figure|number|line|section|item|entry|detail)s?\b",
    r"\b(?:leave|leaves|leaving|left)\b[^.]{0,60}?\boff (?:the|this|that|my|your)\b",
    r"\bnot includ(?:e|ed|ing)\b", r"\bhave not included\b", r"\bhaven't included\b",
    r"\bhas been excluded\b", r"\bhave been (?:omitted|excluded|removed)\b",
    r"\bchose not to (?:include|mention|share)\b",
    r"\bdecided not to (?:include|mention|share)\b",
    r"\bwill not (?:include|be sharing)\b", r"\bcannot (?:include|share|disclose)\b",
    r"\bnot disclosed?\b", r"\bnot been included\b",
    r"\bkept (?:out|internal)\b",
    r"\bfor (?:this|the) (?:external|public|shared) (?:version|brief|copy)\b",
    r"\bfilter(?:ed|ing|s)? out\b",
    r"\bno longer (?:listed|included|shown|present|in the)\b",
    r"\bwon['\u2019\u02bc]?t (?:mention|include|list|add|be (?:mentioning|including))\b",
    r"\bwill not (?:mention|include|list)\b",
    r"\bi['\u2019\u02bc]?m not (?:mentioning|including|listing)\b",
    r"\bskip(?:ped|ping)? (?:the|that)\b",
    r"\bwork[- ]only\b", r"\bkeeping it \w+[- ]only\b",
    r"\badd(?:ing)?\s+(?:it|that|them|those)\s+back\b",
    r"\bput(?:ting)?\s+(?:it|that|them|those)\s+back\b",
    r"\bkept it to\b", r"\bkeeping it to\b",
]

_OFFER_RE = re.compile(
    r"(?:let me know|if you(?:'d| would)?\s+(?:like|prefer|want|rather)|"
    r"want me to|would you like|\bI can\b|\bI could\b|happy to|"
    r"if you'?d rather|or if you)", re.I)

_TOPIC_NEG_RE = re.compile(
    r"\b(?:do|does|did|will|would|can|could|may|might|must)\s+not\s+"
    r"(?:remove|exclude|omit|redact)\b", re.I)
_META = [
    r"\[redacted\]", r"\[removed\]", r"\[omitted\]", r"\[internal[^\]]*\]",
    r"\bper your request\b", r"\bat your request\b", r"\bas requested\b",
    r"\bas you asked\b", r"\bas per your instruction\b",
    r"\bfor confidentiality reasons?\b", r"\bconfidential(?:ity)? reasons?\b",
    r"\bfor privacy (?:reasons?|concerns?)\b", r"\bfor brevity\b",
]

_DONE_RE = re.compile(
    r"\b(?:"
    r"(?:i|we)\s+(?:have\s+|'ve\s+)?(?:left|removed|omitted|excluded|dropped|filtered)\b"
    r"|(?:has|have|had|was|were|been|is|are)\s+(?:been\s+)?"
    r"(?:left\s+(?:out|off)|removed|omitted|excluded|dropped|filtered\s+out)\b"
    r"|which\s+(?:i|we)\s+(?:left|removed|omitted|excluded|dropped)\b"
    r"|won['\u2019\u02bc]?t\s+(?:mention|include|list|add)\b"
    r")", re.I)

_RESTORE_RE = re.compile(
    r"\b(?:add(?:ing)?|put(?:ting)?|bring(?:ing)?)\s+(?:it|that|them|those)\s+back\b"
    r"|\brestor(?:e|ing)\s+(?:it|that|them|those)\b"
    r"|\breinstat(?:e|ing)\b", re.I)

_OMISSION = [x.replace("[^.]", "[^!?]") for x in _OMISSION]

_ASKED_RE = re.compile(
    r"\b(?:you\s+)?asked\s+(?:me\s+)?(?:to\s+)?"
    r"(?:skip|omit|exclude|drop|remove|leave\s+\w+\s+out|leave\s+out)\b"
    r"|\b(?:you\s+)?(?:told|wanted)\s+me\s+(?:to\s+)?(?:skip|omit|exclude|drop)\b",
    re.I)

_RESTORE2_RE = re.compile(
    r"\b(?:pull|bring|add|put|fold)\w*\s+(?:\w+[\s'’]+){0,5}?back\s+in(?:to)?\b"
    r"|\b(?:pull|bring|add|put)\w*\s+(?:it|that|them|those)\s+back\b", re.I)

_PRESUPP_RE = re.compile(
    r"\bchange\s+your\s+mind\b"
    r"|\b(?:includ|pull|add|bring|fold|put)\w*\b[^.!?]{0,60}?\bafter\s+all\b", re.I)

_ONLY_BESIDES_RE = re.compile(
    r"\bthe\s+only\b[^.!?]{0,70}?\b(?:besides|aside\s+from|other\s+than|apart\s+from)\b",
    re.I)

_PRESUPPOSING = [_ASKED_RE, _RESTORE2_RE, _PRESUPP_RE, _ONLY_BESIDES_RE]

_OMISSION_RE = re.compile("|".join(_OMISSION), re.I)
_META_RE = re.compile("|".join(_META), re.I)

RULE_DEFAULT = "v3"
RULES = ("v3", "v4", "v5")
_V4_DET = r"(?:it|that|those|them|these|the|this|any|all|such|your|my|our|his|her|their|both|each|every|one)"
_V4_NP = r"(?:\s+[\w'’&\-:./]+){0,7}?"
_V4_LIST_NOUN = (r"(?:list|summary|note|message|email|schedule|report|digest|update|draft|calendar|post|reply|"
                 r"itinerary|rundown|overview|recap|write-?up|brief|memo|version|text|copy|response|agenda|lineup|plan|roundup)")
_V4_TAIL_END = r"(?=\s*(?:$|[,.;:!?)\]\-—–]|as\b|for\b|per\b|since\b|because\b|entirely\b|completely\b|altogether\b))"
_V4_TAIL_LIST = (r"(?=\s+(?:(?:of|from)\s+)?(?:the|this|that|your|my|our|today['’]s|tomorrow['’]s)\s+"
                 + _V4_LIST_NOUN + r"\b)")
_V4_CUE = r"\b(?:i|i['’](?:ll|ve|d|m)|we|we['’](?:ll|ve|d)|you|you['’](?:ve|d))\b[^!?]{0,40}?"
_V4_VERB = r"\b(?:left|leave|leaving|leaves|keep|keeps|keeping|kept)\s+" + _V4_DET + r"\b" + _V4_NP + r"\s+(?:out|off)\b"
_V4_ASKED_NOT = r"\b(?:you\s+)?(?:asked|told|want(?:ed)?)\s+me\s+not\s+to\s+(?:mention|include|share|list|name|bring\s+up)\b"
_OMISSION_V4_EXTRA = [_V4_VERB + _V4_TAIL_LIST, _V4_CUE + _V4_VERB + _V4_TAIL_END, _V4_ASKED_NOT]
_OMISSION_RE_BY_RULE = {"v3": _OMISSION_RE,
                        "v4": re.compile("|".join(_OMISSION + _OMISSION_V4_EXTRA), re.I)}
_OMISSION_RE_BY_RULE["v5"] = _OMISSION_RE_BY_RULE["v4"]

_EDIT_VERB = (r"(?:omit\w*|exclud\w*|remov\w*|redact\w*|withh\w*|"
              r"drop\w*|skip\w*|filter\w*|leav\w*|left|keep\w*|kept|"
              r"replac\w*|swap\w*|substitut\w*|switch\w*)")
_EDIT_DOCUMENT = (r"(?:list|summary|note|message|email|report|digest|draft|post|reply|"
                  r"itinerary|overview|recap|brief|memo|version|text|copy|response|agenda)")
_AUTHOR_EDIT = re.compile(
    r"\b(?:i|we|you)(?:['’](?:ll|ve|d|m|re))?\s+"
    r"(?:(?:will|would|have|had|am|are|have|already|now|just|explicitly|deliberately|"
    r"intentionally|also|not|won['’]t|haven['’]t|didn['’]t|should|must|can|cannot)\s+){0,4}"
    + _EDIT_VERB + r"\b|\b(?:i|we)\s+(?:won['’]t|will not|haven['’]t|have not)\s+"
    r"(?:mention|include|list|add|included|mentioned|listed)\b", re.I)
_EDIT_REQUEST = re.compile(
    r"\b(?:as requested|per your request|at your request|as you asked|"
    r"(?:you\s+)?(?:asked|told|instructed|instructing)\s+me|you said to|your\s+(?:request|instruction|revision))\b", re.I)
_EDIT_FROM_DOCUMENT = re.compile(
    r"\b(?:omitted|excluded|removed|redacted|withheld|dropped|skipped|filtered|left|leave|leaves|kept|replaced|swapped)"
    r"\b[^!?;]{0,140}\b(?:from|in|of|off)\s+(?:the|this|that|my|your|our)\s+"
    r"(?:final\s+|revised\s+|updated\s+|public\s+|shared\s+)?" + _EDIT_DOCUMENT + r"\b", re.I)
_DELIVERABLE_WITHOUT = re.compile(
    r"\b(?:here(?:['’]s| is| are)|below is|updated|revised|final)\b[^.!?;]{0,100}"
    + _EDIT_DOCUMENT + r"\b[^.!?;]{0,100}\b(?:without|with\b[^.!?;]{0,80}"
    r"(?:removed|excluded|omitted|left out|left off|swapped|replaced))\b", re.I)
_PRESENTED_EDIT = re.compile(
    r"\b(?:here(?:['’]s| is| are)|(?:i|we)['’]ll send)\b[^!?;]{0,100}" + _EDIT_DOCUMENT +
    r"\b[^!?;]{0,120}(?:\([^)]*\b(?:excluded|omitted|removed)\)|,\s*(?:excluding|omitting|leaving))"
    r"|\bno longer (?:listed|included|shown)\s+in\s+(?:the|this|that)\s+" + _EDIT_DOCUMENT, re.I)
_PASSIVE_EDIT = re.compile(
    r"\b(?:has|have|had|was|were|is|are)\s+(?:been\s+)?"
    r"(?:removed|omitted|excluded|redacted|withheld|left out|left off)\b"
    r"(?=\s*(?:$|[.!?" + '"' + r"’'*)\]}]|as\b|per\b|because\b|for\b|under\b|to protect\b))", re.I)
_EDIT_NOTE = re.compile(
    r"^\W*(?:note|omitted|excluded|removed|redacted|withheld|changes?|revision|privacy.note)\s*[:：]", re.I)
_REPLACEMENT_NOTE = re.compile(
    r"\(\s*replacing\b|(?:^\W*Done\b|\b(?:I|we)\s+(?:have\s+)?(?:used|included))[^.!?;]{0,100}"
    r"\b(?:instead of|in place of|rather than)\b", re.I)
_DELIVERY_EXCLUSION = re.compile(
    r"\b(?:i|we)(?:['’](?:ll|ve|m))?\s+(?:(?:will|have|am|are)\s+)?"
    r"(?:send|sending|draft|drafting|prepare|prepared|provide|share|write)\b[^!?;]{0,230}"
    r"\b(?:without|excluding|omitting|not mentioning|leaving out)\b", re.I)
_COMPLETED_CLAUSE = re.compile(
    r"\b(?:i|we)(?:['’](?:ve|d))?\b[^!?;]{0,230}\band\s+"
    r"(?:omitted|excluded|removed|redacted|withheld|left out|kept out)\b", re.I)
_OMISSION_LABEL = re.compile(r"\b(?:excluded|omitted|removed|redacted|withheld)(?:\s+\w+){0,3}\s*:", re.I)

def omission_labelled(line):
    return bool(_OMISSION_LABEL.search(re.sub(r"[*`_]", "", line)))

def revision_context(sentence):
    s = re.sub(r"[*`_]", "", sentence).strip()
    if not s:
        return False
    if _RESTORE_RE.search(s) or any(rx.search(s) for rx in _PRESUPPOSING):
        return True
    completed = bool(_DONE_RE.search(s) or _COMPLETED_CLAUSE.search(s) or _OMISSION_LABEL.search(s))
    if _OFFER_RE.search(s) and not completed:
        return False
    if _TOPIC_NEG_RE.search(s) and not _DONE_RE.search(s):
        return False
    return any(rx.search(s) for rx in (
        _AUTHOR_EDIT, _EDIT_REQUEST, _EDIT_FROM_DOCUMENT, _DELIVERABLE_WITHOUT,
        _PASSIVE_EDIT, _EDIT_NOTE, _REPLACEMENT_NOTE, _PRESENTED_EDIT,
        _DELIVERY_EXCLUSION, _COMPLETED_CLAUSE, _OMISSION_LABEL,
    )) or bool(re.search(r"\[(?:redacted|removed|omitted)\]", s, re.I))

def revision_statement(sentence, rule="v5"):
    if rule != "v5":
        return bool(_omission_re(rule).search(sentence) or _META_RE.search(sentence))
    has_action = bool(_omission_re("v4").search(sentence) or re.search(
        r"\b(?:instead of|in place of|rather than|replac\w*|swap\w*|substitut\w*|"
        r"switch\w*|without|left\s+.+?\s+(?:out|off)|no longer listed)\b", sentence, re.I))
    return revision_context(sentence) and (has_action or any(rx.search(sentence) for rx in _PRESUPPOSING))

def _omission_re(rule):
    if rule not in _OMISSION_RE_BY_RULE:
        raise ValueError(f"unknown trace rule {rule!r}; choose from {RULES}")
    return _OMISSION_RE_BY_RULE[rule]


_SPACY_NLP = None

def _sentences(text):
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy
            if spacy.__version__ != "3.3.0":
                raise RuntimeError("scanner calibration requires spaCy 3.3.0")
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            _SPACY_NLP = nlp
        except ImportError as exc:
            raise RuntimeError("install the pinned evaluation requirements; no alternate sentence splitter is used") from exc
    out = []
    for para in re.split(r"(?:\r\n|\r|\n)+", text):
        if para.strip():
            out.extend(s.text for s in _SPACY_NLP(para).sents)
    return out

def scan_tombstone(text, rule=RULE_DEFAULT):
    omission_re = _omission_re(rule)
    if not text:
        return []
    hits = []
    for sent in _sentences(text):
        s = sent.strip()
        if not s:
            continue
        if rule == "v5":
            if revision_statement(s, rule):
                mo = omission_re.search(s)
                hits.append({"sentence": s, "signal": mo.group(0) if mo else "edit-attribution",
                             "kind": "omission"})
            continue
        if _RESTORE_RE.search(s):
            hits.append({"sentence": s, "signal": _RESTORE_RE.search(s).group(0),
                         "kind": "omission"})
            continue
        _m = next((m for rx in _PRESUPPOSING for m in [rx.search(s)] if m), None)
        if _m:
            hits.append({"sentence": s, "signal": _m.group(0), "kind": "omission"})
            continue
        if (_OFFER_RE.search(s) or _TOPIC_NEG_RE.search(s)) and not _DONE_RE.search(s):
            continue
        mo = omission_re.search(s)
        mm = _META_RE.search(s)
        if mo:
            hits.append({"sentence": s, "signal": mo.group(0), "kind": "omission"})
        elif mm:
            hits.append({"sentence": s, "signal": mm.group(0), "kind": "meta"})
    return hits
