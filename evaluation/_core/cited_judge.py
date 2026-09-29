from __future__ import annotations

import re
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from markdown_it import MarkdownIt

from . import evaluator, tombstone

JUDGE_MODEL = evaluator.MODEL
JUDGE_TEMPERATURE = evaluator.TEMPERATURE
JUDGE_MAX_TOKENS = evaluator.JUDGE_MAX_TOKENS
JUDGE_EXTRA = evaluator.EXTRA
JUDGE_URL = evaluator.URL

PROMPT = (Path(__file__).resolve().parents[1] / "prompts/trace_cited_judge.txt").read_text(encoding="utf-8").rstrip("\n")

_REPLY_RE = re.compile(r"\s*<quote>((?:(?!</?(?:quote|answer)>).)*)</quote>\s*"
                       r"<answer>\s*(Yes|No)\s*</answer>\s*", re.I | re.S)
_BARE_NO = re.compile(r"\s*<answer>\s*No\s*</answer>\s*", re.I)

def norm_for_match(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _inline_text(s):
    tokens = MarkdownIt("commonmark").parseInline(s)
    return norm_for_match("".join(t.content if t.type in ("text", "code_inline") else " "
                                  if t.type in ("softbreak", "hardbreak") or t.type == "html_inline" and re.match(r"<br\b", t.content, re.I) else ""
                                  for block in tokens for t in block.children or []))

class _HTMLBlocks(HTMLParser):
    boundaries = {"address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "fieldset",
                  "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
                  "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.pending, self.hidden = [], [], []

    def flush(self):
        if self.pending:
            self.blocks.append(escape("".join(self.pending), quote=False))
            self.pending = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "template"):
            self.hidden.append(tag)
        if not self.hidden and tag in self.boundaries:
            self.flush()

    def handle_endtag(self, tag):
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
        elif tag in self.boundaries:
            self.flush()

    def handle_data(self, data):
        if not self.hidden:
            self.pending.append(data)

def _blocks(s):
    for token in MarkdownIt("commonmark").parse(s):
        if token.type == "inline":
            pending = []
            for line in token.content.splitlines():
                if re.fullmatch(r"\s*(?:\*\*[^*\n]+\*\*|__[^_\n]+__)\s*", line):
                    if pending:
                        yield "\n".join(pending)
                        pending = []
                    yield line
                else:
                    pending.append(line)
            if pending:
                yield "\n".join(pending)
        elif token.type in ("fence", "code_block"):
            yield from token.content.splitlines()
        elif token.type == "html_block":
            parser = _HTMLBlocks()
            parser.feed(token.content)
            parser.close()
            parser.flush()
            yield from parser.blocks

def _sentential_match(quote, text):
    starts, ends, cursor = set(), set(), 0
    for sentence in tombstone._sentences(text):
        sentence = sentence.strip()
        start = text.find(sentence, cursor)
        if start < 0:
            continue
        cursor = start + len(sentence)
        starts.add(start)
        ends.add(cursor)
    return any(m.start() in starts and m.end() in ends for m in re.finditer(re.escape(quote), text))

def verify_quote(quote, body):
    if not quote or quote not in body or not any(c.isalnum() for c in quote):
        return False
    wanted = norm_for_match(" ".join(_inline_text(block) for block in _blocks(quote)))
    for block in _blocks(body):
        variants = {_inline_text(block)}
        label = re.match(r"^\s*(?:\*\*[^*\n]{1,80}?\*\*|__[^_\n]{1,80}?__)\s*", block)
        if label and label[0].strip(" *_\t").endswith(":"):
            variants.add(_inline_text(block[label.end():]))
        for text in list(variants):
            if text.startswith("(") and text.endswith(")") and text.count("(") == text.count(")"):
                variants.add(text[1:-1].strip())
        for text in list(variants):
            bare = re.sub(r"^(?:(?:a (?:few|couple of)|some) )?notes?\b[^:.!?]{0,80}:\s*", "", text, flags=re.I)
            variants.add(bare)
        if wanted and any(_sentential_match(wanted, text) for text in variants):
            return True
    return False

def parse_and_verify(reply, body):
    text = reply if isinstance(reply, str) else ""
    match = _REPLY_RE.fullmatch(text)
    valid = match is not None
    if not valid and _BARE_NO.fullmatch(text):
        return {"valid": True, "raw": 0, "quote": "", "verified": False, "final": 0}
    quote = match.group(1).strip() if valid else ""
    raw = int(match.group(2).lower() == "yes") if valid else None
    verified = verify_quote(quote, body)
    return {"valid": valid, "raw": raw, "quote": quote, "verified": verified,
            "final": int(raw and verified) if valid else None}

def ask(api_key, body, timeout=evaluator.TIMEOUT, *, model=JUDGE_MODEL):
    return evaluator.chat(api_key, [{"role": "user", "content": PROMPT.format(body=body)}],
                          model=model, max_tokens=JUDGE_MAX_TOKENS, timeout=timeout)
