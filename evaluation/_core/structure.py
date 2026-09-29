from __future__ import annotations

import re
from markdown_it import MarkdownIt

from .alias_match import alias_in, registered_items
from . import tombstone as TS


def _excluded_heading(text):
    label = r"(?:omitted|excluded|removed|redacted|withheld)"
    return (not re.search(r"\b(?:no|not|without)\b", text, re.I)
            and bool(re.fullmatch(label + r"(?:\s+[\w-]+){0,3}|(?:[\w-]+\s+){0,3}" + label, text, re.I)))


def _layout(text, offset=0, heading="", persistent=False, inherited_excluded=False):
    positions, groups, contexts, blocks, stack = {}, {}, {}, {}, []
    sections, containers = [], []
    parsed = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parsed = re.sub(r"(?m)^(\s*)•\s", r"\1- ", parsed)
    tokens = MarkdownIt("commonmark").parse(parsed)
    for i, token in enumerate(tokens):
        if token.type in ("blockquote_open", "list_item_open"):
            containers.append((heading, persistent, sections, inherited_excluded))
            inherited_excluded = (inherited_excluded or _excluded_heading(heading)
                                  or any(_excluded_heading(value) for n, value in sections))
            sections = []
        elif token.type in ("blockquote_close", "list_item_close"):
            heading, persistent, sections, inherited_excluded = containers.pop()
            if token.type == "blockquote_close" and not stack and not persistent:
                heading = ""
            continue
        if token.type == "inline":
            label = token.content.strip(" *#_:\n")
            styled = re.fullmatch(r"\s*(?:\*\*[^*\n]+\*\*|__[^_\n]+__)\s*", token.content)
            is_heading = bool(i and tokens[i - 1].type == "heading_open")
            if is_heading:
                level = int(tokens[i - 1].tag[1:])
                sections = [(n, value) for n, value in sections if n < level]
                sections.append((level, label))
            if (is_heading or token.content.rstrip(" *_").endswith(":")
                    or (styled and not re.search(r"[.!?]$", label))):
                heading, persistent = label, is_heading
        if token.type in ("inline", "html_block") and token.map:
            for line in range(*token.map):
                contexts[offset + line] = (inherited_excluded or _excluded_heading(heading)
                                            or any(_excluded_heading(value) for n, value in sections))
                blocks[offset + line] = (offset + token.map[0], offset + token.map[1])
        if token.type in ("bullet_list_open", "ordered_list_open"):
            identity = offset + token.map[0], len(stack)
            groups[identity] = {"heading": heading, "parent": stack[-1][0] if stack else None,
                                "excluded": inherited_excluded or _excluded_heading(heading)
                                or any(_excluded_heading(value) for n, value in sections)}
            stack.append([identity, 0])
        elif token.type in ("bullet_list_close", "ordered_list_close"):
            stack.pop()
            if not stack and not persistent:
                heading = ""
        elif token.type == "list_item_open":
            stack[-1][1] += 1
            for line in range(*token.map):
                positions[offset + line] = (offset + token.map[0], stack[-1][1], stack[-1][0])
        elif token.type in ("fence", "code_block"):
            inner, nested, labels, paragraphs = _layout(
                token.content, offset + token.map[0] + (token.type == "fence"), heading, persistent,
                inherited_excluded or any(_excluded_heading(value) for n, value in sections))
            positions.update(inner)
            groups.update(nested)
            contexts.update(labels)
            blocks.update(paragraphs)
            if not stack and not persistent:
                heading = ""
        elif token.type == "paragraph_close" and not stack and not persistent:
            previous = tokens[i - 1] if i else None
            if previous and previous.type == "inline" and previous.content.strip(" *#_:\n") != heading:
                heading = ""
    return positions, groups, contexts, blocks


def rows(text):
    positions, groups, contexts, blocks = _layout(text)
    lines, out, index = re.split(r"(\r\n|\r|\n)", text or ""), [], 0
    count = (len(lines) + 1) // 2
    while index < count:
        item, block = positions.get(index), blocks.get(index)
        end = index + 1
        if block is not None:
            while end < count and blocks.get(end) == block and positions.get(end) == item:
                end += 1
        value = "".join(lines[2 * index:2 * end - 1])
        out.append({"text": value, "position": item[1] if item else None,
                    "list": item[2] if item else None,
                    "excluded": contexts.get(index, False)})
        index = end
    return out, groups


def _introduced_spans(text, sentences, aliases, family, rule):
    spans = []
    records = None
    introduction = re.compile(r"\b(?:findings?|results?|facts?|items?|statements?|passages?|paragraphs?|text|details?|content|entries|entry|removed|omitted|excluded|redacted|withheld|left out)(?:\s*:\s*|\s+)", re.I)
    for start, stop, trace in sentences:
        if not trace:
            continue
        for intro in introduction.finditer(text, start, stop):
            payload = intro.end()
            if payload == len(text) or not TS.revision_statement(text[start:payload], rule):
                continue
            closing = {'"': '"', "'": "'", "“": "”", "‘": "’", "`": "`"}.get(text[payload])
            body = payload + bool(closing)
            end = None
            if records is None:
                records = registered_items(aliases, family)
            for item in records:
                pattern = r"\s+".join(re.escape(word) for word in item["canonical"].split())
                match = re.compile(pattern, re.I).match(text, body)
                if match and (match.end() == len(text) or not text[match.end()].isalnum()):
                    candidate = match.end()
                    if closing:
                        while candidate < len(text) and text[candidate].isspace():
                            candidate += 1
                        if candidate == len(text) or text[candidate] != closing:
                            continue
                        candidate += 1
                    end = max(end or 0, candidate)
            if end is None and closing:
                for index in range(body, len(text)):
                    if text[index] != closing or text[index - 1:index] == "\\":
                        continue
                    if closing in ("'", "’") and text[index - 1:index].isalnum() and text[index + 1:index + 2].isalnum():
                        continue
                    end = index + 1
                    break
            if end is not None:
                if text[end:end + 1] in (".", "!", "?"):
                    end += 1
                spans.append((start, end))
                break
    return spans


def classify(text, aliases=None, family=None, *, excluded=False, rule="v5"):
    first = re.split(r"\r\n|\r|\n", text, maxsplit=1)[0].strip(" *#_:\r\n")
    label = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", re.sub(r"[*`_]", "", first)).split(":", 1)[0]
    if excluded or rule == "v5" and (_excluded_heading(first) or _excluded_heading(label)):
        return {"content": False, "spans": [text] if text.strip() else []}
    flat = re.sub(r"[\r\n]", " ", text)
    sentences, cursor = [], 0
    for sentence in TS._sentences(text):
        sentence = sentence.strip()
        start = text.find(sentence, cursor)
        if start < 0:
            continue
        cursor = start + len(sentence)
        trace = TS.revision_statement(sentence, rule) or rule == "v5" and TS.omission_labelled(sentence)
        sentences.append((start, cursor, trace))
    introduced = _introduced_spans(text, sentences, aliases, family, rule)
    sentences = [(a, b, trace or any(start <= a and b <= end for start, end in introduced))
                 for a, b, trace in sentences]
    intervals = introduced + [(a, b) for a, b, trace in sentences if trace]
    content = False
    left = 0
    for right in range(len(sentences)):
        end = sentences[right][1]
        if not alias_in(flat[sentences[left][0]:end], aliases, family):
            continue
        while left < right and alias_in(flat[sentences[left + 1][0]:end], aliases, family):
            left += 1
        start = sentences[left][0]
        if any(trace for a, b, trace in sentences[left:right + 1]):
            intervals.append((start, end))
        else:
            content = True
        left = right + 1
    merged = []
    for a, b in sorted(set(intervals)):
        if merged and a < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return {"content": content, "spans": [text[a:b] for a, b in merged]}
