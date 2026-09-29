from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_P = json.loads((_HERE / "prompts.json").read_text(encoding="utf-8"))
_NOUNS = json.loads((_HERE / "artifact_nouns.json").read_text(encoding="utf-8"))

SYSTEM_DEFENSES = ("restrictive", "recipient_typed", "deliverable_only")

def defense_text(name: str, task_id: str = None) -> str:
    if name not in SYSTEM_DEFENSES:
        raise ValueError("unknown defense %r; available: %s" % (name, list(SYSTEM_DEFENSES)))
    d = _P[name]
    if name == "deliverable_only":
        if not task_id:
            raise ValueError("deliverable_only text is per task; task_id is required")
        noun = _NOUNS.get(task_id)
        if not noun:
            raise KeyError("no artifact noun recorded for %s" % task_id)
        return d["template"].format(artifact=noun)
    return d["text"]

def apply(messages: list, name: str, task_id: str = None) -> list:
    if any(m.get("role") == "system" for m in messages):
        raise ValueError("a system message is already present; do not stack defense prompts")
    return [{"role": "system", "content": defense_text(name, task_id)}] + list(messages)
