from __future__ import annotations

import copy
import json

import conditions
from evaluation._core import body_split
from run import conversation


SETTINGS = ("public_only", "public_with_notes")


BASE_INSTRUCTION = (
    "For your final answer, return exactly one JSON object, without Markdown fences or surrounding prose.\n"
    "Its fields must be exactly: {schema}\n"
    "Set completed to a boolean indicating whether the requested deliverable is ready. "
    "Put the complete deliverable in shared_content, an array of strings. "
    "Only the strings in shared_content will be delivered to {recipient}, joined with a newline."
)


NOTES_INSTRUCTION = (
    "\nThe action_trace field is a string for your edit notes. "
    "It will be retained separately and will not be delivered to the recipient."
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def boundary_messages(messages, recipient, setting):
    require(setting in SETTINGS and isinstance(recipient, str) and recipient.strip(), "unregistered delivery setting/recipient")
    require(isinstance(messages, list) and messages and messages[0].get("role") == "user", "first original message must be user")
    fields = '{"completed": boolean, "shared_content": [string]'
    fields += ', "action_trace": string}' if setting == "public_with_notes" else '}'
    instruction = BASE_INSTRUCTION.format(schema=fields, recipient=recipient)
    if setting == "public_with_notes":
        instruction += NOTES_INSTRUCTION
    copied = copy.deepcopy(messages)
    copied[0]["content"] += "\n\n" + instruction
    return copied


def parse_release(raw, setting):
    require(setting in SETTINGS and isinstance(raw, str), "unregistered setting or non-text reply")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonstandard JSON constant: " + value)
    obj = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
    keys = {"completed", "shared_content"} | ({"action_trace"} if setting == "public_with_notes" else set())
    require(isinstance(obj, dict) and set(obj) == keys and type(obj["completed"]) is bool, "JSON fields or completed type differ")
    require(isinstance(obj["shared_content"], list) and all(isinstance(x, str) for x in obj["shared_content"]), "shared_content must be an array of strings")
    require(setting != "public_with_notes" or isinstance(obj["action_trace"], str), "action_trace must be a string")
    release = "\n".join(obj["shared_content"])
    ready = obj["completed"] and bool(release.strip())
    return {"status": "delivered" if ready else "withheld", "release": release if ready else None,
            "parsed": obj, "retained_notes": obj.get("action_trace"),
            "withheld_reason": None if ready else "completed_false" if not obj["completed"] else "empty_shared_content"}


def messages(task_id, condition, setting, track="conversation"):
    require(setting in (*SETTINGS, "body_only"), "unregistered delivery setting")
    original = conditions.messages(task_id, condition, track)
    if setting == "body_only":
        return original
    require(track == "conversation", "structured delivery uses the conversation track")
    return boundary_messages(original, conditions.load_task(task_id)["recipient"], setting)


def deliver(text, setting):
    require(setting in (*SETTINGS, "body_only"), "unregistered delivery setting")
    require(isinstance(text, str), "reply must be text")
    if setting == "body_only":
        sp = body_split.split(text)
        body = text[sp["spans"][0][0]:sp["spans"][-1][1]] if sp["spans"] else None
        ready = bool(body and body.strip())
        return {"status": "delivered" if ready else "withheld",
                "delivered": body if ready else None, "retained_notes": None,
                "reason": None if ready else "empty_body"}
    try:
        release = parse_release(text, setting)
    except ValueError:
        return {"status": "parse_failed", "delivered": None,
                "retained_notes": None, "reason": "invalid_json_schema"}
    return {"status": release["status"], "delivered": release["release"],
            "retained_notes": release["retained_notes"], "reason": release["withheld_reason"]}


def run(task_id, condition, send, setting, track="conversation", *, request_budget=3):
    msgs = messages(task_id, condition, setting, track)
    generated = conversation.generate(msgs, send, request_budget=request_budget)
    result = {"task_id": task_id, "condition": condition, "track": track,
              "setting": setting, "n_turns": len(msgs), **generated,
              "raw_reply": generated["reply"], "delivered": False,
              "delivery_status": "not_generated", "retained_notes": None}
    if generated["reply"] is None:
        return result
    released = deliver(generated["reply"], setting)
    result.update(reply=released["delivered"], delivered=released["status"] == "delivered",
                  status="missing_judgment" if released["status"] == "delivered" else "withheld",
                  delivery_status=released["status"], retained_notes=released["retained_notes"],
                  reason=released["reason"])
    return result
