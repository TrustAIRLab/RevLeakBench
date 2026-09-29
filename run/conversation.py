from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conditions

def cell(task_id: str, condition: str, track: str = "conversation") -> list:
    return conditions.messages(task_id, condition, track)


def result(response) -> dict:
    content, finish = None, None
    if isinstance(response, dict):
        if response.get("protocol") == "revleak-workspace-agent-v1":
            content = response.get("final")
            finish = "stop" if response.get("ok") and response.get("status") == "success" else response.get("status")
        elif isinstance(response.get("choices"), list) and response["choices"]:
            choice = response["choices"][0]
            if isinstance(choice, dict):
                message = choice.get("message") or {}
                content = message.get("content") if isinstance(message, dict) else None
                finish = choice.get("finish_reason")
                if isinstance(message, dict) and message.get("tool_calls"):
                    finish = "tool_calls"
        else:
            content = response.get("content", response.get("reply"))
            finish = response.get("finish_reason")
    valid = finish == "stop" and isinstance(content, str) and bool(content.strip())
    return {"reply": content if valid else None, "finish_reason": finish,
            "status": "missing_judgment" if valid else "not_generated"}


def generate(messages, send, *, request_budget=3) -> dict:
    if type(request_budget) is not int or request_budget < 1:
        raise ValueError("request_budget must be a positive integer")
    for attempt in range(1, request_budget + 1):
        try:
            response = send(messages)
        except Exception as exc:
            if attempt == request_budget:
                return {"reply": None, "finish_reason": None, "status": "not_generated",
                        "n_requests": attempt, "error": type(exc).__name__}
            continue
        return {**result(response), "n_requests": attempt}


def run(task_id: str, condition: str, send, track: str = "conversation", *, request_budget=3) -> dict:
    msgs = cell(task_id, condition, track)
    return {"task_id": task_id, "condition": condition, "track": track,
            "n_turns": len(msgs), **generate(msgs, send, request_budget=request_budget)}
