from __future__ import annotations

import json
import re
import urllib.request

MODEL = "zai-org/GLM-5.3-Flash:baseten"


EXTRA = {"thinking": {"type": "disabled"}}


TIMEOUT = 240

URL = "https://router.huggingface.co/v1/chat/completions"
TEMPERATURE = 0
JUDGE_MAX_TOKENS = 16384
READER_MAX_TOKENS = 4096


class EvaluatorError(RuntimeError):
    pass


def chat(api_key, messages, *, model=MODEL, max_tokens=JUDGE_MAX_TOKENS,
         json_mode=False, timeout=TIMEOUT):
    if not api_key:
        raise EvaluatorError("HF_TOKEN is required for model evaluation")
    payload = {"model": model, "messages": messages, "temperature": TEMPERATURE,
               "max_tokens": max_tokens, **EXTRA}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        body = _post_json(URL, payload, {"Authorization": "Bearer " + api_key}, timeout)
        choice = body["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise EvaluatorError("evaluator response did not finish normally")
        content = choice["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise EvaluatorError("evaluator returned no usable text")
        return content
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        raise EvaluatorError("invalid evaluator response or transport failure") from exc


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())

def parse_json(raw):
    if not raw:
        return None
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.S)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None
