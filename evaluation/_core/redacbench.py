from __future__ import annotations

import math
from typing import Sequence

from . import evaluator
from .redacbench_prompts import ANSWER_SUFFIX, ANSWER_JSON, EQ_TEMPLATE, EQ_JSON, INSTRUCTION, JSON_NOTE

DEFAULT_EVAL_MODEL = evaluator.MODEL
DEFAULT_THRESHOLD = 0.8


BridgeError = evaluator.EvaluatorError


def _json_call(api_key, messages, *, model, max_tokens, json_mode):
    raw = evaluator.chat(api_key, messages, model=model, max_tokens=max_tokens, json_mode=json_mode)
    result = evaluator.parse_json(raw)
    if not isinstance(result, dict):
        raise BridgeError("evaluator did not return a JSON object")
    return result


def get_answer(api_key: str, context: str, question: str, model: str = DEFAULT_EVAL_MODEL) -> str:
    messages = [{"role": "system", "content": context},
                {"role": "user", "content": question + ANSWER_SUFFIX + ANSWER_JSON}]
    result = _json_call(api_key, messages, model=model, max_tokens=evaluator.READER_MAX_TOKENS,
                        json_mode=True)
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise BridgeError("reader returned no usable answer")
    return answer


def get_equality(api_key: str, value: str, answer: str, model: str = DEFAULT_EVAL_MODEL) -> float:
    messages = [{"role": "user", "content": EQ_TEMPLATE.format(value=value, answer=answer) + EQ_JSON}]
    response = _json_call(api_key, messages, model=model, max_tokens=evaluator.READER_MAX_TOKENS,
                         json_mode=True)
    result = response.get("score")
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise BridgeError("equality judge returned a non-numeric score")
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise BridgeError("equality score is outside [0, 1]")
    return float(result)


def check_propositions(api_key: str, context: str, propositions: Sequence[str | dict],
                       model: str = DEFAULT_EVAL_MODEL) -> list[dict]:
    props = [{"text": p} if isinstance(p, str) else p for p in propositions]
    user = f"context: {context}\n\npropositions: " + "\n".join(
        f"{i + 1}. {p['text']}" for i, p in enumerate(props))
    messages = [{"role": "system", "content": INSTRUCTION + JSON_NOTE}, {"role": "user", "content": user}]
    response = _json_call(api_key, messages, model=model, max_tokens=evaluator.JUDGE_MAX_TOKENS,
                         json_mode=False)
    result = response.get("evaluations")
    if not isinstance(result, list):
        raise BridgeError("proposition judge did not return a list")
    return result
