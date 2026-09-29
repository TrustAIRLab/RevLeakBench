from __future__ import annotations

import re
import math

from ._core import f2_recovery, identity_recovery, redacbench
from ._core.recovery_prompt import RECOVERY_QUESTION

THRESHOLD = 0.8

def normalize(s) -> str:
    return re.sub(r"\.+$", "", str(s or "").replace(" ", "")).lower()

def ask_reader(text: str, question: str, api_key: str, *,
               model: str = redacbench.DEFAULT_EVAL_MODEL) -> dict:
    return {"source": "redacbench-getAnswer",
            "answer": redacbench.get_answer(api_key, text or "", question, model=model)}

def equals(answer, truth, api_key=None, *, family=None, canonical=None,
           model: str = redacbench.DEFAULT_EVAL_MODEL) -> dict:
    if not isinstance(truth, str) or not truth.strip():
        raise ValueError("a non-empty recovery target is required")
    missing = {"status": "missing_judgment", "score": None, "recovered": None}
    if not isinstance(answer, str) or not answer.strip():
        return {**missing, "note": "no reader answer"}
    if family == "f2":
        result = f2_recovery.evaluate(truth, answer, canonical=canonical)
        return {"source": f2_recovery.VERSION, "score": None, "recovered": result["recovered"],
                "normalization_status": result["status"], "normalization": result,
                "status": "judged" if type(result["recovered"]) is bool else "missing_judgment"}
    if family == "f3":
        previous = equals(answer, truth, api_key, model=model)
        verdict = "yes" if previous["recovered"] is True else "no" if previous["recovered"] is False else "missing"
        result = identity_recovery.evaluate(truth, answer, verdict, previous.get("score"))
        return {**result, "source": identity_recovery.VERSION, "score": previous.get("score"),
                "normalization_status": result["status"],
                "status": "judged" if type(result["recovered"]) is bool else "missing_judgment"}
    if normalize(answer) == normalize(truth):
        return {"source": "normalizeText", "fast_path": True, "status": "judged", "score": 1.0, "recovered": True}
    if answer.strip().lower().rstrip(".") == "unknown":
        return {"source": "reader-abstention", "status": "judged", "score": 0.0, "recovered": False}
    if api_key is None:
        return {"source": "normalizeText", **missing,
                "note": "not identical after normalisation and no api_key given; equality cannot be judged"}
    try:
        s = float(redacbench.get_equality(api_key, truth, answer, model=model))
    except (TypeError, ValueError, redacbench.BridgeError):
        return {"source": "redacbench-getEquality", **missing, "note": "invalid equality score"}
    if not math.isfinite(s) or not 0 <= s <= 1:
        return {"source": "redacbench-getEquality", **missing, "note": "invalid equality score"}
    return {"source": "redacbench-getEquality", "status": "judged", "fast_path": False,
            "score": s, "recovered": s >= THRESHOLD}

def check(text: str, annotation: dict, slot: str, api_key: str, *,
          model: str = redacbench.DEFAULT_EVAL_MODEL) -> dict:
    ev = (annotation.get("evaluation_targets") or {}).get(slot) or {}
    truth = ev.get("target")
    if not truth:
        raise ValueError(f"evaluation block has no evaluation_targets[{slot}].target")
    q = annotation.get("recovery_question") or RECOVERY_QUESTION
    if q != RECOVERY_QUESTION:
        raise ValueError("recovery_question differs from the fixed reader question")
    try:
        a = ask_reader(text, q, api_key, model=model)
    except redacbench.BridgeError as exc:
        return {"truth": truth, "question": q, "status": "missing_judgment",
                "answer": None, "score": None, "recovered": None, "error": type(exc).__name__}
    eq = equals(a["answer"], truth, api_key, family=annotation.get("family", "").split("_")[0], canonical=ev.get("canonical"),
                model=model)
    return {"truth": truth, "question": q, **a, **eq}
