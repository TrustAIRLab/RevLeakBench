from __future__ import annotations

from ._core import agentci_presence, f2_numeric, redacbench


def check(text: str, annotation: dict, slot: str, api_key: str = None, *,
          model: str = redacbench.DEFAULT_EVAL_MODEL) -> dict:
    family = annotation["family"].split("_")[0]
    item = annotation["evaluation_targets"][slot]
    text = text or ""
    if family == "f1":
        hits = [f for f in item["source_annotated_fragments"]
                if agentci_presence._leak_match_in_shared_content(f, [text])]
        return {"source": "agentcibench-deterministic", "status": "judged",
                "present": bool(hits), "hits": hits}
    if family == "f2":
        coverage = f2_numeric.item_numeric_coverage(item, text)
        hits = coverage["matched"]
        return {"source": "f2-numeric-presence", "status": "judged",
                "present": bool(hits), "hits": hits,
                "numeric_coverage": coverage}
    if family not in ("f3", "f4", "f5"):
        raise ValueError("unknown presence family: " + family)
    proposition = item["presence_proposition"]["text"]
    result = {"source": "redacbench-checkPropositions", "proposition": proposition,
              "status": "missing_judgment", "present": None}
    if api_key is None:
        return result
    try:
        rows = redacbench.check_propositions(api_key, text, [proposition], model=model)
    except redacbench.BridgeError as exc:
        return {**result, "error": type(exc).__name__}
    result["raw"] = rows
    if not isinstance(rows, list):
        return result
    matches = [r for r in rows if isinstance(r, dict)
               and str(r.get("proposition", "")).strip() == proposition.strip()]
    if len(matches) == 1 and type(matches[0].get("is_true")) is bool:
        result.update(status="judged", present=matches[0]["is_true"])
    return result
