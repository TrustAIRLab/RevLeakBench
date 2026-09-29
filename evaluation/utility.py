from __future__ import annotations

from ._core import redacbench

def propositions(annotation: dict, condition: str = None) -> list:
    out, seen = [], set()

    def add(text, role="required"):
        t = str(text or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append({"text": t, "role": role})

    contracts = annotation.get("terminal_contracts") or {}
    if contracts and not condition:
        raise ValueError("condition is required to select the final content contract")
    contract_key = condition
    for prefix in ("equal_turn_", "late_exclusion_", "model_added_"):
        if condition and condition.startswith(prefix) and condition[len(prefix):] in ("A", "B"):
            contract_key = "static_" + condition[len(prefix):]
    if contract_key == "replace_control_static":
        contract_key = "replace"
    if contract_key and contract_key not in contracts:
        raise ValueError("no final content contract for condition %r" % condition)
    contract = contracts.get(contract_key) if condition else None
    slots = list((contract or {}).get("must_include_slots") or [])

    for p in (annotation.get("required_propositions") or []):
        if isinstance(p, dict):
            if p.get("role") in (None, "required"):
                add(p.get("text"))
        else:
            add(p)
    if not out:
        tmpl = annotation.get("y_proposition")
        ys = annotation.get("required_y")
        if isinstance(ys, list) and tmpl and "{y}" in str(tmpl):
            for y in ys:
                add(str(tmpl).replace("{y}", str(y)))
        elif isinstance(ys, list):
            for y in ys:
                add(y)
        elif ys:
            add(ys)
        elif tmpl:
            add(tmpl)

    if slots and "C" in slots:
        c = annotation.get("replacement_proposition")
        if not isinstance(c, str) or not c.strip():
            raise ValueError("the final contract requires C, but its proposition is missing")
        add(c)
    return out

def check(text: str, annotation: dict, api_key: str, condition: str = None, *,
          model: str = redacbench.DEFAULT_EVAL_MODEL) -> dict:
    props = propositions(annotation, condition)
    if not props:
        return {"source": "redacbench-checkPropositions", "condition": condition, "n_props": 0,
                "pass_all": None, "status": "missing_judgment",
                "note": "no required propositions recorded for this task"}
    texts = [p["text"] for p in props]
    try:
        evals = redacbench.check_propositions(api_key, text or "", texts, model=model)
    except redacbench.BridgeError as exc:
        return {"source": "redacbench-checkPropositions", "condition": condition,
                "n_props": len(texts), "status": "missing_judgment", "pass_all": None,
                "error": type(exc).__name__}
    if not isinstance(evals, list):
        return {"source": "redacbench-checkPropositions", "condition": condition,
                "n_props": len(texts), "status": "missing_judgment", "pass_all": None,
                "note": "judge returned an invalid proposition list"}

    flags, unresolved = {}, []
    got = {}
    for e in evals or []:
        if isinstance(e, dict):
            key = str(e.get("proposition", "")).strip()
            if key in got:
                got[key] = None
            else:
                got[key] = e.get("is_true")
    for t in texts:
        if type(got.get(t)) is bool:
            flags[t] = got[t]
        else:
            unresolved.append(t)

    if unresolved:
        return {"source": "redacbench-checkPropositions", "condition": condition,
                "n_props": len(texts), "raw": evals, "flags": flags,
                "pass_all": None, "status": "missing_judgment",
                "unresolved": unresolved,
                "note": "judge returned no verdict for %d of %d propositions" % (len(unresolved), len(texts))}
    return {"source": "redacbench-checkPropositions", "condition": condition,
            "n_props": len(texts), "raw": evals, "flags": flags,
            "pass_all": all(flags.values()), "status": "judged",
            "missing": [k for k, ok in flags.items() if not ok]}
