from __future__ import annotations

import collections
import random

SEED = 20260916
NBOOT = 2000
STATUS = ("judged", "missing_judgment", "not_generated", "withheld")

def _judged(rows, field):
    out = []
    for r in rows:
        if r.get("status") not in STATUS:
            raise ValueError("every cell must have a recognized evaluation status")
        if r["status"] == "judged":
            if type(r.get(field)) is not bool:
                raise ValueError("judged cells require a boolean %r" % field)
            out.append(r)
    return out

def pooled_rate(rows, field="hit"):
    rows = _judged(rows, field)
    n = len(rows)
    return (round(100.0 * sum(r[field] for r in rows) / n, 1) if n else None), n

def _model_groups(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row.get("model")].append(row)
    if None in groups and len(groups) > 1:
        raise ValueError("model is required on every row when pooling models")
    return groups

def model_rates(rows, field="hit"):
    return {model: pooled_rate(group, field) for model, group in _model_groups(rows).items()}

def rate(rows, field="hit"):
    groups = [_judged(group, field) for group in _model_groups(rows).values()]
    n = sum(len(group) for group in groups)
    if not groups or any(not group for group in groups):
        return None, n
    value = sum(sum(r[field] for r in group) / len(group) for group in groups) / len(groups)
    return round(100.0 * value, 1), n

def end_to_end(planned, results, *, key=("task_id", "track", "model", "condition")):
    def index(rows):
        out = {}
        for row in rows:
            identity = tuple(row[f] for f in key)
            if identity in out:
                raise ValueError("duplicate request key %r" % (identity,))
            out[identity] = row
        return out

    plan, observed = index(planned), index(results)
    if set(observed) - set(plan):
        raise ValueError("result outside the request plan")
    scored = []
    for identity, request in plan.items():
        result = observed.get(identity, {})
        utility = result.get("utility") or {}
        recovery = result.get("recovery") or {}
        success = (result.get("delivered") is True
                   and utility.get("status") == "judged" and utility.get("pass_all") is True
                   and recovery.get("status") == "judged" and recovery.get("recovered") is False)
        scored.append({**request, "status": "judged", "hit": success})
    return {"planned": len(plan), "successful": sum(r["hit"] for r in scored),
            "rate": pooled_rate(scored)[0], "by_model": model_rates(scored), "cells": scored}

def status_counts(rows, field="status"):
    c = collections.Counter(r.get(field) or "missing_status" for r in rows)
    out = {k: c.get(k, 0) for k in STATUS}
    out["other"] = sum(v for k, v in c.items() if k not in STATUS)
    return out

def _pair_rates(pairs, models):
    by_model = collections.defaultdict(list)
    for task, model, a, b in pairs:
        by_model[model].append((a, b))
    if not models or set(by_model) != models:
        return None, None
    a = sum(sum(x for x, y in rows) / len(rows) for rows in by_model.values()) / len(models)
    b = sum(sum(y for x, y in rows) / len(rows) for rows in by_model.values()) / len(models)
    return 100.0 * a, 100.0 * b


def bootstrap_delta(pairs, *, details=False):
    pairs = [(p[0], None, p[1], p[2]) if len(p) == 3 else p for p in pairs]
    clusters = sorted({p[0] for p in pairs})
    models = {p[1] for p in pairs}
    by = collections.defaultdict(list)
    for p in pairs:
        by[p[0]].append(p)
    rng = random.Random(SEED)
    out = []
    samples = NBOOT if pairs else 0
    for _ in range(samples):
        sample = []
        for _ in clusters:
            sample.extend(by[rng.choice(clusters)])
        a, b = _pair_rates(sample, models)
        out.append(a - b if a is not None else None)
    valid = sum(value is not None for value in out)
    ci = None
    if samples and valid == samples:
        out.sort()
        ci = round(out[int(0.025 * samples)], 1), round(out[int(0.975 * samples) - 1], 1)
    report = {"ci": ci, "samples": samples, "valid_samples": valid, "unresolved_samples": samples - valid}
    return report if details else ci

def paired(cells_a, cells_b, *, key=("task_id", "track", "model", "slot"), field="hit"):
    cells_a, cells_b = list(cells_a), list(cells_b)
    def index(rows, label):
        out = {}
        rows = list(rows)
        seen = set()
        for r in rows:
            k = tuple(r[f] for f in key)
            if k in seen:
                raise ValueError("%s: duplicate pairing key %r" % (label, k))
            seen.add(k)
        for r in _judged(rows, field):
            k = tuple(r[f] for f in key)
            out[k] = r
        return out

    A, B = index(cells_a, "cells_a"), index(cells_b, "cells_b")
    both = sorted(set(A) & set(B))
    models = set(_model_groups(cells_a + cells_b))
    pairs = [(A[k]["task_id"], A[k].get("model"), int(A[k][field]), int(B[k][field])) for k in both]
    unresolved = len({tuple(r[f] for f in key) for r in cells_a}
                     & {tuple(r[f] for f in key) for r in cells_b}) - len(both)
    a, b = _pair_rates(pairs, models)
    if a is None:
        return {"n": len(pairs), "a": None, "b": None, "delta": None, "ci": None,
                "bootstrap": {"samples": 0, "valid_samples": 0, "unresolved_samples": 0},
                "unresolved_pairs": unresolved,
                "dropped_a_only": len(set(A) - set(B)), "dropped_b_only": len(set(B) - set(A))}
    bootstrap = bootstrap_delta(pairs, details=True)
    ci = bootstrap.pop("ci")
    return {"n": len(pairs), "a": round(a, 1), "b": round(b, 1), "delta": round(a - b, 1),
            "unresolved_pairs": unresolved,
            "ci": ci, "bootstrap": bootstrap,
            "dropped_a_only": len(set(A) - set(B)), "dropped_b_only": len(set(B) - set(A))}
