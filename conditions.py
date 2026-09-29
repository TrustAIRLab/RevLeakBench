from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
TRACKS = ("conversation", "agent")

FOUR_CELL = ("revoke_A", "static_A", "revoke_B", "static_B")
SEVEN_CELL = FOUR_CELL + ("replace", "replace_control", "direct")

CONTROLS = {
    "equal_turn": ("equal_turn_A", "equal_turn_B"),
    "late_exclusion": ("late_exclusion_A", "late_exclusion_B"),
    "replace_control_static": ("replace_control_static",),
}

DESCRIPTION = {
    "direct": "no exclusion is requested; measures baseline leakage",
    "revoke_A": "A is introduced, then withdrawn anaphorically in the final turn",
    "revoke_B": "B is introduced, then withdrawn anaphorically in the final turn",
    "static_A": "A is excluded by name in turn 1 and never introduced",
    "static_B": "B is excluded by name in turn 1 and never introduced",
    "replace": "an item is introduced, then replaced with C in the final turn",
    "replace_control": "C is requested from the start",
    "equal_turn_A": "static exclusion of A plus the five-turn shell; nothing is introduced",
    "equal_turn_B": "static exclusion of B plus the five-turn shell; nothing is introduced",
    "late_exclusion_A": "A is excluded by name in the final turn and never introduced",
    "late_exclusion_B": "B is excluded by name in the final turn and never introduced",
    "replace_control_static": "replacement control plus the static exclusion sentence",
}

COMPARISONS = {
    "revocation_vs_static": {
        "needs": FOUR_CELL,
        "pairs": (("revoke_A", "static_A"), ("revoke_B", "static_B")),
        "question": "how much more often revocation leaves a trace than static exclusion",
    },
    "replacement": {
        "needs": ("replace", "replace_control"),
        "pairs": (("replace", "replace_control"),),
        "question": "the effect of replacement itself (for f1/f2 use replace_control_static as the control)",
    },
    "baseline_leak": {
        "needs": ("direct",),
        "pairs": (),
        "question": "how much leaks with no exclusion requested at all",
    },
    "turn_count": {
        "needs": ("static_A", "static_B") + CONTROLS["equal_turn"],
        "pairs": (("equal_turn_A", "static_A"), ("equal_turn_B", "static_B")),
        "question": "whether the extra conversation turns themselves produce traces",
    },
    "timing": {
        "needs": CONTROLS["equal_turn"] + CONTROLS["late_exclusion"],
        "pairs": (("late_exclusion_A", "equal_turn_A"), ("late_exclusion_B", "equal_turn_B")),
        "question": "the effect of placing the exclusion instruction in the final turn rather than the first",
    },
    "introduction": {
        "needs": CONTROLS["late_exclusion"] + ("revoke_A", "revoke_B"),
        "pairs": (("revoke_A", "late_exclusion_A"), ("revoke_B", "late_exclusion_B")),
        "question": "the additional effect of introducing then withdrawing",
    },
    "model_added": {
        "needs": ("direct",),
        "pairs": (),
        "question": "traces when model-added content is asked to be removed",
    },
}

def task_ids(family: str | None = None) -> list[str]:
    if family is not None and not re.fullmatch(r"f[1-5]_[a-z]+", family):
        raise ValueError("invalid family identifier")
    pat = f"data/{family}/*/task.json" if family else "data/*/*/task.json"
    return sorted(p.parent.name for p in ROOT.glob(pat))

def load_task(task_id: str) -> dict:
    if not isinstance(task_id, str) or not re.fullmatch(r"[a-z0-9_]+", task_id):
        raise ValueError("invalid task identifier")
    hits = list(ROOT.glob(f"data/*/{task_id}/task.json"))
    if len(hits) != 1:
        raise KeyError(f"expected exactly one task definition for: {task_id}")
    return json.loads(hits[0].read_text(encoding="utf-8"))

def load_evaluation_spec(task_id: str) -> dict:
    return load_task(task_id)["evaluation"]

load_annotation = load_evaluation_spec

def messages(task_id: str, condition: str, track: str = "conversation") -> list[dict]:
    if track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}, got {track!r}")
    block = load_task(task_id)["messages"][track]
    if condition not in block:
        raise KeyError(f"{task_id}/{track} has no condition {condition}; available: {sorted(block)}")
    return block[condition]

def iter_cells(conditions=SEVEN_CELL, tracks=TRACKS, family=None):
    for tid in task_ids(family):
        block = load_task(tid)["messages"]
        for track in tracks:
            for cid in conditions:
                if cid not in block.get(track, {}):
                    raise ValueError("%s/%s has no condition %s" % (tid, track, cid))
                yield tid, track, cid, block[track][cid]

def comparison(name: str, task_id: str = None) -> dict:
    spec = dict(COMPARISONS[name])
    if name == "replacement":
        if task_id is None:
            raise ValueError("replacement comparison requires task_id to choose its control")
        task = load_task(task_id)
        control = "replace_control_static" if task["family"].split("_")[0] in ("f1", "f2") else "replace_control"
        spec.update(needs=("replace", control), pairs=(("replace", control),))
    return spec

def check_comparison(name: str, ran: set[str], task_id: str = None) -> None:
    spec = comparison(name, task_id)
    missing = [c for c in spec["needs"] if c not in ran]
    if missing:
        raise ValueError(
            f"reporting {name} ({spec['question']}) requires {list(spec['needs'])}; "
            f"missing {missing}.")
