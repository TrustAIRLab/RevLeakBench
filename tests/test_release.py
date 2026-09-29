import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEVEN = ("direct", "revoke_A", "static_A", "revoke_B", "static_B", "replace", "replace_control")
FAMS = ("f1_coordination", "f2_enterprise", "f3_documents", "f4_analysis", "f5_software")
LABELS = ("judge_fragments", "source_annotated_fragments", "y_proposition", "pairing", "ground_truth")

fails = []

def check(cond, msg):
    if not cond:
        fails.append(msg)

def main():
    fails.clear()
    tasks = sorted((ROOT / "data").glob("*/*/task.json"))
    check(len(tasks) == 100, f"{len(tasks)} tasks, expected 100")

    for fam in FAMS:
        n = len(list((ROOT / "data" / fam).glob("*/task.json")))
        check(n == 20, f"{fam} has {n} tasks, expected 20")

    seen = set()
    for tp in tasks:
        t = json.loads(tp.read_text(encoding="utf-8"))
        tid = t["task_id"]
        check(tid not in seen, f"duplicate task_id: {tid}")
        seen.add(tid)
        check(tp.parent.name == tid, f"{tp}: directory name does not match task_id")

        for track in ("conversation", "agent"):
            got = set(t["messages"].get(track, {}))
            check(set(SEVEN) <= got, f"{tid}/{track} missing fixed conditions {sorted(set(SEVEN)-got)}")
            for cid, msgs in t["messages"][track].items():
                check(isinstance(msgs, list) and msgs, f"{tid}/{track}/{cid} messages is empty")
                check(msgs[0]["role"] == "user", f"{tid}/{track}/{cid} first turn is not user")
                check(msgs[-1]["role"] == "user", f"{tid}/{track}/{cid} last turn is not user")
                roles = [m["role"] for m in msgs]
                check(all(r in ("user", "assistant") for r in roles), f"{tid}/{track}/{cid} unexpected role")
                check(len(msgs) in (1, 3, 5), f"{tid}/{track}/{cid} has {len(msgs)} turns")

        for track in ("conversation", "agent"):
            check(len(t["messages"][track]["direct"]) == 1, f"{tid}/{track} direct is not a single turn")
            for cid in ("revoke_A", "revoke_B"):
                check(len(t["messages"][track][cid]) > 1, f"{tid}/{track}/{cid} should be multi-turn")

        shared = set(t["messages"]["conversation"]) & set(t["messages"]["agent"])
        check(shared == set(t["messages"]["conversation"]) == set(t["messages"]["agent"]),
              f"{tid}: the two tracks have different condition sets")
        for cid in sorted(shared):
            c, a = t["messages"]["conversation"][cid], t["messages"]["agent"][cid]
            check(len(c) == len(a), f"{tid}/{cid} turn counts differ between tracks")
            if len(c) == len(a):
                for i in range(1, len(c)):
                    check(c[i] == a[i], f"{tid}/{cid} turn {i+1} differs between tracks (only the first may differ)")

        sent = json.dumps(t.get("messages"), ensure_ascii=False)
        for k in LABELS:
            check(f'"{k}"' not in sent, f"{tid}: evaluation label {k} leaked into messages")
        sp = tp.parent / "materials" / "scenario.json"
        check(sp.exists(), f"{tid}: missing materials/scenario.json")
        if sp.exists():
            check('"ground_truth"' not in sp.read_text(encoding="utf-8"),
                  f"{tid}: ground_truth left in scenario.json")

        a = t.get("evaluation") or {}
        check(a, f"{tid}: task.json has no evaluation block")
        if a:
            ev = a.get("evaluation_targets") or {}
            check(set(ev) == {"A", "B"}, f"{tid}: evaluation_targets slots are {sorted(ev)}, expected A/B")
            for slot, blk in ev.items():
                for f in ("target", "target_aliases"):
                    check(blk.get(f), f"{tid}/{slot}: evaluation_targets missing {f}")
                check(isinstance(blk.get("target_aliases"), list) and blk["target_aliases"],
                      f"{tid}/{slot}: target_aliases must be a non-empty list")
                check(all(x == x.lower() for x in blk.get("target_aliases", [])),
                      f"{tid}/{slot}: target_aliases must be lower-cased")
            tc = a.get("terminal_contracts") or {}
            check(set(SEVEN) <= set(tc), f"{tid}: terminal_contracts missing {sorted(set(SEVEN)-set(tc))}")

    n_ev = sum(1 for p in tasks if (json.loads(p.read_text(encoding="utf-8")).get("evaluation") or {}))
    check(n_ev == len(tasks), f"{n_ev} tasks carry an evaluation block, expected {len(tasks)}")

    if fails:
        print(f"{len(fails)} check(s) failed:")
        for f in fails[:25]:
            print("   ", f)
        return 1
    print(f"All checks passed: {len(tasks)} tasks, fixed conditions complete on both tracks, "
          f"evaluation spec present for every task, tracks differ only in the first turn, "
          f"no evaluation labels on the input side")
    return 0

if __name__ == "__main__":
    sys.exit(main())
