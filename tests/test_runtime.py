from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import tokenize
import unittest
from unittest.mock import Mock, patch
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import conditions
from defenses import delivery, output_filter, prompts
from evaluation import metrics, recovery, traces, utility
from evaluation._core import cited_judge, redacbench
from run import conversation, model_added
from run.agent.workspace import Workspace, files_for, materialise


class EvaluationTests(unittest.TestCase):
    def test_xml_verdicts(self):
        for answer, quote, expected in [("Yes", "I removed X.", "yes_verified"),
                                         ("No", "", "no"), ("Yes", "Not in the text.", "yes_unverified")]:
            reply = "<quote>%s</quote><answer>%s</answer>" % (quote, answer)
            with patch.object(cited_judge, "ask", return_value=reply):
                result = traces.judge("I removed X.", "offline")
            self.assertEqual(result["verdict"], expected)
            self.assertEqual(result["judge_reply"], reply)

    def test_malformed_is_missing(self):
        for reply in ["", "Answer: Yes", "Answer: banana", "<answer>Yes</answer>",
                      "<quote></quote><answer>No</answer><answer>Yes</answer>",
                      "<quote></quote><answer>No</answer><answer>banana</answer>"]:
            with patch.object(cited_judge, "ask", return_value=reply):
                self.assertEqual(traces.judge("Text.", "offline")["status"], "missing_judgment")

    def test_upstream_boolean_schema(self):
        annotation = {"required_y": ["Y1", "Y2"]}
        rows = [{"proposition": "Y1", "is_true": True}, {"proposition": "Y2", "is_true": False}]
        with patch.object(redacbench, "check_propositions", return_value=rows):
            result = utility.check("Text", annotation, "offline")
        self.assertEqual(result["flags"], {"Y1": True, "Y2": False})
        self.assertEqual(result["status"], "judged")
        self.assertFalse(result["pass_all"])

    def test_missing_invalid_and_duplicate_propositions(self):
        variants = [[], [{"proposition": "Y1", "is_true": True}],
                    [{"proposition": "Y1", "is_true": True}, {"proposition": "Y2", "is_true": None}],
                    [{"proposition": "Y1", "is_true": True}, {"proposition": "Y2", "is_true": "true"}],
                    [{"proposition": "Y1", "is_true": True}, {"proposition": "Y1", "is_true": False}]]
        for rows in variants:
            with patch.object(redacbench, "check_propositions", return_value=rows):
                result = utility.check("Text", {"required_y": ["Y1", "Y2"]}, "offline")
            self.assertEqual(result["status"], "missing_judgment")
            self.assertIsNone(result["pass_all"])

    def test_failed_bridge_is_missing(self):
        with patch.object(redacbench, "check_propositions", side_effect=redacbench.BridgeError("offline")):
            self.assertIsNone(utility.check("Text", {"required_y": ["Y1"]}, "offline")["pass_all"])

    def test_recovery_missing_and_abstention(self):
        with self.assertRaises(ValueError):
            recovery.equals(None, None)
        self.assertIsNone(recovery.equals("", "X")["recovered"])
        self.assertIsNone(recovery.equals("a paraphrase", "X")["recovered"])
        self.assertFalse(recovery.equals("unknown", "X")["recovered"])
        self.assertTrue(recovery.equals("X", "X")["recovered"])
        for value in [None, float("nan"), 2.0]:
            with patch.object(redacbench, "get_equality", return_value=value):
                self.assertIsNone(recovery.equals("Y", "X", "offline")["recovered"])

    def test_rates_and_duplicates(self):
        rows = [{"status": "judged", "hit": True}, {"status": "missing_judgment", "hit": None}]
        self.assertEqual(metrics.rate(rows), (100.0, 1))
        self.assertEqual(metrics.rate(rows[1:]), (None, 0))
        with self.assertRaises(ValueError):
            metrics.rate([{"status": "judged", "hit": "no"}])
        cell = dict(task_id="t", track="conversation", model="m", slot="A", status="judged", hit=True)
        with self.assertRaises(ValueError):
            metrics.paired([cell, cell], [cell])
        self.assertEqual(metrics.paired([cell], [dict(cell, hit=False)])["delta"], 100.0)
        a = [dict(cell, task_id=str(i), hit=i < 2) for i in range(3)]
        b = [dict(cell, task_id=str(i), hit=i < 1) for i in range(3)]
        self.assertEqual(metrics.paired(a, b)["delta"], 33.3)

    def test_filter_preserves_outside_bytes(self):
        text = "  Kept sentence.  \n\nI removed X.\n  "
        result = output_filter.filter_reply(text)
        self.assertEqual(result["delivered"], text.replace("I removed X.", ""))
        self.assertEqual(output_filter.filter_reply("I removed X; the meeting is at 10 AM.",
                                                  protect=("10 AM",))["status"], "withheld")
        self.assertEqual(output_filter.filter_reply("   ")["status"], "withheld")

    def test_filter_uses_full_scanner_and_literal_protection(self):
        text = "Public information.\n_Removed:_ Private Event"
        result = output_filter.filter_reply(text, aliases=["Private Event"], family="f1")
        self.assertEqual(result["delivered"], "Public information.\n")
        text = "Kept sentence. I removed the 134-unit apartment complex."
        self.assertEqual(output_filter.filter_reply(text, protect=["134 unit apartment complex"], family="f3")["status"], "delivered")
        text = "The meeting is at 10 AM. I removed X; the meeting is at 10 AM."
        self.assertEqual(output_filter.filter_reply(text, protect=["10 AM"])["delivered"], "The meeting is at 10 AM. ")
        text = "Kept.\n_Removed:_ Private Event\n_Removed:_ Private Event"
        self.assertEqual(output_filter.filter_reply(text, aliases=["Private Event"], family="f1")["delivered"], "Kept.\n\n")


class DeliveryTests(unittest.TestCase):
    def test_boundary_only_changes_first_turn(self):
        for tid in conditions.task_ids():
            original = conditions.messages(tid, "revoke_A")
            recipient = conditions.load_task(tid)["recipient"]
            for setting in delivery.SETTINGS:
                messages = delivery.messages(tid, "revoke_A", setting)
                self.assertEqual(messages[1:], original[1:])
                self.assertTrue(messages[0]["content"].startswith(original[0]["content"] + "\n\n"))
                appended = messages[0]["content"][len(original[0]["content"]):]
                self.assertIn(recipient, appended)
                self.assertEqual("action_trace" in appended, setting == "public_with_notes")
                self.assertEqual(conditions.messages(tid, "revoke_A"), original)
            for track in conditions.TRACKS:
                self.assertEqual(delivery.messages(tid, "revoke_A", "body_only", track),
                                 conditions.messages(tid, "revoke_A", track))
        with self.assertRaises(ValueError):
            delivery.messages(tid, "revoke_A", "public_only", "agent")

    def test_notes_never_enter_delivered_reply(self):
        for setting in delivery.SETTINGS:
            obj = {"completed": True, "shared_content": ["First line.", "Second line."]}
            if setting == "public_with_notes":
                obj["action_trace"] = "I removed the private password."
            raw = json.dumps(obj)
            result = delivery.run("f5_env_config", "revoke_A",
                                  lambda _: {"content": raw, "finish_reason": "stop"}, setting)
            self.assertTrue(result["delivered"])
            self.assertEqual(result["reply"], "First line.\nSecond line.")
            self.assertEqual(result["raw_reply"], raw)
            self.assertEqual(result["retained_notes"], obj.get("action_trace"))
            self.assertFalse(traces.analyse(result["reply"])["scanner"]["revision_trace"])

    def test_invalid_structured_response_is_not_resampled(self):
        invalid = ['{"completed":true,"completed":false,"shared_content":["Text"]}',
                   '{"completed":1,"shared_content":["Text"]}',
                   '{"completed":true,"shared_content":"Text"}',
                   '{"completed":true,"shared_content":[NaN]}',
                   '{"completed":true,"shared_content":["Text"],"extra":true}',
                   '```json\n{"completed":true,"shared_content":["Text"]}\n```',
                   '[]', 'null']
        for raw in invalid:
            send = Mock(return_value={"content": raw, "finish_reason": "stop"})
            result = delivery.run("f5_env_config", "revoke_A", send, "public_only")
            self.assertEqual((result["delivery_status"], result["n_requests"], send.call_count),
                             ("parse_failed", 1, 1))
            self.assertFalse(result["delivered"])
            self.assertIsNone(result["reply"])
        raw = '{"completed":true,"shared_content":["Text"]}'
        self.assertEqual(delivery.deliver(raw, "public_with_notes")["status"], "parse_failed")

    def test_withheld_and_failed_generation(self):
        for obj, reason in (({"completed": False, "shared_content": ["Text"]}, "completed_false"),
                            ({"completed": True, "shared_content": [" ", "\n"]}, "empty_shared_content")):
            result = delivery.deliver(json.dumps(obj), "public_only")
            self.assertEqual((result["status"], result["reason"]), ("withheld", reason))
            self.assertIsNone(result["delivered"])
        send = Mock(side_effect=TimeoutError())
        result = delivery.run("f5_env_config", "revoke_A", send, "public_only", request_budget=2)
        self.assertEqual((result["delivery_status"], result["n_requests"], send.call_count),
                         ("not_generated", 2, 2))
        self.assertFalse(result["delivered"])
        send = Mock(return_value={"content": '{"completed":true}', "finish_reason": "length"})
        result = delivery.run("f5_env_config", "revoke_A", send, "public_only")
        self.assertEqual((result["delivery_status"], send.call_count), ("not_generated", 1))

    def test_body_only_preserves_the_selected_text(self):
        text = 'Here is the draft:\n\nDear team,\n\nThe report is ready.\n\nBest,\nAlex\n\nLet me know if you need changes.'
        expected = 'Dear team,\n\nThe report is ready.\n\nBest,\nAlex\n\n'
        self.assertEqual(delivery.deliver(text, "body_only")["delivered"], expected)
        self.assertEqual(delivery.deliver("I removed that one.", "body_only")["status"], "withheld")
        result = delivery.run("f5_env_config", "revoke_A",
                              lambda _: {"protocol": "revleak-workspace-agent-v1", "ok": True,
                                         "status": "success", "final": text}, "body_only", "agent")
        self.assertTrue(result["delivered"])
        self.assertEqual(result["reply"], expected)

    def test_failed_delivery_remains_in_end_to_end_denominator(self):
        replies = ['{"completed":true,"shared_content":["Final text."]}',
                   '{"completed":false,"shared_content":["Final text."]}', 'Invalid JSON']
        plan, results = [], []
        for model, raw in enumerate(replies):
            row = dict(task_id="f5_env_config", condition="revoke_A", track="conversation", model=str(model))
            plan.append(row)
            result = delivery.run(row["task_id"], row["condition"],
                                  lambda _: {"content": raw, "finish_reason": "stop"}, "public_only")
            results.append({**row, **result, "utility": {"status": "judged", "pass_all": True},
                            "recovery": {"status": "judged", "recovered": False}})
        result = metrics.end_to_end(plan, results)
        self.assertEqual((result["planned"], result["successful"], result["rate"]), (3, 1, 33.3))


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = [conditions.load_task(tid) for tid in conditions.task_ids()]

    def test_all_workspaces_and_final_contracts(self):
        self.assertEqual(len(self.tasks), 100)
        for task in self.tasks:
            tid = task["task_id"]
            self.assertTrue(files_for(tid), tid)
            annotation = task["evaluation"]
            required = utility.propositions(annotation, "revoke_A")
            replacement = utility.propositions(annotation, "replace")
            self.assertTrue(required, tid)
            c = annotation["replacement_proposition"]
            self.assertIn(c, [p["text"] for p in replacement], tid)
            self.assertNotIn(c, [p["text"] for p in required], tid)
            for cid in task["messages"]["conversation"]:
                self.assertTrue(utility.propositions(annotation, cid), (tid, cid))
            with self.assertRaises(ValueError):
                utility.propositions(annotation)

    def test_target_registrations_and_metadata(self):
        counts = Counter()
        for task in self.tasks:
            self.assertFalse(set(task) & {"scene_id", "artifact_ref", "task_prompt"})
            self.assertFalse(set(task["evaluation"]) & {"replacement_side", "item_proposition", "required_items"})
            for item in task["evaluation"]["evaluation_targets"].values():
                self.assertNotIn("recovery_aliases", item)
                counts[item["information_type"]] += 1
                self.assertTrue(item["category_cues"])
                self.assertTrue(all(isinstance(cue, str) and cue.strip() for cue in item["category_cues"]))
                self.assertTrue(all(alias == " ".join(alias.split()) for alias in item["target_aliases"]))
        self.assertEqual(counts, {"Security credentials": 6, "Personal information": 72,
                                  "Organizational information": 77, "Content the user chose not to deliver": 45})

    def test_replacement_controls(self):
        for task in self.tasks:
            tid = task["task_id"]
            control = "replace_control_static" if task["family"].split("_")[0] in ("f1", "f2") else "replace_control"
            conditions.check_comparison("replacement", {"replace", control}, tid)
            with self.assertRaises(ValueError):
                conditions.check_comparison("replacement", {"replace"}, tid)

    def test_generation_and_replay_status(self):
        tid = self.tasks[0]["task_id"]
        result = conversation.run(tid, "revoke_A", lambda _: {"content": "Draft.", "finish_reason": "stop"})
        self.assertEqual(result["status"], "missing_judgment")
        built = dict(task_id=tid, slot="A", track="agent", status="constructed",
                     draft="Original draft.\n", removal_turn="Leave that out.", reference_rule={})
        result = model_added.replay(built, lambda _: {"content": "Final.", "finish_reason": "stop"})
        self.assertEqual(result["track"], "agent")
        self.assertEqual(result["status"], "missing_judgment")
        self.assertEqual(result["messages"][1]["content"], built["draft"])
        with self.assertRaises(ValueError):
            model_added.replay(built, lambda _: "Final.", track="conversation")

    def test_generation_failures_and_request_budget(self):
        tid = self.tasks[0]["task_id"]
        response = {"choices": [{"finish_reason": "stop", "message": {"content": "Final."}}]}
        with patch(__name__ + ".conversation.cell", return_value=[]) as messages:
            with patch(__name__ + ".conversation.generate", wraps=conversation.generate):
                from unittest.mock import Mock
                send = Mock(side_effect=[TimeoutError(), OSError(), response])
                generated = conversation.run(tid, "revoke_A", send)
                self.assertEqual((generated["status"], generated["n_requests"], send.call_count), ("missing_judgment", 3, 3))
                send = Mock(side_effect=TimeoutError())
                failed = conversation.run(tid, "revoke_A", send, request_budget=2)
                self.assertEqual((failed["status"], send.call_count), ("not_generated", 2))
                for bad in ({"content": "Partial.", "finish_reason": "length"},
                            {"content": "", "finish_reason": "stop"}, "No finish metadata."):
                    send = Mock(return_value=bad)
                    failed = conversation.run(tid, "revoke_A", send)
                    self.assertEqual((failed["status"], send.call_count), ("not_generated", 1))
                    self.assertIsNone(failed["reply"])

    def test_workspace_paths_and_defense_stacking(self):
        for name in ["../escape.txt", "/escape.txt", "a/../../escape.txt"]:
            with self.assertRaises(ValueError):
                Workspace({name: "Text"})
        with self.assertRaises(ValueError):
            prompts.apply([{"role": "system", "content": "Existing"}], "restrictive")
        with self.assertRaises(ValueError):
            files_for("../task")
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "workspace"
            dest.mkdir()
            (dest / "link").symlink_to(Path(directory), target_is_directory=True)
            with patch("run.agent.workspace.files_for", return_value={"link/outside.txt": "text"}):
                with self.assertRaises(ValueError):
                    materialise("unused", dest)

    def test_model_added_requires_resolvable_reference(self):
        tid = self.tasks[0]["task_id"]
        replies = iter(["A valid reference draft.", "B valid reference draft."])
        def reference(task, annotation, slot, draft):
            return ("Remove that one.", {}) if draft.startswith(slot + " valid") else (None, {})
        direct = {"content": "Invalid reference draft.", "finish_reason": "stop"}
        with patch.object(model_added, "removal_turn", side_effect=reference):
            built = model_added.build(tid, lambda _: {"content": next(replies), "finish_reason": "stop"},
                                      max_drafts=3, direct_reply=direct)
            self.assertEqual(built["n_drafts"], 3)
            self.assertEqual(built["cases"]["A"]["sample"], 1)
            self.assertEqual(built["cases"]["B"]["sample"], 2)
            self.assertEqual(built["cases"]["A"]["draft"], "A valid reference draft.")
            def forbid(_):
                self.fail("a qualifying reused draft must not trigger a request")
            failed = model_added.build(tid, forbid, max_drafts=1, direct_reply=direct)
            self.assertTrue(failed["drafts"][0]["reused"])
            self.assertTrue(all(c["status"] == "construction_failed" for c in failed["cases"].values()))
        with patch.object(model_added, "removal_turn", return_value=("Remove that one.", {})):
            reused = model_added.build(tid, forbid, direct_reply=direct)
            self.assertEqual(reused["n_drafts"], 1)

    def test_model_added_shares_four_attempts_including_failures(self):
        from unittest.mock import Mock
        tid = self.tasks[0]["task_id"]
        direct = {"content": "Unrelated draft.", "finish_reason": "stop"}
        send = Mock(side_effect=TimeoutError())
        with patch.object(model_added, "removal_turn", return_value=(None, {})):
            built = model_added.build(tid, send, direct_reply=direct)
        self.assertEqual((built["n_drafts"], send.call_count), (4, 3))
        self.assertTrue(all(c["status"] == "construction_failed" for c in built["cases"].values()))
        with self.assertRaises(ValueError):
            model_added.build(tid, send, direct_reply=direct, max_drafts=5)
        with self.assertRaises(ValueError):
            model_added.build(tid, send, direct_reply={**direct, "condition": "revoke_A"})
        with self.assertRaises(TypeError):
            model_added.build(tid, send, direct_reply="No completion metadata.")
    def test_source_and_anonymity_checks(self):
        patterns = [re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
                    re.compile(r"hf_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
                    re.compile(r"AIza[0-9A-Za-z_-]{25,}")]
        for path in ROOT.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            self.assertNotIn(path.name, (".env", ".netrc", "credentials.json"))
            if path.suffix == ".py":
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source)
                self.assertFalse(any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(source).readline)))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.assertIsNone(ast.get_docstring(node))
            if path.name.startswith("requirements"):
                self.assertFalse(any(line.lstrip().startswith("#") for line in path.read_text().splitlines()))
            if path.suffix in (".py", ".mjs", ".js", ".md", ".json", ".jsonl", ".txt"):
                text = path.read_text(encoding="utf-8")
                for pattern in patterns:
                    self.assertIsNone(pattern.search(text), str(path.relative_to(ROOT)))


if __name__ == "__main__":
    unittest.main()
