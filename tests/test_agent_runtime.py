from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import conditions
from run.agent.workspace import files_for
from defenses import prompts

class AgentTests(unittest.TestCase):
    def test_persisted_subprocess_and_reuse(self):
        from run.agent.bridge import check_agent_python, run_persisted_job
        self.assertTrue(check_agent_python()["ok"])
        job = dict(model="offline", files={"notes.txt": "Public material."},
                   dialogue=[{"role": "user", "content": "Read the notes."}],
                   fake_script=[{"name": "read_file", "args": {"name": "notes.txt"}}],
                   fake_final="Offline result.")
        with tempfile.TemporaryDirectory() as directory:
            result = run_persisted_job(job, Path(directory), timeout=45)
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["n_model_calls"], 2)
            self.assertEqual(run_persisted_job(job, Path(directory)), result)
            self.assertTrue((Path(directory) / "result.progress.json").is_file())
            with self.assertRaises(ValueError):
                run_persisted_job(dict(job, fake_final="Changed"), Path(directory))

    def test_agent_and_defenses(self):
        from run.agent.runner import run_job, selftest
        self.assertEqual(selftest(), 0)
        for family in sorted((ROOT / "data").iterdir()):
            tid = sorted(family.glob("*/task.json"))[0].parent.name
            files = files_for(tid)
            job = dict(model="offline", files=files, dialogue=conditions.messages(tid, "revoke_A", "agent"),
                       fake_script=[{"name": "read_file", "args": {"name": next(iter(files))}}],
                       fake_final="Offline result.")
            self.assertTrue(run_job(job)["ok"])
            job["dialogue"] = prompts.apply(job["dialogue"], "recipient_typed", tid)
            job["system_prompt_allowed"] = prompts.defense_text("recipient_typed", tid)
            self.assertTrue(run_job(job)["ok"])


if __name__ == "__main__":
    unittest.main()
