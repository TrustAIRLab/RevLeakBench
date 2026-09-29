from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

AGENT_PYTHON = Path(os.environ.get("REVLEAK_AGENT_PYTHON") or sys.executable).resolve()

def check_agent_python(python: Path = None) -> dict:
    py = Path(python or AGENT_PYTHON).resolve()
    if not py.is_file():
        return {"ok": False, "python": str(py), "error": "interpreter not found; set REVLEAK_AGENT_PYTHON"}
    probe = ("import sys, json;"
             "d={'version': '%d.%d' % sys.version_info[:2]};"
             "\ntry:\n import langgraph, langchain_core;"
             " d['langgraph']=getattr(langgraph,'__version__','?')\n"
             "except Exception as e:\n d['error']=str(e)\n"
             "print(json.dumps(d))")
    try:
        out = subprocess.run([str(py), "-B", "-c", probe], capture_output=True, text=True, timeout=60)
        info = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {"ok": False, "python": str(py), "error": f"probe failed: {exc}"}
    info["python"] = str(py)
    info["ok"] = "error" not in info and tuple(int(x) for x in info.get("version", "0.0").split(".")) >= (3, 11)
    if not info["ok"] and "error" not in info:
        info["error"] = f"requires Python 3.11+ with langgraph installed; found {info.get('version')}"
    return info

def _save(path: Path, value: dict) -> None:
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    pending.replace(path)

def recover_episode(directory: Path, reason: str = "interrupted") -> dict:
    directory = Path(directory).resolve()
    result = directory / "result.json"
    if result.is_file():
        return json.loads(result.read_text(encoding="utf-8"))
    progress = directory / "result.progress.json"
    out = json.loads(progress.read_text(encoding="utf-8")) if progress.is_file() else {"n_model_calls": 0}
    if out.get("status") != "success":
        out.update(ok=False, final=None, status=reason, error=f"agent process {reason}; retained last progress snapshot")
    _save(result, out)
    return out

def run_persisted_job(job: dict, directory: Path, *, python: Path = AGENT_PYTHON, timeout: float = 1800, lock_fd=None) -> dict:
    directory = Path(directory).resolve()
    python = Path(python).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    job_path, result_path = directory / "job.json", directory / "result.json"
    if job_path.exists():
        if json.loads(job_path.read_text(encoding="utf-8")) != job:
            raise ValueError(f"episode inputs differ from saved job: {directory}; use a new tag")
        return recover_episode(directory)
    if result_path.exists() or (directory / "result.progress.json").exists():
        raise ValueError(f"orphaned episode output: {directory}; inspect before running")
    with job_path.open("x", encoding="utf-8") as fh:
        json.dump(job, fh, ensure_ascii=False, indent=1)
    try:
        subprocess.run([str(python), "-B", "-m", "run.agent.runner",
                        "--job", str(job_path), "--out", str(result_path)],
                       capture_output=True, text=True, cwd=ROOT, timeout=timeout, check=False,
                       pass_fds=() if lock_fd is None else (lock_fd,))
    except subprocess.TimeoutExpired:
        return recover_episode(directory, "process_timeout")
    except OSError:
        return recover_episode(directory, "process_start_error")
    return recover_episode(directory, "process_error")
