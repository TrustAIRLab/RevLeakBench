from __future__ import annotations


import json
import re


from pathlib import Path
from pathlib import PurePosixPath


from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _material(task_id: str) -> dict:
    if not re.fullmatch(r"[a-z0-9_]+", task_id):
        raise ValueError("task_id must contain only lowercase letters, digits and underscores")
    hits = list(ROOT.glob(f"data/*/{task_id}/materials"))
    if len(hits) != 1:
        raise KeyError(f"expected exactly one material directory for task: {task_id}")
    mat = hits[0]
    ws = mat / "workspace"
    if ws.is_dir():
        files = {p.name: p.read_text(encoding="utf-8", errors="replace")
                 for p in sorted(ws.iterdir()) if p.is_file() and p.name != "material.json"}
        if files:
            return {"workspace_files": files}
        mj = ws / "material.json"
        if mj.is_file():
            m = json.loads(mj.read_text(encoding="utf-8"))
            if m.get("workspace_files"):
                return m
    return json.loads((mat / "scenario.json").read_text(encoding="utf-8"))


class Workspace:

    def __init__(self, files: dict[str, str]):
        for name, content in files.items():
            if (not isinstance(name, str) or not name or "\\" in name
                    or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                    or not isinstance(content, str)):
                raise ValueError("workspace files require safe relative names and string contents")
        self.files = dict(files)

    @classmethod
    def for_task(cls, task_id: str) -> "Workspace":
        material = _material(task_id)
        if "workspace_files" in material:
            files = material["workspace_files"]
            if not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
                raise ValueError("workspace_files must map file names to text")
            return cls(files)
        if "initial_states" in material:
            states = material.get("initial_states") or {}
            return cls({f"{app}.json": json.dumps(state, indent=2) for app, state in states.items()})
        if "source_documents" in material:
            docs = material.get("source_documents") or {}
            return cls({f"{name}.json": json.dumps(doc, indent=2) for name, doc in docs.items()})
        raise ValueError(f"{task_id}: materials have none of workspace_files / initial_states / source_documents")

    def call(self, name: str, args: dict[str, Any]) -> str:
        if name == "list_files":
            return "\n".join(f"{n}\t{len(t)} chars" for n, t in self.files.items()) or "(empty workspace)"
        if name == "read_file":
            n = str(args.get("name", ""))
            if n not in self.files:
                return f"error: no such file: {n!r}. Files: {', '.join(self.files)}"
            return self.files[n]
        if name == "search":
            q = str(args.get("query", "")).strip().lower()
            if not q:
                return "error: empty query"
            hits = [f"{n}:{i + 1}: {ln.strip()}" for n, t in self.files.items() for i, ln in enumerate(t.split("\n")) if q in ln.lower()]
            return "\n".join(hits[:50]) if hits else "(no matches)"
        return f"error: unknown tool {name!r}"


def files_for(task_id: str) -> dict:
    files = Workspace.for_task(task_id).files
    if not files:
        raise ValueError(f"{task_id}: agent workspace is empty; the task cannot be run on the agent track")
    return files


def materialise(task_id: str, dest: Path) -> Path:
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for name, content in files_for(task_id).items():
        target = (dest / name).resolve()
        if dest not in target.parents:
            raise ValueError("workspace file escapes the destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return dest
