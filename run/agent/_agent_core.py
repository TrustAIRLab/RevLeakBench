from __future__ import annotations
import json
import urllib.request

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "List the files in the workspace with their sizes.",
                                      "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "read_file", "description": "Return the full content of one workspace file.",
                                      "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "File name as shown by list_files."}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "search", "description": "Find lines in any workspace file that contain the query (case-insensitive).",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())
