# RevLeakBench

RevLeakBench tests whether an assistant leaves a revision trace in a recipient-visible
deliverable after an item is withdrawn. 100 tasks, five scenarios, conversation and
agent tracks.

Evaluation definitions: [PROTOCOL.md](PROTOCOL.md).

## Install

Evaluation: Python 3.8-3.10, `requirements.txt` with spaCy 3.3.0.
Agent runner: a separate Python 3.11+ environment,
`requirements-agent.txt`. Model evaluation uses `HF_TOKEN` and
GLM-5.3-Flash through the HF router, with thinking disabled and temperature 0.

Paths resolve from the package, not the working directory.

    python -m pip install -r /path/to/RevLeakBench/requirements.txt
    python -B /path/to/RevLeakBench/tests/test_release.py
    python -B /path/to/RevLeakBench/tests/test_runtime.py -v

## Conversation track

    import sys
    sys.path.insert(0, "/path/to/RevLeakBench")
    import conditions
    from evaluation import traces, utility, presence, recovery
    from run import conversation

    generated = conversation.run("f4_archaeology", "revoke_A", your_model)
    reply = generated["reply"]

    if reply is not None:
        evaluation = conditions.load_evaluation_spec("f4_archaeology")
        aliases = evaluation["evaluation_targets"]["A"]["target_aliases"]
        trace_result = traces.analyse(reply, aliases=aliases, family="f4")
        scanner = trace_result["scanner"]
        traces.judge(reply, api_key=hf_token)
        utility.check(reply, evaluation, hf_token, condition="revoke_A")
        presence.check(reply, evaluation, "A", api_key=hf_token)
        recovery.check(reply, evaluation, "A", api_key=hf_token)

`your_model(messages)` returns the chat completion response dictionary, or a dictionary with
`content` and `finish_reason`. Evaluate only generated replies. Endpoint errors are
retried within `request_budget` (default 3); empty and truncated replies fail generation.

`scanner` contains `revision_trace`, `granularity` and `spans`. Trace presence is
independent of disclosure level. `raw_scan` contains the original rule hits;
`levels` includes task-content presence outside traces.

The three `check()` calls accept `model=...`, defaulting to
`zai-org/GLM-5.3-Flash:baseten`. Recovery uses that model for both the reader and
the equality judge.

## Agent track

Files from `run.agent.workspace.files_for(task_id)`, dialogue from
`conditions.messages(task_id, condition, "agent")`, both into
`run.agent.runner.run_job()`. Credentials from `HF_TOKEN` or `OPENAI_API_KEY`.
Use `run.agent.bridge` to run the agent in its own interpreter.

For a prompt defense, prepend `defenses.prompts.apply(...)` and register the same text
in the job's `system_prompt_allowed`.

## Contents

| Path | |
|---|---|
| data/scenario/task_id/task.json | rendered messages, metadata, evaluation block |
| data/scenario/task_id/materials/ | workspace source material |
| conditions.py | conditions, presets, comparison checks |
| run/ | conversation, agent and model-added execution |
| evaluation/ | traces, positions, required content, presence, recovery, statistics |
| defenses/ | prompt defenses, delivery boundaries, output filter |
| tests/ | offline checks |

Model inputs are the selected messages and workspace files. The evaluation block
contains the answers.

For model-added cases, call `model_added.build(task_id, your_model,
direct_reply=direct_result)` once. It selects A/B from the direct reply and at most
three additional drafts. Pass `built["cases"]["A"]` or `built["cases"]["B"]` to
`model_added.replay()`. Agent callbacks return the agent runner's result dictionary.
F2 construction requires `api_key=hf_token` to verify the registered fact with
`checkPropositions`; `model=...` selects the evaluator.
An unavailable case replays as `not_generated`, with `construction_status="construction_failed"`.

`defenses.output_filter.filter_reply(reply, aliases=aliases, family=family,
protect=required_strings)` removes scanner spans and protects literal required content.

Delivery boundaries use `defenses.delivery`:

    from defenses import delivery
    generated = delivery.run("f5_env_config", "revoke_A", your_model,
                             setting="public_with_notes")
    reply = generated["reply"]

| Setting | Delivered text | Track |
|---|---|---|
| `public_only` | JSON `shared_content` joined with newlines | conversation |
| `public_with_notes` | same, with `action_trace` retained separately | conversation |
| `body_only` | extracted body of an ordinary reply | both |

Evaluate `reply` when `delivered=True`. `raw_reply` and `retained_notes` stay private.
`delivery.messages(...)` builds the dialogue; `delivery.deliver(reply, setting)`
extracts delivery from an existing response. Invalid JSON is `parse_failed` and is
not resampled. Failed deliveries count as unsuccessful in `metrics.end_to_end()`.

Each `evaluation_targets` entry records `information_type`, identity-only
`target_aliases`, and separate `category_cues` for descriptive traces.

## 🙏 Citation

If you find this useful in your research, please consider citing:

```bibtex
@article{ZJZ26,
  author = {Yage Zhang and Yukun Jiang and Yang Zhang},
  title = {{"Nothing to See Here'': Unintended Disclosure through Revision Traces of LLM Deliverables}},
  year = {2026},
  journal = {{CoRR abs/2609.35408}},
  url = {https://arxiv.org/abs/2609.35408}
}
```
Licences: [licenses/README.md](licenses/README.md).
