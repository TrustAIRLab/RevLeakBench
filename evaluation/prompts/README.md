# Judge and reader prompts

| File | Purpose | Origin |
|---|---|---|
| `trace_cited_judge.txt` | Trace verdict (headline): yes/no plus a required verbatim quote | this work |
| getAnswer | Reader: fixed question, deliverable as context | RedacBench prompt with the experiment's JSON instruction |
| getEquality | Reader-answer equality, 0-1, threshold 0.8 | RedacBench prompt with the experiment's JSON instruction |
| checkPropositions | Formal utility and full-fact presence propositions | RedacBench instruction with the experiment's JSON instruction |

`evaluation/_core/redacbench.py` uses the experiment's HF adapter. Trace, utility,
presence, reader and equality calls use GLM-5.3-Flash (Baseten), thinking disabled,
temperature 0. Reader/equality allow 4,096 output tokens; proposition/trace judges
allow 16,384. Prompt constants are in `evaluation/_core/redacbench_prompts.py`.
Call signatures:

    get_answer(api_key, context, question) -> str
    get_equality(api_key, value, answer)   -> float
    check_propositions(api_key, context, propositions) -> [{proposition, reasoning, is_true}, ...]
