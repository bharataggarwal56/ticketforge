"""Scoring for TicketForge: pure metric functions plus thin AuditKIT wrappers.

`parse` does json.loads and nothing else: no fence stripping, no repair. If it
fails, every metric is 0. The headline exact-match metric is `exact_match_fields`
(five structured fields); the free-text summary is scored by length and token F1.
`auditkit` is imported lazily so the pure functions work without it.
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt import build_messages  # noqa: E402

STRUCTURED = ("category", "priority", "product", "sentiment", "needs_human")
METRICS = (
    "schema_valid", "category_acc", "priority_acc", "product_acc", "sentiment_acc",
    "needs_human_acc", "summary_len_ok", "summary_f1", "exact_match_fields", "exact_match_strict",
)


def parse(output):
    """Strict parse. Returns a dict, or None if the output is not one JSON object."""
    try:
        obj = json.loads(output.strip())
    except (ValueError, AttributeError):
        return None
    return obj if isinstance(obj, dict) else None


def _norm(text):
    return " ".join(str(text).lower().split())


def _field_ok(pred, gold, key):
    if key not in pred:
        return False
    if key == "needs_human":
        return isinstance(pred[key], bool) and pred[key] is gold[key]
    if key == "product":
        return isinstance(pred[key], str) and _norm(pred[key]) == _norm(gold[key])
    return pred[key] == gold[key]


def _summary_f1(pred_text, gold_text):
    p, g = _norm(pred_text).split(), _norm(gold_text).split()
    overlap = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not overlap:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def score_output(output, gold_json, validator):
    """All metrics for one model output against one gold JSON string."""
    pred, gold = parse(output), json.loads(gold_json)
    if pred is None:
        return {m: 0.0 for m in METRICS}
    s = {"schema_valid": float(validator.is_valid(pred))}
    for key in STRUCTURED:
        s[f"{key}_acc"] = float(_field_ok(pred, gold, key))
    summary = pred.get("summary")
    ok = isinstance(summary, str)
    s["summary_len_ok"] = float(ok and 1 <= len(summary.split()) <= 20)
    s["summary_f1"] = _summary_f1(summary, gold["summary"]) if ok else 0.0
    s["exact_match_fields"] = float(all(_field_ok(pred, gold, k) for k in STRUCTURED))
    s["exact_match_strict"] = float(
        s["exact_match_fields"] == 1.0 and ok and _norm(summary) == _norm(gold["summary"])
    )
    return s


def strip_pattern(schema):
    """Copy of the schema without the `summary` pattern (fallback if xgrammar rejects it)."""
    out = copy.deepcopy(schema)
    out["properties"]["summary"].pop("pattern", None)
    return out


def make_scorers(schema):
    """One AuditKIT scorer per metric, all backed by `score_output`."""
    import auditkit as ak

    validator = Draft202012Validator(schema)

    def build(metric):
        def fn(sample, output):
            return score_output(output, sample.target, validator)[metric]
        return ak.scorer(fn, direction=ak.Direction.MAXIMIZE, name=metric)

    return [build(m) for m in METRICS]


def prompt_adapter():
    """AuditKIT adapter that sends exactly `build_messages(ticket)` as chat messages."""
    from auditkit.adapter import Adapter, _gen_params
    from auditkit.model import Request

    class PromptAdapter(Adapter):
        method = "ticketforge_prompt"

        def adapt(self, sample, config):
            messages = build_messages(sample.input)
            flat = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return [Request(prompt=flat, request_type="chat",
                            params={**_gen_params(config), "messages": messages})]

    return PromptAdapter()


def load_samples(test_jsonl, meta_jsonl):
    """Test split as AuditKIT samples: input = ticket, target = gold JSON, id = uid."""
    import auditkit as ak

    rows = [json.loads(line) for line in Path(test_jsonl).read_text().splitlines() if line.strip()]
    metas = [json.loads(line) for line in Path(meta_jsonl).read_text().splitlines() if line.strip()]
    assert len(rows) == len(metas), (len(rows), len(metas))
    samples = []
    for i, (row, meta) in enumerate(zip(rows, metas)):
        assert meta["row"] == i, f"test_meta row {meta['row']} != position {i}"
        user, gold = row["messages"][1]["content"], row["messages"][2]["content"]
        samples.append(ak.Sample(input=user, target=gold, id=meta["uid"],
                                 metadata={"hard_case": meta["hard_case"]}))
    return samples


def slice_by_hard_case(outputs, samples, schema):
    """Mean of every metric per hard_case (and 'all'), from raw outputs in sample order."""
    validator = Draft202012Validator(schema)
    groups = {"all": []}
    for out, smp in zip(outputs, samples):
        scores = score_output(out, smp.target, validator)
        groups["all"].append(scores)
        groups.setdefault(smp.metadata["hard_case"], []).append(scores)
    return {
        name: {"n": len(rows), **{m: sum(r[m] for r in rows) / len(rows) for m in METRICS}}
        for name, rows in groups.items()
    }
