"""Stage 1: curate the generated ticket pools with CuratorKIT.

Two separate runs so CuratorKIT's random split can never mix test into train:
  train pool -> <out>/train_<tag>/{train,validation}/sft_messages.jsonl
  test pool  -> <out>/test_<tag>/test/sft_messages.jsonl  (+ test_meta.jsonl)

The system prompt comes from src/prompt.py via a `system` column, which
CuratorKIT exports as the leading system turn. Cleaning and PII
pseudonymization touch only the ticket text, so the gold JSON is never changed.
Post-checks fail loudly if an exported row is malformed or a label drifted.

Usage:
    python src/curate.py --tag n50
    python src/curate.py --tag 2026-10-24
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt import SYSTEM_PROMPT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "configs" / "schema.json").read_text())
# Pattern-based types only. spaCy's PERSON tagged product names ("Gift Card",
# "Zenith Smartwatch") and Faker replaced them, breaking the label (risk R7).
# Ticket names are generator fakes, so skipping PERSON exposes nothing real.
PII_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS", "US_SSN"]


def add_system(row: dict) -> dict:
    """CuratorKIT preprocessing_fn: add the system prompt as a `system` column."""
    return {**row, "system": SYSTEM_PROMPT}


def run_curator(dataset: Path, out_dir: Path, split: dict, args) -> None:
    from curatorkit import Curator, CuratorConfig

    Curator(CuratorConfig(
        dataset=str(dataset),
        field_mapping={"ticket_text": "instruction", "label_json": "output"},
        preprocessing_fn=add_system,
        dedup="minhash",
        minhash_threshold=args.minhash_threshold,
        clean=True,
        clean_fields=["instruction"],
        min_tokens=args.min_tokens,
        pii_pseudonymize=True,
        pii_fields=["instruction"],
        pii_entity_types=PII_ENTITIES,
        pii_spacy_model=args.spacy_model,
        export_formats=["messages"],
        output_split=split,
        output_dir=str(out_dir),
    )).run()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_split(rows: list[dict], raw_by_label: dict[str, dict], name: str) -> list[str]:
    """Return problems found in one exported split."""
    validator = Draft202012Validator(SCHEMA)
    problems = []
    for i, row in enumerate(rows):
        msgs = row.get("messages", [])
        roles = [m.get("role") for m in msgs]
        where = f"{name}[{i}]"
        if roles != ["system", "user", "assistant"]:
            problems.append(f"{where}: roles {roles}")
            continue
        system, user, assistant = (m["content"] for m in msgs)
        if system != SYSTEM_PROMPT:
            problems.append(f"{where}: system prompt differs")
        if assistant not in raw_by_label:
            problems.append(f"{where}: assistant JSON matches no raw label_json")
            continue
        errors = list(validator.iter_errors(json.loads(assistant)))
        if errors:
            problems.append(f"{where}: schema errors {[e.message for e in errors][:2]}")
        product = json.loads(assistant)["product"]
        if product not in user:
            problems.append(f"{where}: product {product!r} missing from ticket after curation")
    return problems


def report(out_dir: Path, splits: list[str], n_raw: int) -> dict:
    manifest = json.loads((out_dir / "manifest.json").read_text())
    rejected = read_jsonl(out_dir / "rejected.jsonl")
    counts = {s: len(read_jsonl(out_dir / s / "sft_messages.jsonl")) for s in splits}
    print(f"\n== {out_dir.name}: {n_raw} raw -> {sum(counts.values())} exported {counts}, "
          f"{len(rejected)} rejected")
    for step, c in manifest.get("stage_counts", {}).items():
        print(f"  {step:28s} {c}")
    # Dedup drops are counted here, not in rejected.jsonl.
    print(f"  dedup_stats                  {manifest.get('dedup_stats')}")
    reasons = Counter(r.get("rejection_reason") for r in rejected)
    for reason, n in reasons.most_common():
        print(f"  rejected: {n:4d}  {reason}")
    for r in rejected[:3]:
        print(f"    e.g. [{r.get('rejecting_step')}] {r.get('instruction', '')[:120]!r}")
    return {"raw": n_raw, "exported": counts, "rejected": len(rejected), "reasons": dict(reasons),
            "dedup": manifest.get("dedup_stats")}


def curate_pool(raw_path: Path, out_dir: Path, split: dict, args) -> dict:
    if out_dir.exists():
        if not args.overwrite:
            sys.exit(f"{out_dir} exists; use a new --tag or pass --overwrite")
        shutil.rmtree(out_dir)
    raw = read_jsonl(raw_path)
    raw_by_label = {r["label_json"]: r for r in raw}
    run_curator(raw_path, out_dir, split, args)

    problems, exported = [], {}
    for name in split:
        exported[name] = read_jsonl(out_dir / name / "sft_messages.jsonl")
        problems += check_split(exported[name], raw_by_label, f"{out_dir.name}/{name}")
    stats = report(out_dir, list(split), len(raw))
    if problems:
        print(f"\nPOST-CHECK FAILED ({len(problems)}):", *problems[:20], sep="\n  ", file=sys.stderr)
        sys.exit(1)
    print("  post-checks passed (roles, system prompt, schema-valid labels, products intact)")

    if "test" in split:
        # The export keeps only `messages`; rejoin meta on the (unique) gold JSON.
        if len(raw_by_label) != len(raw):
            sys.exit("test label_json values are not unique; cannot rejoin meta")
        meta = [{**raw_by_label[r["messages"][2]["content"]]["meta"], "row": i}
                for i, r in enumerate(exported["test"])]
        with open(out_dir / "test_meta.jsonl", "w", encoding="utf-8") as f:
            for m in meta:
                f.write(json.dumps(m) + "\n")
        print(f"  wrote test_meta.jsonl; hard_case {dict(Counter(m['hard_case'] for m in meta))}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=ROOT / "data/raw/train_pool.jsonl")
    ap.add_argument("--test", type=Path, default=ROOT / "data/raw/test_pool.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data/curated")
    ap.add_argument("--tag", required=True, help="suffix for the output folders, e.g. a date")
    ap.add_argument("--min-tokens", type=int, default=5)
    ap.add_argument("--minhash-threshold", type=float, default=0.85)
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    stats = {
        "train": curate_pool(args.train, args.out / f"train_{args.tag}",
                             {"train": 0.9, "validation": 0.1}, args),
        "test": curate_pool(args.test, args.out / f"test_{args.tag}", {"test": 1.0}, args),
    }
    (args.out / f"stats_{args.tag}.json").write_text(json.dumps(stats, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
