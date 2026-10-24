import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gen_tickets as g  # noqa: E402

SCHEMA = json.loads((ROOT / "configs" / "schema.json").read_text())
N_TRAIN, N_TEST = 400, 100


def _level(entry_id):
    # ids look like "tr.priority.high.1" or "te.sentiment.negative.0"
    return entry_id.split(".")[2]


class GenTicketsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = g.generate(N_TRAIN, "train", seed=7)
        cls.test = g.generate(N_TEST, "test", seed=7, unique_labels=True)
        cls.rows = cls.train + cls.test

    def labels(self, rows):
        return [json.loads(r["label_json"]) for r in rows]

    def test_deterministic(self):
        again = g.generate(N_TRAIN, "train", seed=7)
        self.assertEqual([r["ticket_text"] for r in again], [r["ticket_text"] for r in self.train])
        other = g.generate(N_TRAIN, "train", seed=8)
        self.assertNotEqual([r["ticket_text"] for r in other], [r["ticket_text"] for r in self.train])

    def test_labels_schema_valid_and_ordered(self):
        validator = Draft202012Validator(SCHEMA)
        for lab in self.labels(self.rows):
            self.assertEqual(list(validator.iter_errors(lab)), [], lab)
            self.assertEqual(list(lab), list(SCHEMA["properties"]))

    def test_summary_word_limit(self):
        for lab in self.labels(self.rows):
            self.assertLessEqual(len(lab["summary"].split()), 20)

    def test_product_matches_category(self):
        for lab in self.labels(self.rows):
            self.assertIn(lab["category"], g.PRODUCTS[lab["product"]])

    def test_pool_ids_disjoint(self):
        tags = {eid.split(".")[0] for eid in g.REGISTRY}
        self.assertEqual(tags, {"tr", "te"})
        for r in self.train:
            self.assertTrue(all(i.startswith("tr.") for i in r["meta"]["ids"]))
        for r in self.test:
            self.assertTrue(all(i.startswith("te.") for i in r["meta"]["ids"]))
        self.assertEqual({r["meta"]["style"] for r in self.train} & {r["meta"]["style"] for r in self.test}, set())

    def test_leakage_check_passes(self):
        self.assertEqual(g.check_leakage(self.train, self.test), [])

    def test_leakage_check_detects_planted_ngram(self):
        test = copy.deepcopy(self.test)
        test[0]["_skeleton"] = test[0]["_skeleton"] + [self.train[0]["_skeleton"][0]]
        problems = g.check_leakage(self.train, test)
        self.assertTrue(any("8-grams" in p for p in problems), problems)

    def test_leakage_check_detects_foreign_id(self):
        test = copy.deepcopy(self.test)
        test[0]["meta"]["ids"].append(self.train[0]["meta"]["ids"][0])
        self.assertTrue(g.check_leakage(self.train, test))

    def test_hard_case_shares(self):
        for rows, share in ((self.train, 0.20), (self.test, 0.30)):
            hard = sum(r["meta"]["hard_case"] != "none" for r in rows)
            self.assertAlmostEqual(hard / len(rows), share, delta=0.02)
            kinds = {r["meta"]["hard_case"] for r in rows}
            self.assertEqual(kinds, {"none", *g.HARD_TYPES})

    def test_multi_priority_is_max(self):
        for r in self.rows:
            if r["meta"]["hard_case"] != "multi":
                continue
            levels = [g.PRIORITIES.index(_level(i)) for i in r["meta"]["ids"] if ".priority." in i]
            self.assertEqual(len(levels), 2)
            self.assertEqual(g.PRIORITIES[max(levels)], json.loads(r["label_json"])["priority"])
            issue_cats = [_level(i) for i in r["meta"]["ids"] if ".issue." in i]
            self.assertEqual(issue_cats[0], json.loads(r["label_json"])["category"])
            self.assertNotEqual(issue_cats[0], issue_cats[1])

    def test_ambiguous_category_is_request(self):
        for r in self.rows:
            if r["meta"]["hard_case"] != "ambiguous":
                continue
            cat = json.loads(r["label_json"])["category"]
            req = [_level(i) for i in r["meta"]["ids"] if ".request." in i]
            issue = [_level(i) for i in r["meta"]["ids"] if ".issue." in i]
            self.assertEqual(req, [cat])
            self.assertNotEqual(issue, [cat])

    def test_mixed_last_sentiment_wins(self):
        for r in self.rows:
            if r["meta"]["hard_case"] != "mixed":
                continue
            cue_ids = [i for i in r["meta"]["ids"] if ".sentiment." in i]
            self.assertEqual(len({_level(i) for i in cue_ids}), 2)
            self.assertEqual(_level(cue_ids[-1]), json.loads(r["label_json"])["sentiment"])
            # the labelled cue must come after the opposite one in the text
            text = r["ticket_text"].lower()
            first, last = (g._transcript(g.REGISTRY[i]["text"]) for i in cue_ids)
            self.assertLess(text.index(first), text.index(last))

    def test_needs_human_iff_escalation(self):
        for r in self.rows:
            has_cue = any(".escalation." in i for i in r["meta"]["ids"])
            self.assertEqual(has_cue, json.loads(r["label_json"])["needs_human"])

    def test_typos_keep_product_names(self):
        typo_rows = [r for r in self.rows if r["meta"]["hard_case"] == "typo"]
        self.assertTrue(typo_rows)
        for r in typo_rows:
            self.assertIn(json.loads(r["label_json"])["product"], r["ticket_text"])

    def test_typos_change_text(self):
        import random
        segs = [("The quick brown fox jumps over the lazy dog again and again. " * 5, False),
                ("Orbit Laptop 14", True)]
        noisy = g.add_typos(segs, random.Random(0))
        self.assertNotEqual(noisy[0][0], segs[0][0])
        self.assertEqual(noisy[1], segs[1])

    def test_test_labels_unique(self):
        labels = [r["label_json"] for r in self.test]
        self.assertEqual(len(labels), len(set(labels)))

    def test_enums_balanced(self):
        for field, values in (("category", g.CATEGORIES), ("priority", g.PRIORITIES),
                              ("sentiment", g.SENTIMENTS)):
            counts = [sum(lab[field] == v for lab in self.labels(self.train)) for v in values]
            self.assertLessEqual(max(counts) - min(counts), 1, field)

    def test_cli_writes_files_without_skeleton(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            self.assertEqual(g.main(["--n", "20", "--out", d]), 0)
            rows = [json.loads(line) for line in (Path(d) / "test_pool.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual(set(rows[0]), {"ticket_text", "label_json", "meta"})


if __name__ == "__main__":
    unittest.main()
