import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import auditkit as ak  # noqa: E402
import scorers as sc  # noqa: E402
from prompt import build_messages  # noqa: E402

SCHEMA = json.loads((ROOT / "configs" / "schema.json").read_text())
CURATED = ROOT / "data" / "curated" / "test_n50"
GOLD = {
    "category": "billing", "priority": "high", "product": "Zenith Smartwatch",
    "sentiment": "negative", "summary": "Customer was charged twice for the watch.", "needs_human": False,
}
V = Draft202012Validator(SCHEMA)


def score(pred, gold=GOLD):
    out = pred if isinstance(pred, str) else json.dumps(pred)
    return sc.score_output(out, json.dumps(gold), V)


class ScoreOutputTest(unittest.TestCase):
    def test_perfect(self):
        self.assertTrue(all(v == 1.0 for v in score(GOLD).values()))

    def test_non_json_and_fenced_json_score_zero(self):
        for bad in ("not json", "```json\n" + json.dumps(GOLD) + "\n```", "[1, 2]", ""):
            self.assertTrue(all(v == 0.0 for v in score(bad).values()), bad)

    def test_wrong_enum_breaks_schema_and_that_field(self):
        s = score({**GOLD, "category": "refund"})
        self.assertEqual((s["schema_valid"], s["category_acc"], s["exact_match_fields"]), (0.0, 0.0, 0.0))
        self.assertEqual(s["priority_acc"], 1.0)

    def test_extra_key_is_schema_invalid_but_fields_still_match(self):
        s = score({**GOLD, "extra": 1})
        self.assertEqual((s["schema_valid"], s["exact_match_fields"]), (0.0, 1.0))

    def test_missing_key(self):
        pred = {k: v for k, v in GOLD.items() if k != "priority"}
        s = score(pred)
        self.assertEqual((s["schema_valid"], s["priority_acc"], s["category_acc"]), (0.0, 0.0, 1.0))

    def test_product_is_normalized(self):
        self.assertEqual(score({**GOLD, "product": "  zenith   SMARTWATCH "})["product_acc"], 1.0)
        self.assertEqual(score({**GOLD, "product": "Zenith"})["product_acc"], 0.0)

    def test_needs_human_must_be_a_bool(self):
        gold = {**GOLD, "needs_human": True}
        self.assertEqual(score({**gold, "needs_human": 1}, gold)["needs_human_acc"], 0.0)
        self.assertEqual(score({**gold, "needs_human": False}, gold)["needs_human_acc"], 0.0)

    def test_summary_length_and_f1(self):
        long = " ".join(["word"] * 21)
        s = score({**GOLD, "summary": long})
        self.assertEqual((s["summary_len_ok"], s["schema_valid"], s["exact_match_fields"]), (0.0, 0.0, 1.0))
        s = score({**GOLD, "summary": "Charged twice for the watch."})
        self.assertEqual((s["summary_len_ok"], s["exact_match_strict"]), (1.0, 0.0))
        self.assertTrue(0.0 < s["summary_f1"] < 1.0)
        self.assertEqual(score({**GOLD, "summary": "  CUSTOMER was charged twice for the   watch. "})["exact_match_strict"], 1.0)

    def test_strip_pattern(self):
        self.assertNotIn("pattern", sc.strip_pattern(SCHEMA)["properties"]["summary"])
        self.assertIn("pattern", SCHEMA["properties"]["summary"])


@unittest.skipUnless(CURATED.exists(), "needs data/curated/test_n50")
class AuditKitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = sc.load_samples(CURATED / "test" / "sft_messages.jsonl", CURATED / "test_meta.jsonl")
        cls.gold = {s.input: s.target for s in cls.samples}
        cls.requests, cls.headers = [], []
        gold, requests, headers = cls.gold, cls.requests, cls.headers

        class Stub(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, payload):
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send({"object": "list", "data": [{"id": "stub", "object": "model"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                headers.append(dict(self.headers))
                reply = gold[body["messages"][1]["content"]]
                self._send({
                    "id": "x", "object": "chat.completion", "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": reply}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                })

        cls.server = HTTPServer(("127.0.0.1", 0), Stub)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}/v1"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.requests.clear()
        self.headers.clear()

    def run_stub(self, **extra):
        return ak.evaluate(
            self.samples, model="api:stub", api_base=self.base, api_key="EMPTY",
            scorers=sc.make_scorers(SCHEMA), adapter=sc.prompt_adapter(),
            config=ak.RunConfig(temperature=0, max_tokens=256), **extra,
        )

    def test_samples_carry_hard_case(self):
        self.assertEqual(len(self.samples), 10)
        self.assertTrue(all("hard_case" in s.metadata for s in self.samples))

    def test_fake_callable_gold_scores_one_and_garbage_scores_zero(self):
        scorers = sc.make_scorers(SCHEMA)
        good = ak.evaluate(self.samples, model=lambda prompts: [self.gold[s.input] for s in self.samples],
                           scorers=scorers, adapter=sc.prompt_adapter())
        self.assertEqual(set(good.headline), set(sc.METRICS))
        self.assertTrue(all(v == 1.0 for v in good.headline.values()), good.headline)
        bad = ak.evaluate(self.samples, model=lambda prompts: ["nope"] * len(prompts),
                          scorers=scorers, adapter=sc.prompt_adapter())
        self.assertTrue(all(v == 0.0 for v in bad.headline.values()), bad.headline)

    def test_response_format_is_forwarded_with_build_messages_and_settings(self):
        rf = {"type": "json_schema", "json_schema": {"name": "ticket", "schema": SCHEMA}}
        result = self.run_stub(response_format=rf)
        self.assertEqual(len(self.requests), len(self.samples))
        for body in self.requests:
            self.assertEqual(body["response_format"], rf)
            ticket = body["messages"][1]["content"]
            self.assertEqual(body["messages"], build_messages(ticket))
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["max_tokens"], 256)
        self.assertTrue(all(h.get("Authorization") == "Bearer EMPTY" for h in self.headers))
        self.assertTrue(all(v == 1.0 for v in result.headline.values()), result.headline)

    def test_no_response_format_when_not_given(self):
        self.run_stub()
        self.assertEqual(len(self.requests), len(self.samples))
        self.assertTrue(all("response_format" not in b for b in self.requests))

    def test_slice_by_hard_case(self):
        outputs = [s.target for s in self.samples]
        sl = sc.slice_by_hard_case(outputs, self.samples, SCHEMA)
        self.assertEqual(sl["all"]["n"], len(self.samples))
        self.assertEqual(sum(v["n"] for k, v in sl.items() if k != "all"), len(self.samples))
        self.assertTrue(all(v["exact_match_fields"] == 1.0 for v in sl.values()))


if __name__ == "__main__":
    unittest.main()
