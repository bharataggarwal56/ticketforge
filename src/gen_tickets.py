"""Label-first synthetic support-ticket generator (stdlib only, seeded).

Labels are sampled first, then a ticket is rendered from phrase banks so the
gold JSON is correct by construction. Every bank entry has an ID that belongs
to exactly one pool (train or test); each pool renders only from its own IDs.
A leakage check runs on every generation and exits non-zero if it fails.

Usage:
    python src/gen_tickets.py --n 50 --samples 3        # smoke run
    python src/gen_tickets.py --n 1500                  # 1,500 train / 300 test
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

CATEGORIES = ["billing", "technical", "account", "shipping", "other"]
PRIORITIES = ["low", "medium", "high", "urgent"]
SENTIMENTS = ["negative", "neutral", "positive"]
HARD_TYPES = ["ambiguous", "mixed", "multi", "typo"]
HARD_SHARE = {"train": 0.20, "test": 0.30}
MAX_SUMMARY_WORDS = 20
NGRAM = 8

# Product catalog is shared by both pools: products are slot values, not templates.
PRODUCTS = {
    "CloudSync Pro": ["technical", "account", "billing"],
    "Nimbus Router X2": ["technical", "shipping"],
    "Zenith Smartwatch": ["technical", "shipping"],
    "PayFlow Wallet": ["billing", "account"],
    "StreamMax Premium": ["billing", "account", "technical"],
    "Orbit Laptop 14": ["shipping", "technical"],
    "HomeHub Speaker": ["shipping", "technical", "other"],
    "FitTrack Band": ["shipping", "technical", "other"],
    "PhotoVault App": ["account", "technical", "billing"],
    "Gift Card": ["billing", "other"],
    "Loyalty Rewards": ["account", "other", "billing"],
}
FIRST_NAMES = ["Asha", "Ben", "Carla", "Dev", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jonas",
               "Kavya", "Liam", "Maya", "Noah", "Olu", "Priya", "Quinn", "Rosa", "Sam", "Tara"]
LAST_NAMES = ["Sharma", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Silva", "Novak", "Reyes",
              "Kowalski", "Mehta", "Brown", "Haddad", "Chen", "Fischer", "Rossi"]

# Each style: (separator around opener/signoff, separator between body sentences, transform).
STYLE_FORMAT = {
    "email": ("\n\n", " ", None),
    "chat": ("\n", "\n", None),
    "tweet": (" ", " ", None),
    "letter": ("\n\n", " ", None),
    "transcript": (" ", " ", "transcript"),
}

# Phrase banks. Issue and request entries are (text, short form used in the summary).
TEXT = {
    "train": {
        "style": {
            "email": {
                "opener": ["Hello support team,", "Hi there, this is {name}.", "Dear support,"],
                "signoff": ["Best regards,\n{name}\n{email}", "Thanks,\n{name}", "Cheers,\n{name} ({email})"],
            },
            "chat": {
                "opener": ["hi", "hey, quick question", "hello? {name} here"],
                "signoff": ["my email is {email}", "- {name}", "that's it from me"],
            },
            "tweet": {
                "opener": ["@support", "Hey @help:"],
                "signoff": ["#customerservice", "DM me at {email}"],
            },
        },
        "issue": {
            "billing": [
                ("I was charged twice for {product} this month.", "double charge"),
                ("My invoice for {product} shows the wrong amount.", "wrong invoice amount"),
                ("The renewal charge for {product} went through even though I cancelled.", "charged after cancelling"),
                ("I need a refund for {product}, the payment was taken by mistake.", "mistaken payment needing refund"),
                ("My card was declined when paying for {product}, but my bank says it is fine.", "card declined at payment"),
            ],
            "technical": [
                ("{product} keeps crashing every time I open it.", "repeated crashes"),
                ("{product} will not connect to my wifi anymore.", "wifi connection failure"),
                ("After the latest update, {product} is extremely slow.", "slowness after update"),
                ("{product} shows an error code and stops working.", "error code and failure"),
                ("The sync feature in {product} has stopped working.", "sync not working"),
            ],
            "account": [
                ("I cannot log in to my {product} account.", "login failure"),
                ("I forgot my password for {product} and the reset email never arrives.", "password reset email missing"),
                ("My {product} account got locked after a few login attempts.", "locked account"),
                ("I want to change the email address on my {product} account.", "email change request"),
                ("I think someone else accessed my {product} account.", "possible unauthorized access"),
            ],
            "shipping": [
                ("My {product} order {order} has not arrived yet.", "order not arrived"),
                ("The tracking for order {order} with {product} has not updated in days.", "tracking not updating"),
                ("{product} arrived damaged, the box was crushed.", "item arrived damaged"),
                ("I received the wrong item instead of {product}.", "wrong item received"),
                ("Order {order} for {product} went to the wrong address.", "delivery to wrong address"),
            ],
            "other": [
                ("I have a general question about {product}.", "general question"),
                ("Do you offer a student discount on {product}?", "student discount question"),
                ("I would like to give some feedback about {product}.", "product feedback"),
                ("Is {product} available in other colours?", "colour availability question"),
                ("Is there a user manual for {product} somewhere?", "user manual request"),
            ],
        },
        "request": {
            "billing": [("Please refund the charge.", "refund request"),
                        ("Please send me a corrected invoice.", "corrected invoice request")],
            "technical": [("Please help me get it working again.", "help fixing a fault"),
                          ("Please tell me how to troubleshoot it.", "troubleshooting help")],
            "account": [("Please reset my login.", "login reset request"),
                        ("Please unlock my account.", "account unlock request")],
            "shipping": [("Please send a replacement unit.", "replacement shipment request"),
                         ("Please tell me where my parcel is.", "parcel location request")],
            "other": [("Please just answer my question.", "information request"),
                      ("Please point me to the right information.", "information pointer request")],
        },
        "priority": {
            "low": ["No rush, whenever you get a chance.", "This is not urgent at all.", "Take your time with this one."],
            "medium": ["It would be good to sort this out this week.", "Please look into it in the next few days."],
            "high": ["This is blocking my work, please prioritise it.", "I need this fixed today."],
            "urgent": ["This is an emergency, everything is down right now.",
                       "URGENT: our whole team is blocked at this moment.",
                       "Production is down and we need help immediately."],
        },
        "sentiment": {
            "negative": ["I am really frustrated with this.", "This is honestly unacceptable.", "Very disappointed so far."],
            "neutral": ["Here are the details.", "Let me know what you need from me."],
            "positive": ["I really love your products otherwise!", "Your team has always been great to me.",
                         "You folks are awesome, by the way."],
        },
        "escalation": ["Please have a human agent contact me.", "I would like to speak to a real person about this.",
                       "Can a supervisor review this case?", "Please escalate this to a manager."],
        "multi": ["Also, separately:", "One more thing:", "On top of that,"],
        "ambiguous": ["Anyway, what I actually need is this.", "That said, here is my request."],
        "summary": [
            "Customer reports {short} for {product}{extra}.",
            "{Short} with {product}{extra}.",
            "Issue: {short} ({product}){extra}.",
            "User says {short} affecting {product}{extra}.",
        ],
    },
    "test": {
        "style": {
            "letter": {
                "opener": ["To whom it may concern,", "Dear Sir or Madam,"],
                "signoff": ["Yours faithfully,\n{name}\n{email}", "Respectfully,\n{name}"],
            },
            "transcript": {
                "opener": ["[voice message transcript] uh hi, this is {name} calling,",
                           "[transcribed call] hello yes it is {name},"],
                "signoff": ["um okay that is all, you can email me at {email}, bye", "right, that is everything, bye"],
            },
        },
        "issue": {
            "billing": [
                ("There is an unexpected fee on my statement relating to {product}.", "unexpected fee"),
                ("The amount debited for {product} does not match the price quoted at checkout.", "debit not matching quoted price"),
                ("I have been billed for {product} after the free trial ended without notice.", "billing after free trial"),
            ],
            "technical": [
                ("The device {product} freezes and must be restarted several times a day.", "frequent freezing"),
                ("Since the firmware upgrade, {product} fails to boot properly.", "boot failure after firmware upgrade"),
                ("Data entered into {product} disappears once the session ends.", "data loss after session"),
            ],
            "account": [
                ("The two-factor code for {product} is never received on my phone.", "two-factor code not received"),
                ("My profile on {product} was deactivated without any explanation.", "profile deactivated"),
                ("I wish to merge two separate {product} profiles into one.", "profile merge request"),
            ],
            "shipping": [
                ("The courier marked parcel {order} containing {product} as delivered, yet nothing came.", "parcel marked delivered but missing"),
                ("Part of the package for {product} was missing when opened.", "incomplete package"),
                ("Delivery of {product}, reference {order}, has been postponed repeatedly.", "repeatedly postponed delivery"),
            ],
            "other": [
                ("Could you share details on the warranty terms for {product}?", "warranty terms question"),
                ("I am curious whether {product} will be sold in physical stores.", "retail availability question"),
                ("Does {product} have any partnership programme?", "partnership programme question"),
            ],
        },
        "request": {
            "billing": [("Kindly reverse the transaction.", "transaction reversal request"),
                        ("Kindly issue a credit to my card.", "card credit request")],
            "technical": [("Kindly guide me through repairing the fault.", "repair guidance request"),
                          ("Kindly send a patch or workaround.", "patch or workaround request")],
            "account": [("Kindly restore access to my profile.", "profile access restoration"),
                        ("Kindly update my sign-in details.", "sign-in details update")],
            "shipping": [("Kindly dispatch a new one by courier.", "new dispatch request"),
                         ("Kindly confirm the delivery date.", "delivery date confirmation")],
            "other": [("Kindly provide the information requested.", "information request"),
                      ("Kindly direct me to the relevant page.", "relevant page request")],
        },
        "priority": {
            "low": ["There is no time pressure on this matter.", "A reply within the next few weeks would be fine."],
            "medium": ["A resolution in the coming days would be appreciated.", "I would like an answer before the weekend."],
            "high": ["This matter is severely affecting my daily operations.", "I require a resolution within twenty-four hours."],
            "urgent": ["Our entire business has come to a complete standstill as of now.",
                       "Immediate action is required, all services are offline."],
        },
        "sentiment": {
            "negative": ["Frankly, my patience has run out.", "The experience so far has been dreadful."],
            "neutral": ["Further information can be supplied on request.", "Awaiting your response."],
            "positive": ["Overall I remain a delighted customer.", "Your service has been wonderful over the years."],
        },
        "escalation": ["Kindly arrange for a staff member to telephone me.",
                       "I request that a senior representative handle this personally.",
                       "A callback from a member of your team is required."],
        "multi": ["In addition, a second matter:", "Furthermore, unrelated to the above,"],
        "ambiguous": ["Nonetheless, the action I require is as follows.", "Be that as it may, my request is below."],
        "summary": [
            "The customer describes {short} concerning {product}{extra}.",
            "Ticket about {short} on {product}{extra}.",
            "Reported problem: {short}, product {product}{extra}.",
        ],
    },
}


def _compile(text: dict) -> tuple[dict, dict]:
    """Give every bank entry an ID like 'tr.issue.billing.0'. Returns (banks, registry)."""
    registry: dict[str, dict] = {}
    banks: dict[str, dict] = {}

    def walk(node, path, pool):
        if isinstance(node, dict):
            return {k: walk(v, path + [k], pool) for k, v in node.items()}
        entries = []
        for i, item in enumerate(node):
            entry_id = ".".join(path + [str(i)])
            body, short = item if isinstance(item, tuple) else (item, None)
            if entry_id in registry:
                raise ValueError(f"duplicate bank id {entry_id}")
            entry = {"id": entry_id, "pool": pool, "text": body, "short": short}
            registry[entry_id] = entry
            entries.append(entry)
        return entries

    for pool, tag in (("train", "tr"), ("test", "te")):
        banks[pool] = walk(text[pool], [tag], pool)
    return banks, registry


BANKS, REGISTRY = _compile(TEXT)
STYLES = {pool: list(BANKS[pool]["style"]) for pool in BANKS}


def _segments(template: str, slots: dict) -> list[tuple[str, bool]]:
    """Split a template into (text, is_slot) pieces with slots filled."""
    out = []
    for i, part in enumerate(re.split(r"\{(\w+)\}", template)):
        if i % 2:
            out.append((slots[part], True))
        elif part:
            out.append((part, False))
    return out


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def skeleton(segs: list[tuple[str, bool]]) -> list[tuple[str, ...]]:
    """Token runs of the template text; slot values act as breaks."""
    runs, cur = [], []
    for text, is_slot in segs:
        if is_slot:
            if cur:
                runs.append(tuple(cur))
            cur = []
        else:
            cur += _tokens(text)
    if cur:
        runs.append(tuple(cur))
    return runs


def ngrams(runs: list[tuple[str, ...]], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {run[i:i + n] for run in runs for i in range(len(run) - n + 1)}


_KEYBOARD = {}
for _row in ("qwertyuiop", "asdfghjkl", "zxcvbnm"):
    for _i, _c in enumerate(_row):
        _KEYBOARD[_c] = _row[max(0, _i - 1):_i] + _row[_i + 1:_i + 2]


def add_typos(segs: list[tuple[str, bool]], rng: random.Random) -> list[tuple[str, bool]]:
    """Swap/drop/keyboard-neighbour noise on 3-8% of letters; slot values are untouched."""
    rate = rng.uniform(0.03, 0.08)
    out = []
    for text, is_slot in segs:
        if is_slot:
            out.append((text, True))
            continue
        chars, i = list(text), 0
        while i < len(chars):
            c = chars[i]
            if c.isalpha() and rng.random() < rate:
                op = rng.choice(["swap", "drop", "neighbour"])
                if op == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                    i += 1
                elif op == "drop":
                    del chars[i]
                    continue
                elif c.lower() in _KEYBOARD:
                    n = rng.choice(_KEYBOARD[c.lower()])
                    chars[i] = n.upper() if c.isupper() else n
            i += 1
        out.append(("".join(chars), False))
    return out


def _transcript(text: str) -> str:
    return re.sub(r"[.!?:;()]", "", text.lower())


def _balanced(values: list, n: int, rng: random.Random) -> list:
    out = [values[i % len(values)] for i in range(n)]
    rng.shuffle(out)
    return out


def _assign_plan(n: int, pool: str, rng: random.Random) -> list[dict]:
    cols = {
        "category": _balanced(CATEGORIES, n, rng),
        "priority": _balanced(PRIORITIES, n, rng),
        "sentiment": _balanced(SENTIMENTS, n, rng),
        "needs_human": _balanced([False, True], n, rng),
        "style": _balanced(STYLES[pool], n, rng),
    }
    hard = ["none"] * n
    order = list(range(n))
    rng.shuffle(order)
    for k in range(round(n * HARD_SHARE[pool])):
        kind = HARD_TYPES[k % len(HARD_TYPES)]
        for idx in order:
            # mixed sentiment needs a non-neutral final sentiment
            if hard[idx] == "none" and (kind != "mixed" or cols["sentiment"][idx] != "neutral"):
                hard[idx] = kind
                break
    return [{**{f: cols[f][i] for f in cols}, "hard_case": hard[i]} for i in range(n)]


def _join(sentences: list[list], sep: str) -> list[tuple[str, bool]]:
    out = []
    for i, s in enumerate(sentences):
        if i:
            out.append((sep, False))
        out += s
    return out


def render(plan: dict, pool: str, rng: random.Random, uid: str) -> dict:
    """Render one ticket for a label plan. Returns a row with an in-memory '_skeleton'."""
    bank = BANKS[pool]
    cat, prio, sent, human, hard = (plan["category"], plan["priority"], plan["sentiment"],
                                    plan["needs_human"], plan["hard_case"])
    product = rng.choice([p for p, cats in PRODUCTS.items() if cat in cats])
    first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
    slots = {
        "product": product,
        "order": f"ORD-{rng.randint(100000, 999999)}",
        "name": f"{first} {last}",
        "email": f"{first}.{last}{rng.randint(1, 99)}@example.com".lower(),
    }
    ids: list[str] = []

    def pick(entries):
        e = rng.choice(entries)
        ids.append(e["id"])
        return e

    def seg(e, **extra):
        return _segments(e["text"], {**slots, **extra})

    body, short2 = [], None
    if hard == "ambiguous":
        # Background issue from another category; the explicit request decides the label.
        other = rng.choice([c for c in PRODUCTS[product] if c != cat])
        issue, conn, req = pick(bank["issue"][other]), pick(bank["ambiguous"]), pick(bank["request"][cat])
        body += [seg(issue), seg(conn), seg(req), seg(pick(bank["priority"][prio]))]
        short = req["short"]
    elif hard == "multi":
        # First issue sets category and product; priority is the max of both cues.
        other = rng.choice([c for c in CATEGORIES if c != cat])
        p2 = product if other in PRODUCTS[product] else rng.choice(
            [p for p, cats in PRODUCTS.items() if other in cats])
        low = PRIORITIES[rng.randint(0, PRIORITIES.index(prio))]
        lv1, lv2 = (prio, low) if rng.random() < 0.5 else (low, prio)
        issue, issue2 = pick(bank["issue"][cat]), None
        body += [seg(issue), seg(pick(bank["priority"][lv1])), seg(pick(bank["multi"]))]
        issue2 = pick(bank["issue"][other])
        body += [seg(issue2, product=p2), seg(pick(bank["priority"][lv2]))]
        short, short2 = issue["short"], issue2["short"]
    else:
        issue = pick(bank["issue"][cat])
        body.append(seg(issue))
        if rng.random() < 0.5:
            body.append(seg(pick(bank["request"][cat])))
        body.append(seg(pick(bank["priority"][prio])))
        short = issue["short"]

    if human:
        body.append(seg(pick(bank["escalation"])))
    if hard == "mixed":
        # Opposite sentiment first; the final sentiment statement is the label.
        opp = "positive" if sent == "negative" else "negative"
        body.insert(0, seg(pick(bank["sentiment"][opp])))
        body.append(seg(pick(bank["sentiment"][sent])))
    else:
        s = seg(pick(bank["sentiment"][sent]))
        if rng.random() < 0.4:
            body.insert(0, s)
        else:
            body.append(s)

    style = plan["style"]
    outer, inner, transform = STYLE_FORMAT[style]
    opener = seg(pick(bank["style"][style]["opener"]))
    signoff = seg(pick(bank["style"][style]["signoff"]))
    segs = opener + [(outer, False)] + _join(body, inner) + [(outer, False)] + signoff
    if transform == "transcript":
        segs = [(t if is_slot else _transcript(t), is_slot) for t, is_slot in segs]
    skel = skeleton(segs)
    if hard == "typo":
        segs = add_typos(segs, rng)

    extra = (f", plus {short2}" if short2 else "") + ("; wants a human agent" if human else "")
    tmpl = pick(bank["summary"])
    summary = tmpl["text"].format(short=short, Short=short[0].upper() + short[1:],
                                  product=product, extra=extra)
    assert len(summary.split()) <= MAX_SUMMARY_WORDS, summary

    label = {"category": cat, "priority": prio, "product": product, "sentiment": sent,
             "summary": summary, "needs_human": human}
    return {
        "ticket_text": "".join(t for t, _ in segs).strip(),
        "label_json": json.dumps(label, ensure_ascii=False),
        "meta": {"uid": uid, "pool": pool, "style": style, "hard_case": hard, "ids": ids},
        "_skeleton": skel,
    }


def generate(n: int, pool: str, seed: int, unique_labels: bool = False) -> list[dict]:
    rng = random.Random(f"{seed}-{pool}")
    rows, seen = [], set()
    for i, plan in enumerate(_assign_plan(n, pool, rng)):
        for _ in range(100):
            row = render(plan, pool, rng, f"{pool}-{i:05d}")
            if not unique_labels or row["label_json"] not in seen:
                break
        else:
            raise RuntimeError(f"could not render a unique label for {pool}-{i:05d}")
        seen.add(row["label_json"])
        rows.append(row)
    return rows


def check_leakage(train: list[dict], test: list[dict], n: int = NGRAM) -> list[str]:
    """Return a list of problems; empty means no leakage."""
    problems = []
    pools = {}
    for entry in REGISTRY.values():
        pools.setdefault(entry["text"], set()).add(entry["pool"])
    shared = sorted(t for t, p in pools.items() if len(p) > 1)
    if shared:
        problems.append(f"{len(shared)} bank texts in both pools, e.g. {shared[:3]}")
    for pool, rows in (("train", train), ("test", test)):
        for r in rows:
            bad = [i for i in r["meta"]["ids"] if REGISTRY.get(i, {}).get("pool") != pool]
            if bad:
                problems.append(f"{r['meta']['uid']} uses ids outside the {pool} pool: {bad[:3]}")
    train_grams = set()
    for r in train:
        train_grams |= ngrams(r["_skeleton"], n)
    leaks = set()
    for r in test:
        leaks |= ngrams(r["_skeleton"], n) & train_grams
    if leaks:
        problems.append(f"{len(leaks)} shared {n}-grams, e.g. {[' '.join(g) for g in sorted(leaks)[:3]]}")
    return problems


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")


def print_stats(name: str, rows: list[dict]) -> None:
    labels = [json.loads(r["label_json"]) for r in rows]
    print(f"\n== {name}: {len(rows)} rows")
    for field in ("category", "priority", "sentiment", "needs_human"):
        print(f"  {field:12s} {dict(Counter(lab[field] for lab in labels))}")
    print(f"  {'hard_case':12s} {dict(Counter(r['meta']['hard_case'] for r in rows))}")
    print(f"  {'style':12s} {dict(Counter(r['meta']['style'] for r in rows))}")
    words = [len(r["ticket_text"].split()) for r in rows]
    sw = [len(lab["summary"].split()) for lab in labels]
    print(f"  ticket words min/mean/max {min(words)}/{sum(words) / len(words):.1f}/{max(words)}")
    print(f"  summary words min/mean/max {min(sw)}/{sum(sw) / len(sw):.1f}/{max(sw)}")
    print(f"  distinct products {len({lab['product'] for lab in labels})}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1500, help="train rows; test gets n/5")
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--samples", type=int, default=0, help="print this many sample rows per pool")
    args = ap.parse_args(argv)

    n_test = args.n_test if args.n_test is not None else max(1, args.n // 5)
    train = generate(args.n, "train", args.seed)
    test = generate(n_test, "test", args.seed, unique_labels=True)

    problems = check_leakage(train, test)
    for name, rows in (("train", train), ("test", test)):
        print_stats(name, rows)
    if problems:
        print("\nLEAKAGE CHECK FAILED:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    print(f"\nLeakage check passed (ids disjoint, no shared {NGRAM}-grams).")

    write_jsonl(train, args.out / "train_pool.jsonl")
    write_jsonl(test, args.out / "test_pool.jsonl")
    print(f"Wrote {len(train)} train and {len(test)} test rows to {args.out}/")

    for name, rows in (("train", train), ("test", test)):
        for r in rows[:args.samples]:
            print(f"\n--- {name} {r['meta']['uid']} style={r['meta']['style']} hard={r['meta']['hard_case']}")
            print(r["ticket_text"])
            print(">>", r["label_json"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
