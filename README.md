# TicketForge
Fine-tuning a small open LLM to turn messy support tickets into schema-valid JSON.
Served and benchmarked with SGLang. Built for Syntax Error 2026 (Local LLM Track).

## Data

Synthetic, label-first tickets from `src/gen_tickets.py` (seed 42): labels are sampled first and the ticket is rendered to match, so gold JSON is correct by construction. Train and test use disjoint phrase banks and styles; a leakage check (no shared bank IDs, no shared 8-grams) runs on every generation. Design: `docs/plan.md` §1.

```bash
python src/gen_tickets.py --n 1500          # -> data/raw/{train,test}_pool.jsonl
python -m unittest discover tests
```

| | Train pool | Test pool |
|---|---|---|
| Rows | 1,500 | 300 |
| Styles | email, chat, tweet (500 each) | formal letter, voice transcript (150 each) |
| category / priority / sentiment / needs_human | balanced (300 / 375 / 500 / 750 per value) | balanced (60 / 75 / 100 / 150 per value) |
| Hard cases | 300 (20%): 75 each ambiguous, mixed, multi, typo | 90 (30%): 23 ambiguous, 23 mixed, 22 multi, 22 typo |
| Ticket words (min / mean / max) | 21 / 38.0 / 72 | 31 / 50.3 / 83 |
| Summary words (min / mean / max) | 5 / 9.3 / 18 | 7 / 11.2 / 20 |
| Leakage check | passed | |
