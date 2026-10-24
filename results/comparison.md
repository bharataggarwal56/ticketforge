## 2x2: base/tuned x plain/constrained (test split, n=300)

| model / mode | schema-valid | exact match | category | priority | product | sentiment | needs_human | summary F1 | mean latency ms | req/s | out tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base / plain | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4824 | 3.25 | 169 |
| base / constrained | 0.973 | 0.027 | 0.660 | 0.263 | 0.987 | 0.537 | 0.557 | 0.245 | 2734 | 5.71 | 297 |
| tuned / plain | 1.000 | 0.533 | 0.937 | 0.697 | 1.000 | 0.937 | 0.840 | 0.423 | 4230 | 3.70 | 192 |
| tuned / constrained | 1.000 | 0.533 | 0.937 | 0.700 | 1.000 | 0.937 | 0.840 | 0.423 | 2738 | 5.74 | 298 |

Latency and throughput are from `sglang.bench_serving` (random prompts, 310 in / 52 out tokens, concurrency 16);
the constrained rows send `response_format` with the schema.

## exact match (5 structured fields) by hard_case

| model / mode | all | none | typo | ambiguous | mixed | multi |
|---|---|---|---|---|---|---|
| base / plain | 0.000 (n=300) | 0.000 (n=210) | 0.000 (n=22) | 0.000 (n=23) | 0.000 (n=23) | 0.000 (n=22) |
| base / constrained | 0.027 (n=300) | 0.033 (n=210) | 0.000 (n=22) | 0.000 (n=23) | 0.000 (n=23) | 0.045 (n=22) |
| tuned / plain | 0.533 (n=300) | 0.548 (n=210) | 0.545 (n=22) | 0.565 (n=23) | 0.435 (n=23) | 0.455 (n=22) |
| tuned / constrained | 0.533 (n=300) | 0.548 (n=210) | 0.545 (n=22) | 0.565 (n=23) | 0.435 (n=23) | 0.455 (n=22) |
