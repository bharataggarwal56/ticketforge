# TicketForge plan

Decisions and pipeline for turning support tickets into JSON that is valid against `configs/schema.json`.
The tool notes are in `docs/tools.md`. Where this plan and the official docs disagree, trust the docs and update this file.

## Decisions

| Question (from tools.md) | Decision |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct`, fp16 with LoRA, sized to fit a Colab T4. Qwen2.5 has no thinking mode, so `enable_thinking` is not needed |
| Gold labels | Label-first generation: sample the label fields, then render a ticket that matches them, so labels are correct by construction |
| Environments | Two Colab notebooks. A trains with CuratorKIT and AlignTune; B serves with SGLang in a uv Python 3.12 venv and evaluates with AuditKIT |
| AuditKIT `api:` and `response_format` | The source says it is forwarded (see "AuditKIT scorers"). This still has to be checked with a stub server |

## Output schema

`configs/schema.json` (JSON Schema 2020-12) lists all six fields as required and sets `additionalProperties: false`.

- `category`, `priority`, and `sentiment` are enums. `needs_human` is a boolean.
- `product` is a free string of 1–60 characters. The generator draws from a fixed catalog, and scoring normalizes case and whitespace.
- JSON Schema cannot count words, so `summary` uses a `pattern` that allows 1–20 whitespace-separated tokens, capped at 160 characters.
- Gold JSON is serialized in schema property order.

## 1. Data generation, label-first (`src/gen_tickets.py`, stdlib only, seeded)

1. **Labels:** sample `category`, `priority`, `sentiment`, `needs_human`, and `product`, where the product is drawn from catalog items valid for that category. Sampling is stratified so every enum value is well covered.
2. **Ticket:** build it from an opener, an issue phrase (category × product), a priority cue ("whenever you can" → low, "production is down" → urgent), a sentiment cue, an escalation cue when `needs_human` is true (legal threat, refund over limit, "let me talk to a person"), and a sign-off. Fake names, emails, and order IDs are added so the CuratorKIT PII step has something to work on.
3. **Summary:** filled from a summary template using the same slots. The generator asserts it is at most 20 words.
4. **Held-out pools:** every template, phrase bank, and style has an ID. The banks are kept small but varied: about 5 train and 3 test issue phrases per category, and 2–4 variants per cue bank. Styles: email, live chat, and short note/tweet (train), plus formal letter and voice-to-text transcript (test). Each ID belongs to either the train pool or the test pool, and each pool's tickets use only that pool's IDs. A leakage check asserts that no ID is in both pools, that every row uses only its own pool's IDs, and that train and test skeletons (text before typo noise, split at slot values) share no 8-gram. The generator runs this check on every run and exits non-zero if it fails.
5. **Hard cases (about 20% of train and about 30% of test):** each pool builds its hard cases from its own pool's phrases only. Train includes them so the model learns the labeling rules below. The test slices measure whether those rules generalize to unseen wording.
   - *Ambiguous:* cues point to two categories. The explicitly requested action decides the category.
   - *Mixed sentiment:* praise and complaint in one ticket. The sentiment of the final request wins.
   - *Multiple issues:* the first-stated issue sets the category, and priority is the highest across the issues.
   - *Typos:* seeded character noise (swap, drop, keyboard neighbor) on 3–8% of characters. Product names stay recognizable.

   Each row has `meta.hard_case` ∈ {none, ambiguous, mixed, multi, typo} so results can be sliced by case type.
6. **Outputs:** `data/raw/train_pool.jsonl` (~1,500 rows) and `data/raw/test_pool.jsonl` (~300 rows). `--n` sets the number of train rows, and test gets `n/5`. Columns are `ticket_text`, `label_json` (canonical JSON string), and `meta` (`uid`, `pool`, `style`, `hard_case`, `ids`). Test `label_json` values are unique so that `meta` can be rejoined after CuratorKIT, whose export drops extra columns. `data/raw` is gitignored. Laptop smoke tests use `--n 50`. Unit tests are in `tests/test_gen_tickets.py` (stdlib `unittest`).

## 2. CuratorKIT (`src/curate.py`, laptop)

- `field_mapping={"ticket_text": "instruction", "label_json": "output"}`. The direction is {source column: DataSample field}.
- **Run 1, train pool:** `dedup="minhash"`, `clean=True`, `clean_fields=["instruction"]` (so the gold JSON is never modified), `pii_pseudonymize=True` with `pii_spacy_model="en_core_web_sm"`, `export_formats=["messages"]`, `output_split={"train": 0.9, "validation": 0.1}`, and a fresh `output_dir` per run (e.g. `data/curated/train_<date>`). Set `min_tokens` low enough for tweet-style tickets.
- **Run 2, test pool:** same cleaning and PII settings, `output_split={"test": 1.0}`, separate `output_dir`. Keeping test in its own run stops CuratorKIT's random split from mixing test into train.
- The instruction holds only the ticket text, so a shared prompt cannot make minhash collapse samples. The system prompt comes from `src/prompt.py` (`SYSTEM_PROMPT`, `build_messages(ticket)`), which eval also uses. A `preprocessing_fn` adds it as a `system` column, and CuratorKIT writes it as the leading system turn.
- Checked in the CuratorKIT source (main, 2026-10-24; the config reference page omits some of this):
  - `output_split` takes any dict whose values sum to 1.0. Samples are shuffled with `output_split_seed`, cut in key order (the last key gets the remainder), and each slice is written to `output_dir/<key>/`. So `{"test": 1.0}` works. Use `validation`, not `val`, as the HF split name.
  - The `messages` exporter (`sft_messages.jsonl`) adds a leading system turn when `metadata["system_prompt"]` is set. The connector fills that from a raw column named `system`, `system_prompt`, or `system_message`.
  - The export keeps only `messages`, so `curate.py` rejoins test `meta` on `label_json` into `test_meta.jsonl`.
- `curate.py` post-checks: every row is system/user/assistant, the system turn equals `SYSTEM_PROMPT`, the assistant turn is schema-valid and equals a raw `label_json`, and product names survive PII pseudonymization.
- After each run, read `rejected.jsonl` and `manifest.json` and record the counts in the README.

## 3. Baseline eval (Notebook B)

Serve the base model and evaluate the test set two ways: **unconstrained** (prompt only) and **constrained** (`response_format` with `json_schema`). Both use the same prompt, `temperature=0`, and `max_tokens=256`.

## 4. Training (Notebook A, Colab T4)

- AlignTune `create_sft_trainer` with `model_name="Qwen/Qwen2.5-1.5B-Instruct"`, `backend="trl"` (switch to unsloth only if it works on a T4), and `task_type="chat_completion"`.
- LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o, plus gate/up/down if VRAM allows.
- `max_seq_length=512`, 3 epochs, lr 2e-4, batch 4, with grad accumulation as needed.
- **Use fp16, not bf16** (the T4 has no bf16). Look up the exact precision argument in the AlignTune docs or source before writing it.
- Run `aligntune advise estimate` on the laptop first to check VRAM.
- Merge with `aligntune.utils.load_finetuned_model` and `save_pretrained` to `outputs/merged`. Push the adapter and the merged model to HF Hub.

## 5. Final eval (Notebook B)

Serve the merged tuned model and run the same two modes. Results:
- a 2×2 table: base/tuned × free/constrained
- `ak.compare(base, tuned)`
- slices by `hard_case`

Measure performance with `python3 -m sglang.bench_serving --backend sglang-oai-chat` and AuditKIT `result.perf`. Write results to `results/*.json` and the README table.

## Which machine runs what

| Where | Runs |
|---|---|
| Laptop (`.venv`, CPU, no CUDA) | Generator, CuratorKIT runs, schema validation, AuditKIT scorers with a fake model callable, `aligntune advise estimate` |
| Notebook A, `notebooks/A_train.ipynb` (Colab T4, default runtime) | `pip install aligntune` (installs CuratorKIT too); pull curated data from HF; AlignTune SFT; val loss; merge; push to HF |
| Notebook B, `notebooks/B_serve_eval.ipynb` (Colab T4, fresh runtime) | SGLang in a uv 3.12 venv; baseline and tuned serving; AuditKIT eval; bench_serving; results JSON |

### Notebook B: SGLang setup (from Colab testing)

- Install SGLang in a uv Python 3.12 venv at `/content/sgl`:
  ```bash
  uv venv --python 3.12 /content/sgl
  uv pip install --python /content/sgl/bin/python --prerelease=allow sglang
  uv pip install --python /content/sgl/bin/python ninja
  ```
- Put the venv's bin directory on PATH: `os.environ["PATH"] = "/content/sgl/bin:" + os.environ["PATH"]`.
- **ninja must be installed in the venv.** FlashInfer JIT-compiles kernels at startup and fails with "No such file: ninja" without it.
- On a T4, launch with `--dtype float16 --attention-backend triton --sampling-backend pytorch`. Launch in the background with output to a log file:
  ```bash
  nohup /content/sgl/bin/python -m sglang.launch_server --model-path MODEL --port 30000 \
    --dtype float16 --attention-backend triton --sampling-backend pytorch \
    > /content/sglang.log 2>&1 &
  ```
  Poll `/health`, or wait for "The server is fired up and ready to roll!" in the log.
- If startup also fails on **nvcc**, fall back to `sglang==0.5.19`, the last release with CUDA 12 support.
- AuditKIT (`auditkit[requests]`, `jsonschema`) runs in the notebook's default Python and reaches SGLang only over HTTP.

## How files move

- **Laptop → GitHub:** code, configs, docs, notebooks, and `results/`. Never `.env`, `data/raw`, or weights.
- **Laptop → HF Hub:** the curated train/val and test folders go to a private dataset repo, `bharataggarwal56/ticketforge-data`, via `huggingface_hub`. The token comes from `.env` or the CLI login.
- **Notebook A:** reads the HF dataset. Writes the adapter to `…/ticketforge-qwen2.5-1.5b-lora` and the merged model to `…/ticketforge-qwen2.5-1.5b-merged`. Checkpoints are pushed every epoch.
- **Notebook B:** pulls the model and test set from HF. Results JSON is either downloaded and committed from the laptop, or pushed from Colab using a token in Colab secrets.

## AuditKIT scorers (`src/scorers.py`)

- `parse(output)` runs `json.loads(output.strip())` with no repair. If it fails, every metric scores 0.
- `schema_valid`: `jsonschema.Draft202012Validator(schema)` on the parsed object.
- Per-field accuracy: `category_acc`, `priority_acc`, `product_acc` (normalized), `sentiment_acc`, `needs_human_acc`.
- Summary: `summary_len_ok` (at most 20 words) and `summary_f1` (token F1 against gold).
- **`exact_match_fields`** is the headline exact-match metric. It compares the parsed dicts on the five structured fields, so key order and whitespace don't matter. The summary is free text and can't be expected to match exactly on held-out templates.
- `exact_match_strict`: all six fields, with the summary normalized. Reported for completeness.
- The built-in string `exact_match` is not used.

### Constrained decoding through `api:`

What the AuditKIT source shows (main branch, read 2026-10-24):

1. `ak.evaluate(..., **opts)` calls `AutoModel.resolve(model, **opts)`, which returns `APIModel(model=..., name=spec, **opts)` for `api:` specs.
2. `APIModel.__init__` calls `reject_generation_kwargs`. It rejects only `_RESERVED_GENERATION_KWARGS`: temperature, top_p, top_k, max_tokens, max_new_tokens, stop_sequences, presence_penalty, frequency_penalty, num_completions, seed. These are set through `RunConfig`.
3. `split_session_kwargs` sends `timeout`, `verify`, `proxies`, and `cert` to the HTTP session. Every other kwarg goes into `_extra_kwargs`, which `body.update(...)` merges into the request body for both `/chat/completions` and `/completions`.

So this should send `response_format` to SGLang:

```python
ak.evaluate(samples, model="api:MODEL", api_base="http://localhost:30000/v1", api_key="EMPTY",
            response_format={"type": "json_schema", "json_schema": {"name": "ticket", "schema": schema}},
            scorers=[...])
```

**Not yet run.** Check it on the laptop before relying on it: point `api_base` at a local `http.server` stub that records the JSON body, then assert that `response_format` is in it. Pin the AuditKIT version that passes this check. If it fails, use a callable that wraps `openai.Client` with `response_format` (risk R1).

## Risks and fallbacks

| # | Risk | Fallback |
|---|---|---|
| R1 | The PyPI AuditKIT release differs from main and drops or rejects `response_format` | Pin the version and verify with the stub. If it fails, pass a callable that wraps `openai.Client(...).chat.completions.create(response_format=...)` and time latency ourselves; bench_serving still gives latency and throughput |
| R2 | SGLang won't start on the T4 (driver older than CUDA 13, missing ninja, nvcc) | Install ninja in the venv with its bin on PATH, then try the triton/pytorch backends, then `sglang==0.5.19`, then a Kaggle T4 runtime |
| R3 | xgrammar rejects the `summary` `pattern` or is slow with it | Serve with a copy of the schema without the pattern (`configs/schema_serving.json`). The full schema in `schema_valid` still enforces word count |
| R4 | The model overfits templates and does poorly on held-out styles | Add more phrase banks and compositional slots, and add noise to train. Optionally paraphrase **train only** with an LLM, keeping a paraphrase only if a rule check confirms its labels |
| R5 | Hard-case gold labels are debatable | Apply the deterministic rules in section 1 in both pools (train ~20%, test ~30%, each from its own pool's phrases). Report hard-case test slices separately |
| R6 | Minhash drops many templated near-duplicates, or `min_tokens` rejects short tickets | Inspect `rejected.jsonl`. Raise the minhash threshold or use `dedup="exact"`, and lower `min_tokens` |
| R7 | PII pseudonymization changes product names or order IDs | Diff a sample. If products are affected, restrict hygiene to names and emails |
| R8 | AlignTune arguments (fp16 flag, combined kwargs) differ from the docs, or the T4 runs out of memory | Read the AlignTune docs and source first. If OOM: batch 2 with grad accumulation 8, gradient checkpointing, `quantization={"load_in_4bit": True}`, `max_seq_length=384` |
| R9 | Colab disconnects or session limits | Push checkpoints to HF every epoch, and make the notebooks re-runnable from HF state |
| R10 | Dependency conflicts | One concern per notebook. A uses a fresh default runtime with only aligntune. B runs SGLang in its venv and AuditKIT in the default Python |
| R11 | The base model wraps its JSON in ``` fences, which lowers baseline schema-valid | Intended: the strict metric shows what fine-tuning and constrained decoding gain. A lenient-parse number can be reported alongside |
| R12 | Prompt or chat template differs between training and serving | Use one `build_messages()` everywhere, and let SGLang use the model's own chat template. Sanity-check one sample's token IDs under both |
