# Tool notes

Notes on the four required tools, read from their docs and GitHub READMEs on 2026-10-24.
Code below is copied from those sources (only the model and dataset names change); none of it has been run in this repo yet.
If something here disagrees with the official docs, trust the docs and update this file.

| Stage | Tool | Role in TicketForge |
|---|---|---|
| 1 | CuratorKIT | Clean, dedup, scrub PII from tickets; export train/val/test in chat `messages` format |
| 2 | AlignTune | LoRA SFT of a small model on the curated `messages` data |
| — | SGLang | Serve the tuned model with JSON-schema constrained decoding; benchmark latency/throughput |
| 3 | AuditKIT | Score the served model: schema-valid rate, per-field accuracy, exact match, latency |

## Environments: keep them separate

The pinned dependencies conflict, so use one venv per tool on the GPU box:

- **SGLang** pins torch and transformers exactly (AuditKIT's SGLang doc lists sglang 0.5.20 → torch==2.13.0, transformers==5.12.1). It runs on Linux only, Python 3.10–3.12. Current Colab is Python 3.13, so build a 3.12 venv with uv.
- **AlignTune** pins `transformers==5.14.1` and `trl==1.7.1`, and installs CuratorKIT as a dependency.
- **AuditKIT** talks to SGLang over HTTP. Its env needs only `auditkit[requests]` and never needs sglang installed.

## Hardware summary

| Tool | Laptop (Intel Arc iGPU, no CUDA) | Colab/Kaggle GPU |
|---|---|---|
| CuratorKIT | Yes. Core (clean/dedup) is pure Python. Hygiene extra pulls torch/spaCy models but runs on CPU. | Not needed unless generating with a local LLM |
| AlignTune | Tiny smoke tests only. Docs list a CUDA GPU as "recommended". Without CUDA, Unsloth is skipped and it falls back to TRL. | Needed for real training |
| SGLang | **No.** NVIDIA path needs CUDA 13. XPU path is built from source and optimized for Arc **B-series** discrete GPUs (iGPUs not mentioned). CPU path targets Xeon with AMX. | Yes. FlashInfer needs sm75+ (T4 or newer). |
| AuditKIT | Yes. Core is stdlib-only, so scorers can be tested on fixed outputs with a callable model. | Only to reach the SGLang server |

---

## SGLang

Docs: https://docs.sglang.io/ · Repo: https://github.com/sgl-project/sglang

**What it does here.** It serves the fine-tuned model through an OpenAI-compatible API and constrains every output to `configs/schema.json` (xgrammar backend by default). Its benchmark tools give the latency and throughput numbers.

### Install

```bash
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install --prerelease=allow sglang
```

`--prerelease=allow` matters: without it, uv older than 0.12.0 silently installs the old 0.5.9.
Docker alternative: `docker pull lmsysorg/sglang:latest`. SGLang 0.5.19 was the last release with CUDA 12 support.

### Hardware

- NVIDIA with a CUDA 13-compatible driver. AuditKIT's notes say the driver must be ≥ 580.
- FlashInfer (default attention) needs sm75 or newer, so a T4 works. A Kaggle P100 (sm60) is below that.
- **T4/V100 have no bf16 support:** serve with `--dtype float16`.
- If FlashInfer fails to build or load, use `--attention-backend triton --sampling-backend pytorch`.
- On Hopper/Blackwell with bf16/fp16 models, set `SGLANG_ENABLE_JIT_DEEPGEMM=0` if startup fails with an nvcc version error.

### Launch

```bash
sglang serve MODEL_PATH --host 0.0.0.0 --port 30000
# older form, used throughout the docs:
python -m sglang.launch_server --model-path MODEL_PATH --port 30000
```

Wait for "The server is fired up and ready to roll!".
To serve a LoRA adapter without merging it, add `--lora-paths NAME=PATH`. Requests then pick it with `model="BASE:NAME"` on the OpenAI API, or `lora_path` on native `/generate`.

### Minimal example: JSON-schema constrained output (OpenAI API)

```python
import json, openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")
schema = json.load(open("configs/schema.json"))

resp = client.chat.completions.create(
    model="MODEL_PATH",
    messages=[{"role": "user", "content": "<ticket text + format instructions>"}],
    temperature=0,
    max_tokens=256,
    response_format={"type": "json_schema", "json_schema": {"name": "ticket", "schema": schema}},
)
print(resp.choices[0].message.content)
```

You can send only one constraint per request: `json_schema`, `regex`, or `ebnf`. The docs still advise describing the JSON format in the prompt as well.

Offline (no HTTP server) with the same constraint:

```python
import json, sglang as sgl

llm = sgl.Engine(model_path="MODEL_PATH")
params = {"temperature": 0, "json_schema": json.dumps(schema)}
outputs = llm.generate(prompts, params)  # output["text"] per prompt
```

### Benchmarking

```bash
# online serving: throughput, TTFT, ITL, end-to-end latency
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
  --num-prompts 1000 --model MODEL_PATH
# smaller run on random data
python3 -m sglang.bench_serving --backend sglang --max-concurrency 16 --num-prompts 80 \
  --random-input-len 256 --random-output-len 32 --dataset-name random
# max offline throughput, no HTTP
python3 -m sglang.benchmark.offline_throughput --model-path MODEL_PATH --num-prompts 10
```

Use `--backend sglang-oai-chat` to benchmark `/v1/chat/completions` instead of native `/generate`.

---

## CuratorKIT (Stage 1)

Docs: https://curatorkit.lexsi.ai/latest/ · Repo: https://github.com/Lexsi-Labs/CuratorKIT · License: LSAL v1.1 (non-commercial)

**What it does here.** It ingests raw tickets (JSONL/CSV) and maps their columns with `field_mapping`. It cleans the text, dedups it (exact or minhash), and runs hygiene gates; PII pseudonymization matters because tickets contain names and emails. It then exports `sft_messages.jsonl` with train/val/test splits. Each run writes `manifest.json`, `rejected.jsonl`, `dataset_card.md`, `checksums.txt`, and `lexsi_provenance.json`.

**Gap for this project.** The eight generation tasks (qa, preference, grpo, multiturn, evol, cot, adversarial_*) do not include "extract fields into JSON". So the gold JSON labels must already be in the data, or come from a `preprocessing_fn` or a separate labelling step. Decide this before writing Stage 1.

### Install

```bash
pip install "curatorkit[all]"   # connectors + generation + embedding + hygiene (excludes pdf)
pip install curatorkit          # core only: cleaning and dedup
curatorkit --version
```

Python 3.11+. Extras: `hygiene` (secrets/PII/toxicity), `generation` (LiteLLM), `embedding` (diversity gate), `connectors` (HF/Parquet).

### Hardware

The core runs on CPU with no LLM or API key. Generation and judge gates need an LLM endpoint, which can be any LiteLLM model or a local OpenAI-compatible server (including SGLang) via `llm_api_base`.

### Minimal example (no LLM)

```python
from curatorkit import Curator, CuratorConfig

result = Curator(CuratorConfig(
    dataset        = "data/raw/tickets.jsonl",
    field_mapping  = {"ticket_text": "instruction", "label_json": "output"},  # {your_column: DataSample field}
    dedup          = "minhash",                  # "exact" (default) | "minhash" | "none"
    clean          = True,
    export_formats = ["messages"],               # -> sft_messages.jsonl
    output_split   = {"train": 0.8, "val": 0.1, "test": 0.1},
    output_dir     = "data/curated",
)).run()
result.print_summary()
```

Gotchas:
- `field_mapping` maps `{source_column: datasample_field}`. Writing it the other way round fails silently.
- `min_tokens` defaults to 10 and `max_tokens` to 2048. Samples outside that range are rejected by the schema gate.
- Exporters overwrite files of the same name. Use a fresh `output_dir` per run.
- `output_split` (from the source): any dict whose values sum to 1.0, and each key becomes a subfolder `output_dir/<key>/`. A single key like `{"test": 1.0}` works. Name the split `validation` rather than `val` so it maps to the HF validation split.
- System prompt (from the source): a raw column named `system`, `system_prompt`, or `system_message` becomes `metadata["system_prompt"]`, and the messages exporter writes it as a leading system turn. `export_formats=["messages"]` is valid even though the config reference page doesn't list it.
- The output folder is loadable as an HF dataset: `load_dataset("data/curated", "sft_messages")`.

CLI form: `curatorkit run pipeline.yaml --output-dir output/`.

---

## AlignTune (Stage 2)

Docs: https://aligntune.lexsi.ai · Repo: https://github.com/Lexsi-Labs/aligntune · License: MIT (the vendored Unsloth backend brings in AGPL components; the TRL backend runs no AGPL code)

**What it does here.** It runs LoRA SFT of a small model (examples use `Qwen/Qwen3-0.6B`) on CuratorKIT's `sft_messages` export, then saves the model or adapter for SGLang to serve. Optionally, a later GRPO stage could reward schema validity: the source has `aligntune.rewards.verifiable.JSONSchemaVerifiableReward`, but the guides don't show how to wire it in. Read the source before relying on it.

### Install

```bash
pip install aligntune
python -c "import aligntune; print(aligntune.__version__)"
aligntune info
```

Requires Python 3.11+, PyTorch 2.0+, and `git` on PATH (some dependencies install from git). It pins `transformers==5.14.1` and `trl==1.7.1`, and bundles Unsloth and mergekit. **Do not `pip install unsloth` separately.**

### Hardware

A CUDA GPU is recommended. Unsloth needs CUDA and falls back to TRL without it. Check what can run with `aligntune advise estimate --model ... --dataset-size N --algorithm sft`, which estimates VRAM, time and cost without a GPU.

### Minimal example: LoRA SFT on CuratorKIT output

```python
from aligntune.core.backend_factory import create_sft_trainer

trainer = create_sft_trainer(
    model_name="Qwen/Qwen3-0.6B",
    dataset_name="data/curated",        # CuratorKIT output folder
    config_name="sft_messages",
    backend="trl",                      # "unsloth" on CUDA for speed
    task_type="chat_completion",
    messages_column="messages",
    chat_template="auto",
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    max_seq_length=512,
    peft_enabled=True,
    lora_r=16, lora_alpha=32, lora_dropout=0.1,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
trainer.train()
print(trainer.evaluate())
path = trainer.save_model(output_dir="outputs/sft")
```

Corrections from running AlignTune 0.1.28 on the laptop (see `docs/plan.md` §4): use `task_type="sft"` (`chat_completion` fails in the DataManager), and set `precision="fp16"` on a T4. The combination above otherwise loads and trains on CPU. For QLoRA, add `quantization={"load_in_4bit": True}`. CLI equivalent: `aligntune train --model ... --dataset ... --type sft`.

### Merging LoRA for serving

```python
from aligntune.utils import load_finetuned_model
model, tokenizer = load_finetuned_model(output_dir="outputs/sft", base_model="Qwen/Qwen3-0.6B")  # merges LoRA if present
model.save_pretrained("outputs/merged"); tokenizer.save_pretrained("outputs/merged")
```

Or skip merging and serve the adapter with SGLang `--lora-paths`.

---

## AuditKIT (Stage 3)

Docs: https://auditkit.lexsi.ai · Repo: https://github.com/Lexsi-Labs/AuditKIT · License: LSAL v1.2 (source-available)

**What it does here.** It runs the test split against the SGLang server and scores schema validity, exact match, and per-field accuracy (via custom scorers), plus latency and throughput. Every run gets a stable fingerprint and can be compared with `ak.compare(base, tuned)` to show base vs. fine-tuned.

### Install

```bash
pip install auditkit                 # core, zero deps
pip install "auditkit[requests]"     # what the SGLang doc says the api: backend needs
```

Python 3.10+. Make sure you have a recent version (`pip show auditkit`): old releases pin `transformers<5`.

### Hardware

The core is CPU-only. Model inference happens wherever the backend runs (the SGLang server here).

### Minimal example

```python
import auditkit as ak

samples = [ak.Sample(input="<prompt with ticket>", target='{"category": "billing", ...}')]

@ak.scorer(direction=ak.Direction.MAXIMIZE)
def category_acc(sample, output):
    import json
    try:
        return float(json.loads(output)["category"] == json.loads(sample.target)["category"])
    except Exception:
        return 0.0

result = ak.evaluate(
    samples,
    model="api:MODEL_PATH",
    api_base="http://localhost:30000/v1",
    api_key="EMPTY",                       # always pass it, or your real OPENAI_API_KEY is sent
    scorers=["exact_match", "is_json", category_acc],
)
print(result.headline)                    # per-metric means; result.perf holds latency/throughput
```

Offline test without a server: `model=lambda prompts: [...]`. Any `list[str] -> list[str]` callable works.

Relevant built-in scorers: `exact_match` (trimmed string equality), `is_json` / `IsJson(require_keys=[...])`, and `regex`. `LatencyStats` and `Throughput` fill `RunResult.perf`.

Gotchas:
- **Constrained decoding:** the AuditKIT docs don't show `api:` sending `response_format`/`json_schema`. To score SGLang's schema-constrained output, wrap your own SGLang client as the callable model, or confirm in the AuditKIT source that `api:` passes it through.
- `exact_match` on raw JSON strings is sensitive to key order and whitespace. For exact match, parse the JSON and compare dicts in a custom scorer.
- Don't put an extra `:` in the model name: SGLang reads `model:x` as LoRA adapter `x`.
- Qwen3-style models think by default. Pass `RunConfig(chat_template_kwargs={"enable_thinking": False})` or answers come back empty.

---

## Open questions before writing code

1. Base model choice and size, given T4 memory (16 GB, fp16 only).
2. Where gold labels come from (CuratorKIT has no extraction generation task).
3. Whether AuditKIT `api:` can forward `response_format` (check the source).
4. `configs/schema.json` still needs to be written.
