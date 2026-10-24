"""TicketForge demo: ticket text in, schema-checked JSON out (CPU, transformers)."""
import html
import json
import os
import time
from pathlib import Path

import gradio as gr
import jsonschema
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import build_messages

HERE = Path(__file__).parent
MODEL_ID = "bharataggarwal56/ticketforge-qwen2.5-1.5b-merged"
GITHUB_URL = "https://github.com/bharataggarwal56/ticketforge"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}"
DATASET_URL = "https://huggingface.co/datasets/bharataggarwal56/ticketforge-data"
NOTE = (
    "Evaluation and benchmarks in this project used SGLang on a Colab T4. "
    "This demo runs the same merged model on CPU with transformers, so it is slower. "
    "Trained on synthetic tickets, so it may not generalize to real tickets."
)
DTYPE = getattr(torch, os.environ.get("TICKETFORGE_DTYPE", "float32"))

SCHEMA = json.loads((HERE / "schema.json").read_text())
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)
EXAMPLES = json.loads((HERE / "examples.json").read_text())

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE)
model.eval()


def run_ticket(ticket: str) -> dict:
    """Generate for one ticket. Strict parse, no repair; validate against the schema."""
    prompt = tokenizer.apply_chat_template(
        build_messages(ticket), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=128)
    seconds = time.perf_counter() - start
    raw = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    parsed, error = None, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        error = f"not valid JSON: {e}"
    if error is None:
        first = next(iter(VALIDATOR.iter_errors(parsed)), None)
        if first is not None:
            path = "/".join(str(p) for p in first.absolute_path) or "(root)"
            error = f"{first.message} (at {path})"
    return {"raw": raw, "parsed": parsed, "valid": error is None, "error": error, "seconds": seconds}


def ui_run(ticket: str):
    if not ticket or not ticket.strip():
        return "", None, "", ""
    r = run_ticket(ticket)
    if r["valid"]:
        badge = '<span style="color:#0a7d32;font-weight:600">schema-valid: yes</span>'
    else:
        badge = f'<span style="color:#b3261e;font-weight:600">schema-valid: no</span><br>{html.escape(r["error"])}'
    return r["raw"], r["parsed"], badge, f'{r["seconds"]:.1f} s'


with gr.Blocks(title="TicketForge demo") as demo:
    gr.Markdown("# TicketForge demo\nPaste a support ticket and get structured JSON back.")
    gr.Markdown(f"**Note:** {NOTE}")
    gr.Markdown(
        f"[GitHub repo]({GITHUB_URL}) · [Model]({MODEL_URL}) · [Dataset]({DATASET_URL})"
    )
    ticket = gr.Textbox(label="Ticket", lines=8)
    run = gr.Button("Run", variant="primary")
    raw = gr.Textbox(label="Raw model output", lines=6, interactive=False)
    parsed = gr.JSON(label="Parsed JSON (strict json.loads)")
    badge = gr.HTML()
    secs = gr.Textbox(label="Generation time", interactive=False)
    outs = [raw, parsed, badge, secs]
    gr.Examples(
        examples=[[e["ticket"]] for e in EXAMPLES],
        example_labels=[e["label"] for e in EXAMPLES],
        inputs=[ticket],
    )
    run.click(ui_run, ticket, outs)
    ticket.submit(ui_run, ticket, outs)

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
