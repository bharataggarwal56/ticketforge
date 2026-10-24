"""Generate notebooks/A_train.ipynb (run: python notebooks/build_A_train.py).

Keeping the notebook as source here means the JSON is always valid and diffs are readable.
"""
import json
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "A_train.ipynb"
params = json.loads((Path(__file__).resolve().parent / "A_train_params.json").read_text())

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Notebook A: LoRA SFT of Qwen2.5-1.5B-Instruct (Colab T4, default runtime)
Reads the curated data from the private HF dataset, trains with AlignTune (fp16, no bf16 on a T4),
pushes a checkpoint to HF every epoch, then pushes the adapter and the merged model.
Re-running resumes from the latest checkpoint on HF, or skips training if the final models are already there.
Needs a Colab secret named `HF_TOKEN` (notebook access on). SGLang is never installed here.
""")

code("""
!nvidia-smi | head -12
!pip install -q aligntune
import aligntune, trl, transformers, torch
print("aligntune", aligntune.__version__, "| trl", trl.__version__, "| transformers", transformers.__version__, "| torch", torch.__version__)
""")

code(f"""
from google.colab import userdata
from huggingface_hub import HfApi, login

HF_USER = "{params['hf_user']}"
HF_TOKEN = userdata.get("HF_TOKEN")
assert HF_TOKEN, "Add a Colab secret named HF_TOKEN and enable notebook access"
login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi(token=HF_TOKEN)
assert api.whoami()["name"] == HF_USER, "token belongs to a different account"

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_REPO = f"{{HF_USER}}/ticketforge-data"
TAG = "{params['tag']}"
ADAPTER_REPO = f"{{HF_USER}}/ticketforge-qwen2.5-1.5b-lora"
MERGED_REPO = f"{{HF_USER}}/ticketforge-qwen2.5-1.5b-merged"
""")

code("""
from huggingface_hub import snapshot_download
DATA_DIR = snapshot_download(DATA_REPO, repo_type="dataset", local_dir="/content/ticketforge-data",
                             allow_patterns=[f"train_{TAG}/*"], token=HF_TOKEN)
TRAIN_DIR = f"{DATA_DIR}/train_{TAG}"
!ls -R {TRAIN_DIR} | head -20
""")

md("## Token lengths: pick `max_seq_length` so no assistant answer is truncated")
code("""
import json, math
import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(BASE)
totals, answers = [], []
for split in ("train", "validation"):
    for line in open(f"{TRAIN_DIR}/{split}/sft_messages.jsonl"):
        msgs = json.loads(line)["messages"]
        ids = tok.apply_chat_template(msgs, tokenize=True)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        ans = tok(msgs[-1]["content"] + "<|im_end|>\\n", add_special_tokens=False)["input_ids"]
        assert ids[-len(ans):] == ans, "assistant answer is not the tail of the templated example"
        totals.append(len(ids)); answers.append(len(ans))

for name, v in (("full example", totals), ("assistant answer", answers)):
    print(f"{name:17s} n={len(v)} max={max(v)} p99={int(np.percentile(v, 99))} mean={np.mean(v):.1f}")
MAX_SEQ_LEN = math.ceil((max(totals) + 8) / 64) * 64
assert max(totals) <= MAX_SEQ_LEN
print("max_seq_length =", MAX_SEQ_LEN)
""")

md("""
## Training config
Settings come from reading the AlignTune source (see `docs/plan.md` §4); `task_type="sft"` because AlignTune 0.1.28 rejects `chat_completion`:
- `precision="fp16"` sets `SFTConfig(fp16=True, bf16=False)` and loads the weights in float16.
- AlignTune derives `max_steps` from epochs, so "every epoch" means `save_interval = eval_interval = steps_per_epoch`.
- Hub options reach `SFTConfig` as extra kwargs; the assert after training confirms they were applied.
""")
code(f"""
N_TRAIN = sum(1 for _ in open(f"{{TRAIN_DIR}}/train/sft_messages.jsonl"))
N_VAL = sum(1 for _ in open(f"{{TRAIN_DIR}}/validation/sft_messages.jsonl"))
assert (N_TRAIN, N_VAL) == ({params['n_train']}, {params['n_val']}), (N_TRAIN, N_VAL)

BATCH, GRAD_ACCUM, EPOCHS = {params['batch_size']}, {params['grad_accum']}, 3
STEPS_PER_EPOCH = N_TRAIN // (BATCH * GRAD_ACCUM)
print("steps/epoch", STEPS_PER_EPOCH, "total steps", STEPS_PER_EPOCH * EPOCHS)

TRAIN_KW = dict(
    model_name=BASE,
    dataset_name=TRAIN_DIR,
    config_name="sft_messages",
    backend="trl",
    task_type="sft",  # AlignTune 0.1.28 DataManager rejects "chat_completion"
    messages_column="messages",
    chat_template="auto",
    output_dir="/content/outputs/sft",
    num_epochs=EPOCHS,
    batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=2e-4,
    max_seq_length=MAX_SEQ_LEN,
    precision="fp16",
    peft_enabled=True,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    save_interval=STEPS_PER_EPOCH,
    eval_interval=STEPS_PER_EPOCH,
    eval_strategy="steps",
    save_strategy="steps",
    push_to_hub=True,
    hub_model_id=ADAPTER_REPO,
    hub_strategy="all_checkpoints",
    hub_token=HF_TOKEN,
    hub_private_repo=True,
)
""")

md("## Re-run state: skip if finished, resume if checkpoints exist on HF")
code("""
import re
def repo_files(repo):
    try:
        return api.list_repo_files(repo, repo_type="model")
    except Exception:
        return []

adapter_files, merged_files = repo_files(ADAPTER_REPO), repo_files(MERGED_REPO)
FINAL_DONE = "adapter_model.safetensors" in adapter_files and any(f.endswith(".safetensors") for f in merged_files)
ckpts = sorted({int(m.group(1)) for f in adapter_files if (m := re.match(r"checkpoint-(\\d+)/", f))})
RESUME_DIR = None
if FINAL_DONE:
    print("Adapter and merged model already on HF; skipping training and merge.")
elif ckpts:
    RESUME_DIR = snapshot_download(ADAPTER_REPO, repo_type="model", token=HF_TOKEN,
                                   local_dir="/content/resume", allow_patterns=[f"checkpoint-{ckpts[-1]}/*"]) + f"/checkpoint-{ckpts[-1]}"
    print("Resuming from", RESUME_DIR)
else:
    print("Fresh run")
""")

code("""
if not FINAL_DONE:
    # AlignTune calls trainer.train() without arguments; inject the resume path.
    from trl import SFTTrainer
    if RESUME_DIR and not getattr(SFTTrainer, "_resume_patched", False):
        _orig_train = SFTTrainer.train
        def _train(self, resume_from_checkpoint=None, **kw):
            return _orig_train(self, resume_from_checkpoint=resume_from_checkpoint or RESUME_DIR, **kw)
        SFTTrainer.train = _train
        SFTTrainer._resume_patched = True

    from aligntune.core.backend_factory import create_sft_trainer
    trainer = create_sft_trainer(**TRAIN_KW)
    result = trainer.train()
    print(result)

    args = trainer.trainer.args
    assert args.fp16 and not args.bf16, (args.fp16, args.bf16)
    assert args.push_to_hub and args.hub_model_id == ADAPTER_REPO and str(args.hub_strategy).endswith("all_checkpoints"), "hub options were not applied"
    assert args.max_length == MAX_SEQ_LEN
    print(trainer.evaluate())
""")

md("## Save, push the adapter, merge, push the merged model")
code("""
if not FINAL_DONE:
    from aligntune.utils import load_finetuned_model
    adapter_dir = trainer.save_model(output_dir="/content/outputs/sft")
    api.create_repo(ADAPTER_REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(repo_id=ADAPTER_REPO, repo_type="model", folder_path="/content/outputs/sft",
                      ignore_patterns=["checkpoint-*", "runs/*", "*.pt", "optimizer.*"],
                      commit_message="Final LoRA adapter")

    model, tokenizer = load_finetuned_model(output_dir="/content/outputs/sft", base_model=BASE)
    model.save_pretrained("/content/outputs/merged"); tokenizer.save_pretrained("/content/outputs/merged")
    api.create_repo(MERGED_REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(repo_id=MERGED_REPO, repo_type="model", folder_path="/content/outputs/merged",
                      commit_message="Merged fp16 model")
""")

md("## Sanity check on the merged model (one validation ticket)")
code("""
import json, torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(MERGED_REPO, torch_dtype=torch.float16, token=HF_TOKEN).to("cuda")
t = AutoTokenizer.from_pretrained(MERGED_REPO, token=HF_TOKEN)
row = json.loads(open(f"{TRAIN_DIR}/validation/sft_messages.jsonl").readline())["messages"]
prompt = t.apply_chat_template(row[:2], add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
out = m.generate(**prompt, max_new_tokens=128, do_sample=False)
text = t.decode(out[0][prompt["input_ids"].shape[1]:], skip_special_tokens=True)
print(text)
print("gold:", row[2]["content"])
print("parses:", set(json.loads(text)) == set(json.loads(row[2]["content"])))
print(f"https://huggingface.co/{ADAPTER_REPO}\\nhttps://huggingface.co/{MERGED_REPO}")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
}
nbf.validate(nb)
nbf.write(nb, OUT)
print("wrote", OUT)
