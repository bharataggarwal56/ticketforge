# TicketForge local demo

Gradio app that turns a support ticket into JSON with the merged model
(`bharataggarwal56/ticketforge-qwen2.5-1.5b-merged`), checks the output against
`schema.json` and shows the raw output, the parsed JSON and the generation time.
It runs on CPU with transformers. The model is public and downloads on first run
(about 6 GB of RAM in float32).

```bash
cd demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:7860 in your browser. The server listens on localhost only.

`examples.json` holds four tickets taken from the test split
(`data/curated/test_2026-10-24`).
