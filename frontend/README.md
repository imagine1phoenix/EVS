# Frontend (Ask Bar)

## What this UI does

- Shows a single question bar
- Sends your question to AI endpoint at `POST /api/ask`
- Displays the returned answer on screen for any question format

## Run locally

From project root:

```bash
/Users/pritthacker/EVS/.venv/bin/python scripts/run_ai_api.py
```

Open:

- `http://localhost:8080/frontend/`

Ask your question in the bar and click **Ask AI**.

For broad general AI responses (without local GPU Mixtral), set HF token before starting server:

```bash
export HF_TOKEN=your_huggingface_token
/Users/pritthacker/EVS/.venv/bin/python scripts/run_ai_api.py --disable-model
```

## Backend integration (optional)

Frontend sends:

```json
{
  "question": "...",
}
```

If Mixtral cannot be loaded (for example no CUDA GPU), backend tries Hugging Face Inference API (when `HF_TOKEN` is set), then local fallback.

Expected response:

```json
{
  "answer": "detailed response with data representation"
}
```
