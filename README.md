# EVS Project: Climate Data Trend Analysis + Mixtral-8x7B AI Assistant

This project is built for your EVS topic:

- Concept: Climate change
- Activity:
- Use 10-20 years temperature and rainfall data
- Plot line graphs (Python output can also be opened in Excel)
- Interpret increasing/decreasing trends
- Train/use Mixtral-8x7B so users can ask questions and get detailed answers with data representation

## 1. Project Objective

You are analyzing how climate variables (temperature and rainfall) have changed over time and using those findings to power an AI assistant.

The assistant is designed to:

- answer both general questions and climate questions,
- include numeric evidence for climate-data queries,
- adapt to user question format (no fixed response format required).

## 2. Recommended Kaggle Sources

Default configuration uses:

- `berkeleyearth/climate-change-earth-surface-temperature-data` (temperature)
- `rajanand/rainfall-in-india` (rainfall)

You can change dataset slugs in `config/project_config.yaml`.

## 3. Folder Overview

- `scripts/download_kaggle_data.py`: Downloads datasets from Kaggle
- `scripts/run_analysis.py`: Builds yearly 10-20 year climate dataset, trend stats, and plots
- `scripts/generate_qa_pairs.py`: Generates climate Q&A training data
- `scripts/train_mixtral_lora.py`: Fine-tunes Mixtral-8x7B (LoRA)
- `scripts/ask_climate_ai.py`: Answers user questions with data representation
- `src/climate_evs/analysis.py`: Data parsing, trend metrics, interpretation text
- `src/climate_evs/plots.py`: Plot generation
- `src/climate_evs/qa.py`: Q&A generation and prompt helpers
- `frontend/`: Web UI for charts, trend cards, and Q&A presentation

## 4. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set Kaggle API key:

1. Download `kaggle.json` from your Kaggle account settings.
2. Place it at `~/.kaggle/kaggle.json`
3. Run:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

(Optional) Hugging Face login for model download:

```bash
huggingface-cli login
```

## 5. Run Pipeline

### Step A: Download data

```bash
python scripts/download_kaggle_data.py
```

### Step B: Perform climate trend analysis

```bash
python scripts/run_analysis.py
```

Optional custom range:

```bash
python scripts/run_analysis.py --start-year 2001 --end-year 2020
```

### Step C: Build Q&A training dataset

```bash
python scripts/generate_qa_pairs.py --min-pairs 100
```

### Step D: Fine-tune Mixtral-8x7B (LoRA)

```bash
python scripts/train_mixtral_lora.py --epochs 2 --batch-size 1 --grad-accum 16
```

If you have a compatible setup and want 4-bit loading:

```bash
python scripts/train_mixtral_lora.py --load-in-4bit
```

### Step E: Ask questions to your climate AI

Single question mode:

```bash
python scripts/ask_climate_ai.py --question "Show the temperature trend and explain climate impact"
```

Interactive mode:

```bash
python scripts/ask_climate_ai.py --interactive
```

### Step F: Launch frontend dashboard

```bash
/Users/pritthacker/EVS/.venv/bin/python scripts/run_ai_api.py
```

Open:

- `http://localhost:8080/frontend/`

This command serves both frontend and API endpoint:

- `POST /api/ask`

Answer routing:

- Uses local Mixtral when GPU/model is available.
- If local Mixtral is unavailable, it can use Hugging Face Inference API if `HF_TOKEN` is set.
- If neither is available, it falls back to local response logic.

To enable general AI answers without local GPU, set token and restart:

```bash
export HF_TOKEN=your_huggingface_token
/Users/pritthacker/EVS/.venv/bin/python scripts/run_ai_api.py --disable-model
```

The backend still uses climate outputs when available for climate-specific responses:

- `data/processed/yearly_climate_india.csv`
- `outputs/tables/trend_summary.json`

If these files are not available yet, the API runs on demo climate data.

## 6. Output Files for Your EVS Report

After running analysis:

- `data/processed/yearly_climate_india.csv`
- `outputs/tables/yearly_climate_india.xlsx` (open in Excel)
- `outputs/tables/trend_summary.json`
- `outputs/tables/trend_interpretation.md`
- `outputs/plots/temperature_trend.png`
- `outputs/plots/rainfall_trend.png`
- `outputs/plots/dual_axis_trend.png`
- `outputs/plots/normalized_comparison.png`

## 7. How to Present Interpretation in EVS

Use these points in your report discussion:

- If temperature slope is positive, warming trend is present.
- If rainfall slope is negative, drought risk can increase.
- If rainfall variability increases, flood and water-management planning become important.
- Correlate trends with local context: land use, urbanization, policy, adaptation measures.

## 8. Resource Notes for Mixtral-8x7B

- Full fine-tuning is expensive.
- LoRA is used for practical adaptation.
- Strong GPU resources are required for training and inference.
- If resources are limited, run analysis outputs first and train on a cloud GPU.

## 9. Suggested EVS Viva Questions You Can Test

- "Is temperature increasing in the selected period?"
- "How did rainfall change from first year to last year?"
- "Which year was hottest and what does it imply?"
- "Show a trend graph and explain climate-change significance."

This setup gives you a full progression from dataset to analysis to AI-based Q&A.

## 10. Deploy On Vercel

This repository now includes a Vercel serverless API endpoint at `api/ask.js` and a root redirect via `vercel.json`.

### What gets deployed

- Frontend at `/frontend/` (root `/` redirects here)
- API endpoint at `POST /api/ask`

### Steps

1. Push this repository to GitHub.
2. Import the repo into Vercel.
3. Keep default framework setting (`Other`) and deploy.
4. Optional but recommended: add environment variable `HF_TOKEN` in Vercel project settings for richer general AI answers.
5. Redeploy after adding env vars.

### Notes

- If `data/processed/yearly_climate_india.csv` exists, API responds in `project` data mode.
- If not, it automatically falls back to built-in demo climate data.
- If `HF_TOKEN` is missing or unavailable, it falls back to local deterministic response logic.
