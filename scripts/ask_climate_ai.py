#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from climate_evs.analysis import context_for_llm  # noqa: E402
from climate_evs.plots import save_question_plot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask climate questions to Mixtral with data-representation-first answers"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/project_config.yaml"),
        help="Path to project config file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Base Mixtral model",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("models/mixtral_climate_lora"),
        help="LoRA adapter directory (used if present)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to answer",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=450,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive Q&A loop",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def infer_year_slice(df: pd.DataFrame, question: str) -> pd.DataFrame:
    years = [int(match) for match in re.findall(r"\b(?:19|20)\d{2}\b", question)]

    if len(years) >= 2:
        start_year, end_year = sorted(years[:2])
        subset = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
        if not subset.empty:
            return subset

    if len(years) == 1:
        subset = df[df["year"] == years[0]]
        if not subset.empty:
            return subset

    return df.tail(min(10, len(df)))


def infer_chart_variable(question: str) -> str:
    lowered = question.lower()
    if "rain" in lowered or "precip" in lowered:
        return "rainfall"
    return "temperature"


def build_prompt(question: str, context: str) -> str:
    return (
        "<s>[INST] You are an EVS climate project assistant. Use only the provided context. "
        "Your answer must be detailed and include a data table representation. "
        "Respond in this exact section format:\n"
        "1) Trend Insight\n2) Data Representation\n3) Climate-change Meaning\n\n"
        f"Question: {question}\n\n"
        "Context:\n"
        f"{context}\n"
        "[/INST]"
    )


def load_model_and_tokenizer(model_name: str, adapter_dir: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run Mixtral-8x7B inference.")

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    if adapter_dir.exists():
        print(f"Loading LoRA adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = output_ids[0][encoded["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return text.strip()


def response_table(df: pd.DataFrame) -> str:
    display = df[["year", "avg_temperature_c", "total_rainfall_mm"]].copy()
    display = display.rename(
        columns={
            "year": "Year",
            "avg_temperature_c": "AvgTemperatureC",
            "total_rainfall_mm": "TotalRainfallMm",
        }
    )
    display["AvgTemperatureC"] = display["AvgTemperatureC"].round(3)
    display["TotalRainfallMm"] = display["TotalRainfallMm"].round(3)
    return display.to_markdown(index=False)


def answer_single_question(
    question: str,
    yearly_df: pd.DataFrame,
    summary: dict,
    plots_dir: Path,
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
) -> str:
    subset = infer_year_slice(yearly_df, question)
    context = context_for_llm(subset, summary, max_rows=min(15, len(subset)))

    prompt = build_prompt(question, context)
    generated = generate_answer(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    include_chart = any(
        token in question.lower() for token in ["plot", "graph", "chart", "visual", "trend"]
    )
    chart_note = ""
    if include_chart:
        variable = infer_chart_variable(question)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = plots_dir / f"question_{variable}_{timestamp}.png"
        chart_path = save_question_plot(subset, variable=variable, output_path=output_path)
        chart_note = f"\n\nGenerated chart: {Path(chart_path).relative_to(PROJECT_ROOT)}"

    forced_table = response_table(subset)
    return (
        f"{generated}\n\n"
        "Data Representation (source values used):\n"
        f"{forced_table}"
        f"{chart_note}"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]

    yearly_path = PROJECT_ROOT / paths["yearly_csv"]
    summary_path = PROJECT_ROOT / paths["summary_json"]
    plots_dir = PROJECT_ROOT / paths["plots_dir"]

    if not yearly_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "Missing analysis outputs. Run scripts/run_analysis.py before asking questions."
        )

    yearly_df = pd.read_csv(yearly_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        adapter_dir=PROJECT_ROOT / args.adapter_dir,
    )

    if args.interactive or args.question is None:
        print("Interactive climate AI mode. Type 'exit' to quit.")
        while True:
            user_question = input("\nQuestion: ").strip()
            if user_question.lower() in {"exit", "quit"}:
                break
            if not user_question:
                continue
            answer = answer_single_question(
                question=user_question,
                yearly_df=yearly_df,
                summary=summary,
                plots_dir=plots_dir,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            print("\n" + answer)
        return

    answer = answer_single_question(
        question=args.question,
        yearly_df=yearly_df,
        summary=summary,
        plots_dir=plots_dir,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(answer)


if __name__ == "__main__":
    main()
