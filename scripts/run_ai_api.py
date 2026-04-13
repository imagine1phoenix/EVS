#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from climate_evs.analysis import context_for_llm  # noqa: E402


MONTHLY_CLIMATE_KEYWORDS = {
    "climate",
    "temperature",
    "temp",
    "rain",
    "rainfall",
    "precip",
    "weather",
    "monsoon",
    "trend",
    "hottest",
    "wettest",
    "warming",
    "cooling",
    "drought",
    "flood",
    "flooding",
    "heatwave",
    "evs",
    "celsius",
    "fahrenheit",
    "mm",
    "year",
    "years",
    "compare",
    "difference",
    "anomaly",
    "regression",
    "correlation",
    "emission",
    "emissions",
    "co2",
    "greenhouse",
    "ozone",
    "sustainability",
    "environment",
    "environmental",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve frontend and AI endpoint (POST /api/ask)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/project_config.yaml"),
        help="Path to project config YAML",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Local Mixtral model name",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("models/mixtral_climate_lora"),
        help="LoRA adapter path (optional)",
    )
    parser.add_argument(
        "--disable-model",
        action="store_true",
        help="Disable local model loading (HF API can still be used)",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Hugging Face Inference model for general responses",
    )
    parser.add_argument("--max-new-tokens", type=int, default=450)
    parser.add_argument("--temperature", type=float, default=0.25)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config_path = path if path.is_absolute() else PROJECT_ROOT / path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
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


def response_table(df: pd.DataFrame) -> str:
    display = df[["year", "avg_temperature_c", "total_rainfall_mm"]].rename(
        columns={
            "year": "Year",
            "avg_temperature_c": "AvgTemperatureC",
            "total_rainfall_mm": "TotalRainfallMm",
        }
    )
    display = display.round({"AvgTemperatureC": 3, "TotalRainfallMm": 3})
    return display.to_markdown(index=False)


def build_climate_prompt(question: str, context: str) -> str:
    return (
        "<s>[INST] You are an EVS climate assistant. Use the provided context. "
        "Answer clearly and directly based on the question format.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n"
        "[/INST]"
    )


def build_general_prompt(question: str) -> str:
    return (
        "<s>[INST] You are a helpful AI assistant. Answer directly, clearly, and naturally. "
        "Adapt to the question format.\n\n"
        f"Question: {question}\n"
        "[/INST]"
    )


def demo_yearly_dataframe() -> pd.DataFrame:
    records = [
        {"year": 2005, "avg_temperature_c": 24.91, "total_rainfall_mm": 1152},
        {"year": 2006, "avg_temperature_c": 24.96, "total_rainfall_mm": 1108},
        {"year": 2007, "avg_temperature_c": 25.01, "total_rainfall_mm": 1197},
        {"year": 2008, "avg_temperature_c": 25.04, "total_rainfall_mm": 1079},
        {"year": 2009, "avg_temperature_c": 25.11, "total_rainfall_mm": 1023},
        {"year": 2010, "avg_temperature_c": 25.14, "total_rainfall_mm": 1249},
        {"year": 2011, "avg_temperature_c": 25.20, "total_rainfall_mm": 1182},
        {"year": 2012, "avg_temperature_c": 25.19, "total_rainfall_mm": 1120},
        {"year": 2013, "avg_temperature_c": 25.27, "total_rainfall_mm": 1164},
        {"year": 2014, "avg_temperature_c": 25.34, "total_rainfall_mm": 1095},
        {"year": 2015, "avg_temperature_c": 25.42, "total_rainfall_mm": 1048},
        {"year": 2016, "avg_temperature_c": 25.48, "total_rainfall_mm": 1230},
        {"year": 2017, "avg_temperature_c": 25.51, "total_rainfall_mm": 1138},
        {"year": 2018, "avg_temperature_c": 25.57, "total_rainfall_mm": 1176},
        {"year": 2019, "avg_temperature_c": 25.66, "total_rainfall_mm": 1117},
        {"year": 2020, "avg_temperature_c": 25.73, "total_rainfall_mm": 1204},
        {"year": 2021, "avg_temperature_c": 25.78, "total_rainfall_mm": 1143},
        {"year": 2022, "avg_temperature_c": 25.82, "total_rainfall_mm": 1211},
        {"year": 2023, "avg_temperature_c": 25.87, "total_rainfall_mm": 1184},
        {"year": 2024, "avg_temperature_c": 25.93, "total_rainfall_mm": 1168},
    ]
    return pd.DataFrame(records)


def summary_from_yearly(yearly_df: pd.DataFrame) -> dict[str, Any]:
    years = yearly_df["year"].to_list()
    temps = yearly_df["avg_temperature_c"].to_list()
    rains = yearly_df["total_rainfall_mm"].to_list()

    year_diff = (years[-1] - years[0]) or 1
    temp_slope = (temps[-1] - temps[0]) / year_diff
    rain_slope = (rains[-1] - rains[0]) / year_diff

    hottest_idx = int(yearly_df["avg_temperature_c"].idxmax())
    wettest_idx = int(yearly_df["total_rainfall_mm"].idxmax())

    return {
        "year_span": {
            "start": int(years[0]),
            "end": int(years[-1]),
            "count": int(len(yearly_df)),
        },
        "temperature_trend": {
            "slope_per_year": float(temp_slope),
            "absolute_change": float(temps[-1] - temps[0]),
            "direction": "increasing" if temp_slope > 0 else "decreasing" if temp_slope < 0 else "stable",
        },
        "rainfall_trend": {
            "slope_per_year": float(rain_slope),
            "absolute_change": float(rains[-1] - rains[0]),
            "direction": "increasing" if rain_slope > 0 else "decreasing" if rain_slope < 0 else "stable",
        },
        "hottest_year": {
            "year": int(yearly_df.loc[hottest_idx, "year"]),
            "avg_temperature_c": float(yearly_df.loc[hottest_idx, "avg_temperature_c"]),
        },
        "wettest_year": {
            "year": int(yearly_df.loc[wettest_idx, "year"]),
            "total_rainfall_mm": float(yearly_df.loc[wettest_idx, "total_rainfall_mm"]),
        },
    }


def direction_from_change(change: float, tolerance: float = 1e-9) -> str:
    if change > tolerance:
        return "increasing"
    if change < -tolerance:
        return "decreasing"
    return "stable"


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    import torch

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


class ClimateAIService:
    def __init__(
        self,
        config: dict[str, Any],
        model_name: str,
        hf_model: str,
        adapter_dir: Path,
        disable_model: bool,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        paths = config["paths"]
        yearly_path = PROJECT_ROOT / paths["yearly_csv"]
        summary_path = PROJECT_ROOT / paths["summary_json"]

        if yearly_path.exists() and summary_path.exists():
            self.yearly_df = pd.read_csv(yearly_path)
            self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.data_mode = "project"
        else:
            self.yearly_df = demo_yearly_dataframe()
            self.summary = summary_from_yearly(self.yearly_df)
            self.data_mode = "demo"

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.model = None
        self.tokenizer = None
        self.hf_client = None
        self.hf_model = hf_model
        self.mode = "fallback"
        self.model_error = "Model not initialized"
        self.hf_error = "HF provider not initialized"

        if disable_model:
            self.model_error = "Local model disabled by flag --disable-model"
        else:
            self._try_load_mixtral(model_name=model_name, adapter_dir=PROJECT_ROOT / adapter_dir)

        if self.model is None:
            self._try_load_hf_api(hf_model=hf_model)
            if self.hf_client is not None:
                self.mode = "hf-api"

    def _try_load_mixtral(self, model_name: str, adapter_dir: Path) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            self.model_error = f"Mixtral dependencies unavailable: {exc}"
            return

        if not torch.cuda.is_available():
            self.model_error = "CUDA GPU is not available for Mixtral inference"
            return

        try:
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
                model = PeftModel.from_pretrained(model, str(adapter_dir))

            model.eval()
            self.model = model
            self.tokenizer = tokenizer
            self.mode = "mixtral"
            self.model_error = ""
        except Exception as exc:  # pragma: no cover
            self.model_error = f"Failed to load Mixtral: {exc}"

    def _try_load_hf_api(self, hf_model: str) -> None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            self.hf_error = "HF token not found (set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN)"
            return

        try:
            from huggingface_hub import InferenceClient
        except Exception as exc:  # pragma: no cover
            self.hf_error = f"huggingface_hub unavailable: {exc}"
            return

        try:
            self.hf_client = InferenceClient(model=hf_model, token=token)
            self.hf_error = ""
        except Exception as exc:  # pragma: no cover
            self.hf_error = f"Failed to initialize HF client: {exc}"

    def _answer_with_hf(self, question: str, *, is_climate: bool, context: str | None = None) -> str:
        if self.hf_client is None:
            raise RuntimeError("HF client is not initialized")

        if is_climate and context:
            prompt = (
                "You are an EVS climate assistant. Use climate context for data-backed answers. "
                "Answer naturally and directly.\n\n"
                f"Question: {question}\n\n"
                f"Climate context:\n{context}\n"
            )
        else:
            prompt = (
                "You are a helpful AI assistant. Answer clearly, directly, and in the user's format.\n\n"
                f"Question: {question}\n"
            )

        result = self.hf_client.text_generation(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            return_full_text=False,
        )

        if isinstance(result, str) and result.strip():
            return result.strip()
        raise RuntimeError("HF provider returned empty response")

    @staticmethod
    def _normalize_question(question: str) -> str:
        return re.sub(r"\s+", " ", question).strip().lower()

    @staticmethod
    def sanitize_answer_text(answer: str) -> str:
        if not isinstance(answer, str):
            return ""

        lines = answer.splitlines()
        diagnostic_line = re.compile(
            r"^\s*(?:response mode:\s*.*|(?:climate-)?fallback(?: mode enabled)?\s*\(.*\))\s*$",
            re.IGNORECASE,
        )

        removed_prefix = False
        while lines and diagnostic_line.match(lines[0]):
            lines.pop(0)
            removed_prefix = True

        if removed_prefix:
            while lines and not lines[0].strip():
                lines.pop(0)

        cleaned = "\n".join(lines).strip()
        return cleaned if cleaned else answer.strip()

    @staticmethod
    def _is_climate_question(normalized_question: str) -> bool:
        if re.search(r"\b(?:19|20)\d{2}\b", normalized_question):
            return True
        return any(keyword in normalized_question for keyword in MONTHLY_CLIMATE_KEYWORDS)

    @staticmethod
    def _is_greeting_or_smalltalk(normalized_question: str) -> bool:
        greetings = {
            "hi",
            "hii",
            "hello",
            "hey",
            "yo",
            "hola",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "help",
            "thanks",
            "thank you",
        }
        if normalized_question in greetings:
            return True
        words = normalized_question.split()
        return len(words) <= 2 and any(token in {"hi", "hello", "hey"} for token in words)

    @staticmethod
    def _pick_phrase(question: str, options: list[str]) -> str:
        if not options:
            return ""
        idx = sum(ord(ch) for ch in question) % len(options)
        return options[idx]

    def _smalltalk_answer(self, question: str) -> str:
        opener = self._pick_phrase(question, ["Hi!", "Hello!", "Hey!"])
        suggestion = self._pick_phrase(
            question,
            [
                "Try: Explain photosynthesis in 3 points.",
                "Try: Compare rainfall between 2010 and 2020.",
                "Try: Solve 125*48.",
            ],
        )
        return (
            f"{opener} I can answer general questions and climate-data questions. "
            "For climate questions, I include data representation.\n\n"
            f"{suggestion}"
        )

    @staticmethod
    def _extract_related_texts(related_topics: list[Any], limit: int = 3) -> list[str]:
        texts: list[str] = []

        def walk(items: list[Any]) -> None:
            for item in items:
                if len(texts) >= limit:
                    return
                if isinstance(item, dict):
                    text = item.get("Text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
                    nested = item.get("Topics")
                    if isinstance(nested, list):
                        walk(nested)

        walk(related_topics)
        return texts[:limit]

    def _web_lookup(self, question: str) -> dict[str, Any] | None:
        query = quote_plus(question)
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1&no_html=1"
        request = Request(url, headers={"User-Agent": "EVS-AI-Assistant/1.0"})

        try:
            with urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return None

        answer = payload.get("Answer")
        abstract = payload.get("AbstractText")
        definition = payload.get("Definition")
        heading = payload.get("Heading")
        abstract_url = payload.get("AbstractURL")

        direct = ""
        for candidate in [answer, abstract, definition]:
            if isinstance(candidate, str) and candidate.strip():
                direct = candidate.strip()
                break

        related = self._extract_related_texts(payload.get("RelatedTopics") or [], limit=3)

        if not direct and not related:
            return None

        return {
            "heading": heading if isinstance(heading, str) else "",
            "direct": direct,
            "related": related,
            "source": abstract_url if isinstance(abstract_url, str) else "",
        }

    @staticmethod
    def _safe_eval_math(expression: str) -> float | None:
        try:
            node = ast.parse(expression, mode="eval")
        except Exception:
            return None

        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.UAdd,
            ast.USub,
        )

        if not all(isinstance(n, allowed_nodes) for n in ast.walk(node)):
            return None

        try:
            value = eval(compile(node, "<expr>", "eval"), {"__builtins__": {}}, {})
        except Exception:
            return None

        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _math_answer(self, question: str) -> str | None:
        normalized = self._normalize_question(question)
        if not any(token in normalized for token in ["solve", "calculate", "what is", "math"]):
            return None

        expression = re.sub(r"[^0-9\+\-\*\/\(\)\.% ]", "", normalized)
        expression = re.sub(r"\s+", "", expression)
        if not expression:
            return None

        value = self._safe_eval_math(expression)
        if value is None:
            return None

        if value.is_integer():
            result = str(int(value))
        else:
            result = f"{value:.6f}".rstrip("0").rstrip(".")
        return f"Result: {result}"

    @staticmethod
    def _subset_metrics(subset: pd.DataFrame) -> dict[str, Any]:
        ordered = subset.sort_values("year").reset_index(drop=True)
        first = ordered.iloc[0]
        last = ordered.iloc[-1]

        years_span = int(last["year"] - first["year"])
        safe_year_span = years_span if years_span > 0 else 1

        temp_change = float(last["avg_temperature_c"] - first["avg_temperature_c"])
        rain_change = float(last["total_rainfall_mm"] - first["total_rainfall_mm"])

        hottest_row = ordered.loc[ordered["avg_temperature_c"].idxmax()]
        wettest_row = ordered.loc[ordered["total_rainfall_mm"].idxmax()]

        return {
            "start_year": int(first["year"]),
            "end_year": int(last["year"]),
            "row_count": int(len(ordered)),
            "temp_start": float(first["avg_temperature_c"]),
            "temp_end": float(last["avg_temperature_c"]),
            "temp_change": temp_change,
            "temp_slope": temp_change / safe_year_span,
            "temp_direction": direction_from_change(temp_change),
            "rain_start": float(first["total_rainfall_mm"]),
            "rain_end": float(last["total_rainfall_mm"]),
            "rain_change": rain_change,
            "rain_slope": rain_change / safe_year_span,
            "rain_direction": direction_from_change(rain_change),
            "avg_temp": float(ordered["avg_temperature_c"].mean()),
            "avg_rain": float(ordered["total_rainfall_mm"].mean()),
            "hottest_year": int(hottest_row["year"]),
            "hottest_temp": float(hottest_row["avg_temperature_c"]),
            "wettest_year": int(wettest_row["year"]),
            "wettest_rain": float(wettest_row["total_rainfall_mm"]),
        }

    @staticmethod
    def _implication_text(temp_direction: str, rain_direction: str) -> str:
        if temp_direction == "increasing" and rain_direction == "decreasing":
            return "This suggests warming with moisture stress, increasing drought and heat-risk pressure."
        if temp_direction == "increasing" and rain_direction == "increasing":
            return "This suggests warmer and wetter conditions; adaptation should cover heat and intense rainfall."
        if temp_direction == "stable" and rain_direction == "stable":
            return "Both variables are relatively stable in this selected period; monitor anomalies closely."
        return "The variables move differently, so local seasonal and land-use context is important."

    @staticmethod
    def _extract_focus(question: str) -> dict[str, bool]:
        lowered = question.lower()
        focus_temperature = any(token in lowered for token in ["temperature", "temp", "heat", "warming"])
        focus_rainfall = any(token in lowered for token in ["rain", "rainfall", "precip", "monsoon"])
        focus_extreme = any(
            token in lowered for token in ["hottest", "wettest", "highest", "lowest", "extreme", "max", "min"]
        )
        focus_compare = any(token in lowered for token in ["compare", "difference", "vs", "versus", "between", "change"])

        if not focus_temperature and not focus_rainfall:
            focus_temperature = True
            focus_rainfall = True

        return {
            "temperature": focus_temperature,
            "rainfall": focus_rainfall,
            "extreme": focus_extreme,
            "compare": focus_compare,
        }

    def _fallback_climate_answer(self, question: str, subset: pd.DataFrame) -> str:
        focus = self._extract_focus(question)
        metrics = self._subset_metrics(subset)

        lines = [
            "1) Trend Insight",
            (
                f"Selected period: {metrics['start_year']} to {metrics['end_year']} "
                f"({metrics['row_count']} years)."
            ),
        ]

        if focus["temperature"]:
            lines.append(
                (
                    f"Temperature: {metrics['temp_start']:.2f} C -> {metrics['temp_end']:.2f} C "
                    f"({metrics['temp_change']:+.2f} C), {metrics['temp_direction']} "
                    f"at about {metrics['temp_slope']:+.4f} C/year."
                )
            )

        if focus["rainfall"]:
            lines.append(
                (
                    f"Rainfall: {metrics['rain_start']:.2f} mm -> {metrics['rain_end']:.2f} mm "
                    f"({metrics['rain_change']:+.2f} mm), {metrics['rain_direction']} "
                    f"at about {metrics['rain_slope']:+.2f} mm/year."
                )
            )

        if focus["extreme"]:
            lines.append(
                (
                    f"Extremes: hottest year {metrics['hottest_year']} ({metrics['hottest_temp']:.2f} C), "
                    f"wettest year {metrics['wettest_year']} ({metrics['wettest_rain']:.2f} mm)."
                )
            )

        if focus["compare"] and metrics["row_count"] > 1:
            lines.append(
                (
                    f"Start vs end comparison: temperature {metrics['temp_start']:.2f} -> {metrics['temp_end']:.2f}, "
                    f"rainfall {metrics['rain_start']:.2f} -> {metrics['rain_end']:.2f}."
                )
            )

        lines.extend(["", "2) Data Representation", response_table(subset), "", "3) Climate-change Meaning"])
        lines.append(
            self._implication_text(
                temp_direction=metrics["temp_direction"],
                rain_direction=metrics["rain_direction"],
            )
        )
        lines.append(
            (
                f"Average in this period: {metrics['avg_temp']:.2f} C temperature and "
                f"{metrics['avg_rain']:.2f} mm rainfall."
            )
        )

        return "\n".join(lines)

    def _generic_fallback_answer(self, question: str) -> str:
        math_answer = self._math_answer(question)
        if math_answer is not None:
            return math_answer

        lookup = self._web_lookup(question)
        if lookup is not None:
            lines: list[str] = []
            if lookup["heading"]:
                lines.append(f"Topic: {lookup['heading']}")
            if lookup["direct"]:
                lines.append(lookup["direct"])
            if lookup["related"]:
                lines.append("")
                lines.append("More context:")
                for item in lookup["related"]:
                    lines.append(f"- {item}")
            if lookup["source"]:
                lines.append("")
                lines.append(f"Source: {lookup['source']}")
            return "\n".join(lines).strip()

        normalized = self._normalize_question(question)
        if any(token in normalized for token in ["code", "python", "program", "bug", "error"]):
            return (
                "Share your exact code snippet and full error message. I will give a direct corrected version "
                "and explain what to change."
            )

        return (
            "I could not fetch a reliable specific answer right now. Rephrase with clearer detail, for example: "
            "'Explain photosynthesis in 5 bullet points' or 'Compare SQL and NoSQL with examples'."
        )

    def _build_charts(self, subset: pd.DataFrame, question: str) -> list[dict]:
        """Generate chart configs for the frontend to render with Chart.js."""
        charts = []
        q_lower = question.lower()
        years = subset["year"].tolist()
        temps = [round(float(v), 2) for v in subset["avg_temperature_c"]]
        rains = [round(float(v), 1) for v in subset["total_rainfall_mm"]]

        # Temperature line chart
        if any(kw in q_lower for kw in ["temp", "hot", "warm", "heat", "celsius", "cool"]):
            charts.append({
                "id": "tempLine",
                "type": "line",
                "title": "Temperature Trend (°C)",
                "labels": years,
                "datasets": [{
                    "label": "Avg Temperature (°C)",
                    "data": temps,
                    "borderColor": "#ef5350",
                    "backgroundColor": "rgba(239,83,80,0.1)",
                    "fill": True,
                    "tension": 0.35,
                }],
            })

        # Rainfall bar chart
        if any(kw in q_lower for kw in ["rain", "precip", "monsoon", "flood", "drought", "water", "wet"]):
            charts.append({
                "id": "rainBar",
                "type": "bar",
                "title": "Annual Rainfall (mm)",
                "labels": years,
                "datasets": [{
                    "label": "Total Rainfall (mm)",
                    "data": rains,
                    "backgroundColor": "rgba(66,165,245,0.6)",
                    "borderColor": "#42a5f5",
                    "borderWidth": 1,
                    "borderRadius": 4,
                }],
            })

        # Dual trend — for general trend / compare / overview / analysis
        if any(kw in q_lower for kw in ["trend", "compare", "overall", "analysis", "climate", "both", "data", "show"]):
            charts.append({
                "id": "dualLine",
                "type": "line",
                "title": "Temperature & Rainfall Trends",
                "labels": years,
                "datasets": [
                    {
                        "label": "Avg Temperature (°C)",
                        "data": temps,
                        "borderColor": "#ef5350",
                        "backgroundColor": "rgba(239,83,80,0.08)",
                        "fill": False,
                        "tension": 0.35,
                        "yAxisID": "y",
                    },
                    {
                        "label": "Total Rainfall (mm)",
                        "data": rains,
                        "borderColor": "#42a5f5",
                        "backgroundColor": "rgba(66,165,245,0.08)",
                        "fill": False,
                        "tension": 0.35,
                        "yAxisID": "y1",
                    },
                ],
                "dualAxis": True,
            })

        # Donut for distribution breakdown (latest year or average)
        if any(kw in q_lower for kw in ["distribution", "breakdown", "agriculture", "impact", "sector"]):
            avg_temp = round(float(subset["avg_temperature_c"].mean()), 2)
            avg_rain = round(float(subset["total_rainfall_mm"].mean()), 1)
            # Show proportion of hot months vs mild months
            hot_years = len(subset[subset["avg_temperature_c"] > avg_temp])
            mild_years = len(subset) - hot_years
            charts.append({
                "id": "tempDonut",
                "type": "doughnut",
                "title": f"Years Above vs Below Average ({avg_temp}°C)",
                "labels": ["Above Average", "Below Average"],
                "datasets": [{
                    "data": [hot_years, mild_years],
                    "backgroundColor": ["rgba(239,83,80,0.7)", "rgba(66,165,245,0.7)"],
                    "borderColor": ["#ef5350", "#42a5f5"],
                    "borderWidth": 2,
                }],
            })

        # Radar chart for seasonal/decade analysis
        if any(kw in q_lower for kw in ["pattern", "seasonal", "radar", "variab", "extreme"]):
            # Compute decade-level stats
            subset_copy = subset.copy()
            subset_copy["decade"] = (subset_copy["year"] // 10) * 10
            decade_stats = subset_copy.groupby("decade").agg(
                temp_mean=("avg_temperature_c", "mean"),
                rain_mean=("total_rainfall_mm", "mean"),
            ).reset_index()
            charts.append({
                "id": "decadeRadar",
                "type": "radar",
                "title": "Decade-Level Climate Profile",
                "labels": [f"{int(d)}s" for d in decade_stats["decade"]],
                "datasets": [
                    {
                        "label": "Avg Temp (°C)",
                        "data": [round(float(v), 2) for v in decade_stats["temp_mean"]],
                        "borderColor": "#ef5350",
                        "backgroundColor": "rgba(239,83,80,0.15)",
                    },
                    {
                        "label": "Avg Rainfall (mm)/10",
                        "data": [round(float(v) / 10, 1) for v in decade_stats["rain_mean"]],
                        "borderColor": "#42a5f5",
                        "backgroundColor": "rgba(66,165,245,0.15)",
                    },
                ],
            })

        # If no specific chart matched, give a default dual-line
        if not charts:
            charts.append({
                "id": "defaultLine",
                "type": "line",
                "title": "Climate Overview",
                "labels": years,
                "datasets": [
                    {
                        "label": "Temperature (°C)",
                        "data": temps,
                        "borderColor": "#ef5350",
                        "fill": False,
                        "tension": 0.35,
                        "yAxisID": "y",
                    },
                    {
                        "label": "Rainfall (mm)",
                        "data": rains,
                        "borderColor": "#42a5f5",
                        "fill": False,
                        "tension": 0.35,
                        "yAxisID": "y1",
                    },
                ],
                "dualAxis": True,
            })

        return charts

    def answer(self, question: str) -> tuple[str, str, list]:
        """Returns (answer_text, mode, charts_list)."""
        normalized_question = self._normalize_question(question)

        if self._is_greeting_or_smalltalk(normalized_question):
            return self._smalltalk_answer(question), "greeting", []

        is_climate = self._is_climate_question(normalized_question)
        subset = infer_year_slice(self.yearly_df, question) if is_climate else None
        charts = self._build_charts(subset, question) if is_climate and subset is not None else []

        if self.model is not None and self.tokenizer is not None:
            try:
                if is_climate and subset is not None:
                    context = context_for_llm(subset, self.summary, max_rows=min(15, len(subset)))
                    prompt = build_climate_prompt(question, context)
                else:
                    prompt = build_general_prompt(question)

                generated = generate_answer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )

                if is_climate and subset is not None:
                    generated = (
                        f"{generated}\n\n"
                        "Data Representation (source values used):\n"
                        f"{response_table(subset)}"
                    )
                return generated, "mixtral", charts
            except Exception as exc:  # pragma: no cover
                self.model_error = f"Mixtral inference failed: {exc}"

        if self.hf_client is not None:
            try:
                context = None
                if is_climate and subset is not None:
                    context = context_for_llm(subset, self.summary, max_rows=min(15, len(subset)))
                generated = self._answer_with_hf(question=question, is_climate=is_climate, context=context)

                if is_climate and subset is not None:
                    generated = (
                        f"{generated}\n\n"
                        "Data Representation (source values used):\n"
                        f"{response_table(subset)}"
                    )
                return generated, "hf-api", charts
            except Exception as exc:  # pragma: no cover
                self.hf_error = f"HF inference failed: {exc}"

        if is_climate and subset is not None:
            return self._fallback_climate_answer(question, subset), "fallback", charts

        return self._generic_fallback_answer(question), "fallback", []


class ClimateAPIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, ai_service: ClimateAIService, directory: str, **kwargs):
        self.ai_service = ai_service
        super().__init__(*args, directory=directory, **kwargs)

    def _write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/ask":
            self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            question = str(payload.get("question", "")).strip()

            if not question:
                self._write_json({"error": "question is required"}, status=HTTPStatus.BAD_REQUEST)
                return

            answer, mode, charts = self.ai_service.answer(question)
            answer = self.ai_service.sanitize_answer_text(answer)
            self._write_json(
                {
                    "answer": answer,
                    "mode": mode,
                    "data_mode": self.ai_service.data_mode,
                    "charts": charts,
                }
            )
        except json.JSONDecodeError:
            self._write_json({"error": "Invalid JSON payload"}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover
            self._write_json(
                {"error": f"Unexpected server error: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", ""} or self.path.startswith("/?"):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/frontend/")
            self.end_headers()
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if self.path in {"/", ""} or self.path.startswith("/?"):
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/frontend/")
            self.end_headers()
            return
        super().do_HEAD()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    ai_service = ClimateAIService(
        config=config,
        model_name=args.model_name,
        hf_model=args.hf_model,
        adapter_dir=args.adapter_dir,
        disable_model=args.disable_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    def handler_factory(*handler_args, **handler_kwargs):
        return ClimateAPIHandler(
            *handler_args,
            ai_service=ai_service,
            directory=str(PROJECT_ROOT),
            **handler_kwargs,
        )

    server = ThreadingHTTPServer((args.host, args.port), handler_factory)

    if ai_service.mode == "mixtral":
        mode_info = "Mixtral mode enabled"
    elif ai_service.mode == "hf-api":
        mode_info = f"HF API mode enabled (model={ai_service.hf_model})"
    else:
        mode_info = (
            f"Fallback mode enabled (local={ai_service.model_error}; hf={ai_service.hf_error})"
        )

    print(f"Serving on http://{args.host}:{args.port}")
    print(f"Frontend: http://{args.host}:{args.port}/frontend/")
    print("API endpoint: POST /api/ask")
    print(mode_info)
    server.serve_forever()


if __name__ == "__main__":
    main()
