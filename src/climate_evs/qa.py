from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def build_system_prompt() -> str:
    return (
        "You are a climate-analysis assistant for an EVS project. "
        "Always provide numeric data and concise interpretation. "
        "Every answer must include: "
        "1) a clear trend explanation, "
        "2) at least one data representation block (markdown table), "
        "3) practical climate-change implications."
    )


def _single_row_markdown(row: pd.Series) -> str:
    table = pd.DataFrame(
        {
            "Year": [int(row["year"])],
            "AvgTemperatureC": [round(float(row["avg_temperature_c"]), 3)],
            "TotalRainfallMm": [round(float(row["total_rainfall_mm"]), 3)],
        }
    )
    return table.to_markdown(index=False)


def _window_markdown(df: pd.DataFrame) -> str:
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


def build_context_payload(yearly_df: pd.DataFrame, summary: dict[str, Any]) -> str:
    t = summary["temperature_trend"]
    r = summary["rainfall_trend"]
    latest_table = _window_markdown(yearly_df.tail(10))

    lines = [
        f"Years covered: {summary['year_span']['start']} to {summary['year_span']['end']}",
        (
            "Temperature trend slope (C/year): "
            f"{t['slope_per_year']:.4f} ({t['direction']})"
        ),
        (
            "Rainfall trend slope (mm/year): "
            f"{r['slope_per_year']:.2f} ({r['direction']})"
        ),
        "Data representation (latest 10 years):",
        latest_table,
    ]
    return "\n".join(lines)


def generate_training_pairs(
    yearly_df: pd.DataFrame,
    summary: dict[str, Any],
    min_pairs: int = 80,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []

    base_questions = [
        "Explain the overall climate trend in this dataset.",
        "Is temperature increasing or decreasing in the selected years?",
        "Is rainfall increasing or decreasing in the selected years?",
        "Which year was the hottest and why is it important for climate-change analysis?",
        "Which year was the wettest and what could that imply for flood risk?",
        "Compare the first and last year for both temperature and rainfall.",
        "Give me a detailed EVS interpretation using data table representation.",
    ]

    t = summary["temperature_trend"]
    r = summary["rainfall_trend"]

    first = yearly_df.iloc[0]
    last = yearly_df.iloc[-1]

    for question in base_questions:
        answer_lines = [
            "Trend summary:",
            (
                f"- Temperature is {t['direction']} at about {t['slope_per_year']:.4f} C/year."
            ),
            (
                f"- Rainfall is {r['direction']} at about {r['slope_per_year']:.2f} mm/year."
            ),
            (
                "- This indicates climate variability that can affect agriculture, water resources, "
                "and local heat stress conditions."
            ),
            "",
            "Data representation:",
            _window_markdown(yearly_df.tail(8)),
            "",
            "First vs last year snapshot:",
            _window_markdown(pd.DataFrame([first, last])),
        ]
        pairs.append({"question": question, "answer": "\n".join(answer_lines)})

    for _, row in yearly_df.iterrows():
        year = int(row["year"])
        question = f"Give a climate summary for year {year}."
        answer = "\n".join(
            [
                f"For {year}, here is the observed climate data.",
                _single_row_markdown(row),
                "Interpretation: Compare this year with surrounding years to assess anomalies.",
            ]
        )
        pairs.append({"question": question, "answer": answer})

    years = yearly_df["year"].tolist()
    for start_index in range(0, max(1, len(years) - 4)):
        window = yearly_df.iloc[start_index : start_index + 5]
        if len(window) < 3:
            continue
        start_year = int(window.iloc[0]["year"])
        end_year = int(window.iloc[-1]["year"])
        question = (
            f"Analyze temperature and rainfall trend between {start_year} and {end_year} "
            "with data representation."
        )
        temp_change = (
            float(window.iloc[-1]["avg_temperature_c"])
            - float(window.iloc[0]["avg_temperature_c"])
        )
        rain_change = (
            float(window.iloc[-1]["total_rainfall_mm"])
            - float(window.iloc[0]["total_rainfall_mm"])
        )
        answer = "\n".join(
            [
                (
                    f"From {start_year} to {end_year}, temperature changed by {temp_change:.2f} C "
                    f"and rainfall changed by {rain_change:.2f} mm."
                ),
                "Data representation:",
                _window_markdown(window),
                "EVS implication: Track both long-term direction and short-term fluctuations.",
            ]
        )
        pairs.append({"question": question, "answer": answer})

    while len(pairs) < min_pairs:
        sample = yearly_df.sample(n=min(6, len(yearly_df)), random_state=len(pairs))
        sample = sample.sort_values("year")
        start_year = int(sample.iloc[0]["year"])
        end_year = int(sample.iloc[-1]["year"])
        question = (
            f"Prepare an EVS-ready explanation for sampled years {start_year} to {end_year}, "
            "including a data table."
        )
        answer = "\n".join(
            [
                "Climate pattern explanation with data representation:",
                _window_markdown(sample),
                (
                    "Interpretation: Use this table to discuss warming trends, rainfall shifts, "
                    "and adaptation planning in your project."
                ),
            ]
        )
        pairs.append({"question": question, "answer": answer})

    return pairs


def split_train_eval(
    pairs: list[dict[str, str]],
    eval_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)

    eval_size = max(1, int(len(shuffled) * eval_ratio))
    eval_set = shuffled[:eval_size]
    train_set = shuffled[eval_size:]
    return train_set, eval_set


def save_jsonl(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
