from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MONTH_COLUMNS = {
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
}


@dataclass
class TrendResult:
    slope_per_year: float
    intercept: float
    r2: float
    absolute_change: float
    percent_change: float
    direction: str


def _normalize_columns(df: pd.DataFrame) -> dict[str, str]:
    return {column.lower().strip(): column for column in df.columns}


def _first_existing(raw_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(raw_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_temperature_yearly(path: Path, country: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_map = _normalize_columns(df)

    if {"dt", "averagetemperature", "country"}.issubset(col_map):
        dt_col = col_map["dt"]
        temp_col = col_map["averagetemperature"]
        country_col = col_map["country"]

        filtered = df[df[country_col].astype(str).str.lower() == country.lower()].copy()
        filtered[dt_col] = pd.to_datetime(filtered[dt_col], errors="coerce")
        filtered[temp_col] = pd.to_numeric(filtered[temp_col], errors="coerce")
        filtered = filtered.dropna(subset=[dt_col, temp_col])
        filtered["year"] = filtered[dt_col].dt.year

        yearly = (
            filtered.groupby("year", as_index=False)[temp_col]
            .mean()
            .rename(columns={temp_col: "avg_temperature_c"})
        )
        return yearly

    date_col = None
    year_col = None
    temp_col = None

    for candidate in ["date", "dt", "datetime"]:
        if candidate in col_map:
            date_col = col_map[candidate]
            break

    for candidate in ["year"]:
        if candidate in col_map:
            year_col = col_map[candidate]
            break

    for column in df.columns:
        if "temp" in column.lower():
            temp_col = column
            break

    if temp_col is None:
        raise ValueError(
            f"No temperature column found in {path.name}. Expected a column containing 'temp'."
        )

    working = df.copy()
    working[temp_col] = pd.to_numeric(working[temp_col], errors="coerce")

    if year_col is None:
        if date_col is None:
            raise ValueError(
                f"Could not infer year information from {path.name}. Add either a year or date column."
            )
        working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
        working["year"] = working[date_col].dt.year
    else:
        working["year"] = pd.to_numeric(working[year_col], errors="coerce")

    yearly = (
        working.dropna(subset=["year", temp_col])
        .groupby("year", as_index=False)[temp_col]
        .mean()
        .rename(columns={temp_col: "avg_temperature_c"})
    )
    yearly["year"] = yearly["year"].astype(int)
    return yearly


def _infer_month_columns(columns: list[str]) -> list[str]:
    month_cols: list[str] = []
    for column in columns:
        token = column.upper().strip()[:3]
        if token in MONTH_COLUMNS:
            month_cols.append(column)
    return month_cols


def load_rainfall_yearly(path: Path, subdivision: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_map = _normalize_columns(df)

    year_col = col_map.get("year")
    annual_col = col_map.get("annual")
    subdivision_col = col_map.get("subdivision")

    if year_col and annual_col:
        working = df.copy()
        if subdivision_col and subdivision:
            filtered = working[
                working[subdivision_col].astype(str).str.lower() == subdivision.lower()
            ]
            if not filtered.empty:
                working = filtered
        working[year_col] = pd.to_numeric(working[year_col], errors="coerce")
        working[annual_col] = pd.to_numeric(working[annual_col], errors="coerce")
        yearly = (
            working.dropna(subset=[year_col, annual_col])
            .groupby(year_col, as_index=False)[annual_col]
            .mean()
            .rename(columns={year_col: "year", annual_col: "total_rainfall_mm"})
        )
        yearly["year"] = yearly["year"].astype(int)
        return yearly

    month_cols = _infer_month_columns(list(df.columns))
    if year_col and month_cols:
        working = df.copy()
        if subdivision_col and subdivision:
            filtered = working[
                working[subdivision_col].astype(str).str.lower() == subdivision.lower()
            ]
            if not filtered.empty:
                working = filtered
        for month_col in month_cols:
            working[month_col] = pd.to_numeric(working[month_col], errors="coerce")
        working[year_col] = pd.to_numeric(working[year_col], errors="coerce")
        working["total_rainfall_mm"] = working[month_cols].sum(axis=1, skipna=True)
        yearly = (
            working.dropna(subset=[year_col, "total_rainfall_mm"])
            .groupby(year_col, as_index=False)["total_rainfall_mm"]
            .mean()
            .rename(columns={year_col: "year"})
        )
        yearly["year"] = yearly["year"].astype(int)
        return yearly

    date_col = None
    rain_col = None
    for candidate in ["date", "dt", "datetime"]:
        if candidate in col_map:
            date_col = col_map[candidate]
            break
    for column in df.columns:
        if "rain" in column.lower() or "precip" in column.lower():
            rain_col = column
            break

    if date_col and rain_col:
        working = df.copy()
        working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
        working[rain_col] = pd.to_numeric(working[rain_col], errors="coerce")
        working = working.dropna(subset=[date_col, rain_col])
        working["year"] = working[date_col].dt.year
        yearly = (
            working.groupby("year", as_index=False)[rain_col]
            .sum()
            .rename(columns={rain_col: "total_rainfall_mm"})
        )
        yearly["year"] = yearly["year"].astype(int)
        return yearly

    raise ValueError(
        f"No rainfall structure recognized in {path.name}. Expected annual/monthly or date+rainfall columns."
    )


def discover_default_sources(raw_dir: Path) -> tuple[Path, Path]:
    temperature_path = _first_existing(
        raw_dir,
        [
            "*GlobalLandTemperaturesByCountry*.csv",
            "*temperature*.csv",
            "*temp*.csv",
        ],
    )
    rainfall_path = _first_existing(
        raw_dir,
        [
            "*rainfall*.csv",
            "*precip*.csv",
            "*rain*.csv",
        ],
    )

    if temperature_path is None:
        raise FileNotFoundError(
            "Temperature file not found in data/raw. Download dataset and confirm CSV is present."
        )
    if rainfall_path is None:
        raise FileNotFoundError(
            "Rainfall file not found in data/raw. Download dataset and confirm CSV is present."
        )

    return temperature_path, rainfall_path


def compute_linear_trend(yearly_df: pd.DataFrame, value_column: str) -> TrendResult:
    x = yearly_df["year"].to_numpy(dtype=float)
    y = yearly_df[value_column].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = intercept + slope * x

    residual_ss = np.sum((y - fitted) ** 2)
    total_ss = np.sum((y - np.mean(y)) ** 2)
    r2 = 0.0 if total_ss == 0 else 1.0 - residual_ss / total_ss

    absolute_change = float(y[-1] - y[0])
    base = y[0]
    percent_change = float(np.nan) if base == 0 else float((absolute_change / abs(base)) * 100)

    if slope > 0:
        direction = "increasing"
    elif slope < 0:
        direction = "decreasing"
    else:
        direction = "stable"

    return TrendResult(
        slope_per_year=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        absolute_change=absolute_change,
        percent_change=percent_change,
        direction=direction,
    )


def build_yearly_climate_dataframe(
    raw_dir: Path,
    country: str,
    rainfall_subdivision: str,
    recent_year_window: int = 20,
    minimum_year_window: int = 10,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    temp_file, rain_file = discover_default_sources(raw_dir)

    temp_yearly = load_temperature_yearly(temp_file, country=country)
    rain_yearly = load_rainfall_yearly(rain_file, subdivision=rainfall_subdivision)

    merged = pd.merge(temp_yearly, rain_yearly, on="year", how="inner").sort_values("year")
    merged = merged.dropna(subset=["avg_temperature_c", "total_rainfall_mm"]).copy()

    if start_year is not None:
        merged = merged[merged["year"] >= start_year]
    if end_year is not None:
        merged = merged[merged["year"] <= end_year]

    merged = merged.sort_values("year")

    if start_year is None and end_year is None and len(merged) > recent_year_window:
        merged = merged.tail(recent_year_window)

    if len(merged) < minimum_year_window:
        raise ValueError(
            "Not enough overlapping temperature/rainfall years after filtering. "
            f"Need at least {minimum_year_window} years, found {len(merged)}."
        )

    return merged.reset_index(drop=True)


def summarize_trends(yearly_df: pd.DataFrame) -> dict[str, Any]:
    temp_trend = compute_linear_trend(yearly_df, "avg_temperature_c")
    rain_trend = compute_linear_trend(yearly_df, "total_rainfall_mm")

    hottest_row = yearly_df.loc[yearly_df["avg_temperature_c"].idxmax()]
    wettest_row = yearly_df.loc[yearly_df["total_rainfall_mm"].idxmax()]

    summary = {
        "year_span": {
            "start": int(yearly_df["year"].min()),
            "end": int(yearly_df["year"].max()),
            "count": int(len(yearly_df)),
        },
        "temperature_trend": asdict(temp_trend),
        "rainfall_trend": asdict(rain_trend),
        "hottest_year": {
            "year": int(hottest_row["year"]),
            "avg_temperature_c": float(hottest_row["avg_temperature_c"]),
        },
        "wettest_year": {
            "year": int(wettest_row["year"]),
            "total_rainfall_mm": float(wettest_row["total_rainfall_mm"]),
        },
    }
    return summary


def interpretation_markdown(summary: dict[str, Any]) -> str:
    span = summary["year_span"]
    t = summary["temperature_trend"]
    r = summary["rainfall_trend"]
    hottest = summary["hottest_year"]
    wettest = summary["wettest_year"]

    temp_pct = "N/A" if np.isnan(t["percent_change"]) else f"{t['percent_change']:.2f}%"
    rain_pct = "N/A" if np.isnan(r["percent_change"]) else f"{r['percent_change']:.2f}%"

    lines = [
        "# Climate Trend Interpretation",
        "",
        f"- Analysis period: {span['start']} to {span['end']} ({span['count']} years)",
        (
            f"- Temperature trend: {t['direction']} "
            f"({t['slope_per_year']:.4f} C/year, total change {t['absolute_change']:.2f} C, "
            f"{temp_pct}, R2={t['r2']:.3f})"
        ),
        (
            f"- Rainfall trend: {r['direction']} "
            f"({r['slope_per_year']:.2f} mm/year, total change {r['absolute_change']:.2f} mm, "
            f"{rain_pct}, R2={r['r2']:.3f})"
        ),
        (
            f"- Hottest year in the selected period: {hottest['year']} "
            f"({hottest['avg_temperature_c']:.2f} C)"
        ),
        (
            f"- Wettest year in the selected period: {wettest['year']} "
            f"({wettest['total_rainfall_mm']:.2f} mm)"
        ),
        "",
        "## EVS interpretation",
        "",
        "- If temperature is increasing, it indicates long-term warming pressure in the selected region.",
        "- If rainfall is decreasing, it may indicate drought risk and pressure on water resources.",
        "- If rainfall is increasing with high variability, flood and extreme-weather preparedness becomes important.",
        "- Use these trends with local context (urbanization, land use, policy changes) before final conclusions.",
    ]
    return "\n".join(lines)


def context_for_llm(yearly_df: pd.DataFrame, summary: dict[str, Any], max_rows: int = 15) -> str:
    clipped = yearly_df.tail(max_rows).copy()
    clipped = clipped.rename(
        columns={
            "year": "Year",
            "avg_temperature_c": "AvgTemperatureC",
            "total_rainfall_mm": "TotalRainfallMm",
        }
    )

    table_markdown = clipped.to_markdown(index=False)

    t = summary.get("temperature_trend", {})
    r = summary.get("rainfall_trend", {})
    hottest = summary.get("hottest_year", {})
    wettest = summary.get("wettest_year", {})
    span = summary.get("year_span", {})

    # Compute variability metrics from the provided DataFrame
    temp_std = float(yearly_df["avg_temperature_c"].std()) if len(yearly_df) > 1 else 0.0
    rain_std = float(yearly_df["total_rainfall_mm"].std()) if len(yearly_df) > 1 else 0.0
    temp_range = (
        float(yearly_df["avg_temperature_c"].max() - yearly_df["avg_temperature_c"].min())
        if len(yearly_df) > 1 else 0.0
    )
    rain_range = (
        float(yearly_df["total_rainfall_mm"].max() - yearly_df["total_rainfall_mm"].min())
        if len(yearly_df) > 1 else 0.0
    )

    context_lines = [
        f"Analysis period: {span.get('start', '?')} to {span.get('end', '?')} "
        f"({span.get('count', '?')} years)",
        "",
        "## Temperature Trend",
        f"- Direction: {t.get('direction', 'unknown')}",
        f"- Slope: {t.get('slope_per_year', 0):.4f} C/year",
        f"- Absolute change: {t.get('absolute_change', 0):.2f} C",
        f"- R² (goodness of fit): {t.get('r2', 0):.3f}",
        f"- Standard deviation: {temp_std:.3f} C",
        f"- Range (max-min): {temp_range:.2f} C",
        f"- Hottest year: {hottest.get('year', '?')} at {hottest.get('avg_temperature_c', '?'):.2f} C",
        "",
        "## Rainfall Trend",
        f"- Direction: {r.get('direction', 'unknown')}",
        f"- Slope: {r.get('slope_per_year', 0):.2f} mm/year",
        f"- Absolute change: {r.get('absolute_change', 0):.2f} mm",
        f"- R² (goodness of fit): {r.get('r2', 0):.3f}",
        f"- Standard deviation: {rain_std:.1f} mm",
        f"- Range (max-min): {rain_range:.1f} mm",
        f"- Wettest year: {wettest.get('year', '?')} at {wettest.get('total_rainfall_mm', '?'):.1f} mm",
        "",
        "## EVS Interpretation Hints",
        "- If temperature slope is positive, a warming trend is confirmed.",
        "- If rainfall slope is negative, drought risk and water stress increase.",
        "- High variability (large std dev) means year-to-year anomalies matter more than the trend.",
        "- Connect findings to: agriculture, urbanization, water resources, IPCC thresholds.",
        "",
        "## Yearly Data Table",
        table_markdown,
    ]
    return "\n".join(context_lines)


def persist_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def persist_yearly_data(yearly_df: pd.DataFrame, csv_path: Path, xlsx_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    yearly_df.to_csv(csv_path, index=False)
    yearly_df.to_excel(xlsx_path, index=False)


def persist_interpretation(markdown: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
