from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def _save_figure(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return str(path)


def plot_temperature_trend(yearly_df: pd.DataFrame, output_path: Path) -> str:
    plt.figure(figsize=(11, 5.5))
    plt.plot(
        yearly_df["year"],
        yearly_df["avg_temperature_c"],
        color="#c0392b",
        marker="o",
        linewidth=2,
        label="Avg Temperature (C)",
    )
    rolling = yearly_df["avg_temperature_c"].rolling(window=3, min_periods=1).mean()
    plt.plot(
        yearly_df["year"],
        rolling,
        color="#e67e22",
        linestyle="--",
        linewidth=2,
        label="3-year moving average",
    )

    plt.title("Yearly Temperature Trend")
    plt.xlabel("Year")
    plt.ylabel("Temperature (C)")
    plt.legend()
    return _save_figure(output_path)


def plot_rainfall_trend(yearly_df: pd.DataFrame, output_path: Path) -> str:
    plt.figure(figsize=(11, 5.5))
    plt.plot(
        yearly_df["year"],
        yearly_df["total_rainfall_mm"],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label="Total Rainfall (mm)",
    )
    rolling = yearly_df["total_rainfall_mm"].rolling(window=3, min_periods=1).mean()
    plt.plot(
        yearly_df["year"],
        rolling,
        color="#17becf",
        linestyle="--",
        linewidth=2,
        label="3-year moving average",
    )

    plt.title("Yearly Rainfall Trend")
    plt.xlabel("Year")
    plt.ylabel("Rainfall (mm)")
    plt.legend()
    return _save_figure(output_path)


def plot_dual_axis_trend(yearly_df: pd.DataFrame, output_path: Path) -> str:
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_temp = "#c0392b"
    color_rain = "#1f77b4"

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Temperature (C)", color=color_temp)
    ax1.plot(
        yearly_df["year"],
        yearly_df["avg_temperature_c"],
        color=color_temp,
        linewidth=2,
        marker="o",
    )
    ax1.tick_params(axis="y", labelcolor=color_temp)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Rainfall (mm)", color=color_rain)
    ax2.plot(
        yearly_df["year"],
        yearly_df["total_rainfall_mm"],
        color=color_rain,
        linewidth=2,
        marker="s",
    )
    ax2.tick_params(axis="y", labelcolor=color_rain)

    plt.title("Temperature vs Rainfall Trends")
    return _save_figure(output_path)


def plot_normalized_comparison(yearly_df: pd.DataFrame, output_path: Path) -> str:
    normalized = yearly_df.copy()
    temp_std = normalized["avg_temperature_c"].std(ddof=0)
    rain_std = normalized["total_rainfall_mm"].std(ddof=0)
    if temp_std == 0:
        temp_std = 1.0
    if rain_std == 0:
        rain_std = 1.0

    normalized["temp_norm"] = (
        normalized["avg_temperature_c"] - normalized["avg_temperature_c"].mean()
    ) / temp_std
    normalized["rain_norm"] = (
        normalized["total_rainfall_mm"] - normalized["total_rainfall_mm"].mean()
    ) / rain_std

    plt.figure(figsize=(11, 5.5))
    plt.plot(
        normalized["year"],
        normalized["temp_norm"],
        marker="o",
        linewidth=2,
        color="#8e44ad",
        label="Temperature (z-score)",
    )
    plt.plot(
        normalized["year"],
        normalized["rain_norm"],
        marker="s",
        linewidth=2,
        color="#16a085",
        label="Rainfall (z-score)",
    )

    plt.axhline(0, color="#555555", linewidth=1)
    plt.title("Normalized Climate Signal Comparison")
    plt.xlabel("Year")
    plt.ylabel("Standardized value (z-score)")
    plt.legend()
    return _save_figure(output_path)


def save_standard_plots(yearly_df: pd.DataFrame, plots_dir: Path) -> dict[str, str]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "temperature_plot": plot_temperature_trend(
            yearly_df, plots_dir / "temperature_trend.png"
        ),
        "rainfall_plot": plot_rainfall_trend(yearly_df, plots_dir / "rainfall_trend.png"),
        "dual_axis_plot": plot_dual_axis_trend(yearly_df, plots_dir / "dual_axis_trend.png"),
        "normalized_plot": plot_normalized_comparison(
            yearly_df, plots_dir / "normalized_comparison.png"
        ),
    }
    return outputs


def save_question_plot(yearly_df: pd.DataFrame, variable: str, output_path: Path) -> str:
    plt.figure(figsize=(10, 5))

    if variable == "rainfall":
        y_col = "total_rainfall_mm"
        color = "#1f77b4"
        y_label = "Rainfall (mm)"
        title = "Rainfall Trend for Question Context"
    else:
        y_col = "avg_temperature_c"
        color = "#c0392b"
        y_label = "Temperature (C)"
        title = "Temperature Trend for Question Context"

    plt.plot(yearly_df["year"], yearly_df[y_col], marker="o", linewidth=2, color=color)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(y_label)

    return _save_figure(output_path)
