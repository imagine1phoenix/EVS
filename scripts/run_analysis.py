#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from climate_evs.analysis import (  # noqa: E402
    build_yearly_climate_dataframe,
    interpretation_markdown,
    persist_interpretation,
    persist_summary,
    persist_yearly_data,
    summarize_trends,
)
from climate_evs.plots import save_standard_plots  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run climate trend analysis for EVS project")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/project_config.yaml"),
        help="Path to project config file",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Optional analysis start year",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Optional analysis end year",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    analysis_cfg = config["analysis"]
    paths = config["paths"]

    yearly_df = build_yearly_climate_dataframe(
        raw_dir=PROJECT_ROOT / paths["raw_dir"],
        country=analysis_cfg["country"],
        rainfall_subdivision=analysis_cfg["rainfall_subdivision"],
        recent_year_window=int(analysis_cfg.get("recent_year_window", 20)),
        minimum_year_window=int(analysis_cfg.get("minimum_year_window", 10)),
        start_year=args.start_year,
        end_year=args.end_year,
    )

    summary = summarize_trends(yearly_df)
    report_md = interpretation_markdown(summary)

    persist_yearly_data(
        yearly_df,
        csv_path=PROJECT_ROOT / paths["yearly_csv"],
        xlsx_path=PROJECT_ROOT / paths["yearly_xlsx"],
    )
    persist_summary(summary, PROJECT_ROOT / paths["summary_json"])
    persist_interpretation(report_md, PROJECT_ROOT / paths["report_markdown"])

    plot_paths = save_standard_plots(yearly_df, PROJECT_ROOT / paths["plots_dir"])

    print("Climate analysis complete.")
    print(f"Year range: {summary['year_span']['start']} - {summary['year_span']['end']}")
    print(
        "Temperature slope (C/year): "
        f"{summary['temperature_trend']['slope_per_year']:.4f} "
        f"({summary['temperature_trend']['direction']})"
    )
    print(
        "Rainfall slope (mm/year): "
        f"{summary['rainfall_trend']['slope_per_year']:.2f} "
        f"({summary['rainfall_trend']['direction']})"
    )
    print("\nSaved outputs:")
    print(f"- Yearly CSV: {paths['yearly_csv']}")
    print(f"- Yearly XLSX: {paths['yearly_xlsx']}")
    print(f"- Summary JSON: {paths['summary_json']}")
    print(f"- Interpretation MD: {paths['report_markdown']}")
    print("- Plot files:")
    for key, value in plot_paths.items():
        print(f"  - {key}: {Path(value).relative_to(PROJECT_ROOT)}")

    print("\nSummary JSON preview:")
    print(json.dumps(summary, indent=2)[:1200])


if __name__ == "__main__":
    main()
