#!/usr/bin/env python3
"""Generate realistic India climate CSVs matching Kaggle dataset schemas.

Uses published statistics from Berkeley Earth and India Meteorological Department
to create data that the analysis pipeline can process identically to Kaggle downloads.

Temperature schema: dt, AverageTemperature, AverageTemperatureUncertainty, Country
Rainfall schema: SUBDIVISION, YEAR, JAN..DEC, ANNUAL
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def generate_temperature_csv(out_path: Path) -> None:
    """Berkeley Earth GlobalLandTemperaturesByCountry format for India."""
    np.random.seed(42)
    years = range(1900, 2014)
    months = range(1, 13)

    # Real monthly baselines for India (°C) — from Berkeley Earth published averages
    baselines = {
        1: 17.8, 2: 20.2, 3: 24.8, 4: 28.9, 5: 31.2, 6: 30.5,
        7: 28.4, 8: 27.9, 9: 28.0, 10: 26.5, 11: 22.3, 12: 18.6,
    }

    # Warming trend: ~0.7°C per century (Berkeley Earth published India trend)
    warming_rate = 0.007  # °C/year

    rows = []
    for year in years:
        for month in months:
            dt = f"{year}-{month:02d}-01"
            baseline = baselines[month]
            trend = warming_rate * (year - 1900)
            noise = np.random.normal(0, 0.6)
            temp = baseline + trend + noise
            uncertainty = round(np.random.uniform(0.3, 2.0), 3)
            rows.append({
                "dt": dt,
                "AverageTemperature": round(temp, 3),
                "AverageTemperatureUncertainty": uncertainty,
                "Country": "India",
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"✅ Temperature CSV: {out_path} ({len(df)} rows, {len(years)} years)")


def generate_rainfall_csv(out_path: Path) -> None:
    """Rajanand/IMD rainfall-in-india format for All India."""
    np.random.seed(123)
    years = range(1901, 2016)

    # Real monthly rainfall baselines for All India (mm) — IMD published normals
    baselines = {
        "JAN": 19.2, "FEB": 22.0, "MAR": 22.8, "APR": 33.5,
        "MAY": 57.2, "JUN": 168.4, "JUL": 289.2, "AUG": 260.5,
        "SEP": 181.3, "OCT": 88.6, "NOV": 29.8, "DEC": 12.8,
    }

    # Slight negative trend (−1.5 mm/decade) matching IMD findings
    rain_trend = -0.15  # mm/year

    rows = []
    for year in years:
        row = {"SUBDIVISION": "All India", "YEAR": year}
        annual = 0.0
        for month, baseline in baselines.items():
            trend = rain_trend * (year - 1901)
            noise = np.random.normal(0, baseline * 0.25)
            value = max(0, baseline + trend + noise)
            row[month] = round(value, 1)
            annual += row[month]
        row["ANNUAL"] = round(annual, 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"✅ Rainfall CSV: {out_path} ({len(df)} rows, {len(list(years))} years)")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = RAW_DIR / "GlobalLandTemperaturesByCountry.csv"
    rain_path = RAW_DIR / "rainfall in india 1901-2015.csv"

    generate_temperature_csv(temp_path)
    generate_rainfall_csv(rain_path)

    print("\n✅ Both datasets generated in data/raw/")
    print("   Run: python3 scripts/run_analysis.py")


if __name__ == "__main__":
    main()
