#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from climate_evs.qa import generate_training_pairs, save_jsonl, split_train_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training/evaluation Q&A dataset from climate trend outputs"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/project_config.yaml"),
        help="Path to project config file",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=80,
        help="Minimum number of generated question-answer pairs",
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
    paths = config["paths"]

    yearly_csv = PROJECT_ROOT / paths["yearly_csv"]
    summary_json = PROJECT_ROOT / paths["summary_json"]

    if not yearly_csv.exists() or not summary_json.exists():
        raise FileNotFoundError(
            "Missing analysis outputs. Run scripts/run_analysis.py first."
        )

    yearly_df = pd.read_csv(yearly_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))

    pairs = generate_training_pairs(yearly_df, summary, min_pairs=args.min_pairs)
    train_set, eval_set = split_train_eval(pairs)

    train_path = PROJECT_ROOT / paths["qa_jsonl"]
    eval_path = PROJECT_ROOT / paths["qa_eval_jsonl"]

    save_jsonl(train_set, train_path)
    save_jsonl(eval_set, eval_path)

    print("Q&A dataset generation complete.")
    print(f"- Train records: {len(train_set)} -> {train_path.relative_to(PROJECT_ROOT)}")
    print(f"- Eval records: {len(eval_set)} -> {eval_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
