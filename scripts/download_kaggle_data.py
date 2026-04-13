#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_download(dataset_slug: str, output_dir: Path, force: bool) -> None:
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(output_dir),
        "--unzip",
    ]
    if force:
        command.append("--force")

    print(f"Downloading: {dataset_slug}")
    subprocess.run(command, check=True)


def validate_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "kaggle CLI is not installed. Install with: pip install kaggle"
        )

    credential_path = Path.home() / ".kaggle" / "kaggle.json"
    if not credential_path.exists():
        raise RuntimeError(
            "Missing ~/.kaggle/kaggle.json. Create Kaggle API token and place it there."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download climate datasets from Kaggle into data/raw"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/project_config.yaml"),
        help="Path to project config YAML",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override raw data directory",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset slug (repeatable), e.g., owner/dataset-name",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_kaggle_cli()

    config = load_config(args.config)
    raw_dir = args.raw_dir or Path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.dataset or config.get("kaggle", {}).get("datasets", [])
    if not datasets:
        raise ValueError("No Kaggle datasets configured. Add them in config/project_config.yaml")

    for dataset_slug in datasets:
        run_download(dataset_slug, output_dir=raw_dir, force=args.force)

    print("Done. Files are available in data/raw")


if __name__ == "__main__":
    main()
