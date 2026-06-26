"""
data_download.py
================
Automatically fetches the SAMSum dialogue-summarization dataset from the
HuggingFace Hub and caches it locally into the ``data/`` folder.

The raw dataset is NOT committed to git (see .gitignore); running this
script reproduces it on any fresh clone.

Usage:
    python src/data_download.py --config configs/config.yaml
"""
import argparse
import os

import yaml
from datasets import load_dataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_data(config: dict) -> str:
    """Download the dataset to the local data/ cache directory."""
    data_dir = config["paths"]["data_dir"]
    dataset_name = config["dataset"]["name"]
    os.makedirs(data_dir, exist_ok=True)

    cache_dir = os.path.join(data_dir, "hf_cache")
    print(f"[data_download] Downloading '{dataset_name}' into '{cache_dir}' ...")

    dataset = load_dataset(dataset_name, cache_dir=cache_dir)

    # Persist an arrow copy under data/ so reviewers can inspect splits.
    safe_name = dataset_name.replace("/", "__")
    save_path = os.path.join(data_dir, safe_name)
    dataset.save_to_disk(save_path)

    print("[data_download] Done. Splits available:")
    for split, ds in dataset.items():
        print(f"    - {split}: {len(ds)} examples")
    print(f"[data_download] Saved to: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Download SAMSum dataset.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    download_data(config)


if __name__ == "__main__":
    main()
