"""
data_utils.py
=============
Shared helpers for loading the SAMSum dataset and tokenizing examples.
Used by train.py and evaluate.py to avoid code duplication.
"""
import os

import yaml
from datasets import load_dataset, load_from_disk


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset(config: dict):
    """Load SAMSum from the local data/ cache, downloading if necessary."""
    data_dir = config["paths"]["data_dir"]
    dataset_name = config["dataset"]["name"]
    safe_name = dataset_name.replace("/", "__")
    save_path = os.path.join(data_dir, safe_name)
    cache_dir = os.path.join(data_dir, "hf_cache")

    if os.path.isdir(save_path):
        print(f"[data_utils] Loading cached dataset from '{save_path}'")
        return load_from_disk(save_path)

    print(f"[data_utils] Cache miss. Downloading '{dataset_name}' ...")
    os.makedirs(data_dir, exist_ok=True)
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    dataset.save_to_disk(save_path)
    return dataset


def maybe_subset(split_ds, n: int):
    """Return the first ``n`` examples (0 or None means full split)."""
    if n and n > 0 and n < len(split_ds):
        return split_ds.select(range(n))
    return split_ds


def build_tokenize_fn(tokenizer, config: dict):
    """Create a batched tokenization function for the dataset."""
    prefix = config["model"]["source_prefix"]
    max_in = config["model"]["max_input_length"]
    max_out = config["model"]["max_target_length"]
    text_col = config["dataset"]["text_column"]
    summary_col = config["dataset"]["summary_column"]

    def tokenize(batch):
        inputs = [prefix + (t or "") for t in batch[text_col]]
        model_inputs = tokenizer(
            inputs, max_length=max_in, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            text_target=batch[summary_col],
            max_length=max_out,
            truncation=True,
            padding="max_length",
        )
        # Replace pad token ids in labels by -100 so they are ignored in loss
        label_ids = []
        for seq in labels["input_ids"]:
            label_ids.append(
                [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
            )
        model_inputs["labels"] = label_ids
        return model_inputs

    return tokenize
