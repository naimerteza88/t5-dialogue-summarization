"""
predict.py
==========
Inference utilities for the fine-tuned T5 dialogue summarizer.

Can be used as a library (imported by app/app.py) or from the CLI:

    python src/predict.py --input path/to/dialogue.txt
    python src/predict.py --text "Amanda: I baked cookies. Jerry: Sure!"
"""
import argparse
import functools
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from data_utils import load_config
except ImportError:  # when imported from a different working dir
    import yaml

    def load_config(config_path: str) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")


@functools.lru_cache(maxsize=2)
def _load_model(model_dir: str):
    """Load (and cache) the tokenizer + model from a directory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model, device


def resolve_model_dir(config: dict, model_dir: str = None) -> str:
    """Return a usable model directory, falling back to the base checkpoint."""
    if model_dir and os.path.isdir(model_dir):
        return model_dir
    latest = os.path.join(config["training"]["output_dir"], "latest")
    if os.path.isdir(latest):
        return latest
    # Fall back to the pretrained checkpoint so the app still works pre-training
    print("[predict] No fine-tuned model found; using base checkpoint.")
    return config["model"]["checkpoint"]


def summarize(text: str, config: dict = None, model_dir: str = None,
              max_length: int = None, num_beams: int = 4) -> str:
    """Generate a summary for a single dialogue string."""
    if config is None:
        config = load_config(DEFAULT_CONFIG)
    resolved = resolve_model_dir(config, model_dir)
    tokenizer, model, device = _load_model(resolved)

    prefix = config["model"]["source_prefix"]
    max_in = config["model"]["max_input_length"]
    max_out = max_length or config["model"]["max_target_length"]

    inputs = tokenizer(
        prefix + text, return_tensors="pt", max_length=max_in, truncation=True
    ).to(device)
    with torch.no_grad():
        ids = model.generate(
            **inputs, max_length=max_out, num_beams=num_beams, length_penalty=1.0
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser(description="Summarize a dialogue with T5.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--input", default=None, help="Path to a text file with a dialogue.")
    p.add_argument("--text", default=None, help="Raw dialogue text.")
    args = p.parse_args()

    if not args.input and not args.text:
        args.text = (
            "Syahmi: what work you planning to give Tom?\n"
            "Putri: i was hoping to send him on a business trip first.\n"
            "Syahmi: cool. is there any suitable work for him?\n"
            "Putri: he did excellent in last quarter. i will assign new project, once he is back."
        )
        print("[predict] No input given; using a sample dialogue.\n")

    if args.input:
        with open(args.input, "r") as f:
            text = f.read()
    else:
        text = args.text

    config = load_config(args.config)
    summary = summarize(text, config=config, model_dir=args.model_dir)
    print("----- DIALOGUE -----")
    print(text.strip())
    print("\n----- SUMMARY -----")
    print(summary)


if __name__ == "__main__":
    main()
