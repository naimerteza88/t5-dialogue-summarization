"""
evaluate.py
===========
Compute ROUGE metrics for a fine-tuned T5 model on the SAMSum test split
and log the results to MLflow.

Usage:
    python src/evaluate.py --config configs/config.yaml
    python src/evaluate.py --model_dir models/t5-samsum/latest
"""
import argparse
import os

import mlflow
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from data_utils import get_dataset, load_config, maybe_subset


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate T5 summarizer with ROUGE.")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model_dir", default=None, help="Path to fine-tuned model dir.")
    return p.parse_args()


def generate_summary(model, tokenizer, text, config, device):
    prefix = config["model"]["source_prefix"]
    inputs = tokenizer(
        prefix + text,
        return_tensors="pt",
        max_length=config["model"]["max_input_length"],
        truncation=True,
    ).to(device)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_length=config["model"]["max_target_length"],
            num_beams=4,
            length_penalty=1.0,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def main():
    args = parse_args()
    config = load_config(args.config)

    model_dir = args.model_dir or os.path.join(config["training"]["output_dir"], "latest")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model dir '{model_dir}' not found. Train the model first (python src/train.py)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[evaluate] Loading model from '{model_dir}' on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    dataset = get_dataset(config)
    test_ds = maybe_subset(dataset["test"], config["dataset"]["test_subset"])

    text_col = config["dataset"]["text_column"]
    summary_col = config["dataset"]["summary_column"]

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(test_ds)
    print(f"[evaluate] Scoring {n} test examples ...")

    for i, ex in enumerate(test_ds):
        pred = generate_summary(model, tokenizer, ex[text_col], config, device)
        scores = scorer.score(ex[summary_col], pred)
        for k in totals:
            totals[k] += scores[k].fmeasure
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{n} done")

    results = {f"{k}_f": v / n for k, v in totals.items()}
    print("\n[evaluate] ===== ROUGE F1 (test) =====")
    for k, v in results.items():
        print(f"    {k}: {v:.4f}")

    # Log to MLflow under a dedicated evaluation run
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"eval-{os.path.basename(model_dir)}"):
        mlflow.log_param("eval_model_dir", model_dir)
        mlflow.log_param("test_examples", n)
        for k, v in results.items():
            mlflow.log_metric(k, v)

    # Save a small text report into artifacts/
    os.makedirs(config["paths"]["artifacts_dir"], exist_ok=True)
    report_path = os.path.join(config["paths"]["artifacts_dir"], "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("ROUGE F1 scores on SAMSum test split\n")
        f.write(f"Model: {model_dir}\n")
        f.write(f"Examples: {n}\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"[evaluate] Report written to '{report_path}'")


if __name__ == "__main__":
    main()
