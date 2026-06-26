"""
train.py
========
Fine-tune a T5 transformer on the SAMSum dialogue-summarization dataset
using HuggingFace Trainer, with full MLflow experiment tracking.

Techniques used here:
  * Transfer learning  - start from the pretrained ``t5-small`` checkpoint.
  * Transformers       - encoder-decoder seq2seq model (T5).
  * MLflow tracking    - logs hyperparameters, metrics and the model artifact.
  * Hyperparameter tuning - CLI overrides allow running multiple experiments
                            with different learning rates / batch sizes.

Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --learning_rate 5e-4 \
        --batch_size 8 --run_name run-lr5e4-bs8
"""
import argparse
import os

import mlflow
import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from data_utils import build_tokenize_fn, get_dataset, load_config, maybe_subset

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune T5 for summarization.")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_train_epochs", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # ----- Apply CLI overrides (enables multiple HPO experiments) ----------
    lr = args.learning_rate or config["training"]["learning_rate"]
    batch_size = args.batch_size or config["training"]["per_device_train_batch_size"]
    epochs = args.num_train_epochs or config["training"]["num_train_epochs"]
    max_steps = args.max_steps if args.max_steps is not None else config["training"]["max_steps"]
    run_name = args.run_name or f"run-lr{lr}-bs{batch_size}"

    seed = config["training"]["seed"]
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device} | lr={lr} bs={batch_size} epochs={epochs} max_steps={max_steps}")

    # ----- Load model & tokenizer (transfer learning) ---------------------
    checkpoint = config["model"]["checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # ----- Data ------------------------------------------------------------
    dataset = get_dataset(config)
    train_ds = maybe_subset(dataset["train"], config["dataset"]["train_subset"])
    eval_ds = maybe_subset(dataset["validation"], config["dataset"]["eval_subset"])

    tokenize_fn = build_tokenize_fn(tokenizer, config)
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    output_dir = os.path.join(config["training"]["output_dir"], run_name)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps and max_steps > 0 else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        learning_rate=lr,
        weight_decay=config["training"]["weight_decay"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy="no",
        report_to=[],
        seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ----- MLflow tracking -------------------------------------------------
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_checkpoint": checkpoint,
                "learning_rate": lr,
                "batch_size": batch_size,
                "num_train_epochs": epochs,
                "max_steps": max_steps,
                "weight_decay": config["training"]["weight_decay"],
                "train_subset": len(train_tok),
                "eval_subset": len(eval_tok),
                "max_input_length": config["model"]["max_input_length"],
                "max_target_length": config["model"]["max_target_length"],
                "seed": seed,
            }
        )

        print("[train] Starting training ...")
        train_result = trainer.train()
        train_metrics = train_result.metrics

        print("[train] Evaluating ...")
        eval_metrics = trainer.evaluate()

        # Log metrics to MLflow
        mlflow.log_metric("train_loss", float(train_metrics.get("train_loss", np.nan)))
        mlflow.log_metric("train_runtime_sec", float(train_metrics.get("train_runtime", np.nan)))
        mlflow.log_metric("eval_loss", float(eval_metrics.get("eval_loss", np.nan)))

        # ----- Persist the fine-tuned model -------------------------------
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[train] Model saved to '{output_dir}'")

        # Also export a "latest" copy that the app / predict use by default
        latest_dir = os.path.join(config["training"]["output_dir"], "latest")
        trainer.save_model(latest_dir)
        tokenizer.save_pretrained(latest_dir)

        # Log the model directory as an MLflow artifact
        mlflow.log_artifacts(output_dir, artifact_path="model")

        print("[train] MLflow run complete.")
        print(f"    train_loss={train_metrics.get('train_loss')}")
        print(f"    eval_loss={eval_metrics.get('eval_loss')}")


if __name__ == "__main__":
    main()
