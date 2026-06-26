The model.safetensors weight file (~230 MB) for this MLflow run was intentionally
NOT committed because it exceeds GitHub's 100 MB file-size limit.

All experiment evidence is preserved: run parameters, metrics, tags, and the
small artifacts (tokenizer.json, spiece.model, configs, training_args.bin).

To regenerate the weights, run the corresponding training command from the README:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --learning_rate 5e-4 --batch_size 8 --run_name run-lr5e4-bs8
