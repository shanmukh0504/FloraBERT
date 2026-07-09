"""Train a small FloraBERT prediction checkpoint for end-to-end validation.

This is intentionally lightweight. It trains from scratch on a shard of the
checked-in NAM transformer data and saves a local checkpoint that can be loaded
by scripts/predict.py and the evaluation utilities.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from florabert import config, dataio, training, utils
from florabert import transformers as tr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.models / "transformer" / "prediction-model-smoke",
    )
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--nshards", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    settings = utils.get_model_settings(
        config.settings, model_name="roberta-pred-mean-pool"
    )
    _, tokenizer, model = tr.load_model(
        "roberta-pred-mean-pool",
        config.models / "byte-level-bpe-tokenizer",
        **settings,
    )

    data_dir = config.data_final / "transformer" / "genex" / "nam"
    datasets = dataio.load_datasets(
        tokenizer,
        data_dir / "train.tsv",
        eval_data=data_dir / "eval.tsv",
        file_type="csv",
        delimiter="\t",
        seq_key="sequence",
        shuffle=True,
        nshards=args.nshards,
        n_workers=1,
    )

    trainer = training.make_trainer(
        model,
        dataio.load_data_collator("pred"),
        datasets["train"],
        datasets["eval"],
        args.output_dir,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        logging_steps=1,
        eval_steps=max(1, args.max_steps),
        save_steps=max(1, args.max_steps),
        save_total_limit=1,
        fp16=False,
        learning_rate=args.learning_rate,
        report_to=[],
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"OK: saved smoke prediction model to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
