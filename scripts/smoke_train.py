"""Run a one-step training smoke test on a small NAM data shard."""
from __future__ import annotations

from pathlib import Path

from florabert import config, dataio, training, utils
from florabert import transformers as tr


def main() -> int:
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
        shuffle=False,
        nshards=100,
        n_workers=1,
    )

    trainer = training.make_trainer(
        model,
        dataio.load_data_collator("pred"),
        datasets["train"],
        datasets["eval"],
        Path("/tmp/florabert-smoke-trainer"),
        num_train_epochs=1,
        max_steps=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        eval_steps=1,
        save_steps=1000,
        save_total_limit=1,
        fp16=False,
        learning_rate=1e-4,
        report_to=[],
    )
    trainer.train()
    print("OK: one-step training smoke test completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
