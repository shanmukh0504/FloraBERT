"""Smoke test current FloraBERT data loading and model forward pass.

This validates the minimum path needed before training:
  1. load project settings
  2. load the byte-level BPE tokenizer
  3. load a small shard of the NAM train TSV
  4. instantiate the mean-pool RoBERTa regressor
  5. run one labeled forward pass
"""
from __future__ import annotations

from torch.utils.data import DataLoader
from transformers import default_data_collator

from florabert import config, dataio, utils
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

    datasets = dataio.load_datasets(
        tokenizer,
        config.data_final / "transformer" / "genex" / "nam" / "train.tsv",
        file_type="csv",
        delimiter="\t",
        seq_key="sequence",
        shuffle=False,
        nshards=100,
        n_workers=1,
    )

    loader = DataLoader(
        datasets["train"],
        batch_size=2,
        collate_fn=default_data_collator,
    )
    batch = next(iter(loader))
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )

    expected_shape = (2, len(config.tissues))
    actual_shape = tuple(outputs.logits.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(f"Expected logits shape {expected_shape}, got {actual_shape}")
    if outputs.loss is None:
        raise RuntimeError("Expected a training loss from labeled forward pass")

    print(f"OK: forward pass loss={outputs.loss.item():.4f}, logits_shape={actual_shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
