"""Predict tissue expression values for a promoter sequence or gene ID.

Examples:
    python scripts/predict.py --gene-id Zm00001eb002390
    python scripts/predict.py --sequence GTCCCGTGCCTA...
    python scripts/predict.py --gene-id Zm00001eb002390 --checkpoint models/transformer/prediction-model
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from florabert import config, utils
from florabert import transformers as tr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence", help="Raw promoter DNA sequence")
    source.add_argument("--gene-id", help="Gene ID to look up in merged_seq_genex.csv")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional trained Hugging Face checkpoint/model directory",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=config.models / "byte-level-bpe-tokenizer",
        help="Directory containing vocab.json and merges.txt",
    )
    return parser.parse_args()


def load_gene_sequence(gene_id: str) -> str:
    merged_path = config.data_final / "nam_data" / "merged_seq_genex.csv"
    with merged_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["gene_id"] == gene_id:
                return row["seq"]
    raise ValueError(f"Could not find gene_id {gene_id!r} in {merged_path}")


def main() -> int:
    args = parse_args()
    sequence = args.sequence or load_gene_sequence(args.gene_id)
    sequence = sequence.strip().upper()
    unexpected = sorted(set(sequence) - set("ACGTN"))
    if unexpected:
        raise ValueError(f"Sequence contains unexpected bases: {unexpected}")

    settings = utils.get_model_settings(
        config.settings, model_name="roberta-pred-mean-pool"
    )
    _, tokenizer, model = tr.load_model(
        "roberta-pred-mean-pool",
        args.tokenizer_dir,
        pretrained_model=args.checkpoint,
        **settings,
    )
    model.eval()

    inputs = tokenizer(
        [sequence],
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    values = outputs.logits.squeeze(0).tolist()
    result = {
        "source": args.gene_id if args.gene_id else "sequence",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "trained_model": args.checkpoint is not None,
        "predictions": dict(zip(config.tissues, values)),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
