"""Lightweight repository health checks for FloraBERT.

This script intentionally uses only the Python standard library so it can run
before the heavy ML/data-science environment is installed.
"""
from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TISSUES = ["tassel", "base", "anther", "middle", "ear", "shoot", "tip", "root"]


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"OK: {message}")


def read_config_value(key: str) -> str:
    config_path = ROOT / "config.yaml"
    for line in config_path.read_text().splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    fail(f"missing config key {key!r}")


def check_config() -> None:
    num_tissues = int(read_config_value("num_tissues"))
    if num_tissues != len(EXPECTED_TISSUES):
        fail(f"config num_tissues={num_tissues}, expected {len(EXPECTED_TISSUES)}")
    ok(f"config num_tissues matches expected tissue count ({num_tissues})")


def check_merged_data() -> None:
    merged_path = ROOT / "data" / "final" / "nam_data" / "merged_seq_genex.csv"
    with merged_path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        first_row = next(reader)

    tissue_cols = header[4:]
    if tissue_cols != EXPECTED_TISSUES:
        fail(f"merged tissue columns are {tissue_cols}, expected {EXPECTED_TISSUES}")
    if not first_row[2] or set(first_row[2]) - set("ACGTNacgtn"):
        fail("first merged promoter sequence is empty or contains unexpected bases")
    ok(f"merged data has expected tissue columns ({len(tissue_cols)})")


def check_transformer_split(split: str) -> None:
    split_path = ROOT / "data" / "final" / "transformer" / "genex" / "nam" / f"{split}.tsv"
    with split_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["sequence", "labels"]:
            fail(f"{split}.tsv header is {reader.fieldnames}, expected ['sequence', 'labels']")
        row = next(reader)

    labels = ast.literal_eval(row["labels"])
    if len(labels) != len(EXPECTED_TISSUES):
        fail(f"{split}.tsv first label vector has {len(labels)} labels")
    if not row["sequence"] or set(row["sequence"]) - set("ACGTNacgtn"):
        fail(f"{split}.tsv first sequence is empty or contains unexpected bases")
    ok(f"{split}.tsv has expected sequence/label shape")


def check_tokenizer_files() -> None:
    tokenizer_dir = ROOT / "models" / "byte-level-bpe-tokenizer"
    missing = [
        str(path.relative_to(ROOT))
        for path in [tokenizer_dir / "vocab.json", tokenizer_dir / "merges.txt"]
        if not path.exists()
    ]
    if missing:
        fail(f"missing tokenizer files: {', '.join(missing)}")
    ok("byte-level BPE tokenizer files are present")


def check_checkpoint_hint() -> None:
    transformer_dir = ROOT / "models" / "transformer"
    if not transformer_dir.exists():
        print("WARN: models/transformer is missing; train or download checkpoints before evaluation")
        return
    checkpoints = list(transformer_dir.glob("**/pytorch_model.bin")) + list(
        transformer_dir.glob("**/model.safetensors")
    )
    if checkpoints:
        ok(f"found {len(checkpoints)} transformer checkpoint file(s)")
    else:
        print("WARN: no transformer checkpoint weights found under models/transformer")


def main() -> int:
    check_config()
    check_merged_data()
    for split in ["train", "eval", "test"]:
        check_transformer_split(split)
    check_tokenizer_files()
    check_checkpoint_hint()
    return 0


if __name__ == "__main__":
    sys.exit(main())
