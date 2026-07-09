"""Prepare text files for masked-language-model pretraining.

Each output line contains one promoter sequence. By default this reads the
checked-in processed NAM FASTA files and writes the files expected by
scripts/1-modeling/pretrain.py.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from florabert import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=config.data_processed / "Maize_nam",
        help="Directory containing processed FASTA files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.data_final / "transformer" / "seq",
        help="Directory where all_seqs_train.txt and all_seqs_test.txt are written.",
    )
    parser.add_argument("--test-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=config.settings["random_seed"])
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="Optional cap for quick pretraining experiments.",
    )
    return parser.parse_args()


def iter_fasta_sequences(paths: list[Path]):
    for path in paths:
        sequence_parts: list[str] = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if sequence_parts:
                        yield "".join(sequence_parts).upper()
                        sequence_parts = []
                else:
                    sequence_parts.append(line)
            if sequence_parts:
                yield "".join(sequence_parts).upper()


def write_sequences(path: Path, sequences: list[str]) -> None:
    with path.open("w") as handle:
        for sequence in sequences:
            handle.write(sequence)
            handle.write("\n")


def main() -> int:
    args = parse_args()
    fasta_paths = sorted(args.input_dir.glob("*/*.fa"))
    if not fasta_paths:
        raise FileNotFoundError(f"No FASTA files found under {args.input_dir}")

    sequences = [seq for seq in iter_fasta_sequences(fasta_paths) if seq]
    if args.max_seqs is not None:
        sequences = sequences[: args.max_seqs]
    if len(sequences) < 2:
        raise ValueError("Need at least two sequences to create train/test files")

    rng = random.Random(args.seed)
    rng.shuffle(sequences)
    test_count = max(1, int(len(sequences) * args.test_size))
    test_sequences = sequences[:test_count]
    train_sequences = sequences[test_count:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "all_seqs_train.txt"
    test_path = args.output_dir / "all_seqs_test.txt"
    write_sequences(train_path, train_sequences)
    write_sequences(test_path, test_sequences)

    print(f"OK: wrote {len(train_sequences):,} train sequences to {train_path}")
    print(f"OK: wrote {len(test_sequences):,} test sequences to {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
