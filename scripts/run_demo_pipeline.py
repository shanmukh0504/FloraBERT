"""Run FloraBERT's local demo pipeline.

The default run is quick and validates the current repository setup without
retraining. Use --train-smoke to regenerate the local smoke checkpoint before
evaluating it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SMOKE_CHECKPOINT = ROOT / "models" / "transformer" / "prediction-model-smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-smoke",
        action="store_true",
        help="Regenerate the local smoke checkpoint before evaluation.",
    )
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--nshards", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    if not PYTHON.exists():
        print("Missing .venv. Run `make python_requirements` first.", file=sys.stderr)
        return 1

    run([str(PYTHON), "scripts/check_project_health.py"])
    run([str(PYTHON), "scripts/smoke_test_model.py"])

    if args.train_smoke or not SMOKE_CHECKPOINT.exists():
        run(
            [
                str(PYTHON),
                "scripts/1-modeling/train_prediction_smoke.py",
                "--max-steps",
                str(args.max_steps),
                "--nshards",
                str(args.nshards),
                "--batch-size",
                str(args.batch_size),
            ]
        )

    run(
        [
            str(PYTHON),
            "scripts/predict.py",
            "--gene-id",
            "Zm00001eb002390",
            "--checkpoint",
            str(SMOKE_CHECKPOINT.relative_to(ROOT)),
        ]
    )
    run(
        [
            str(PYTHON),
            "scripts/1-modeling/evaluate_checkpoint.py",
            "--checkpoint",
            str(SMOKE_CHECKPOINT.relative_to(ROOT)),
            "--nshards",
            str(args.nshards),
            "--output",
            "output/model_eval/prediction-model-smoke.csv",
            "--report-dir",
            "output/reports/prediction-model-smoke",
        ]
    )
    print("\nOK: demo pipeline completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
