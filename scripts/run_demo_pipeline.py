"""Run FloraBERT's local demo pipeline.

The default run is quick and validates the current repository setup without
retraining. Use --train-smoke to regenerate the local smoke checkpoint before
evaluating it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
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


def run(command: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def main() -> int:
    args = parse_args()
    print(f"Using Python: {PYTHON}", flush=True)
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    env.setdefault("HF_DATASETS_CACHE", str(ROOT / ".cache" / "huggingface" / "datasets"))
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

    run([str(PYTHON), "scripts/check_project_health.py"], env)
    run([str(PYTHON), "scripts/smoke_test_model.py"], env)

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
            ],
            env,
        )

    run(
        [
            str(PYTHON),
            "scripts/predict.py",
            "--gene-id",
            "Zm00001eb002390",
            "--checkpoint",
            str(SMOKE_CHECKPOINT.relative_to(ROOT)),
        ],
        env,
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
        ],
        env,
    )
    print("\nOK: demo pipeline completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
