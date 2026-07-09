"""Evaluate a saved FloraBERT prediction checkpoint on NAM test data."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from florabert import config, dataio, metrics, utils
from florabert import transformers as tr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Saved Hugging Face model directory",
    )
    parser.add_argument("--nshards", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.output / "model_eval" / "checkpoint_metrics.csv",
        help="Path for aggregate metrics CSV. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for metrics, predictions, and plots.",
    )
    return parser.parse_args()


def make_report_dir(args: argparse.Namespace) -> Path:
    if args.report_dir is not None:
        return args.report_dir
    return config.output / "reports" / args.checkpoint.name


def load_gene_metadata(sequences: list[str]) -> pd.DataFrame:
    """Attach gene metadata to final evaluated sequences where possible."""
    merged_path = config.data_final / "nam_data" / "merged_seq_genex.csv"
    merged = pd.read_csv(
        merged_path,
        usecols=["gene_id", "gene_id_full", "seq", "fold"],
    )
    merged = merged.loc[merged["fold"] == "test"].drop_duplicates("seq")
    metadata = pd.DataFrame({"sequence": sequences})
    metadata = metadata.merge(merged, left_on="sequence", right_on="seq", how="left")
    metadata = metadata.drop(columns=["seq"])
    metadata.insert(0, "example_idx", np.arange(len(metadata)))
    return metadata


def make_predictions_frame(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    true_np = y_true.cpu().numpy()
    pred_np = y_pred.cpu().numpy()

    rows = []
    for i in range(true_np.shape[0]):
        base = metadata.iloc[i].to_dict()
        base["sequence"] = base["sequence"][:80]
        for tissue_idx, tissue in enumerate(config.tissues):
            rows.append(
                {
                    **base,
                    "tissue": tissue,
                    "true": true_np[i, tissue_idx],
                    "pred": pred_np[i, tissue_idx],
                    "error": pred_np[i, tissue_idx] - true_np[i, tissue_idx],
                    "abs_error": abs(pred_np[i, tissue_idx] - true_np[i, tissue_idx]),
                }
            )
    return pd.DataFrame(rows)


def make_tissue_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tissue, df_tissue in predictions.groupby("tissue", sort=False):
        true = df_tissue["true"].to_numpy()
        pred = df_tissue["pred"].to_numpy()
        if len(df_tissue) > 1 and np.std(true) > 0 and np.std(pred) > 0:
            r2 = float(np.corrcoef(true.ravel(), pred.ravel())[0, 1] ** 2)
        else:
            r2 = np.nan
        rows.append(
            {
                "tissue": tissue,
                "mse": float(np.mean((pred - true) ** 2)),
                "mae": float(np.mean(np.abs(pred - true))),
                "r2": r2,
            }
        )
    return pd.DataFrame(rows)


def save_plots(
    predictions: pd.DataFrame,
    metrics_df: pd.DataFrame,
    tissue_metrics: pd.DataFrame,
    report_dir: Path,
) -> None:
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.scatter(predictions["true"], predictions["pred"], s=12, alpha=0.45)
    lo = min(predictions["true"].min(), predictions["pred"].min())
    hi = max(predictions["true"].max(), predictions["pred"].max())
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xlabel("True expression")
    plt.ylabel("Predicted expression")
    plt.title("All Tissues")
    plt.tight_layout()
    plt.savefig(plots_dir / "true_vs_pred_all.png", dpi=140)
    plt.close()

    for tissue, df_tissue in predictions.groupby("tissue", sort=False):
        plt.figure(figsize=(5, 4))
        plt.scatter(df_tissue["true"], df_tissue["pred"], s=14, alpha=0.55)
        lo = min(df_tissue["true"].min(), df_tissue["pred"].min())
        hi = max(df_tissue["true"].max(), df_tissue["pred"].max())
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        plt.xlabel("True expression")
        plt.ylabel("Predicted expression")
        plt.title(tissue)
        plt.tight_layout()
        plt.savefig(plots_dir / f"true_vs_pred_{tissue.replace(' ', '_')}.png", dpi=140)
        plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(tissue_metrics["tissue"], tissue_metrics["mae"])
    plt.ylabel("MAE")
    plt.xticks(rotation=35, ha="right")
    plt.title("MAE by Tissue")
    plt.tight_layout()
    plt.savefig(plots_dir / "mae_by_tissue.png", dpi=140)
    plt.close()

    plt.figure(figsize=(5, 3.5))
    plt.bar(metrics_df["metric"], metrics_df["value"])
    plt.ylabel("Value")
    plt.title("Aggregate Metrics")
    plt.tight_layout()
    plt.savefig(plots_dir / "aggregate_metrics.png", dpi=140)
    plt.close()


def main() -> int:
    args = parse_args()
    report_dir = make_report_dir(args)
    report_dir.mkdir(parents=True, exist_ok=True)

    settings = utils.get_model_settings(
        config.settings, model_name="roberta-pred-mean-pool"
    )
    _, tokenizer, model = tr.load_model(
        "roberta-pred-mean-pool",
        config.models / "byte-level-bpe-tokenizer",
        pretrained_model=args.checkpoint,
        **settings,
    )

    datasets = dataio.load_datasets(
        tokenizer,
        config.data_final / "transformer" / "genex" / "nam" / "test.tsv",
        file_type="csv",
        delimiter="\t",
        seq_key="sequence",
        shuffle=False,
        nshards=args.nshards,
        n_workers=1,
    )
    y_true, y_pred = metrics.get_predictions(model, datasets["train"])
    metadata = load_gene_metadata(datasets["train"]["sequence"])

    metric_fns = [
        torch.nn.MSELoss(),
        metrics.make_mae_loss(),
        utils.compute_r2,
        utils.compute_pseudo_r2,
    ]
    metric_names = ["mse", "mae", "r2", "pseudo_r2"]
    values = metrics.evaluate_model(y_true, y_pred, metric_fns)
    metrics_df = pd.DataFrame(
        {"metric": metric_names, "value": [float(v) for v in values]}
    ).assign(checkpoint=str(args.checkpoint), n_examples=len(datasets["train"]))
    predictions = make_predictions_frame(y_true, y_pred, metadata)
    tissue_metrics = make_tissue_metrics(predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.output, index=False)
    metrics_df.to_csv(report_dir / "metrics.csv", index=False)
    predictions.to_csv(report_dir / "predictions_long.csv", index=False)
    tissue_metrics.to_csv(report_dir / "tissue_metrics.csv", index=False)
    save_plots(predictions, metrics_df, tissue_metrics, report_dir)

    print(metrics_df.to_string(index=False))
    print(f"OK: saved metrics to {args.output}")
    print(f"OK: saved report to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
