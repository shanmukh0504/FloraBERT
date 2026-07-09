"""Smoke test prediction and metric computation on a small NAM test shard."""
from __future__ import annotations

from florabert import config, dataio, metrics, utils
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
        config.data_final / "transformer" / "genex" / "nam" / "test.tsv",
        file_type="csv",
        delimiter="\t",
        seq_key="sequence",
        shuffle=False,
        nshards=100,
        n_workers=1,
    )
    y_true, y_pred = metrics.get_predictions(model, datasets["train"])
    if tuple(y_true.shape) != tuple(y_pred.shape):
        raise RuntimeError(f"Prediction shape {y_pred.shape} != label shape {y_true.shape}")
    if y_pred.shape[1] != len(config.tissues):
        raise RuntimeError(f"Expected {len(config.tissues)} outputs, got {y_pred.shape[1]}")

    mse = metrics.evaluate_model(
        y_true, y_pred, [metrics.make_tissue_loss(0, metric="mse")]
    )[0]
    print(f"OK: evaluation smoke test completed, shape={tuple(y_pred.shape)}, mse0={mse:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
