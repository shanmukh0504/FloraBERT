"""
Pretraining on masked language model task.
"""
import torch

from florabert import config, utils, training, dataio
from florabert import transformers as tr

DATA_DIR = config.data_final / "transformer" / "seq"
TOKENIZER_DIR = config.models / "byte-level-bpe-tokenizer"

OUTPUT_DIR = config.models / "transformer" / "language-model"


def main():
    args = utils.get_args(
        data_dir=DATA_DIR,
        train_data="all_seqs_train.txt",
        test_data="all_seqs_test.txt",
        output_dir=OUTPUT_DIR,
        model_name="roberta-lm",
    )

    settings = utils.get_model_settings(config.settings, args)

    print(args)

    config_obj, tokenizer, model = tr.load_model(
        args.model_name,
        TOKENIZER_DIR,
        pretrained_model=args.pretrained_model,
        **settings,
    )

    num_params = utils.count_model_parameters(model, trainable_only=True)
    print(f"Loaded {args.model_name} model with {num_params:,} trainable parameters")

    datasets = dataio.load_datasets(
        tokenizer,
        args.train_data,
        test_data=args.test_data,
        file_type="text",
        seq_key="text",
        nshards=args.nshards,
        n_workers=args.n_workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_train = datasets["train"]
    dataset_test = datasets["test"]
    print(f"Loaded training data with {len(dataset_train):,} examples")
    data_collator = dataio.load_data_collator(
        "language-model",
        tokenizer=tokenizer,
        mlm_prob=config.settings["training"]["pretrain"]["mlm_prob"],
    )

    training_settings = config.settings["training"]["pretrain"].copy()
    training_settings.pop("mlm_prob", None)
    if args.learning_rate is not None:
        training_settings["learning_rate"] = args.learning_rate
    if args.num_train_epochs is not None:
        training_settings["num_train_epochs"] = args.num_train_epochs
    if args.max_steps is not None:
        training_settings["max_steps"] = args.max_steps
    if args.per_device_train_batch_size is not None:
        training_settings["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.per_device_eval_batch_size is not None:
        training_settings["per_device_eval_batch_size"] = args.per_device_eval_batch_size
    if args.gradient_accumulation_steps is not None:
        training_settings["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    training_settings["report_to"] = []
    training_settings["prediction_loss_only"] = True
    if training_settings.get("load_best_model_at_end", True):
        eval_steps = training_settings.get("eval_steps")
        save_steps = training_settings.get("save_steps")
        if eval_steps and save_steps and save_steps % eval_steps != 0:
            training_settings["save_steps"] = eval_steps
    if training_settings.get("fp16") and not torch.cuda.is_available():
        print("CUDA is unavailable; disabling fp16")
        training_settings["fp16"] = False

    trainer = training.make_trainer(
        model,
        data_collator,
        dataset_train,
        dataset_test,
        args.output_dir,
        **training_settings,
    )

    print(f"Starting training on {torch.cuda.device_count()} GPUs")
    training.do_training(trainer, args, args.output_dir)

    print("Saving model")

    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
