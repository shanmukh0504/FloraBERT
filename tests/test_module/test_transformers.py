import pytest

from torch.optim import Adam

from florabert import config, utils
from florabert import training
from florabert import transformers as tr


# Helper functions
def load_roberta_model(**kwargs):
    return tr.load_model(
        "roberta-pred-mean-pool", config.models / "byte-level-bpe-tokenizer", **kwargs
    )


# Tests
def test_roberta_mean_pool_load_new():
    try:
        load_roberta_model()
    except:
        raise Exception(
            "Failed to load new RobertaForSequenceClassificationMeanPool model"
        )


def test_roberta_mean_pool_load_pretrained():
    pretrained_path = config.models / "transformer" / "language-model"
    if not pretrained_path.exists():
        pytest.skip("Pretrained transformer checkpoint is not checked in")

    try:
        load_roberta_model(pretrained_model=pretrained_path)
    except:
        raise Exception(
            "Failed to load pretrained RobertaForSequenceClassificationMeanPool model"
        )


def test_get_lamb_optimizer():
    _, _, model = load_roberta_model()
    optimizer = training._get_optimizer(
        "lamb",
        model,
        num_param_groups=None,
        param_group_size=None,
        learning_rate=1e-4,
    )
    assert optimizer is not None, "Failed to load optimizer"


def test_linear_scheduler():
    param = load_roberta_model()[2].classifier.out_proj.parameters()
    optimizer = Adam(param)
    training._get_scheduler("linear", optimizer, 10000, num_warmup_steps=500)


def test_delay_scheduler():
    model = load_roberta_model()[2]
    optimizer = training._get_optimizer(
        "adam",
        model,
        num_param_groups=4,
        param_group_size=2,
        learning_rate=1e-4,
    )
    training._get_scheduler(
        "delay",
        optimizer,
        10000,
        num_warmup_steps=500,
        num_param_groups=len(optimizer.param_groups),
        delay_size=400,
    )


def test_make_trainer_simple():
    pass


def test_make_trainer_delay():
    pass


def test_load_datasets():
    pass


def test_convert_str_to_list():
    data = ["[32.0, 430.5]", "[20.0, 0.01]", "[-1419, 4]"]
    lists = utils.convert_str_to_tnsr({"labels": data})["labels"]

    assert type(lists) == list, "Result is not a list"
    assert all([type(li) == list] for li in lists), "Inner lists are not lists"
    assert all(
        [all([type(d) == float for d in li]) for li in lists]
    ), "Elements are not float"
