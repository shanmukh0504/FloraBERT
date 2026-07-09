"""Testing tokenizers in the nlp.py module.
"""
import pytest

from florabert import config, nlp


def test_dnabert_tokenizer():
    tokenizer_dir = config.models / "dnabert" / "tokenizer"
    if not tokenizer_dir.exists():
        pytest.skip("DNABERT tokenizer files are not checked in")

    ex_seq = "AAATCGTCGCGGGCGCTCGCTATATATCGGCTAGCTAACTCGCCCG"
    tokenizer = nlp.DNABERTTokenizer.from_pretrained(
        tokenizer_dir, k=6, max_len=512
    )
    tokenized = tokenizer(ex_seq)
    decoded = tokenizer.decode(tokenized["input_ids"])

    assert (
        ex_seq == decoded
    ), f"Input ({ex_seq}) does not match decoded sequence ({decoded})."
