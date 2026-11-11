"""Quick checks: tokenizer round trip, model shapes, sampling length.

Run with pytest, or directly:
    python tests/test_smoke.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402


def test_tokenizer_round_trip():
    tok = CharTokenizer("hello world")
    text = "hello world"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.vocab_size == len(set(text))


def test_tokenizer_save_load(tmp_path=None):
    tok = CharTokenizer("abcdef")
    path = os.path.join(tmp_path or ".", "vocab_test.json")
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.chars == tok.chars
    assert loaded.encode("face") == tok.encode("face")
    os.remove(path)


def test_model_forward_and_generate():
    cfg = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)

    x = torch.randint(0, 32, (2, 16))
    logits, loss = model(x, x)
    assert logits.shape == (2, 16, 32)
    assert loss.item() > 0
    assert loss.item() > 3.0  # untrained model should be near uniform (~log 32)

    out = model.generate(x[:, :1], max_new_tokens=5, temperature=1.0, top_k=10)
    assert out.shape == (2, 6)


def test_block_size_guard():
    cfg = GPTConfig(vocab_size=8, block_size=4, n_layer=1, n_head=1, n_embd=8)
    model = GPT(cfg)
    try:
        model(torch.zeros(1, 5, dtype=torch.long))
    except ValueError:
        return
    raise AssertionError("expected ValueError for sequence longer than block_size")


if __name__ == "__main__":
    test_tokenizer_round_trip()
    test_tokenizer_save_load()
    test_model_forward_and_generate()
    test_block_size_guard()
    print("All smoke tests passed.")
