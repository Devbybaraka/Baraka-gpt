"""A tiny character-level tokenizer.

Each unique character in the training text gets its own integer id. That keeps
the vocabulary small (usually 60-100 tokens) and makes the model easy to train
on a laptop. It is the same idea nanoGPT uses with ``data/shakespeare_char``.

Usage:
    python tokenizer.py data/sample.txt
"""

from __future__ import annotations

import json
import sys


class CharTokenizer:
    def __init__(self, text: str | None = None, chars: list[str] | None = None):
        if chars is None:
            if text is None:
                raise ValueError("Pass either text= or chars=")
            chars = sorted(set(text))

        self.chars = list(chars)
        self.vocab_size = len(self.chars)

        # Translation tables, both directions.
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_file(cls, filepath: str) -> "CharTokenizer":
        """Build a tokenizer from a training text file."""
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(text)

    def encode(self, s: str) -> list[int]:
        """String -> list of integer ids."""
        try:
            return [self.stoi[c] for c in s]
        except KeyError as exc:
            raise KeyError(
                f"Character {exc.args[0]!r} is not in the vocabulary. "
                "The tokenizer was built from a different text."
            ) from None

    def decode(self, nums) -> str:
        """List of integer ids -> string."""
        return "".join(self.itos[int(i)] for i in nums)

    def save(self, filepath: str = "tokenizer_vocab.json") -> None:
        """Write the vocabulary to disk as JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str = "tokenizer_vocab.json") -> "CharTokenizer":
        """Read a vocabulary saved with ``save()``."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(chars=data["chars"])

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.txt"
    tok = CharTokenizer.from_file(path)
    print(f"Loaded {path}")
    print(f"Vocabulary size: {tok.vocab_size}")
    print(f"Characters: {''.join(tok.chars)!r}")

    test_sentence = "Hello"
    numbers = tok.encode(test_sentence)
    back_to_text = tok.decode(numbers)
    print(f"\nEncode: {test_sentence!r} -> {numbers}")
    print(f"Decode: {numbers} -> {back_to_text!r}")
    print(f"Round trip ok: {back_to_text == test_sentence}")

    tok.save("tokenizer_vocab.json")
    print("\nSaved vocabulary to tokenizer_vocab.json")


if __name__ == "__main__":
    main()
