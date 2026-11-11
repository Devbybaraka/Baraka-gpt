"""Get a bigger training file (optional).

The repo ships with a small ``data/sample.txt`` so it runs offline straight
away. For noticeably better text, download Tiny Shakespeare (~1MB, public
domain) or just point ``--data`` at any plain text file you own.

Usage:
    python prepare_data.py
    python prepare_data.py --out data/my_book.txt --url <direct-download-url>
"""

from __future__ import annotations

import argparse
import os
import urllib.request

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=TINY_SHAKESPEARE_URL)
    p.add_argument("--out", default="data/tinyshakespeare.txt")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out):
        print(f"{args.out} already exists, nothing to do.")
        return

    print(f"Downloading {args.url} ...")
    try:
        urllib.request.urlretrieve(args.url, args.out)
    except Exception as exc:  # offline, blocked URL, bad link...
        print(f"Download failed: {exc}")
        print("No problem - data/sample.txt still works for training.")
        return

    size = os.path.getsize(args.out)
    print(f"Saved {args.out} ({size:,} bytes)")
    print(f"Train with: python train.py --data {args.out}")


if __name__ == "__main__":
    main()
