"""Sample text from a trained checkpoint.

Examples:
    python generate.py --prompt "The fox"
    python generate.py --prompt "Once" --max-new-tokens 500 --temperature 0.8 --top-k 40
"""

from __future__ import annotations

import argparse

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/model.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.9,
                   help="lower = safer, higher = more surprising")
    p.add_argument("--top-k", type=int, default=40,
                   help="only sample from the k most likely tokens (0 = off)")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)

    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    tok = CharTokenizer(chars=ckpt["chars"])

    # Unknown characters in a prompt would crash encode(), so drop them.
    prompt = "".join(ch for ch in args.prompt if ch in tok.stoi)
    if args.prompt and not prompt:
        print(f"Note: prompt {args.prompt!r} has no characters the model knows.")

    idx = torch.tensor([tok.encode(prompt) or [0]], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k or None,
        )
        print(f"\n--- sample {i + 1} ---")
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
