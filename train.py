"""Train the small GPT on a plain text file.

Example:
    python train.py --data data/sample.txt --max-steps 1000

The checkpoint written to ``checkpoints/model.pt`` contains the weights, the
model config and the tokenizer vocabulary, so ``generate.py`` needs nothing else.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/sample.txt", help="training text file")
    p.add_argument("--out", default="checkpoints/model.pt", help="checkpoint path")

    # Model size
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)

    # Optimisation
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.1)

    # Bookkeeping
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device):
    # Random start offsets, leaving room for block_size inputs + 1 targets.
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, eval_batches, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Linear warmup, then cosine decay to ``min_lr_frac * lr``."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    progress = min(progress, 1.0)
    min_lr = args.lr * args.min_lr_frac
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    # --- data -------------------------------------------------------------
    if not os.path.isfile(args.data):
        raise SystemExit(
            f"Data file not found: {args.data}\n"
            "Use the bundled data/sample.txt, run prepare_data.py, or point "
            "--data at your own .txt file."
        )

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    if len(text) < args.block_size + 2:
        raise SystemExit("Training text is too short for this block size.")

    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(len(ids) * (1.0 - args.val_fraction))
    splits = {"train": ids[:split], "val": ids[split:]}
    if len(splits["val"]) < args.block_size + 2:
        splits["val"] = splits["train"]  # tiny corpus: validate on the same text

    print(f"Data: {args.data}  ({len(text):,} chars, {tok.vocab_size} token vocab)")
    print(f"Device: {device}")

    # --- model ------------------------------------------------------------
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"Model: {model.num_params() / 1e6:.2f}M parameters")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.99),
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_val = float("inf")
    start = time.time()

    for step in range(args.max_steps):
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(splits["train"], cfg.block_size, args.batch_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        log_now = step % args.eval_interval == 0 or step == args.max_steps - 1
        if log_now:
            losses = estimate_loss(
                model, splits, cfg.block_size, args.batch_size,
                args.eval_batches, device,
            )
            elapsed = time.time() - start
            print(
                f"step {step:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | lr {lr:.2e} | {elapsed:6.1f}s"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": cfg.to_dict(),
                        "chars": tok.chars,
                        "step": step,
                        "val_loss": best_val,
                    },
                    args.out,
                )

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Checkpoint saved to {args.out}")


if __name__ == "__main__":
    main()
