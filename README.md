# A small GPT that runs on your laptop

A complete, readable character-level GPT in plain PyTorch. No API keys, no cloud,
no GPU required — it trains and samples on a normal laptop CPU in minutes.

The idea is the same one behind nanoGPT: predict the next character, over and
over, with a decoder-only transformer. Everything is in ~400 lines across four
small files, so you can read the whole thing.

```
train.py  ->  tokenizer.py  (text -> integers)
          ->  model.py      (a small transformer)
          ->  checkpoints/model.pt  (weights + vocab + config)
                         |
generate.py  <-----------+
```

## What each file does

| File | Purpose |
| --- | --- |
| `tokenizer.py` | Character tokenizer: text <-> integer ids, save/load vocabulary |
| `model.py` | The GPT: embeddings, causal self-attention, MLP blocks, sampling |
| `train.py` | Training loop: batching, AdamW, warmup + cosine LR, validation, checkpointing |
| `generate.py` | Loads a checkpoint and samples text |
| `prepare_data.py` | Optional: downloads Tiny Shakespeare (~1 MB) for better results |
| `data/sample.txt` | Small bundled corpus so the project runs offline out of the box |
| `tests/test_smoke.py` | Fast checks for tokenizer and model |

The checkpoint is self-contained: it stores the weights, the model config and the
vocabulary, so `generate.py` needs nothing else.

## Quickstart

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# CPU-only PyTorch is much smaller than the default build:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 1) Train on the bundled sample (~3 minutes on a laptop CPU)
python train.py --data data/sample.txt

# 2) Generate text
python generate.py --prompt "The fox" --max-new-tokens 300
```

`train.py` keeps the checkpoint with the best validation loss in
`checkpoints/model.pt`.

## Better results: train on a bigger book

The bundled `data/sample.txt` is only ~8k characters; it is enough to prove the
pipeline works, but a model this size memorises it quickly (train loss falls far
below val loss).

For text that actually reads like English, use a larger corpus:

```bash
python prepare_data.py        # downloads data/tinyshakespeare.txt (~1.1 MB)
python train.py --data data/tinyshakespeare.txt --max-steps 600 --dropout 0.1
python generate.py --checkpoint checkpoints/model.pt \
    --prompt "ROMEO:" --max-new-tokens 300 --temperature 0.8 --top-k 40
```

Reference run on a CPU-only cloud VM with the default 0.82M-parameter model:

```
step     0 | train 4.1401 | val 4.1363
step   200 | train 2.3428 | val 2.3509
step   400 | train 2.0318 | val 2.1023
step   599 | train 1.8923 | val 1.9773
Best val loss: 1.9773        (600 steps, ~3.4 minutes)
```

Output is character-level and still rough at 600 steps — expect genuine words
and speaker names, not polished sentences. Train longer (`--max-steps 3000`) or
use a bigger model for cleaner text; the loss curve keeps improving steadily.

You can also point `--data` at any `.txt` file: a book, your notes, source code.

## Using your own text

```bash
python train.py --data path/to/book.txt --max-steps 2000
python generate.py --prompt "Once upon a time" --max-new-tokens 500
```

Tips:

- Bigger corpus beats bigger model. ~1 MB is a decent starting point.
- If train loss drops while val loss climbs, you are overfitting: add
  `--dropout 0.1`, lower `--n-embd`/`--n-layer`, or get more text.
- Characters the tokenizer never saw cannot be produced or prompted with. The
  vocabulary is built from whatever file you train on.

## Knobs that matter

Model (`model.py` defaults, changeable from `train.py`):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--n-layer` | 4 | transformer blocks |
| `--n-head` | 4 | attention heads per block |
| `--n-embd` | 128 | embedding width |
| `--block-size` | 128 | context length in characters |
| `--dropout` | 0.0 | use 0.1+ for small datasets |

Training:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--max-steps` | 1000 | optimisation steps |
| `--batch-size` | 32 | sequences per step |
| `--lr` | 3e-3 | peak learning rate |
| `--eval-interval` | 100 | print/evaluate cadence |
| `--device` | auto | `cpu`, `cuda` or `mps` |

Generation: `--temperature` (lower = safer, higher = more surprising),
`--top-k` (sample only from the k most likely characters, `0` disables),
`--num-samples` to draw several.

## How it works, briefly

1. **Tokenize.** Every unique character gets an id. A 1 MB book usually has
   60–100 unique characters, so the vocabulary is tiny.
2. **Embed.** Each id becomes a vector, and a learned position vector is added
   so the model knows where it is in the sequence.
3. **Attend.** In each block, every position produces a query, key and value.
   Attention lets each position pull information from earlier positions only
   (causal mask) — that is what makes it a language model rather than a bag of
   characters.
4. **Predict.** A final linear layer maps each position back to scores over the
   vocabulary, and cross-entropy against the next character gives the loss.
5. **Generate.** Feed a prompt, take the last position's probabilities, sample a
   character (optionally with temperature and top-k), append it, repeat.

## Tests

```bash
python tests/test_smoke.py        # or: pytest
```

## Troubleshooting

- `ModuleNotFoundError: No module named 'torch'` — install the CPU wheel as
  shown in Quickstart (and activate the venv).
- `Failed to initialize NumPy: No module named 'numpy'` — harmless warning from
  PyTorch; this project does not use NumPy. Ignore it, or `pip install numpy`.
- `Data file not found` — run `prepare_data.py`, or pass a correct `--data` path.
- Training feels slow — lower `--max-steps`, `--batch-size` or `--n-embd`.
