"""Step 4: train the dual encoder with in-batch negatives.

For a batch of B (email, reply) pairs we score every email against every reply:
a B x B matrix. The diagonal holds the true pairs; the rest of each row are the other
emails' replies, used as free negatives. Softmax cross-entropy on each row pulls the
diagonal up and pushes the rest down. (log P(y) is NOT used here: it's added only at
suggestion time.)

Run:  uv run train.py
      uv run train.py --layers 2 --frac 0.5 --out checkpoints/l2_half.pt   (experiments)
"""
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluate import load_split, norm, one_of_k_accuracy
from features import Featurizer
from model import DualEncoder, pack


def batch_loss(model, batch):
    """batch = list of (email_rows, reply_rows, normalized_reply_text)."""
    E = model.encode_emails(*pack([b[0] for b in batch]))
    R = model.encode_replies(*pack([b[1] for b in batch]))
    scores = E @ R.T                                          # (B, B)
    # If two emails in the batch got the identical reply ("Got it, thanks!"), that
    # off-diagonal reply is also correct: hide it instead of pushing it away.
    texts = [b[2] for b in batch]
    same = torch.tensor([[a == b for b in texts] for a in texts])
    same.fill_diagonal_(False)
    scores = scores.masked_fill(same, float("-inf"))
    return F.cross_entropy(scores, torch.arange(len(batch)))


def one_reply_per_email(encoded, rng):
    # One random reply per email each epoch: the same email never appears twice in a
    # batch, so its other replies can't become wrong "negatives". Over epochs all
    # replies get used.
    return [(e, *rng.choice(replies)) for e, replies in encoded]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3, help="stop after N epochs without val improvement")
    ap.add_argument("--frac", type=float, default=1.0, help="use this fraction of train emails (learning curve)")
    ap.add_argument("--out", default="checkpoints/model.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    train, val = load_split("train"), load_split("val")
    train = train[:int(len(train) * args.frac)]

    # The vocabulary comes from TRAINING text only; val/test words unseen in training map to <unk>.
    feat = Featurizer.build([e["email"] for e in train] + [r["text"] for e in train for r in e["replies"]])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    vocab_path = str(Path(args.out).with_suffix(".vocab.json"))
    feat.save(vocab_path)

    def encode(emails):
        return [(feat.encode(e["email"]), [(feat.encode(r["text"]), norm(r["text"])) for r in e["replies"]])
                for e in emails]
    train_enc, val_enc = encode(train), encode(val)
    val_pairs = one_reply_per_email(val_enc, random.Random(0))  # fixed, so epochs are comparable

    config = {"emb_dim": args.emb_dim, "hidden": args.hidden, "layers": args.layers}
    model = DualEncoder(feat.num_rows, **config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"train {len(train)} emails | vocab {len(feat.vocab)} words + {feat.num_rows - len(feat.vocab) - 1} "
          f"bigram buckets | {n_params / 1e6:.1f}M params ({model.embed.weight.numel() / n_params:.0%} in the embedding table)")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc, bad_epochs = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        pairs = one_reply_per_email(train_enc, rng)
        rng.shuffle(pairs)
        total = 0.0
        for i in range(0, len(pairs) - args.batch_size + 1, args.batch_size):
            loss = batch_loss(model, pairs[i:i + args.batch_size])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        train_loss = total / (len(pairs) // args.batch_size)

        model.eval()
        with torch.no_grad():
            val_losses = [batch_loss(model, val_pairs[i:i + args.batch_size]).item()
                          for i in range(0, len(val_pairs) - args.batch_size + 1, args.batch_size)]
            val_acc = one_of_k_accuracy(model, feat, val)
        val_loss = sum(val_losses) / len(val_losses)
        print(f"epoch {epoch:2d} | train loss {train_loss:.3f} | val loss {val_loss:.3f} | "
              f"val 1-of-100 {val_acc:.1%} | {time.time() - t0:.0f}s", flush=True)

        # Early stopping on val 1-of-100: keep the best model, stop when it stops improving.
        if val_acc > best_acc:
            best_acc, bad_epochs = val_acc, 0
            torch.save({"state_dict": model.state_dict(), "config": config, "vocab_path": vocab_path}, args.out)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"no improvement for {args.patience} epochs, stopping")
                break
    print(f"best val 1-of-100: {best_acc:.1%} -> saved {args.out}")


if __name__ == "__main__":
    main()
