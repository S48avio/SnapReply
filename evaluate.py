"""Step 5: measure how good the dual encoder is.

1-of-100 accuracy (Henderson's metric): for each (email, true reply) pair, mix the true
reply with 99 random replies from OTHER emails. Correct if the true one scores highest.
Random guessing gives 1%.

Recall@3 by intent: score each email against the whole reply bank (all unique training
replies), take the top 3, and check which of the email's true intents appear among them.
This is closer to the real product, and shows which intents the model under-suggests.

Run:  uv run evaluate.py               (test split)
      uv run evaluate.py --split val
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict

import torch

from features import Featurizer
from model import DualEncoder, encode_all


def load_split(name):
    with open(f"data/{name}.jsonl") as f:
        return [json.loads(line) for line in f]


def norm(text):
    # "Sure, I'll be there!" and "sure i'll be there" count as the same reply
    return re.sub(r"[^a-z0-9' ]", "", text.lower()).strip()


def one_of_k_accuracy(model, feat, emails, k=100, seed=0):
    pairs = [(i, r["text"]) for i, e in enumerate(emails) for r in e["replies"]]
    E = encode_all(model, [feat.encode(e["email"]) for e in emails], "email")
    R = encode_all(model, [feat.encode(t) for _, t in pairs], "reply")
    owner = torch.tensor([i for i, _ in pairs])
    text_ids = {t: n for n, t in enumerate({norm(t) for _, t in pairs})}
    text = torch.tensor([text_ids[norm(t)] for _, t in pairs])

    gen = torch.Generator().manual_seed(seed)
    correct = 0
    for start in range(0, len(pairs), 512):
        idx = torch.arange(start, min(start + 512, len(pairs)))
        neg = torch.randint(len(pairs), (len(idx), k - 1), generator=gen)
        cand = torch.cat([idx[:, None], neg], dim=1)                 # column 0 = true reply
        scores = (E[owner[idx]][:, None, :] * R[cand]).sum(-1)       # (batch, k)
        # A "negative" that is another reply of the same email, or the very same text,
        # isn't really wrong, so it doesn't count against the model.
        fake_neg = (owner[cand] == owner[idx][:, None]) | (text[cand] == text[idx][:, None])
        fake_neg[:, 0] = False
        scores[fake_neg] = float("-inf")
        correct += (scores.argmax(1) == 0).sum().item()
    return correct / len(pairs)


def build_bank(train_emails, min_count=1):
    """Unique training replies seen at least min_count times, with intent, count and
    the most common spelling (e.g. "Got it, thanks!" rather than "got it thanks")."""
    intents, spellings = defaultdict(Counter), defaultdict(Counter)
    for e in train_emails:
        for r in e["replies"]:
            key = norm(r["text"])
            intents[key][r["intent"]] += 1
            spellings[key][r["text"]] += 1
    return [{"text": spellings[k].most_common(1)[0][0], "intent": c.most_common(1)[0][0],
             "count": sum(c.values())}
            for k, c in intents.items() if sum(c.values()) >= min_count]


def recall_at_3(model, feat, emails, bank, show=3):
    E = encode_all(model, [feat.encode(e["email"]) for e in emails], "email")
    B = encode_all(model, [feat.encode(b["text"]) for b in bank], "reply")
    top3 = (E @ B.T).topk(3, dim=1).indices
    hits, totals = Counter(), Counter()
    for e, idx in zip(emails, top3.tolist()):
        suggested = {bank[j]["intent"] for j in idx}
        for intent in {r["intent"] for r in e["replies"]}:
            totals[intent] += 1
            hits[intent] += intent in suggested
    for e, idx in list(zip(emails, top3.tolist()))[:show]:
        print(f"\nEMAIL: {e['email'][:160]}")
        for j in idx:
            print(f"   -> [{bank[j]['intent']}] {bank[j]['text']}   (seen {bank[j]['count']}x)")
    return {i: hits[i] / totals[i] for i in totals}, sum(hits.values()) / sum(totals.values())


def load_model(path="checkpoints/model.pt"):
    ckpt = torch.load(path)
    feat = Featurizer.load(ckpt["vocab_path"])
    model = DualEncoder(feat.num_rows, **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="checkpoints/model.pt")
    args = ap.parse_args()

    model, feat = load_model(args.checkpoint)
    emails = load_split(args.split)
    bank = build_bank(load_split("train"))
    print(f"{args.split}: {len(emails)} emails | reply bank: {len(bank)} unique replies")
    print(f"1-of-100 accuracy: {one_of_k_accuracy(model, feat, emails):.1%}   (random = 1%)")
    per_intent, overall = recall_at_3(model, feat, emails, bank)
    print(f"\nRecall@3 overall: {overall:.1%}")
    for intent, r in sorted(per_intent.items(), key=lambda x: -x[1]):
        print(f"   {intent:15s} {r:.1%}")


if __name__ == "__main__":
    main()
