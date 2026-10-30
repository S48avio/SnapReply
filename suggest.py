"""Step 6: turn model scores into 3 Smart Reply suggestions.

    final = dot(email, reply) + alpha * log P(reply)   # fixes the rare-reply (PMI) bias
    -> keep the best reply per intent                   # diversity: no "Sure" + "Sure!"
    -> polarity rule (Kannan et al. 2016)               # offer a real yes AND no choice
    -> up to 3 suggestions

Run:  uv run suggest.py "Can you join tomorrow's meeting at 10?"
      uv run suggest.py --tune       (pick alpha and min_count on val, report on test)
"""
import argparse
import json
import math
from pathlib import Path

import torch

from evaluate import build_bank, load_model, load_split
from model import encode_all

POSITIVE = {"accept", "will_do"}
NEGATIVE = {"decline"}
CONFIG = Path("checkpoints/suggest.json")


class SmartReply:
    def __init__(self, model, feat, train_emails, alpha=1.0, min_count=2):
        self.model, self.feat, self.alpha = model, feat, alpha
        # Only replies seen min_count+ times: drops one-off oddities and makes P(y) reliable.
        self.bank = build_bank(train_emails, min_count)
        total = sum(b["count"] for b in self.bank)
        self.log_prior = torch.tensor([math.log(b["count"] / total) for b in self.bank])
        # Reply vectors don't depend on the email, so compute them once.
        self.B = encode_all(model, [feat.encode(b["text"]) for b in self.bank], "reply")
        self.intents = sorted({b["intent"] for b in self.bank})
        self.columns = [torch.tensor([j for j, b in enumerate(self.bank) if b["intent"] == i])
                        for i in self.intents]
        self.email_cache = {}

    def best_per_intent(self, email_texts):
        """For every email at once: the best reply of each intent -> (N, intents) scores and bank ids."""
        missing = [t for t in email_texts if t not in self.email_cache]
        if missing:  # email vectors don't depend on alpha, so cache them while tuning
            E = encode_all(self.model, [self.feat.encode(t) for t in missing], "email")
            self.email_cache.update(zip(missing, E))
        E = torch.stack([self.email_cache[t] for t in email_texts])
        S = E @ self.B.T + self.alpha * self.log_prior            # (N, bank_size)
        best = [S[:, cols].max(dim=1) for cols in self.columns]   # one max per intent
        scores = torch.stack([b.values for b in best], dim=1)
        ids = torch.stack([cols[b.indices] for cols, b in zip(self.columns, best)], dim=1)
        return scores, ids

    def pick(self, scores, ids, k=3):
        """scores, ids: one email's row from best_per_intent. Returns up to k bank entries."""
        order = scores.argsort(descending=True).tolist()
        ranked = [self.intents[i] for i in order]
        top = ranked[:k]
        # Kannan's rule: if the top 2 contain a "yes" but no "no" is shown, swap the
        # 3rd for the best "no" (and the other way round), so the user has a real choice.
        for have, need in ((POSITIVE, NEGATIVE), (NEGATIVE, POSITIVE)):
            if len(top) == k and set(top[:2]) & have and not set(top) & need:
                swap = next((i for i in ranked if i in need), None)
                if swap is not None:
                    top[-1] = swap
                break
        return [self.bank[ids[self.intents.index(i)].item()] for i in top]

    def suggest_many(self, email_texts):
        scores, ids = self.best_per_intent(email_texts)
        return [self.pick(s, i) for s, i in zip(scores, ids)]

    def suggest(self, email_text):
        return self.suggest_many([email_text])[0]


def recall_at_3(sr, emails):
    """Share of (email, true intent) pairs where that intent is among the 3 suggestions."""
    hits = total = 0
    for e, shown in zip(emails, sr.suggest_many([e["email"] for e in emails])):
        shown = {b["intent"] for b in shown}
        for intent in {r["intent"] for r in e["replies"]}:
            total += 1
            hits += intent in shown
    return hits / total


def tune(model, feat, train):
    val, test = load_split("val"), load_split("test")
    best = None
    print(f"{'min_count':>9} {'alpha':>6} {'bank':>6} {'val R@3':>8}  avg count of shown replies")
    for min_count in (1, 2, 3, 5):
        sr = SmartReply(model, feat, train, alpha=0.0, min_count=min_count)
        for alpha in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
            sr.alpha = alpha
            r = recall_at_3(sr, val)
            # How "common" the shown replies are: 1 = one-off oddities, big = generic.
            shown = [b["count"] for s in sr.suggest_many([e["email"] for e in val[:300]]) for b in s]
            print(f"{min_count:9d} {alpha:6.1f} {len(sr.bank):6d} {r:8.1%}  {sum(shown) / len(shown):8.1f}")
            if best is None or r > best[0]:
                best = (r, alpha, min_count)
    _, alpha, min_count = best
    sr = SmartReply(model, feat, train, alpha=alpha, min_count=min_count)
    print(f"\nbest on val: alpha={alpha}, min_count={min_count}")
    print(f"TEST Recall@3: {recall_at_3(sr, test):.1%}")
    CONFIG.write_text(json.dumps({"alpha": alpha, "min_count": min_count}))
    for e in test[:4]:
        print(f"\nEMAIL: {e['email'][:150]}")
        for b in sr.suggest(e["email"]):
            print(f"   -> [{b['intent']}] {b['text']}   (seen {b['count']}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("email", nargs="?", help="email text to reply to")
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()

    model, feat = load_model()
    train = load_split("train")
    if args.tune:
        tune(model, feat, train)
        return
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    sr = SmartReply(model, feat, train, **cfg)
    for b in sr.suggest(args.email or input("Email: ")):
        print(f"[{b['intent']}] {b['text']}")


if __name__ == "__main__":
    main()
