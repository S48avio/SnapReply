"""Step 1: clean the Kaggle-generated data and split it into train / val / test.

Input : data/smart_reply_data.jsonl   (one generated email per line)
Output: data/{train,val,test}.jsonl   (one email per line: id, email text, replies)

Run:  uv run prepare.py
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

RAW = Path("data/smart_reply_data.jsonl")
OUT_DIR = Path("data")
DROP_INTENTS = {"apologize"}  # only 81 examples in the generated data: too few to learn
SPLIT = (0.8, 0.1, 0.1)
SEED = 42

GREETING_NAME = re.compile(
    r"\b(Hi|Hello|Hey|Dear)\s+(?:(?:Mr|Mrs|Ms|Dr)\.?\s+)?(?!Team\b|All\b|Everyone\b)[A-Z][a-z]+")


def placeholder_token(match):
    # "[Recipient's Name]" -> "[NAME]", "[Date]" -> "[DATE]", ... (Google normalized
    # names/dates the same way, so thousands of variants share one token)
    inside = match.group(0).lower()
    if "company" in inside:
        return "[COMPANY]"
    if "name" in inside:
        return "[NAME]"
    if "date" in inside or "day" in inside:
        return "[DATE]"
    if "time" in inside:
        return "[TIME]"
    return "[THING]"


def clean(text):
    text = re.sub(r"\[[^\]\n]{1,40}\]", placeholder_token, text)
    text = GREETING_NAME.sub(r"\1 [NAME]", text)
    return " ".join(text.split())  # collapse newlines / repeated spaces


def main():
    emails = []
    for line in RAW.open():
        row = json.loads(line)
        if "replies" not in row:  # generation failed for this id
            continue
        replies = [{"intent": r["intent"], "text": clean(r["text"])}
                   for r in row["replies"] if r["intent"] not in DROP_INTENTS]
        if replies:
            emails.append({"id": row["id"],
                           "email": clean(f"{row['subject']}. {row['body']}"),
                           "replies": replies})

    # Split by EMAIL, not by pair: all replies of one email stay in the same split,
    # so the test set never contains an email the model saw during training.
    random.Random(SEED).shuffle(emails)
    n_train = int(SPLIT[0] * len(emails))
    n_val = int(SPLIT[1] * len(emails))
    splits = {"train": emails[:n_train],
              "val": emails[n_train:n_train + n_val],
              "test": emails[n_train + n_val:]}

    for name, rows in splits.items():
        with (OUT_DIR / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_pairs = sum(len(r["replies"]) for r in rows)
        print(f"{name:5s}: {len(rows):6d} emails | {n_pairs:6d} (email, reply) pairs")

    intents = Counter(r["intent"] for e in emails for r in e["replies"])
    print("replies per intent:", dict(intents.most_common()))
    print("\nexample:", json.dumps(splits["train"][0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
