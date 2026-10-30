"""Step 2: turn text into row numbers of the embedding table (unigrams + hashed bigrams).

Table layout:
    row 0                       -> <unk>  (any word not in the vocabulary)
    rows 1 .. V                 -> one row per vocabulary word
    rows V+1 .. V+NUM_BUCKETS   -> hashed bigrams (several bigrams may share a row)
"""
import json
import re
import zlib
from collections import Counter

NUM_BUCKETS = 50_000
MIN_COUNT = 2  # a word must appear at least this often in training to get its own row

# "can't" stays one token, and "?" is kept: a question mark changes the right reply a lot.
TOKEN = re.compile(r"\[[a-z]+\]|[a-z0-9]+(?:'[a-z]+)?|[?!]")


def tokenize(text):
    return TOKEN.findall(text.lower())


class Featurizer:
    def __init__(self, vocab):
        self.vocab = vocab  # word -> row (1..V)

    @classmethod
    def build(cls, texts):
        counts = Counter(tok for t in texts for tok in tokenize(t))
        words = sorted(w for w, c in counts.items() if c >= MIN_COUNT)
        return cls({w: i + 1 for i, w in enumerate(words)})

    @property
    def num_rows(self):
        return 1 + len(self.vocab) + NUM_BUCKETS

    def bigram_row(self, a, b):
        # zlib.crc32, not Python's hash(): hash() changes every time Python starts,
        # which would scramble the rows between training and inference.
        return 1 + len(self.vocab) + zlib.crc32(f"{a}_{b}".encode()) % NUM_BUCKETS

    def encode(self, text):
        toks = tokenize(text)
        rows = [self.vocab.get(t, 0) for t in toks]
        rows += [self.bigram_row(a, b) for a, b in zip(toks, toks[1:])]
        return rows or [0]  # empty text still needs one row to average

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))
