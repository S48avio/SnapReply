"""Step 3: the dual encoder (Henderson et al. 2017): a Deep Averaging Network per side.

    text -> rows -> embedding table (shared) -> mean -> [Linear -> tanh] x N -> vector
    score(email, reply) = dot(email_vector, reply_vector)
"""
import torch
from torch import nn


def tower(emb_dim, hidden, layers):
    blocks, size = [], emb_dim
    for _ in range(layers):
        blocks += [nn.Linear(size, hidden), nn.Tanh()]
        size = hidden
    return nn.Sequential(*blocks)


class DualEncoder(nn.Module):
    def __init__(self, num_rows, emb_dim=128, hidden=256, layers=3):
        super().__init__()
        # EmbeddingBag = lookup rows + average them, in one fast step.
        # Shared by both towers: "meeting" means the same in an email and a reply.
        self.embed = nn.EmbeddingBag(num_rows, emb_dim, mode="mean")
        # Separate towers: an email asks, a reply answers; each learns its own job.
        self.email_tower = tower(emb_dim, hidden, layers)
        self.reply_tower = tower(emb_dim, hidden, layers)

    def encode_emails(self, rows, offsets):
        return self.email_tower(self.embed(rows, offsets))

    def encode_replies(self, rows, offsets):
        return self.reply_tower(self.embed(rows, offsets))


def pack(row_lists):
    """[[3, 7], [5, 1, 9]] -> rows [3, 7, 5, 1, 9], offsets [0, 2]: EmbeddingBag's input format."""
    offsets, total = [], 0
    for r in row_lists:
        offsets.append(total)
        total += len(r)
    flat = [x for r in row_lists for x in r]
    return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


@torch.no_grad()
def encode_all(model, row_lists, side, batch_size=1024):
    """Encode many texts (side = "email" or "reply") into one (N, hidden) tensor."""
    fn = model.encode_emails if side == "email" else model.encode_replies
    out = [fn(*pack(row_lists[i:i + batch_size])) for i in range(0, len(row_lists), batch_size)]
    return torch.cat(out)
