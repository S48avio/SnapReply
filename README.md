<p align="center">
  <img src="assets/logo.svg" alt="SnapReply logo" width="140">
</p>

<h1 align="center">SnapReply</h1>

<p align="center">
  Gmail-style <b>Smart Reply</b>, built from scratch: a dual-encoder retrieval model that suggests
  3 short replies to an email, and trains and runs on a laptop CPU.
</p>

---

## What it does

```
$ uv run suggest.py "Please send me the Q3 sales report by Friday."
[will_do] Sure, I'll send it over right away.
[accept]  Sure, I'll send it over now.
[decline] I'm sorry, I won't be able to do that.
```

The model never generates text. It **picks** the best replies from a fixed bank of replies
seen in training, just like Google's production system did.

## How it works

The design follows two Google papers:
[Kannan et al., 2016](https://arxiv.org/abs/1606.04870) (Smart Reply in Inbox) and
[Henderson et al., 2017](https://arxiv.org/abs/1705.00652) (the faster dual-encoder version).

```
email ─► unigrams + hashed bigrams ─► shared embedding table ─► mean ─► 3 × (Linear → tanh) ─► email vector ─┐
                                                                                                              ├─► dot product
reply ─► unigrams + hashed bigrams ─► shared embedding table ─► mean ─► 3 × (Linear → tanh) ─► reply vector ─┘
```

1. **Encoder:** a Deep Averaging Network ([Iyyer et al., 2015](https://aclanthology.org/P15-1162/)) per
   side. Words and hashed bigrams (`zlib.crc32`, 50k buckets) are looked up in one shared table and
   averaged. Each side then has its own feedforward tower. About 7.4M parameters, 96% of them in the
   embedding table.
2. **Training:** in-batch negatives. In a batch of 64 pairs, each email's 63 *other* replies act as
   negatives, and softmax cross-entropy pulls the true pair together. Each epoch uses one reply per email,
   and identical reply texts are masked, so correct replies are never pushed away.
3. **Suggesting** (`suggest.py`):
   - `score = dot(email, reply) + α · log P(reply)`. In-batch training learns roughly PMI, which favors
     rare, oddly specific replies. The frequency prior corrects that (α is tuned on validation).
   - **Diversity:** at most one reply per intent, so you never see "Sure!" next to "Sure.".
   - **Polarity rule** (Kannan 2016): if the top 2 contain a "yes" and no "no" is shown, the 3rd slot
     becomes the best "no", so the user always has a real choice.

## Data

There's no public Gmail data, so the training set is **synthetic**:
[`kaggle_generate_data.py`](kaggle_generate_data.py) runs **Qwen2.5-7B-Instruct** with vLLM on Kaggle's
free 2× T4 GPUs.

- It first writes an email from a random setup (83 topics × 22 senders × 8 tones × 5 lengths).
- It then writes 3 replies with different intents for that email. The email is written *before* the
  intents are chosen, so the email itself gives away no hints.
- 20,000 emails took about 99 minutes and produced 17,819 valid emails and about 53k (email, reply) pairs.
- 9 intents: accept, decline, thanks, ask_question, will_do, reschedule, acknowledge, need_more_info,
  follow_up. (`apologize` was dropped: only 81 examples.)

## Results

Test split: 1,783 emails never seen during training.

| | Score |
|---|---|
| 1-of-100 accuracy (raw model; random = 1%) | **25.3%** |
| Recall@3 by intent, raw top-3 | 42.0% |
| Recall@3 by intent, full `suggest.py` (α = 0.5, replies seen ≥ 2×) | **64.0%** |

Most of the jump from 42% to 64% comes from the one-reply-per-intent rule. The α · log P(y) prior adds
about 2 points and, more visibly, swaps one-off replies for natural, common ones. In the tuning output,
the average training count of a shown reply goes from about 4 (α = 0) to about 22 (α = 0.5).

### Limitations (honest ones)
- The test emails come from the same generator as the training data, so **real-world quality is lower**
  than 64% suggests. On hand-written emails the suggestions are mixed.
- Pure thank-you or FYI emails are handled poorly, because the generator was told every email must need a reply.
- Rare intents (reschedule, follow_up) are under-suggested, because the data is imbalanced.
- The model overfits after about 15 epochs (train loss 1.2 vs val 2.7). Dropout or weight decay should help.

## Quick start

```bash
uv sync                                   # CPU-only PyTorch
# put the generated data at data/smart_reply_data.jsonl (see kaggle_generate_data.py)
uv run prepare.py                         # clean + split into train/val/test
uv run train.py                           # ~22 s/epoch on a laptop i5
uv run evaluate.py                        # 1-of-100 and Recall@3 on the test split
uv run suggest.py --tune                  # pick alpha and min_count on val
uv run suggest.py "Can you send me the slides?"
```

## Project layout

| File | Role |
|---|---|
| `kaggle_generate_data.py` | One-cell Kaggle script that generates the synthetic dataset |
| `prepare.py` | Normalizes placeholders and names to `[NAME]` etc., then splits by email 80/10/10 |
| `features.py` | Tokenizer, vocabulary and hashed bigrams: text → embedding-table rows |
| `model.py` | Dual encoder (shared `EmbeddingBag` and two tanh towers) |
| `train.py` | In-batch softmax training with early stopping |
| `evaluate.py` | 1-of-100 accuracy and Recall@3 by intent |
| `suggest.py` | α · log P(y), intent diversity, polarity rule; CLI and tuning |

## Next steps
- Dropout and weight decay against overfitting; experiments with layer count, tanh vs ReLU, and a learning curve
- More data for rare intents and for thank-you/FYI emails
- A small set of **real** emails as an honest test set
- A confidence threshold, and a FastAPI demo
