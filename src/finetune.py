"""Fine-tune the text encoder on inspector comments (deep learning on unstructured data).

M2 uses the encoder frozen: it turns comments into numbers without ever seeing our labels.
Here the same encoder keeps training, with a small classification head on top, so that it
learns which words in a previous visit's comments go with a failed next inspection.
(Standard recipe: AutoModelForSequenceClassification, AdamW, a few epochs.)

Leakage rule: a text model only ever scores inspections dated AFTER everything it was
trained on. `forward_text_scores` enforces this with time blocks.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.text import MODEL_NAME, MODEL_REVISION, get_device

SEED = 42
MAX_TOKENS = 128        # 3x faster than 256; comment length made no measurable difference before
EPOCHS = 2
LEARNING_RATE = 5e-5
BATCH_SIZE = 32

# (train on inspections dated before, score inspections from, up to but not including)
BLOCKS_VALIDATION = [("2016-01-01", "2016-01-01", "2019-01-01"),
                     ("2019-01-01", "2019-01-01", "2022-01-01"),
                     ("2022-01-01", "2022-01-01", "2023-01-01"),
                     ("2023-01-01", "2023-01-01", "2024-01-01")]     # the last block scores the validation year
BLOCK_TEST = ("2024-01-01", "2024-01-01", "2100-01-01")             # trained on everything before the test period


def fine_tune(texts, labels, seed=SEED, epochs=EPOCHS, device=None, verbose=True):
    """Train the encoder plus a 2-class head on (text, failed or not). Returns (tokenizer, model)."""
    device = device or get_device()
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, num_labels=2).to(device)

    texts, labels = list(texts), torch.tensor(np.asarray(labels, dtype=np.int64))
    steps = epochs * math.ceil(len(texts) / BATCH_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # Learning rate rises for the first 6% of steps, then falls to zero: the usual recipe for fine-tuning.
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    rng = np.random.default_rng(seed)

    model.train()
    for epoch in range(epochs):
        order, losses = rng.permutation(len(texts)), []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            batch = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_TOKENS,
                              return_tensors="pt").to(device)
            loss = model(**batch, labels=labels[idx].to(device)).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        if verbose:
            print(f"    epoch {epoch + 1}/{epochs}: mean training loss {np.mean(losses):.3f}", flush=True)
    return tokenizer, model.eval()


@torch.no_grad()
def predict_fail_probability(tokenizer, model, texts, batch_size=128):
    """P(next inspection fails) for each text, from a fine-tuned model."""
    texts = list(texts)
    device = next(model.parameters()).device
    order = np.argsort([len(t) for t in texts])          # similar lengths together: less padding
    out = np.zeros(len(texts), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_TOKENS,
                          return_tensors="pt").to(device)
        out[idx] = torch.softmax(model(**batch).logits, dim=-1)[:, 1].float().cpu().numpy()
    return out


def forward_text_scores(pairs, blocks, text_col="prior_comments", name="previous_visit", cache_dir="../data/processed"):
    """Text scores that never look ahead.

    `pairs` = routine inspections that have earlier comments, with columns inspection_id, date,
    label_fail and the text column `text_col`. For each block, a fresh model is fine-tuned on
    the pairs dated before the cutoff and scores the pairs in the block's window.
    Returns a table (inspection_id, text_score). Each block is cached under `name`, because
    training takes minutes.
    """
    cache_dir, parts = Path(cache_dir), []
    for train_before, score_from, score_before in blocks:
        cache = cache_dir / f"finetune_scores_{name}_{score_from[:4]}_{score_before[:4]}.csv"
        train = pairs[pairs["date"] < train_before]
        score = pairs[(pairs["date"] >= score_from) & (pairs["date"] < score_before)]
        assert train["date"].max() < score["date"].min(), "a text model must be trained on the past only"
        if cache.exists() and len(pd.read_csv(cache)) == len(score):
            print(f"  loaded cached text scores ({name}) for {score_from[:4]} to {score_before[:4]}")
            parts.append(pd.read_csv(cache, dtype={"inspection_id": str}))
            continue
        print(f"  fine-tuning ({name}) on {len(train):,} inspections before {train_before[:4]}, "
              f"scoring {len(score):,} from {score_from[:4]} to {score_before[:4]}", flush=True)
        tokenizer, model = fine_tune(train[text_col], train["label_fail"])
        part = pd.DataFrame({"inspection_id": score["inspection_id"].to_numpy(),
                             "text_score": predict_fail_probability(tokenizer, model, score[text_col])})
        part.to_csv(cache, index=False)
        parts.append(part)
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return pd.concat(parts, ignore_index=True)


def self_test():
    """Smoke test: on a toy problem where one word decides the label, a fine-tuned model must learn it."""
    rng = np.random.default_rng(SEED)
    words = ["clean", "soiled"]
    labels = rng.integers(0, 2, 400)
    texts = [f"Floor under the cook line was {words[y]} at the time of inspection." for y in labels]
    tokenizer, model = fine_tune(texts, labels, epochs=4, verbose=False)      # 50 steps: enough for a toy problem
    p = predict_fail_probability(tokenizer, model, ["The walk-in cooler was soiled.", "The walk-in cooler was clean."])
    assert p[0] > 0.8 and p[1] < 0.2, p
    print(f"finetune.py self-test passed: P(fail | 'soiled') = {p[0]:.2f}, P(fail | 'clean') = {p[1]:.2f}")


if __name__ == "__main__":
    self_test()
