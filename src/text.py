"""Turn inspector comments into numbers with a frozen pre-trained BERT-style encoder.

Lecture 12 pattern: AutoTokenizer + AutoModel, no training, mean pooling over the
real (non-padding) tokens. The 384-dimensional sentence vectors are then compressed
to 32 principal components (PCA fitted on training rows only).
"""
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from huggingface_hub.utils import disable_progress_bars
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

disable_progress_bars()            # keep download progress bars out of executed notebooks
hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()

SEED = 42
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"   # pinned, so a later upload cannot change the vectors
MAX_TOKENS = 256
N_COMPONENTS = 32
PC_COLUMNS = [f"text_pc{i + 1:02d}" for i in range(N_COMPONENTS)]
TEXT_FEATURES = PC_COLUMNS + ["has_prior_comments"]


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def embed_texts(texts, max_tokens=MAX_TOKENS, batch_size=64, device=None, verbose=True):
    """One vector per text. Empty texts get a zero vector. Identical texts are embedded once."""
    device = device or get_device()
    texts = pd.Series(list(texts), dtype="object").fillna("")
    unique = sorted({t for t in texts if t.strip()}, key=len)   # by length: batches pad less
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    encoder = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).to(device).eval()
    dim = encoder.config.hidden_size

    vectors = {}
    started = time.time()
    for start in range(0, len(unique), batch_size):
        batch = unique[start:start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_tokens,
                            return_tensors="pt").to(device)
        hidden = encoder(**encoded).last_hidden_state            # (batch, tokens, 384)
        # Mean pooling: average the token vectors, counting only real tokens (mask = 1).
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        for text, vec in zip(batch, pooled.float().cpu().numpy()):
            vectors[text] = vec
        if verbose and (start // batch_size) % 100 == 0:
            print(f"  embedded {start + len(batch):,} of {len(unique):,} unique texts "
                  f"({time.time() - started:.0f}s)", flush=True)

    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        if t in vectors:
            out[i] = vectors[t]
    return out


def _fingerprint(ids, texts):
    """inspection_id + a hash of its text, so a stale cache is detected when the text changes."""
    return pd.DataFrame({
        "inspection_id": list(ids),
        "text_md5": [hashlib.md5(t.encode("utf-8")).hexdigest() for t in texts],
    })


def embed_cached(ids, texts, cache_dir="../data/processed", name="embeddings", force=False, **kwargs):
    """Embed `texts` (aligned with `ids`), or load them from <name>.npy if the cache matches."""
    cache_dir = Path(cache_dir)
    npy, id_file = cache_dir / f"{name}.npy", cache_dir / f"{name}_ids.csv"
    texts = ["" if t is None else str(t) for t in texts]
    fingerprint = _fingerprint(ids, texts)
    if npy.exists() and id_file.exists() and not force:
        if pd.read_csv(id_file, dtype=str).equals(fingerprint.astype(str)):
            print(f"Loaded cached embeddings: {npy}")
            return np.load(npy)
        print("Cache does not match the current texts, embedding again.")
    emb = embed_texts(texts, **kwargs)
    np.save(npy, emb)
    fingerprint.to_csv(id_file, index=False)
    print(f"Saved embeddings {emb.shape} to {npy}")
    return emb


def fit_pca(embeddings, fit_mask):
    """PCA fitted ONLY on the rows in `fit_mask` (training rows that have comments)."""
    return PCA(n_components=N_COMPONENTS, random_state=SEED).fit(embeddings[fit_mask])


def add_text_features(df, embeddings, pca):
    """Add the 32 principal components and the has_prior_comments flag to a copy of df."""
    out = df.copy()
    out[PC_COLUMNS] = pca.transform(embeddings)
    out["has_prior_comments"] = (df["prior_comments"].fillna("").str.strip() != "").astype(int).to_numpy()
    return out
