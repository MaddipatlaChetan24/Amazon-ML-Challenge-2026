"""
Third stage: a cross-encoder that re-scores only the pairs the gradient-boosted
model is unsure about (probability in [BAND_LO, BAND_HI]).

It reads the raw "name | address" text of both records together, so it can see
differences the hand-built similarity features cannot express. Base model:
paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M parameters), fine-tuned
as a pair classifier. Its logit is combined with the GBDT's logit by a
two-feature logistic regression.

Measured before adoption (v7, validation split in two halves by business,
cross-encoder trained on one half's uncertain pairs, scored on the other):
AUC on uncertain pairs 0.865 (GBDT) -> 0.947 (combined); macro F0.5 on the
held-out half 0.9576 -> 0.9660.

Usage (after train_model.py has produced a model):
    python3 cross_encoder.py --model matching_model_v8.joblib
It scores the fixed validation entities with that model, trains the
cross-encoder in two halves (each half's scores come from the model trained on
the other half), fits the combiner and threshold on those out-of-sample scores,
then trains a final cross-encoder on all uncertain validation pairs and saves
everything to cache/cross_encoder_<tag>/.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from posthoc_features import add_posthoc_features
from train_model import CACHE_DIR, DATASET_DIR, VAL_ENTITIES_PATH

BASE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BAND_LO, BAND_HI = 0.05, 0.95
MAXLEN, BS = 128, 32


def device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def raw_texts(split: str, ids=None) -> dict:
    out = {}
    for k in (1, 2, 3):
        x = pd.read_csv(f"{DATASET_DIR}/{split}/{split}_source{k}.tsv", sep="\t", keep_default_na=False,
                        usecols=["entity_id", "business_name", "business_address"])
        if ids is not None:
            x = x[x["entity_id"].isin(ids)]
        out.update(zip(x["entity_id"], x["business_name"] + " | " + x["business_address"]))
    return out


def _batches(tok, pairs, text, labels=None, shuffle=False, seed=0):
    idx = np.random.RandomState(seed).permutation(len(pairs)) if shuffle else np.arange(len(pairs))
    for s in range(0, len(pairs), BS):
        sel = idx[s:s + BS]
        enc = tok([text[pairs[j][0]] for j in sel], [text[pairs[j][1]] for j in sel],
                  truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        y = None if labels is None else torch.tensor(labels[sel], dtype=torch.float32)
        yield enc, y


def train(pairs, labels, text, epochs=2, lr=3e-5, seed=0, base=BASE):
    """base: any Hugging Face encoder allowed by the contest licence rules
    (MIT/Apache-2.0, <= 8B params), e.g. the default MiniLM or
    FacebookAI/xlm-roberta-base (MIT, 278M)."""
    torch.manual_seed(seed)
    dev = device()
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * ((len(pairs) + BS - 1) // BS)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        for i, (enc, y) in enumerate(_batches(tok, pairs, text, labels, shuffle=True, seed=seed + ep)):
            enc = {k: v.to(dev) for k, v in enc.items()}
            loss = lossf(model(**enc).logits.squeeze(-1), y.to(dev))
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            if i % 500 == 0:
                print(f"    epoch {ep+1} step {i}/{steps // epochs} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    return tok, model


@torch.no_grad()
def score(tok, model, pairs, text) -> np.ndarray:
    dev = next(model.parameters()).device
    out = []
    for enc, _ in _batches(tok, pairs, text):
        out.append(model(**{k: v.to(dev) for k, v in enc.items()}).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def combine(prob, ce_logits, combiner) -> np.ndarray:
    """ce_logits: one array, or a list of arrays (one per cross-encoder in an ensemble)."""
    if not isinstance(ce_logits, (list, tuple)):
        ce_logits = [ce_logits]
    return combiner.predict_proba(np.column_stack([logit(prob)] + list(ce_logits)))[:, 1]


def _half(e: str) -> int:
    return int(hashlib.md5(e.encode()).hexdigest(), 16) % 2


def main():
    from train_model import macro_f05_at_threshold
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="stage-1 (or stacked) model file in cache/")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    tag = args.model.replace(".joblib", "")
    outdir = f"{CACHE_DIR}/cross_encoder_{tag}"
    os.makedirs(outdir, exist_ok=True)

    val_ents = pd.read_csv(VAL_ENTITIES_PATH, header=None)[0].tolist()
    df = pq.read_table(f"{CACHE_DIR}/train_features.parquet",
                       filters=[("source1_entity_id", "in", val_ents)]).to_pandas()
    df = add_posthoc_features(df, "train")
    bundle = joblib.load(f"{CACHE_DIR}/{args.model}")
    if "stage1" in bundle:
        from train_stacked import add_prob_group_features
        p1 = bundle["stage1"]["model"].predict_proba(df[bundle["stage1_features"]].values)[:, 1]
        df = add_prob_group_features(df, p1)
    df["prob"] = bundle["model"].predict_proba(df[bundle["features"]].values)[:, 1]
    df = df[["source1_entity_id", "candidate_entity_id", "label", "prob"]].reset_index(drop=True)
    gt = pd.read_csv(os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv"), sep="\t", keep_default_na=False)
    n_true = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"].map(lambda x: len(x.split(",")) if x else 0)))
    df["_n_true"] = df["source1_entity_id"].map(n_true).fillna(0).astype(int)

    band = df["prob"].between(BAND_LO, BAND_HI).to_numpy()
    half = df["source1_entity_id"].map(_half).to_numpy()
    B = df[band].reset_index(drop=True)
    hb = half[band]
    text = raw_texts("train", set(B["source1_entity_id"]) | set(B["candidate_entity_id"]))
    pairs = list(zip(B["source1_entity_id"], B["candidate_entity_id"]))
    y = B["label"].to_numpy()
    print(f"validation pairs {len(df):,}; uncertain band {len(B):,} ({band.mean()*100:.2f}%)", flush=True)

    # out-of-sample cross-encoder scores: train on one half, score the other
    oos = np.zeros(len(B), dtype=np.float32)
    for h in (0, 1):
        tr, te = np.flatnonzero(hb != h), np.flatnonzero(hb == h)
        print(f"  fold {h}: train {len(tr):,} score {len(te):,}", flush=True)
        tok, m = train([pairs[j] for j in tr], y[tr], text, epochs=args.epochs)
        oos[te] = score(tok, m, [pairs[j] for j in te], text)
        del m
    combiner = LogisticRegression().fit(np.c_[logit(B["prob"].to_numpy()), oos], y)

    df["prob2"] = df["prob"]
    df.loc[band, "prob2"] = combine(B["prob"].to_numpy(), oos, combiner)
    best = {}
    for col in ("prob", "prob2"):
        bt, bf = 0.5, -1.0
        for t in np.arange(0.40, 0.92, 0.02):
            f = macro_f05_at_threshold(df, col, "label", t)
            if f > bf:
                bf, bt = f, float(t)
        best[col] = (bt, bf)
        print(f"validation macro F0.5 ({'model alone' if col == 'prob' else 'model + cross-encoder'}): "
              f"{bf:.4f} at threshold {bt:.2f}", flush=True)

    print("  final cross-encoder on all uncertain validation pairs...", flush=True)
    tok, m = train(pairs, y, text, epochs=args.epochs)
    m.save_pretrained(outdir); tok.save_pretrained(outdir)
    joblib.dump({"combiner": combiner, "threshold": best["prob2"][0], "band": (BAND_LO, BAND_HI),
                 "validation_f05": best["prob2"][1], "validation_f05_without": best["prob"][1]},
                f"{outdir}/combiner.joblib")
    print(f"saved {outdir}", flush=True)


if __name__ == "__main__":
    main()
