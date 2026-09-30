"""
Stronger re-checker, trained on a CUDA machine (e.g. Kaggle 2x T4).
Self-contained: needs only the kit written by export_rechecker_kit.py.

  python kaggle_rechecker.py --data /kaggle/input/<kit> [--model ...] [--epochs 2]

Writes to --out: val_scores.npy, test_scores.npy (row-aligned with val_band /
test_band) and prints validation macro F0.5 for
  classifier + MiniLM            (current best)
  classifier + MiniLM + new model (ensemble)
with the logistic combiner fitted on one half of the validation businesses and
scored on the other, so the comparison is out-of-sample.
"""
import argparse
import hashlib
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="/kaggle/working")
ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--maxlen", type=int, default=128)
ap.add_argument("--max-train", type=int, default=0, help="0 = all pairs")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--score-only", default="", help="directory of an already-trained re-checker: skip training")
ap.add_argument("--test", default="", help="test_band.parquet path (default: <data>/test_band.parquet)")
a = ap.parse_args()
torch.manual_seed(a.seed)
dev = "cuda"
print(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}", flush=True)

tr = pd.read_parquet(f"{a.data}/train_pairs.parquet").sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
if a.max_train:
    tr = tr.iloc[:a.max_train]
vb = pd.read_parquet(f"{a.data}/val_band.parquet")
print(f"train {len(tr):,} pairs, validation band {len(vb):,}", flush=True)

src_model = a.score_only or a.model
tok = AutoTokenizer.from_pretrained(src_model)
model = AutoModelForSequenceClassification.from_pretrained(src_model, num_labels=1).to(dev)
net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model


def enc(s1, c):
    e = tok(list(s1), list(c), truncation=True, max_length=a.maxlen, padding=True, return_tensors="pt")
    return {k: v.to(dev) for k, v in e.items()}


opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps = a.epochs * ((len(tr) + a.bs - 1) // a.bs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))
scaler = torch.cuda.amp.GradScaler()
lossf = torch.nn.BCEWithLogitsLoss()
t0, step = time.time(), 0
net.train()
for ep in range(0 if a.score_only else a.epochs):
    order = np.random.RandomState(a.seed + ep).permutation(len(tr))
    for s in range(0, len(tr), a.bs):
        b = tr.iloc[order[s:s + a.bs]]
        with torch.autocast("cuda", dtype=torch.float16):
            logits = net(**enc(b.s1_text, b.c_text)).logits.squeeze(-1)
            loss = lossf(logits.float(), torch.tensor(b.label.values, dtype=torch.float32, device=dev))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        step += 1
        if step % 500 == 0:
            print(f"  epoch {ep+1} step {step}/{steps} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
net.eval()


@torch.no_grad()
def score(df, bs=512):
    out = []
    for s in range(0, len(df), bs):
        b = df.iloc[s:s + bs]
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(net(**enc(b.s1_text, b.c_text)).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


vs = score(vb)
np.save(f"{a.out}/val_scores.npy", vs)
print(f"trained + scored validation in {time.time()-t0:.0f}s", flush=True)

# macro F0.5 on ALL validation pairs, exact contest aggregation
va = pd.read_parquet(f"{a.data}/val_all.parquet")
key_all = va.source1_entity_id + "|" + va.candidate_entity_id
key_b = vb.source1_entity_id + "|" + vb.candidate_entity_id
pos_in_all = pd.Series(np.arange(len(va)), index=key_all.values).reindex(key_b.values).to_numpy()


def macro_f05(p, t):
    pred = (p >= t).astype(int)
    g = va.source1_entity_id.values
    df = pd.DataFrame({"e": g, "pred": pred, "tp": pred * va.label.values})
    agg = df.groupby("e").agg(pp=("pred", "sum"), tp=("tp", "sum"))
    nt = va.groupby("source1_entity_id").n_true.first().reindex(agg.index).values
    pr = np.where(agg.pp > 0, agg.tp / np.maximum(agg.pp, 1), 1.0)
    rc = np.where(nt > 0, agg.tp / np.maximum(nt, 1), 1.0)
    f = np.where((0.25 * pr + rc) > 0, 1.25 * pr * rc / np.maximum(0.25 * pr + rc, 1e-12), 0.0)
    f = np.where(nt == 0, (agg.pp.values == 0).astype(float), f)
    return f.mean()


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


half = vb.source1_entity_id.map(lambda e: int(hashlib.md5(e.encode()).hexdigest(), 16) % 2).to_numpy()
y = vb.label.to_numpy()
for name, X in (("classifier + MiniLM (current best)", np.c_[logit(vb.prob), vb.minilm]),
                ("classifier + new model", np.c_[logit(vb.prob), vs]),
                ("classifier + MiniLM + new model", np.c_[logit(vb.prob), vb.minilm, vs])):
    comb = np.zeros(len(vb))
    for h in (0, 1):
        comb[half == h] = LogisticRegression(max_iter=1000).fit(X[half != h], y[half != h]).predict_proba(X[half == h])[:, 1]
    p = va.prob.to_numpy().copy()
    p[pos_in_all] = comb
    best = max((macro_f05(p, t), t) for t in np.arange(0.5, 0.92, 0.02))
    print(f"{name}: validation macro F0.5 {best[0]:.4f} (threshold {best[1]:.2f})", flush=True)

test_path = a.test or f"{a.data}/test_band.parquet"
if os.path.exists(test_path):
    tb = pd.read_parquet(test_path)
    t1 = time.time()
    np.save(f"{a.out}/test_scores.npy", score(tb))
    print(f"scored {len(tb):,} test pairs in {time.time()-t1:.0f}s", flush=True)
if not a.score_only:
    model.save_pretrained(f"{a.out}/rechecker_model"); tok.save_pretrained(f"{a.out}/rechecker_model")
print("done", flush=True)
