"""
Learned search (bi-encoder) on Kaggle GPUs.

1. Fine-tune a small multilingual sentence encoder on (business, true record)
   text pairs with in-batch negatives.
2. Encode validation businesses + all training records; nearest records per
   business (same country) -> val_topk.parquet.
3. Encode test businesses + all test records -> test_topk.parquet.

  python kaggle_dense.py --data /kaggle/input/.../dense-kit
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="/kaggle/working")
ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--maxlen", type=int, default=64)
ap.add_argument("--max-train", type=int, default=0)
ap.add_argument("--k", type=int, default=50)
ap.add_argument("--scale", type=float, default=30.0)
ap.add_argument("--skip-test", action="store_true")
ap.add_argument("--init-model", default="", help="already fine-tuned dense_model dir: skip training")
a = ap.parse_args()
t00 = time.time()
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() and not os.environ.get("DENSE_CPU") else "cpu")
GPU = dev == "cuda"
print(f"GPUs: {torch.cuda.device_count()}", flush=True)

src_model = a.init_model if a.init_model and os.path.isdir(a.init_model) else a.model
print(f"encoder from {src_model}", flush=True)
tok = AutoTokenizer.from_pretrained(src_model)
enc_model = AutoModel.from_pretrained(src_model).to(dev)


def embed(model, texts):
    e = tok(list(texts), truncation=True, max_length=a.maxlen, padding=True, return_tensors="pt")
    e = {k: v.to(dev, non_blocking=True) for k, v in e.items()}
    h = model(**e).last_hidden_state
    m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
    return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)


if src_model == a.model:
    # ---- 1. fine-tune ----
    tr = pd.read_parquet(f"{a.data}/train_pairs.parquet").sample(frac=1.0, random_state=0).reset_index(drop=True)
    if a.max_train:
        tr = tr.iloc[:a.max_train]
    opt = torch.optim.AdamW(enc_model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(tr) // a.bs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * max(0.0, 1 - s / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=GPU)
    enc_model.train()
    t0 = time.time()
    for s in range(steps):
        b = tr.iloc[s * a.bs:(s + 1) * a.bs]
        with torch.autocast("cuda", dtype=torch.float16, enabled=GPU):
            qa = embed(enc_model, b.a); qb = embed(enc_model, b.b)
            sim = qa @ qb.T * a.scale
            lab = torch.arange(len(b), device=dev)
            loss = (F.cross_entropy(sim.float(), lab) + F.cross_entropy(sim.float().T, lab)) / 2
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(enc_model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if (s + 1) % 500 == 0:
            print(f"  step {s+1}/{steps} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    enc_model.eval()
    enc_model.save_pretrained(f"{a.out}/dense_model"); tok.save_pretrained(f"{a.out}/dense_model")
    print(f"trained in {time.time()-t0:.0f}s", flush=True)

else:
    enc_model.eval()

# encoding on both GPUs
enc_models = [enc_model.half() if dev != "cpu" else enc_model]
if torch.cuda.device_count() > 1:
    import copy
    m2 = copy.deepcopy(enc_model).to("cuda:1"); enc_models.append(m2)


@torch.no_grad()
def encode_all(texts, bs=1024):
    texts = list(texts)
    # sort by length for fast padding
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = torch.empty((len(texts), enc_model.config.hidden_size), dtype=torch.float16 if dev != "cpu" else torch.float32)
    t0 = time.time()
    for i, start in enumerate(range(0, len(texts), bs)):
        idx = order[start:start + bs]
        g = i % len(enc_models)
        d = f"cuda:{g}" if GPU else dev
        e = tok([texts[j] for j in idx], truncation=True, max_length=a.maxlen, padding=True, return_tensors="pt")
        e = {k: v.to(d, non_blocking=True) for k, v in e.items()}
        h = enc_models[g](**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)
        out[torch.as_tensor(idx)] = v.cpu()
        if i % 2000 == 0 and i:
            print(f"    encoded {start:,}/{len(texts):,} ({time.time()-t0:.0f}s)", flush=True)
    return out


@torch.no_grad()
def search(q, r, k, qchunk=1024, rchunk=500_000):
    """top-k inner product of each q row among r rows; returns (idx int64, score f16)."""
    best_s = torch.full((len(q), k), -10.0, dtype=q.dtype)
    best_i = torch.zeros((len(q), k), dtype=torch.int64)
    for rs in range(0, len(r), rchunk):
        rg = r[rs:rs + rchunk].to(dev)
        kk = min(k, len(rg))
        for qs in range(0, len(q), qchunk):
            qg = q[qs:qs + qchunk].to(dev)
            sc = qg @ rg.T
            s, i = sc.topk(kk, dim=1)
            cs = torch.cat([best_s[qs:qs + qchunk].to(dev), s], 1)
            ci = torch.cat([best_i[qs:qs + qchunk].to(dev), i + rs], 1)
            s2, j = cs.topk(k, dim=1)
            best_s[qs:qs + qchunk] = s2.cpu(); best_i[qs:qs + qchunk] = ci.gather(1, j).cpu()
        del rg
        if GPU:
            torch.cuda.empty_cache()
    return best_i.numpy(), best_s.numpy()


def run(split):
    t0 = time.time()
    qd = pd.read_parquet(f"{a.data}/{split}_queries.parquet")
    rd = pd.read_parquet(f"{a.data}/{split}_records.parquet")
    print(f"{split}: {len(qd):,} businesses, {len(rd):,} records", flush=True)
    qe = encode_all(qd.text); re_ = encode_all(rd.text)
    print(f"  encoded in {time.time()-t0:.0f}s", flush=True)
    parts = []
    for c in sorted(qd.country.unique()):
        qi = np.flatnonzero((qd.country == c).to_numpy()); ri = np.flatnonzero((rd.country == c).to_numpy())
        if not len(ri):
            continue
        idx, sc = search(qe[qi], re_[ri], a.k)
        parts.append(pd.DataFrame({"source1_entity_id": np.repeat(qd.entity_id.to_numpy()[qi], a.k),
                                   "candidate_entity_id": rd.entity_id.to_numpy()[ri][idx.ravel()],
                                   "rank": np.tile(np.arange(a.k, dtype=np.int16), len(qi)),
                                   "score": sc.ravel()}))
        print(f"  {c}: searched {len(qi):,} x {len(ri):,} ({time.time()-t0:.0f}s)", flush=True)
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(f"{a.out}/{split}_topk.parquet", index=False)
    print(f"  saved {split}_topk.parquet ({len(res):,} rows, {time.time()-t0:.0f}s)", flush=True)
    return res


val = run("train")
vt = pd.read_parquet(f"{a.data}/val_truth.parquet")
g = vt.assign(m=vt.matched_entity_ids.str.split(",")).explode("m")
g = g[g.m != ""]
got = set(zip(val.source1_entity_id, val.candidate_entity_id))
for k in (10, 20, 30, 50):
    gk = set(zip(val[val["rank"] < k].source1_entity_id, val[val["rank"] < k].candidate_entity_id))
    rec = np.mean([(x, y) in gk for x, y in zip(g.source1_entity_id, g.m)])
    print(f"validation recall of true pairs @ {k}: {rec:.4f}", flush=True)
if not a.skip_test:
    run("test")
print(f"done ({time.time()-t00:.0f}s)", flush=True)
