"""How much does learned search add on top of the existing candidates (validation)?"""
import sys, numpy as np, pandas as pd
topk = pd.read_parquet(sys.argv[1])
va = pd.read_parquet("../cache/rechecker_kit/val_all.parquet", columns=["source1_entity_id", "candidate_entity_id", "label"])
vt = pd.read_parquet("../cache/dense_kit/val_truth.parquet")
g = vt.assign(m=vt.matched_entity_ids.str.split(",")).explode("m"); g = g[g.m != ""]
truth = set(zip(g.source1_entity_id, g.m))
have = set(zip(va.source1_entity_id, va.candidate_entity_id))
missed = truth - have
print(f"true pairs {len(truth):,}; existing search misses {len(missed):,} ({1-len(missed)/len(truth):.4f} recall)")
for k in (5, 10, 20, 30, 50):
    t = topk[topk["rank"] < k]
    pairs = set(zip(t.source1_entity_id, t.candidate_entity_id))
    new = pairs - have
    found = len(missed & pairs)
    print(f"top-{k}: dense recall {len(truth & pairs)/len(truth):.4f} | union recall {(len(truth)-len(missed)+found)/len(truth):.4f} "
          f"| recovers {found:,} of {len(missed):,} misses | new pairs {len(new):,} ({len(new)/topk.source1_entity_id.nunique():.1f}/business)")
