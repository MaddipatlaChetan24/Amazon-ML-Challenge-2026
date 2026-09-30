"""Learned-search top-k pairs that are not already candidates in a test feature table.

  python3 make_new_pairs.py <test_features.parquet> <test_dense_topk.parquet> <k> <out.parquet>
"""
import sys, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))
from generate_predictions import pair_keys

table, topk, k, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
o = pq.read_table(table, columns=["source1_entity_id", "candidate_entity_id"])
ok = pair_keys(o.column(0), o.column(1)); del o
d = pd.read_parquet(topk, columns=["source1_entity_id", "candidate_entity_id", "rank"]); d = d[d["rank"] < k]
kk = pair_keys(d.source1_entity_id.values, d.candidate_entity_id.values)
new = d[~np.isin(kk, ok)][["source1_entity_id", "candidate_entity_id"]].drop_duplicates()
new.to_parquet(out, index=False)
print(f"top-{k} rows {len(d):,}; new pairs {len(new):,} ({len(new)/max(d.source1_entity_id.nunique(),1):.2f}/business)")
