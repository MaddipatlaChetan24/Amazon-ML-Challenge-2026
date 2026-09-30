"""
Combine an existing feature table with feature rows built for extra pairs only
(build_features.py --dense-new), recompute the per-business ambiguity features,
and restore the row order a full rebuild would produce (businesses in their
original order, candidates sorted by id within each business).
"""
import sys, os, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))
from build_features import add_ambiguity_features
from features import AMBIGUITY_FEATURE_NAMES

old_path, new_path, out_path = sys.argv[1:4]
t0 = time.time()
new = pd.read_parquet(new_path)
new = new.drop(columns=[c for c in new.columns if c in AMBIGUITY_FEATURE_NAMES or c == "label" and False], errors="ignore")
old_s1 = pq.read_table(old_path, columns=["source1_entity_id"]).column(0)
order = pd.unique(old_s1.to_numpy(zero_copy_only=False))
print(f"old table {len(old_s1):,} rows, {len(order):,} businesses; new rows {len(new):,}", flush=True)
del old_s1
pos = pd.Series(np.arange(len(order)), index=order)
f = pq.ParquetFile(old_path)
writer = None
n_out = 0
# stream the old table in blocks of whole businesses
buf = []
def flush(block_old, block_new):
    global writer, n_out
    df = pd.concat([block_old, block_new], ignore_index=True)
    df["_b"] = pos.reindex(df.source1_entity_id.values).to_numpy()
    df = df.sort_values(["_b", "candidate_entity_id"], kind="stable").drop(columns="_b").reset_index(drop=True)
    df = add_ambiguity_features(df.drop(columns=AMBIGUITY_FEATURE_NAMES, errors="ignore"))
    t = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, t.schema)
    writer.write_table(t.cast(writer.schema))
    n_out += len(df)

missing = pd.unique(new.source1_entity_id[~new.source1_entity_id.isin(pos.index)].to_numpy())
if len(missing):
    # businesses with no candidates in the old table: appended after all others
    print(f"  {len(missing):,} businesses had no old candidates; their new pairs go at the end", flush=True)
    pos = pd.concat([pos, pd.Series(np.arange(len(order), len(order) + len(missing)), index=missing)])
new_by_b = new.assign(_b=pos.reindex(new.source1_entity_id.values).to_numpy())
assert new_by_b._b.notna().all()
carry = None
for rg in range(f.num_row_groups):
    d = f.read_row_group(rg).to_pandas()
    d = d.drop(columns=[c for c in d.columns if c.startswith("__index")], errors="ignore")
    if carry is not None:
        d = pd.concat([carry, d], ignore_index=True)
    last = d.source1_entity_id.iloc[-1]
    if rg < f.num_row_groups - 1:
        carry = d[d.source1_entity_id == last]; d = d[d.source1_entity_id != last]
    else:
        carry = None
    bmin, bmax = pos[d.source1_entity_id.iloc[0]], pos[d.source1_entity_id.iloc[-1]]
    blk = new_by_b[(new_by_b._b >= bmin) & (new_by_b._b <= bmax)].drop(columns="_b")
    flush(d, blk)
    if rg % 10 == 0:
        print(f"  row group {rg+1}/{f.num_row_groups}: {n_out:,} rows written ({time.time()-t0:.0f}s)", flush=True)
tail = new_by_b[new_by_b._b >= len(order)].drop(columns="_b")
if len(tail):
    flush(tail.iloc[0:0], tail)
writer.close()
print(f"wrote {n_out:,} rows to {out_path} ({time.time()-t0:.0f}s); expected {len(order) and (f.metadata.num_rows + len(new)):,}", flush=True)
