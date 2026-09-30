"""Export texts for the learned (bi-encoder) search experiment on Kaggle."""
import pandas as pd, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from train_model import CACHE_DIR, DATASET_DIR, VAL_ENTITIES_PATH
out = f"{CACHE_DIR}/dense_kit"
def load(split, i):
    d = pd.read_csv(f"{DATASET_DIR}/{split}/{split}_source{i}.tsv", sep="\t", keep_default_na=False, dtype=str)
    return pd.DataFrame({"entity_id": d.entity_id, "country": d.country,
                         "text": (d.business_name.str.strip() + " | " + d.business_address.str.strip()).str.slice(0, 300)})
val = set(pd.read_csv(VAL_ENTITIES_PATH, header=None)[0])
for split in ("train", "test"):
    s1 = load(split, 1)
    if split == "train":
        s1 = s1[s1.entity_id.isin(val)]
    s1.to_parquet(f"{out}/{split}_queries.parquet", index=False)
    pd.concat([load(split, 2), load(split, 3)], ignore_index=True).to_parquet(f"{out}/{split}_records.parquet", index=False)
    print(split, len(s1), flush=True)
gt = pd.read_csv(f"{DATASET_DIR}/train/train_ground_truth.tsv", sep="\t", keep_default_na=False)
g = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
g = g[(g.m != "") & ~g.source1_entity_id.isin(val)]
s1 = load("train", 1).set_index("entity_id").text
rec = pd.concat([load("train", 2), load("train", 3)]).set_index("entity_id").text
g = g.sample(min(1_500_000, len(g)), random_state=0)
pd.DataFrame({"a": s1.reindex(g.source1_entity_id).values, "b": rec.reindex(g.m).values}).to_parquet(f"{out}/train_pairs.parquet", index=False)
vgt = gt[gt.source1_entity_id.isin(val)]
vgt.to_parquet(f"{out}/val_truth.parquet", index=False)
print("pairs", len(g))
