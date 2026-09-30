import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
import cross_encoder as ce
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train_model import CACHE_DIR
d=pd.read_parquet(f"{CACHE_DIR}/valdense10_pairs.parquet")
t=pd.read_parquet(f"{CACHE_DIR}/val_dense_topk.parquet"); t=t[t["rank"]<10]
x=d[d.prob<0.01].merge(t[["source1_entity_id","candidate_entity_id"]])
print("rescue pairs",len(x),"true",x.label.sum(),flush=True)
mb=f"{CACHE_DIR}/cross_encoder_big2_v8"
tok=AutoTokenizer.from_pretrained(mb); mdl=AutoModelForSequenceClassification.from_pretrained(mb).to(ce.device()).eval()
text=ce.raw_texts("train",set(x.source1_entity_id)|set(x.candidate_entity_id))
s=ce.score(tok,mdl,list(zip(x.source1_entity_id,x.candidate_entity_id)),text)
np.savez(f"{CACHE_DIR}/val_rescue_minilm.npz",s1=x.source1_entity_id.values,c=x.candidate_entity_id.values,score=s)
print("saved",flush=True)
