import sys, hashlib, numpy as np, pandas as pd
sys.path.insert(0,'.')
import final_stage as fs
from train_model import CACHE_DIR, DATASET_DIR
K=fs.K; tag=sys.argv[1]
d=pd.read_parquet(f"{CACHE_DIR}/{tag}_pairs.parquet")
vb=pd.read_parquet(f"{K}/val_band.parquet",columns=["source1_entity_id","candidate_entity_id","minilm"]); vb["mp"]=np.load(f"{K}/mpnet_val_scores.npy")
m=np.load(f"{CACHE_DIR}/{tag}_minilm.npz",allow_pickle=True)
sc=pd.concat([vb,pd.DataFrame({"source1_entity_id":m["s1"],"candidate_entity_id":m["c"],"minilm":m["score"],"mp":np.nan})]).drop_duplicates(["source1_entity_id","candidate_entity_id"])
band=d[(d.prob>=0.01)&(d.prob<=0.99)].merge(sc,on=["source1_entity_id","candidate_entity_id"],how="left")
y=band.label.to_numpy(); X2=np.c_[fs.lg(band.prob.to_numpy()),band.minilm]; mpv=band.mp.to_numpy(); has=~np.isnan(mpv)
half=band.source1_entity_id.map(lambda e:int(hashlib.md5(e.encode()).hexdigest(),16)%2).to_numpy()
oof=np.zeros(len(band))
for h in (0,1):
    fi=half!=h
    s2=fs.lr().fit(X2[fi],y[fi]).predict_proba(X2)[:,1]; nar=(s2>0.05)&(s2<0.95)&has
    X3=np.c_[X2,np.nan_to_num(mpv)]; s3=fs.lr().fit(X3[fi&nar],y[fi&nar]).predict_proba(X3)[:,1]
    G=fs.group_features(band.source1_entity_id.values,band.prob.to_numpy(),band.minilm.to_numpy(),s2,s3,np.nan_to_num(mpv),nar)
    out=np.where(nar,fs.hgb().fit(G[fi&nar],y[fi&nar]).predict_proba(G)[:,1],s2); oof[half==h]=out[half==h]
pos=pd.Series(np.arange(len(d)),index=(d.source1_entity_id+"|"+d.candidate_entity_id).values).reindex((band.source1_entity_id+"|"+band.candidate_entity_id).values).to_numpy()
d["p"]=d.prob.to_numpy(); d.loc[pos,"p"]=oof
x=d[d.p>=0.74].sort_values("p",ascending=False).drop_duplicates("candidate_entity_id")
d["pred"]=0; d.loc[x.index,"pred"]=1
d[["source1_entity_id","candidate_entity_id","label","p","pred"]].to_parquet(f"{CACHE_DIR}/{tag}_oof.parquet")
gt=pd.read_csv(f"{DATASET_DIR}/train/train_ground_truth.tsv",sep="\t",keep_default_na=False)
nt=pd.Series(gt.matched_entity_ids.map(lambda s:len(s.split(",")) if s else 0).values,index=gt.source1_entity_id)
g=d.groupby("source1_entity_id").agg(ret=("label","sum"),pp=("pred","sum")); g["tp"]=(d.pred*d.label).groupby(d.source1_entity_id).sum(); g["n"]=nt.reindex(g.index).values
pr=np.where(g.pp>0,g.tp/np.maximum(g.pp,1),1.0); rc=np.where(g.n>0,g.tp/np.maximum(g.n,1),1.0)
g["loss"]=1-np.where(g.n==0,(g.pp==0).astype(float),np.where(g.tp>0,1.25*pr*rc/np.maximum(0.25*pr+rc,1e-12),0.0))
N=len(g); print("macro F0.5",round(1-g.loss.mean(),5))
def cat(r):
    if r.n==0: return "singleton, predicted something"
    fp=r.pp>r.tp; miss=r.ret<r.n; fn=r.tp<r.ret
    return ("FP " if fp else "")+("search-miss " if miss else "")+("rejected-FN" if fn else "")
b=g[g.loss>0].copy(); b["cat"]=b.apply(cat,axis=1)
print(b.groupby("cat").agg(businesses=("loss","size"),points=("loss",lambda s:s.sum()/N)).sort_values("points",ascending=False))
# FN pairs: score distribution
fnp=d[(d.label==1)&(d.pred==0)]; print("rejected true pairs",len(fnp)); print(pd.cut(fnp.p,[0,0.01,0.1,0.3,0.5,0.74,1]).value_counts().sort_index())
fpp=d[(d.label==0)&(d.pred==1)]; print("false-positive pairs",len(fpp)); print(pd.cut(fpp.p,[0.74,0.8,0.9,0.95,0.99,1]).value_counts().sort_index())
