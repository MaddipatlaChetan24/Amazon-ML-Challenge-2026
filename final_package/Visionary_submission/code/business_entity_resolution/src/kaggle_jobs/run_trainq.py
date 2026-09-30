import os, subprocess, glob
kit = glob.glob("/kaggle/input/**/dense-kit", recursive=True) or [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train_pairs.parquet", recursive=True) if "dense" in p]
kit = kit[0]
tq = glob.glob("/kaggle/input/**/dense-trainq/train_queries.parquet", recursive=True) or glob.glob("/kaggle/input/**/train_queries.parquet", recursive=True)
tq = [p for p in tq if "trainq" in p][0]
print("kit", kit, "train queries", tq, flush=True)
os.makedirs("/kaggle/working/kit", exist_ok=True)
for f in ["train_pairs.parquet", "train_records.parquet", "val_truth.parquet"]:
    dst = f"/kaggle/working/kit/{f}"
    if not os.path.exists(dst):
        os.symlink(f"{kit}/{f}", dst)
if not os.path.exists("/kaggle/working/kit/train_queries.parquet"):
    os.symlink(tq, "/kaggle/working/kit/train_queries.parquet")
script = glob.glob("/kaggle/input/**/dense-trainq/kaggle_dense.py", recursive=True)
script = script[0] if script else "/kaggle/src/kaggle_dense.py"
subprocess.run(["python", script, "--data", "/kaggle/working/kit", "--out", "/kaggle/working", "--skip-test", "--k", "10"], check=True)
os.remove("/kaggle/working/kit/train_queries.parquet")
print("finished", flush=True)
