"""Oracle-lens TEACHER (paper A.9.2 stage 3, K=1): for each REAL layer-42 activation, find the dictionary span whose AR vector
best reconstructs it in mean-centred WHITENED space (nearest neighbour by whitened cosine = the K=1 NN-OMP step), and write
SFT seed rows (h42, roll_ids = teacher span, seed_cos, seed_wfve) in the harvest schema so onelayer_av.py can train on them.
Dictionary = the AR cache (span -> AR vector) of --dict-data; queries = h42 of --query-data.

    modal run --detach modal_app.py --task train --gpus 1 --script seed_teacher.py --args "--dict-data '...' --query-data '...' --out /vol/data/seed_k1"
"""
import argparse, glob, os, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch, torch.nn.functional as F
from common import D_MODEL, Whitener

ap = argparse.ArgumentParser()
ap.add_argument("--dict-data", required=True); ap.add_argument("--ar-cache", default="/vol/data/ar_v3"); ap.add_argument("--max-dict-files", type=int, default=1200)
ap.add_argument("--query-data", required=True); ap.add_argument("--max-query-files", type=int, default=250); ap.add_argument("--exclude", default="shard00_part0000")
ap.add_argument("--whitener", default="/vol_data/data/whitener_v2.pt"); ap.add_argument("--out", default="/vol/data/seed_k1"); ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
args = ap.parse_args(); dev = "cuda"; os.makedirs(args.out, exist_ok=True); WHT = Whitener.load(args.whitener, dev)
def cpath(f): return args.ar_cache + "/" + f.replace("/vol_data/data/", "").replace("/vol/data/", "").replace("/", "__")
dfiles = [f for f in sorted(sum((glob.glob(g.strip()) for g in args.dict_data.split(",")), [])) if args.exclude not in f and os.path.exists(cpath(f))][: args.max_dict_files]
qfiles = [f for f in sorted(sum((glob.glob(g.strip()) for g in args.query_data.split(",")), [])) if args.exclude not in f][args.worker::args.nworkers][: args.max_query_files]
# ---- dictionary: whitened, unit-normalised AR vectors on GPU (fp16) + their spans
t0 = time.time(); D, S = [], []
for f in dfiles:
    tb = pq.ParquetFile(cpath(f)).read(columns=["roll_ids", "ar_vec"]); n = tb.num_rows
    v = torch.tensor(tb.column("ar_vec").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float32), device=dev)
    D.append(F.normalize(WHT(v), dim=-1).half()); S.append(tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, 12).astype(np.int32))
D = torch.cat(D); S = np.concatenate(S); print(f"[teacher] dictionary {D.shape[0]} whitened AR vectors from {len(dfiles)} files in {(time.time()-t0)/60:.1f} min", flush=True)
# ---- queries: real h42, whitened + normalised; nearest neighbour by whitened cosine
tot_cos, tot_fve, tot_true, n_q = 0.0, 0.0, 0.0, 0
for qi, f in enumerate(qfiles):
    tb = pq.ParquetFile(f).read(columns=["h42", "roll_ids"]); n = tb.num_rows
    H = tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float32)
    R = tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, 12).astype(np.int32)
    best_i, best_c = [], []
    with torch.no_grad():
        for a in range(0, n, 2048):
            q = F.normalize(WHT(torch.tensor(H[a:a + 2048], device=dev)), dim=-1).half()
            sims = q @ D.T                                                        # [b, N_dict] whitened cosines
            c, i = sims.max(1); best_i.append(i.cpu().numpy()); best_c.append(c.float().cpu().numpy())
    best_i = np.concatenate(best_i); best_c = np.concatenate(best_c); wfve = np.clip(best_c, 0, None) ** 2
    # reference: whitened cos of the TRUE span's own AR vector (if this query file is in the cache)
    true_c = np.full(n, np.nan, dtype=np.float32)
    if os.path.exists(cpath(f)):
        tv = torch.tensor(pq.ParquetFile(cpath(f)).read(columns=["ar_vec"]).column("ar_vec").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float32), device=dev)
        with torch.no_grad(): true_c = F.cosine_similarity(WHT(tv), WHT(torch.tensor(H, device=dev))).cpu().numpy()
    out = pa.table({"h42": pa.FixedSizeListArray.from_arrays(pa.array(H.astype(np.float16).reshape(-1)), D_MODEL),
                    "roll_ids": pa.FixedSizeListArray.from_arrays(pa.array(S[best_i].reshape(-1)), 12),
                    "true_roll_ids": pa.FixedSizeListArray.from_arrays(pa.array(R.reshape(-1)), 12),
                    "seed_cos": pa.array(best_c), "seed_wfve": pa.array(wfve.astype(np.float32)), "true_cos": pa.array(true_c)})
    pq.write_table(out, f"{args.out}/{os.path.basename(f).replace('.parquet', '')}__{os.path.basename(os.path.dirname(f))}.parquet")
    tot_cos += best_c.sum(); tot_fve += wfve.sum(); tot_true += np.nan_to_num(np.clip(true_c, 0, None) ** 2).sum(); n_q += n
    print(f"[teacher] {qi+1}/{len(qfiles)} files | {n_q} activations | mean best whitened cos {tot_cos/n_q:.4f} | mean best wFVE {tot_fve/n_q:.4f} | true-span wFVE {tot_true/n_q:.4f} | {(time.time()-t0)/60:.1f} min", flush=True)
print("TEACHER_DONE", flush=True)
