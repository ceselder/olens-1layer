"""Closed-form adapter P: real h42 -> expected AR vector (ridge on (h42, cached AR(span)) pairs), so a reader trained ONLY on
AR vectors can be fed real activations as P(h42). Then evaluates the reader through P: CE(true span | P(h42)) vs CE(true span | h42)
vs CE(true span | AR(span)), and the RL metric: whitened FVE of AR(greedy text) against the real h42 (no refit / NNLS refit).

    modal run --detach modal_app.py --task train --gpus 1 --script fit_h42_to_ar.py --args "--ckpt /vol/ckpt/small_d1024L1_30m/final.pt"
"""
import argparse, glob, json, os, time
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F
from common import D_MODEL, load_base, load_tokenizer, Layer42Hook, ar_read, Whitener
from av1_model import load_small

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/vol_data/data/harvest_v1/*.parquet,/vol/data/harvest_v2/*.parquet"); ap.add_argument("--ar-cache", default="/vol/data/ar_v3")
ap.add_argument("--n-rows", type=int, default=2000000); ap.add_argument("--lam", type=float, default=1e3)
ap.add_argument("--ckpt", required=True); ap.add_argument("--frozen", default="/vol/frozen/qwen36_27b_embed_head.pt"); ap.add_argument("--ar", default="/vol_data/ckpt/ar_mse_v3/final")
ap.add_argument("--whitener", default="/vol_data/data/whitener_v2.pt"); ap.add_argument("--eval-file", default="/vol/data/harvest_v3/shard00_part0000.parquet"); ap.add_argument("--n-eval", type=int, default=512)
ap.add_argument("--out", default="/vol/ckpt/adapter_h42_to_ar"); ap.add_argument("--reward", default="lora", choices=["lora", "affine"]); ap.add_argument("--affine", default="/vol/ckpt/ar_affine_lastmean/affine.pt"); args = ap.parse_args(); dev = "cuda"; os.makedirs(args.out, exist_ok=True); tok = load_tokenizer()
def cpath(f): return args.ar_cache + "/" + f.replace("/vol_data/data/", "").replace("/vol/data/", "").replace("/", "__")
files = [f for f in sorted(sum((glob.glob(g.strip()) for g in args.data.split(",")), [])) if os.path.exists(cpath(f))]
Sxx = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64, device=dev); Sxy = torch.zeros_like(Sxx); sx = torch.zeros(D_MODEL, dtype=torch.float64, device=dev); sy = torch.zeros_like(sx); n = 0; t0 = time.time()
for f in files:
    X = torch.tensor(pq.ParquetFile(f).read(columns=["h42"]).column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, D_MODEL).astype(np.float32), device=dev).double()
    Y = torch.tensor(pq.ParquetFile(cpath(f)).read(columns=["ar_vec"]).column("ar_vec").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, D_MODEL).astype(np.float32), device=dev).double()
    Sxx += X.T @ X; Sxy += X.T @ Y; sx += X.sum(0); sy += Y.sum(0); n += len(X)
    if n >= args.n_rows: break
xbar, ybar = sx / n, sy / n; Cxx = Sxx.addr_(xbar, xbar, alpha=-n); Cxy = Sxy.addr_(xbar, ybar, alpha=-n)
Cxx.diagonal().add_(args.lam); P = torch.linalg.solve(Cxx, Cxy); b = ybar - xbar @ P; P, b = P.float(), b.float()
torch.save({"P": P.half().cpu(), "b": b.cpu(), "lam": args.lam, "n_rows": n}, f"{args.out}/adapter.pt"); print(f"[adapter] fit on {n} rows in {(time.time()-t0)/60:.1f} min", flush=True)
# ---- evaluation on an unseen file
tb = pq.ParquetFile(args.eval_file).read(columns=["h42", "roll_ids"]); m = min(tb.num_rows, args.n_eval)
H = torch.tensor(tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, D_MODEL).astype(np.float32)[:m], device=dev)
R = torch.tensor(tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, 12).astype(np.int64)[:m], device=dev)
V_true = torch.tensor(pq.ParquetFile(cpath(args.eval_file)).read(columns=["ar_vec"]).column("ar_vec").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, D_MODEL).astype(np.float32)[:m], device=dev)
PH = H @ P + b
print(f"[adapter] P(h42) vs AR(span): cos {F.cosine_similarity(PH, V_true).mean():.4f} | FVE {1 - ((PH - V_true)**2).sum() / ((V_true - V_true.mean(0))**2).sum():.4f} | norms P(h42) {PH.norm(dim=-1).mean():.1f} AR {V_true.norm(dim=-1).mean():.1f} h42 {H.norm(dim=-1).mean():.1f}", flush=True)
reader = load_small(args.ckpt, args.frozen, dev); WHT = Whitener.load(args.whitener, dev)
base = load_base(dev)
if args.reward == "lora":
    from peft import PeftModel
    AR = PeftModel.from_pretrained(base, args.ar, adapter_name="ar").eval(); hook = Layer42Hook(AR)
    vh = nn.Linear(D_MODEL, D_MODEL, bias=True).to(dev, torch.float32); vh.load_state_dict(torch.load(f"{args.ar}/value_head.pt", map_location=dev))
else:
    AR = base.eval(); hook = Layer42Hook(AR); vh = None; _a = torch.load(args.affine, map_location=dev); W_, b_, feat_ = _a["W"].to(dev).float(), _a["b"].to(dev).float(), _a["feat"]
@torch.no_grad()
def ar_vec(ids):
    out = []
    for a in range(0, ids.shape[0], 256):
        Hs = ar_read(AR, hook, ids[a:a+256], torch.ones_like(ids[a:a+256]), vh).float()
        out.append(vh(Hs[:, -1]) if vh is not None else (torch.cat([Hs[:, -1], Hs.mean(1)], -1) if feat_ == "last+mean" else Hs[:, -1]) @ W_ + b_)
    return torch.cat(out)
@torch.no_grad()
def ce(X): lg = reader(X, R)[:, :-1].float(); return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), R.reshape(-1)).item()
@torch.no_grad()
def wfve(X, name):
    g = reader.generate(X, 12); vg = ar_vec(g); wg, wh = WHT(vg), WHT(H)
    refit = (F.cosine_similarity(wg, wh).clamp_min(0) ** 2).mean().item(); raw = (1 - ((wg - wh) ** 2).sum(-1) / (wh ** 2).sum(-1)).mean().item()
    print(f"[eval] input={name:10s} CE(true span) {ce(X):.3f} | greedy text -> AR -> whitened FVE vs REAL h42: refit {refit:.4f}, no-refit {raw:.4f}", flush=True)
    for i in range(3): print(f"    TRUE {tok.decode(R[i])!r}\n    GEN  {tok.decode(g[i])!r}", flush=True)
    return refit
res = {"input": {}}
res["input"]["AR(span)"] = wfve(V_true, "AR(span)"); res["input"]["real h42"] = wfve(H, "real h42"); res["input"]["P(h42)"] = wfve(PH, "P(h42)")
wt = WHT(V_true); wh = WHT(H); res["ceiling_true_span_wfve"] = (F.cosine_similarity(wt, wh).clamp_min(0) ** 2).mean().item()
print(f"[eval] ceiling: AR(true span) whitened FVE vs real h42 = {res['ceiling_true_span_wfve']:.4f}", flush=True)
json.dump(res, open(f"{args.out}/eval.json", "w"), indent=1); print("ADAPTER_DONE", flush=True)
