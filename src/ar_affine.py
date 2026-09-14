"""AFFINE AR: predict the real h42 (activation preceding a span) from the FROZEN 27B's own block-42 state of the bare
span, with a closed-form ridge regression (linear + bias). No LoRA. Features: last-token block-42 state (optionally
concat mean over span positions). Accumulates second moments on GPU in float64, solves for several lambdas, reports raw
FVE / cosine on a held-out slice and on the common unseen file, saves the best (W, b).

    modal run --detach modal_app.py --task train --gpus 1 --script ar_affine.py --args "--data '...' --exclude shard00_part0000 --n-rows 2000000"
"""
import argparse, glob, json, os, time
import numpy as np, pyarrow.parquet as pq, torch, torch.nn.functional as F
from common import D_MODEL, load_base, Layer42Hook, ar_read

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--exclude", default="shard00_part0000")
ap.add_argument("--n-rows", type=int, default=2_000_000); ap.add_argument("--batch", type=int, default=256); ap.add_argument("--n-tok", type=int, default=12)
ap.add_argument("--feat", default="last", help="last | last+mean | firstK (concat block-42 states of the first K span tokens, e.g. first4) | allK (all K positions, e.g. all12)")
ap.add_argument("--lams", default="1e2,1e3,1e4,1e5,1e6"); ap.add_argument("--heldout-files", type=int, default=5)
ap.add_argument("--common", default="/vol/data/harvest_v3/shard00_part0000.parquet"); ap.add_argument("--frozen", default="/vol/frozen/qwen36_27b_embed_head.pt", help="--feat embed: frozen Qwen token embedding table (no 27B forward)")
ap.add_argument("--out", default="/vol/ckpt/ar_affine")
args = ap.parse_args(); dev = "cuda"; os.makedirs(args.out, exist_ok=True)
files = sorted(sum((glob.glob(g.strip()) for g in args.data.split(",")), [])); files = [f for f in files if args.exclude not in f]
rng = np.random.default_rng(0); files = [files[i] for i in rng.permutation(len(files))]
held_files, train_files = files[: args.heldout_files], files[args.heldout_files:]
if args.feat == "embed":
    EMB = torch.load(args.frozen, map_location="cpu")["embed"].to(dev).float(); base = hook = None
else:
    base = load_base(dev); hook = Layer42Hook(base)
def _k(): return int(args.feat[5:]) if args.feat.startswith("first") else int(args.feat[3:]) if args.feat.startswith("all") and args.feat[3:].isdigit() else 0
Dx = D_MODEL * (2 if args.feat in ("last+mean", "embed", "dot+mean") else _k() if _k() else 1)

def read(f):
    tb = pq.ParquetFile(f).read(columns=["h42", "roll_ids"]); n = tb.num_rows
    return (tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float32),
            tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, 12).astype(np.int64)[:, : args.n_tok])

from common import load_tokenizer as _lt
_tok = _lt(); DOT = _tok(".", add_special_tokens=False)["input_ids"]; assert len(DOT) == 1, DOT; DOT = DOT[0]
@torch.no_grad()
def feats(r_np):
    t = torch.tensor(r_np, device=dev)
    if args.feat.startswith("dot"):                             # append a "." after the span; read the frozen block-42 state AT the "."
        t2 = torch.cat([t, torch.full((t.shape[0], 1), DOT, device=dev, dtype=t.dtype)], 1)
        h = ar_read(base, hook, t2, torch.ones_like(t2), None).float()
        return torch.cat([h[:, -1], h[:, :-1].mean(1)], -1) if args.feat == "dot+mean" else h[:, -1]
    if args.feat == "embed":                                    # NO 27B: frozen token embeddings of the span, [mean ‖ last]
        e = EMB[t]; return torch.cat([e.mean(1), e[:, -1]], -1)
    h = ar_read(base, hook, t, torch.ones_like(t), None).float()   # [B,T,d] frozen block-42 states
    if args.feat.startswith("first") or args.feat.startswith("all"):
        k = _k(); return h[:, :k].reshape(h.shape[0], -1)                    # concat states of the first k span tokens
    x = h[:, -1]
    return torch.cat([x, h.mean(1)], -1) if args.feat == "last+mean" else x

Sxx = torch.zeros(Dx, Dx, dtype=torch.float64, device=dev); Sxy = torch.zeros(Dx, D_MODEL, dtype=torch.float64, device=dev)
sx = torch.zeros(Dx, dtype=torch.float64, device=dev); sy = torch.zeros(D_MODEL, dtype=torch.float64, device=dev); n_seen = 0; t0 = time.time()
for f in train_files:
    H, R = read(f)
    for a in range(0, len(R), args.batch):
        x = feats(R[a:a + args.batch]).double(); y = torch.tensor(H[a:a + args.batch], device=dev).double()
        Sxx += x.T @ x; Sxy += x.T @ y; sx += x.sum(0); sy += y.sum(0); n_seen += len(y)
    if n_seen % 100000 < 5000 or True:
        print(f"[affine] {n_seen} rows | {(time.time()-t0)/60:.1f} min", flush=True)
    if n_seen >= args.n_rows:
        break
xbar, ybar = sx / n_seen, sy / n_seen
Cxx = Sxx.addr_(xbar, xbar, alpha=-n_seen); Cxy = Sxy.addr_(xbar, ybar, alpha=-n_seen)   # centre IN PLACE (61k-dim Gram = 30 GB)
print(f"[affine] moments from {n_seen} rows in {(time.time()-t0)/60:.1f} min", flush=True)

def evalset(fl, cap=20000):
    Hs, Rs = zip(*[read(f) for f in fl]); H = np.concatenate(Hs)[:cap]; R = np.concatenate(Rs)[:cap]
    X = torch.cat([feats(R[a:a + args.batch]) for a in range(0, len(R), args.batch)]).double(); Y = torch.tensor(H, device=dev).double()
    return X, Y
Xh, Yh = evalset(held_files); Xc, Yc = evalset([args.common], 1024)
def score(W, b, X, Y):
    P = X @ W + b; mu = Y.mean(0, keepdim=True)
    fve = 1 - ((P - Y) ** 2).sum() / ((Y - mu) ** 2).sum()
    return {"fve": fve.item(), "cos": F.cosine_similarity(P, Y, dim=-1).mean().item(), "cos_centred": F.cosine_similarity(P - mu, Y - mu, dim=-1).mean().item(),
            "norm_ratio": (P.norm(dim=-1) / Y.norm(dim=-1)).mean().item()}
res = {"n_rows": n_seen, "feat": args.feat, "lams": {}}
best = None
for lam in [float(v) for v in args.lams.split(",")]:
    Cxx.diagonal().add_(lam)                                   # in place: no 30 GB temporaries for the 61k-dim case
    try:
        L_ = torch.linalg.cholesky(Cxx); W = torch.cholesky_solve(Cxy, L_); del L_
    except Exception:
        W = torch.linalg.solve(Cxx, Cxy)
    Cxx.diagonal().sub_(lam); torch.cuda.empty_cache(); b = ybar - xbar @ W
    r = {"heldout": score(W, b, Xh, Yh), "common_file": score(W, b, Xc, Yc)}; res["lams"][f"{lam:g}"] = r
    print(f"[affine] lam={lam:g}: held FVE {r['heldout']['fve']:.4f} cos {r['heldout']['cos']:.4f} (centred {r['heldout']['cos_centred']:.4f}) | common FVE {r['common_file']['fve']:.4f} cos {r['common_file']['cos']:.4f}", flush=True)
    if best is None or r["heldout"]["fve"] > best[0]:
        best = (r["heldout"]["fve"], lam, W.float().cpu(), b.float().cpu())
res["best_lam"] = best[1]
torch.save({"W": best[2].half(), "b": best[3], "feat": args.feat, "lam": best[1], "n_rows": n_seen}, f"{args.out}/affine.pt")
json.dump(res, open(f"{args.out}/results.json", "w"), indent=1)
# reference: mean predictor and the LoRA AR (ar_mse_v3) on the same common rows are in data/ar_vs_h42.json (FVE 0.21)
print("AFFINE_DONE", flush=True)
