"""AR(span) vs the REAL h42 it approximates — what falls out?
On the common unseen file: vector stats (cosine, centred cosine, norm ratio, FVE, residual size, per-dim variance
shrinkage), then a cross-feed CE matrix: two verbalizers (trained on real h42 / trained on AR(span)) x inputs
{real h42, AR(span), AR(span) norm-matched, residual h42-AR(span) norm-matched, zero, shuffled h42}, plus decodes.

    modal run --detach modal_app.py --task train --gpus 1 --script ar_vs_h42.py --args "--av-h42 ... --av-ar ..."
"""
import argparse, json
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F
from av1_model import OneLayerAV
from common import D_MODEL, load_base, load_tokenizer, Layer42Hook, ar_read

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="/vol/data/harvest_v3/shard00_part0000.parquet")
ap.add_argument("--n", type=int, default=1024); ap.add_argument("--n-tok", type=int, default=12)
ap.add_argument("--av-h42", required=True, help="verbalizer trained on real h42")
ap.add_argument("--av-ar", required=True, help="verbalizer trained on AR(span)")
ap.add_argument("--ar", default="/vol_data/ckpt/ar_mse_v3/final"); ap.add_argument("--out", default="/vol/results/ar_vs_h42.json")
args = ap.parse_args(); dev = "cuda"; tok = load_tokenizer()

tb = pq.ParquetFile(args.file).read(columns=["h42", "roll_ids"]); n = tb.num_rows
H = torch.tensor(tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float32)[: args.n], device=dev)
R = torch.tensor(tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, 12).astype(np.int64)[: args.n, : args.n_tok], device=dev)
N = H.shape[0]; print(f"[ar-vs-h42] {N} unseen rows", flush=True)

base = load_base(dev)
def build(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    m = OneLayerAV(base, sd.get("init_layer", 63), sd.get("n_layers", 1), sd.get("src_layers")).to(dev)
    m.blocks.load_state_dict({k: v.to(dev) for k, v in sd["blocks"].items()}); m.act_proj.load_state_dict({k: v.to(dev) for k, v in sd["act_proj"].items()})
    m.embed_scale.data = sd["embed_scale"].to(dev).float(); return m.eval()
avh, ava = build(args.av_h42), build(args.av_ar)          # blocks deep-copied BEFORE the LoRA goes in
from peft import PeftModel
AR = PeftModel.from_pretrained(base, args.ar, adapter_name="ar").eval()
value_head = nn.Linear(D_MODEL, D_MODEL, bias=True).to(dev, torch.float32)
value_head.load_state_dict(torch.load(f"{args.ar}/value_head.pt", map_location=dev)); hook = Layer42Hook(AR)

with torch.no_grad():
    V = torch.cat([value_head(ar_read(AR, hook, R[s:s + 64], torch.ones_like(R[s:s + 64]), value_head)[:, -1].float()) for s in range(0, N, 64)])
mu = H.mean(0, keepdim=True); Res = H - V
def cos(a, b): return F.cosine_similarity(a, b, dim=-1)
stats = {
    "cos(AR, h42)": cos(V, H).mean().item(), "cos_centred(AR-mu, h42-mu)": cos(V - mu, H - mu).mean().item(),
    "cos(mu, h42)": cos(mu.expand_as(H), H).mean().item(),
    "norm_ratio |AR|/|h42|": (V.norm(dim=-1) / H.norm(dim=-1)).mean().item(), "|h42| mean": H.norm(dim=-1).mean().item(), "|AR| mean": V.norm(dim=-1).mean().item(),
    "FVE raw (1 - |AR-h|^2 / |h-mu|^2)": (1 - ((V - H) ** 2).sum() / ((H - mu) ** 2).sum()).item(),
    "residual |h42-AR| / |h42|": (Res.norm(dim=-1) / H.norm(dim=-1)).mean().item(),
    "residual |h42-AR| / |h42-mu|": (Res.norm(dim=-1) / (H - mu).norm(dim=-1)).mean().item(),
    "per-dim var ratio var(AR)/var(h42) (median)": (V.var(0) / H.var(0)).median().item(),
    "per-dim var ratio (mean)": (V.var(0) / H.var(0)).mean().item(),
    "cos(residual, h42-mu)": cos(Res, H - mu).mean().item(),
}
for k, v in stats.items(): print(f"[stat] {k}: {v:.4f}", flush=True)

perm = torch.randperm(N, device=dev)
inputs = {"real h42": H, "AR(span)": V, "AR(span) norm-matched to |h42|": V * (H.norm(dim=-1, keepdim=True) / V.norm(dim=-1, keepdim=True)),
          "residual h42-AR(span), norm-matched": Res * (H.norm(dim=-1, keepdim=True) / Res.norm(dim=-1, keepdim=True)),
          "zero": torch.zeros_like(H), "shuffled h42": H[perm]}
@torch.no_grad()
def ce(model, X):
    tot = 0.0
    for s in range(0, N, 64):
        lg = model(X[s:s + 64], R[s:s + 64])[:, :-1].float()
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), R[s:s + 64].reshape(-1), reduction="sum").item()
    return tot / (N * args.n_tok)
matrix = {}
for mname, m in (("AV trained on real h42", avh), ("AV trained on AR(span)", ava)):
    matrix[mname] = {k: ce(m, X) for k, X in inputs.items()}
    print(f"[ce] {mname}: " + " | ".join(f"{k}={v:.3f}" for k, v in matrix[mname].items()), flush=True)
decodes = []
with torch.no_grad():
    for i in range(6):
        d = {"true": tok.decode(R[i])}
        for mname, m in (("h42-AV", avh), ("AR-AV", ava)):
            for iname, X in (("real h42", H), ("AR(span)", V)):
                d[f"{mname} <- {iname}"] = tok.decode(m.generate(X[i:i + 1], args.n_tok)[0])
        decodes.append(d); print("[decode]", json.dumps(d, ensure_ascii=False), flush=True)
import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({"file": args.file, "n": N, "av_h42": args.av_h42, "av_ar": args.av_ar, "ar": args.ar, "stats": stats, "ce_matrix": matrix, "decodes": decodes}, open(args.out, "w"), indent=1, ensure_ascii=False)
print("ARVSH42_DONE", flush=True)
