"""Does the ONE-LAYER AV actually read h42, or did it just learn a language model over rollouts?
On unseen spans: CE(correct h42) vs CE(zero vector) vs CE(shuffled h42). Conditioning on the
SPECIFIC activation <=> CE(correct) << CE(shuffled) ≈ CE(zero). Plus greedy decodes to eyeball.

    modal run modal_app.py --task train --gpus 1 --script conditioning_test_1layer.py \
      --args "--ckpt /vol/ckpt/av1_l63/final.pt --data '/vol_data/data/harvest_v1/*.parquet' --n 512"
"""
import argparse, glob
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from av1_model import load_av1, load_small
from common import D_MODEL, load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=512); ap.add_argument("--n-tok", type=int, default=12)
ap.add_argument("--file-offset", type=int, default=300, help="use a file well past the training shards")
ap.add_argument("--frozen", default="/vol/frozen/qwen36_27b_embed_head.pt")
ap.add_argument("--input-ar", default=None, help="override the checkpoint metadata: lora | affine:<affine.pt> (which AR vector this reader was trained on)")
ap.add_argument("--scales", default="", help="also report CE(correct) with h42 multiplied by each of these scales")
args = ap.parse_args()
dev = "cuda"; tok = load_tokenizer()

files = sorted(glob.glob(args.data))
f = files[min(args.file_offset, len(files) - 1)]
tb = pq.ParquetFile(f).read(columns=["h42", "roll_ids"]); n = tb.num_rows
H = tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float32)[: args.n]
R = tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, 12).astype(np.int64)[: args.n, : args.n_tok]
print(f"[cond1] ckpt={args.ckpt} test={f} n={len(H)}", flush=True)

model = load_small(args.ckpt, args.frozen, dev) if torch.load(args.ckpt, map_location="cpu").get("block_type") in ("small", "fullattn", "linrnn") else load_av1(args.ckpt, dev)
_sd = torch.load(args.ckpt, map_location="cpu"); AR_ADAPTER, AR_AFFINE = _sd.get("ar"), _sd.get("ar_affine")
if args.input_ar:
    AR_ADAPTER, AR_AFFINE = ("/vol_data/ckpt/ar_mse_v3/final", None) if args.input_ar == "lora" else (None, args.input_ar.split(":", 1)[1])
elif (_sd.get("ar_cache") or "").find("affine") >= 0:
    AR_ADAPTER, AR_AFFINE = None, "/vol/ckpt/ar_affine_lastmean/affine.pt"
V = None
if AR_ADAPTER or AR_AFFINE:                      # INVERTER checkpoint: its input is AR(span), not the real h42
    from common import load_base, Layer42Hook, ar_read
    import torch.nn as nn
    base = load_base(dev); hook = Layer42Hook(base)
    Rt = torch.tensor(R, device=dev)
    if AR_ADAPTER:
        from peft import PeftModel
        ARm = PeftModel.from_pretrained(base, AR_ADAPTER, adapter_name="ar").eval(); hook = Layer42Hook(ARm)
        vh = nn.Linear(D_MODEL, D_MODEL, bias=True).to(dev, torch.float32); vh.load_state_dict(torch.load(f"{AR_ADAPTER}/value_head.pt", map_location=dev))
        with torch.no_grad():
            V = torch.cat([vh(ar_read(ARm, hook, Rt[s:s + 64], torch.ones_like(Rt[s:s + 64]), vh)[:, -1].float()) for s in range(0, len(R), 64)])
    else:
        aff = torch.load(AR_AFFINE, map_location=dev); W, b = aff["W"].to(dev).float(), aff["b"].to(dev).float()
        with torch.no_grad():
            hs = [ar_read(base, hook, Rt[s:s + 64], torch.ones_like(Rt[s:s + 64]), None).float() for s in range(0, len(R), 64)]
            V = torch.cat([(torch.cat([h[:, -1], h.mean(1)], -1) if aff["feat"] == "last+mean" else h[:, -1]) @ W + b for h in hs])
    print(f"[cond1] INVERTER checkpoint: input = AR(span) from {AR_ADAPTER or AR_AFFINE}", flush=True)


@torch.no_grad()
def ce(h, t):
    lg = model(h, t)[:, :-1].float()
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1)).item()


on = off = shuf = real = 0.0; nb = 0
for s in range(0, len(H), 64):
    hr = torch.tensor(H[s:s + 64], device=dev); t = torch.tensor(R[s:s + 64], device=dev)
    h = V[s:s + 64] if V is not None else hr                        # inverter: score in AR-space
    if h.shape[0] < 2:
        continue
    on += ce(h, t); off += ce(torch.zeros_like(h), t); shuf += ce(h[torch.randperm(h.shape[0], device=dev)], t); nb += 1
    real += ce(hr, t)                                                # reference: the real h42 fed in
if V is not None:
    print(f"[cond1] AR-SPACE: CE(correct AR(span))={on/nb:.4f} | CE(zero)={off/nb:.4f} | CE(shuffled AR(span))={shuf/nb:.4f} | CE(real h42 in)={real/nb:.4f}", flush=True)
print(f"[cond1] CE(correct h42)={on/nb:.4f} | CE(zero)={off/nb:.4f} | CE(shuffled h42)={shuf/nb:.4f}", flush=True)
print(f"[cond1] conditioning gap (zero-correct)={(off-on)/nb:.4f} | specificity (shuffled-correct)={(shuf-on)/nb:.4f}", flush=True)
for sc in [float(x) for x in args.scales.split(",") if x]:
    acc = 0.0; nb2 = 0
    for s in range(0, len(H), 64):
        h = torch.tensor(H[s:s + 64], device=dev) * sc; t = torch.tensor(R[s:s + 64], device=dev)
        if h.shape[0] < 2:
            continue
        acc += ce(h, t); nb2 += 1
    print(f"[scale] h42 x {sc:.2f}: CE(correct)={acc/nb2:.4f}", flush=True)
h = V[:6] if V is not None else torch.tensor(H[:6], device=dev); g = model.generate(h, args.n_tok)
for i in range(6):
    print(f"[decode] TRUE {tok.decode(R[i])!r}\n         GEN  {tok.decode(g[i])!r}", flush=True)
import json, os
os.makedirs("/vol/results/cond", exist_ok=True)
tag = args.ckpt.split("/ckpt/")[-1].replace("/final.pt", "").replace("/", "_") + "__" + f.split("/data/")[-1].replace("/", "_").replace(".parquet", "")
json.dump({"ckpt": args.ckpt, "test_file": f, "n": len(H), "input": ("AR(span)" if V is not None else "real h42"), "ar": AR_ADAPTER or AR_AFFINE, "ce_correct": on / nb, "ce_zero": off / nb, "ce_shuffled": shuf / nb, "ce_real_h42": real / nb,
           "decodes": [{"true": tok.decode(R[i]), "gen": tok.decode(g[i])} for i in range(6)]}, open(f"/vol/results/cond/{tag}.json", "w"), indent=1)
print("COND1_DONE", flush=True)
