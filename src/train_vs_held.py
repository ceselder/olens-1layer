"""Measure train-vs-held divergence properly: for each saved checkpoint of a run, CE on a fixed
TRAINED-ON subset (rows rank0 trained on) vs the fixed HELD-OUT subset, identical protocol."""
import argparse, glob, re
import numpy as np, pyarrow.parquet as pq, torch, torch.nn.functional as F
from common import D_MODEL, load_base
from av1_model import OneLayerAV
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--heldout", type=int, default=2000); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--world", type=int, default=4)
args = ap.parse_args(); dev = "cuda"
files = sorted(glob.glob(args.data)); r0 = files[0::args.world][:2]        # rank0's first two shard files
H = []; R = []
for f in r0:
    tb = pq.ParquetFile(f).read(columns=["h42", "roll_ids"]); n = tb.num_rows
    H.append(tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float32))
    R.append(tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, 12).astype(np.int64))
H = np.concatenate(H); R = np.concatenate(R)
held = (H[:args.n], R[:args.n]); trained = (H[args.heldout:args.heldout + args.n], R[args.heldout:args.heldout + args.n])
print(f"[tvh] held rows [0,{args.n}) | trained-on rows [{args.heldout},{args.heldout+args.n})", flush=True)
base = load_base(dev); ckpts = sorted(glob.glob(f"{args.run}/step_*.pt") + glob.glob(f"{args.run}/final.pt"),
                                      key=lambda p: int(re.search(r"step_(\d+)", p).group(1)) if "step_" in p else 10**9)
sd0 = torch.load(ckpts[0], map_location="cpu")
model = OneLayerAV(base, sd0.get("init_layer", 63), sd0.get("n_layers", 1), sd0.get("src_layers")).to(dev).eval()
@torch.no_grad()
def ce(Hs, Rs):
    tot = 0.0; nb = 0
    for s in range(0, len(Hs), 64):
        h = torch.tensor(Hs[s:s+64], device=dev); t = torch.tensor(Rs[s:s+64], device=dev)
        lg = model(h, t)[:, :-1].float(); tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t.reshape(-1)).item(); nb += 1
    return tot / nb
print(f"{'step':>7s} {'trained-on CE':>14s} {'held-out CE':>12s} {'gap':>7s}")
for ck in ckpts:
    sd = torch.load(ck, map_location="cpu")
    (model.blocks.load_state_dict if "blocks" in sd else model.blocks[0].load_state_dict)({k: v.to(dev) for k, v in (sd.get("blocks") or sd["block"]).items()})
    model.act_proj.load_state_dict({k: v.to(dev) for k, v in sd["act_proj"].items()}); model.embed_scale.data = sd["embed_scale"].to(dev).float()
    st = re.search(r"step_(\d+)", ck); st = int(st.group(1)) if st else "final"
    a, b = ce(*trained), ce(*held); print(f"{str(st):>7s} {a:14.3f} {b:12.3f} {b-a:7.3f}", flush=True)
print("TVH_DONE", flush=True)
