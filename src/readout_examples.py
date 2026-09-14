"""Qualitative readouts of a shallow AV: (a) per-token readouts on probe sentences (capture h42 with the
frozen 27B, verbalize with the shallow AV); (b) TRUE continuation vs readout on unseen harvest rows."""
import argparse, glob
import numpy as np, pyarrow.parquet as pq, torch
from common import D_MODEL, Layer42Hook, backbone, load_base, load_tokenizer
from av1_model import OneLayerAV
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--file-offset", type=int, default=530); ap.add_argument("--n-rows", type=int, default=10); args = ap.parse_args()
dev = "cuda"; tok = load_tokenizer(); base = load_base(dev); bb = backbone(base); hook = Layer42Hook(base)
sd = torch.load(args.ckpt, map_location="cpu")
av = OneLayerAV(base, sd.get("init_layer", 63), sd.get("n_layers", 1), sd.get("src_layers")).to(dev)
(av.blocks.load_state_dict if "blocks" in sd else av.blocks[0].load_state_dict)({k: v.to(dev) for k, v in (sd.get("blocks") or sd["block"]).items()})
av.act_proj.load_state_dict({k: v.to(dev) for k, v in sd["act_proj"].items()}); av.embed_scale.data = sd["embed_scale"].to(dev).float(); av.eval()
PROBES = ["The Eiffel Tower is located in the city of", "Paris is the capital of",
          "The detective slowly realized that the real killer was", "Sarah hid the letter so her brother would not",
          "To make the sauce, first melt the butter, then add", "3 times 7 equals"]
for text in PROBES:
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    with torch.no_grad():
        hook.capture(); bb(input_ids=ids, use_cache=False); hook.off()
        h = hook.captured[0].float()                                  # [T,d]
        g = av.generate(h, 10)
    toks = tok.convert_ids_to_tokens(ids[0].tolist())
    print(f"\n### {text}")
    for t in range(len(toks)):
        print(f"  {toks[t].replace('Ġ',' ')!r:14s} -> {tok.decode(g[t]).strip()!r}")
files = sorted(glob.glob(args.data)); f = files[min(args.file_offset, len(files)-1)]
tb = pq.ParquetFile(f).read(columns=["h42", "roll_ids"]); n = tb.num_rows
H = tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float32)[:args.n_rows]
R = tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, 12).astype(np.int64)[:args.n_rows]
with torch.no_grad():
    g = av.generate(torch.tensor(H, device=dev), 12)
print("\n### UNSEEN rows: TRUE continuation || readout")
for i in range(len(H)):
    print(f"  TRUE {tok.decode(R[i])!r}\n  READ {tok.decode(g[i])!r}\n")
print("READOUT_DONE", flush=True)
