"""Inverter readers: text span -> AR(span) -> reader -> text.  (The reader's input is AR(span), NOT a raw activation.)

    python feed_inverter_example.py --ar ar_mse_v3 --ckpt av4_5m_arinv_b/final.pt --spans "the recipe calls for two cups of flour and" "Behold, while the child was still alive,"
"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ar_span import ARSpan
from av1_model import OneLayerAV

ap = argparse.ArgumentParser(); ap.add_argument("--ar", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--spans", nargs="+", required=True); ap.add_argument("--n-tok", type=int, default=12)
args = ap.parse_args(); dev = "cuda"
ar = ARSpan(args.ar, device=dev)                                        # loads the 27B once
sd = torch.load(args.ckpt, map_location="cpu")
# build the reader on the SAME base (its blocks were copied before any LoRA; the frozen embed/norm/head are shared)
av = OneLayerAV(ar.base.get_base_model() if hasattr(ar.base, "get_base_model") else ar.base, sd.get("init_layer", 63), sd.get("n_layers", 1), sd.get("src_layers"), sd.get("block_type", "attn")).to(dev)
av.blocks.load_state_dict({k: v.to(dev) for k, v in sd["blocks"].items()}); av.act_proj.load_state_dict({k: v.to(dev) for k, v in sd["act_proj"].items()})
av.embed_scale.data = sd["embed_scale"].to(dev).float(); av.eval()
V = ar(args.spans, n_tok=args.n_tok)                                    # [B, 5120] AR(span)
with torch.no_grad():
    out = av.generate(V, args.n_tok)
for s, o in zip(args.spans, out):
    print(f"SPAN {s!r}\n  -> {ar.tok.decode(o)!r}")
