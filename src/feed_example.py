"""End-to-end: capture a Qwen3.6-27B layer-42 activation for any text and verbalize it with a shallow reader.

    python feed_example.py --ckpt av2_30m/final.pt --text "The recipe calls for two cups of flour and" --positions -1,-4

What the reader expects
  * ONE vector per read: the OUTPUT of decoder block 42 (= hidden_states[43] with output_hidden_states=True), i.e. the
    raw residual stream after block 42, at the token position you want to read. Shape [B, 5120], float32, norm ~89.
  * Do NOT normalize, whiten, mean-centre or scale it; the reader's own act_proj (5120->5120 linear) is the only map.
  * The 27B forward that produces it: plain token ids of the context, no chat template, no BOS prepended (the training
    harvest fed bare 32-256-token windows of web text; contexts shorter than ~32 tokens are out of distribution).
  * Inside the reader the vector becomes the soft token at position 0 (act_proj(h42)); positions 1..T are the ordinary
    token embeddings of what it has generated so far. Greedy 12-token readout = av.generate(h42, 12).
"""
import argparse, sys, os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_base, load_tokenizer, Layer42Hook
from av1_model import OneLayerAV

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--text", required=True)
ap.add_argument("--positions", default="-1", help="comma-separated token positions to read (negative = from the end)")
ap.add_argument("--n-tok", type=int, default=12)
args = ap.parse_args(); dev = "cuda"

tok = load_tokenizer(); base = load_base(dev)                       # Qwen3.6-27B, bf16, ~54 GB

# --- 1) capture h42 for every position of the text (ONE forward pass of the 27B) --------------------------------
ids = torch.tensor([tok(args.text, add_special_tokens=False)["input_ids"]], device=dev)   # bare ids, no BOS/template
hook = Layer42Hook(base).capture()
with torch.no_grad():
    base(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
hook.off()
H = hook.captured[0].float()                                           # [T, 5120]  block-42 output, raw residual
# equivalent without the hook:  base(..., output_hidden_states=True).hidden_states[43][0].float()
print(f"text has {H.shape[0]} tokens; h42 norms {H.norm(dim=-1).mean():.1f} +- {H.norm(dim=-1).std():.1f}")

# --- 2) build the reader from the same base (frozen embed/norm/head) + the checkpoint's trainable blocks --------
sd = torch.load(args.ckpt, map_location="cpu")
av = OneLayerAV(base, sd.get("init_layer", 63), sd.get("n_layers", 1), sd.get("src_layers")).to(dev)
av.blocks.load_state_dict({k: v.to(dev) for k, v in sd["blocks"].items()})
av.act_proj.load_state_dict({k: v.to(dev) for k, v in sd["act_proj"].items()})
av.embed_scale.data = sd["embed_scale"].to(dev).float(); av.eval()

# --- 3) read chosen positions: soft token = act_proj(h42[pos]), then greedy decode -----------------------------
pos = [int(p) % H.shape[0] for p in args.positions.split(",")]
h42 = H[pos]                                                           # [B, 5120] float32 -- feed as is
with torch.no_grad():
    out = av.generate(h42, args.n_tok)                                 # [B, n_tok] token ids
for p, o in zip(pos, out):
    print(f"pos {p:3d} ({tok.decode(ids[0, :p + 1])[-40:]!r})  ->  {tok.decode(o)!r}")
# teacher-forced scoring of a candidate continuation instead of decoding:
#   logits = av(h42, cand_ids)      # [B, T+1, V]; logits[:, i] predicts cand_ids[:, i]
