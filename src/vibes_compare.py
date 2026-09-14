"""VIBES: do the tiny readers surface intermediates like the modulation oracle does?
For WorkspaceBench multihop items: run Qwen3.6-27B on the plain prompt, capture the layer-42 activation at the FINAL prompt token
(the read site the bench uses), then read it out with (a) the modulation oracle (27B LoRA, REPLACE-mode injection at layers[1],
its own prompt.txt + chat template, 4 bullets), (b) our RL'd tiny 1-layer reader (adapter P + 12-token greedy/sampled spans),
(c) our 27B Karvonen-injection reader (raw h42). Prints side by side with the gold target + intermediates; saves JSON.
"""
import argparse, json, os
import torch, torch.nn.functional as F
from peft import PeftModel
from common import D_MODEL, InjectL1, Layer42Hook, MARKER_ID, backbone, build_av_prompt, load_base, load_tokenizer
from av1_model import load_small

ap = argparse.ArgumentParser()
ap.add_argument("--bench", default="/root/src/bench_multihop.json"); ap.add_argument("--n", type=int, default=24)
ap.add_argument("--modlens", default="/vol/modlens/rl-step25"); ap.add_argument("--karvonen", default="/vol/ckpt/av27b_karvonen_12m/step_005000")
ap.add_argument("--tiny", default="/vol/ckpt/olrl_L1_loraAR_2048x16/step_001100.pt"); ap.add_argument("--adapter", default="/vol/ckpt/adapter_h42_to_ar/adapter.pt"); ap.add_argument("--frozen", default="/vol/frozen/qwen36_27b_embed_head.pt")
ap.add_argument("--out", default="/vol/results/vibes_multihop.json"); args = ap.parse_args(); dev = "cuda"; tok = load_tokenizer()
items = json.load(open(args.bench))[: args.n]
base = load_base(dev); hook42 = Layer42Hook(base)
# ---- 1) capture h42 at the final prompt token for every item (plain render, no chat template) ----
H = []
with torch.no_grad():
    for it in items:
        ids = torch.tensor([tok(it["prompt"], add_special_tokens=False)["input_ids"]], device=dev)
        hook42.capture(); base(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False); hook42.off()
        H.append(hook42.captured[0, -1].float())
H = torch.stack(H); print(f"[vibes] {len(items)} items, h42 norms {H.norm(dim=-1).mean():.1f}", flush=True)
# ---- 2) readers on the same base ----
model = PeftModel.from_pretrained(base, args.modlens, adapter_name="modlens").eval(); model.load_adapter(args.karvonen, adapter_name="kar")
class ReplaceInject:                                  # modulation oracle: h'_p = v at the marker (direction AND magnitude), layers[1]
    def __init__(self, m, layer=1): self.vec = None; self.ids = None; self.h = backbone(m).layers[layer].register_forward_hook(self)
    def set(self, vec, ids): self.vec, self.ids = vec, ids
    def off(self): self.vec = None
    def __call__(self, _m, _i, out):
        if self.vec is None: return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1: return out
        b, t = (self.ids == MARKER_ID).nonzero(as_tuple=True); h = h.clone(); h[b, t] = self.vec[b].to(h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h
rep = ReplaceInject(model); kar_inj = InjectL1(model)
ML_PROMPT = tok.apply_chat_template([{"role": "user", "content": open(os.path.join(args.modlens, "prompt.txt")).read()}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
ML_IDS = torch.tensor([tok(ML_PROMPT, add_special_tokens=False)["input_ids"]], device=dev); assert MARKER_ID in ML_IDS[0].tolist()
KAR_IDS = torch.tensor([build_av_prompt(tok)], device=dev)
@torch.no_grad()
def modlens(h):
    model.set_adapter("modlens"); rep.set(h[None], ML_IDS)
    g = model.generate(input_ids=ML_IDS, attention_mask=torch.ones_like(ML_IDS), max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id); rep.off()
    return tok.decode(g[0, ML_IDS.shape[1]:], skip_special_tokens=True).strip()
@torch.no_grad()
def karvonen(h, n_samp=4):
    model.set_adapter("kar")
    ids1 = KAR_IDS; kar_inj.set(h[None], ids1)                                      # batch sizes of vec and ids must match per call
    g0 = model.generate(input_ids=ids1, attention_mask=torch.ones_like(ids1), max_new_tokens=12, min_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id); kar_inj.off()
    idsn = KAR_IDS.repeat(n_samp, 1); kar_inj.set(h[None].repeat(n_samp, 1), idsn)
    gs = model.generate(input_ids=idsn, attention_mask=torch.ones_like(idsn), max_new_tokens=12, min_new_tokens=12, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id); kar_inj.off()
    return tok.decode(g0[0, KAR_IDS.shape[1]:]), [tok.decode(x[KAR_IDS.shape[1]:]) for x in gs]
tiny = load_small(args.tiny, args.frozen, dev); ad = torch.load(args.adapter, map_location=dev); P_, b_ = ad["P"].to(dev).float(), ad["b"].to(dev).float()
@torch.no_grad()
def tinyread(h, n_samp=4):
    v = (h[None] @ P_ + b_); g0 = tiny.generate(v, 12); gs = tiny.generate(v.repeat(n_samp, 1), 12, temperature=1.0)
    return tok.decode(g0[0]), [tok.decode(x) for x in gs]
out = []
for it, h in zip(items, H):
    ml = modlens(h); kg, ks = karvonen(h); tg, ts = tinyread(h)
    row = {"name": it["name"], "prompt": it["prompt"], "target": it["target"], "intermediates": it.get("intermediates", []), "modulation_oracle_4bullets": ml, "karvonen27b_greedy": kg, "karvonen27b_samples": ks, "tiny1L_rl_greedy": tg, "tiny1L_rl_samples": ts}
    out.append(row)
    print(f"\n### {it['name']} | target: {it['target']} | intermediates: {it.get('intermediates')}\n  PROMPT: {it['prompt']!r}\n  MODLENS: {ml!r}\n  27B-KARV greedy: {kg!r}\n           samples: {ks}\n  TINY-1L  greedy: {tg!r}\n           samples: {ts}", flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False); print("VIBES_DONE", flush=True)
