"""The modulation oracle's OWN metric on held-out REAL activations, for several readers:
   target = unit(J h - amu) (J-lens[42], amu = natural mean in J-space); each reader gives up to 4 bullets/spans per activation;
   AR (their LoRA text->J-vector, mean-pool + head) embeds each; NNLS over all non-empty supports -> cos of the reconstruction.
   matched = vs own target, permuted = vs a rolled target; delta = matched - permuted (their headline).
Readers: modulation oracle adapters (replace-mode injection at layers[1], own prompt, greedy 4 bullets); our 27B Karvonen reader
(4 sampled 12-token spans); our tiny readers (4 sampled spans via adapter P). Also K=1 greedy for the span readers.
"""
import argparse, itertools, json, os, re
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F
from peft import PeftModel
from common import D_MODEL, InjectL1, Layer42Hook, MARKER_ID, ar_read, backbone, build_av_prompt, load_base, load_tokenizer
from av1_model import load_small

ap = argparse.ArgumentParser()
ap.add_argument("--eval-file", default="/vol/data/harvest_v5/shard00_part0000.parquet"); ap.add_argument("--n", type=int, default=256)
ap.add_argument("--modlens", default="/vol/modlens/rl-step25,/vol/modlens/rl-step50,/vol/modlens/sft"); ap.add_argument("--karvonen", default="/vol/ckpt/av27b_karvonen_12m/step_005000")
ap.add_argument("--tiny", default="/vol/ckpt/small_d1024L1_30m/final.pt,/vol/ckpt/olrl_L1_loraAR_2048x16/step_001100.pt"); ap.add_argument("--adapter", default="/vol/ckpt/adapter_h42_to_ar/adapter.pt"); ap.add_argument("--frozen", default="/vol/frozen/qwen36_27b_embed_head.pt")
ap.add_argument("--modlens-ar", default="/vol_modlens/ar_l42_text2vec"); ap.add_argument("--jlens", default="/vol/jlens/qwen36_27b_jlens.pt"); ap.add_argument("--amu", default="/vol_modlens/data/natural_whitener_jspace.npz")
ap.add_argument("--out", default="/vol/results/modlens_metric.json"); args = ap.parse_args(); dev = "cuda"; tok = load_tokenizer()
tb = pq.ParquetFile(args.eval_file).read(columns=["h42", "roll_ids"]); m = min(tb.num_rows, args.n)
H = torch.tensor(tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, D_MODEL).astype(np.float32)[:m], device=dev)
R = torch.tensor(tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, 12).astype(np.int64)[:m], device=dev)
base = load_base(dev)
model = PeftModel.from_pretrained(base, args.modlens_ar, adapter_name="ar").eval(); hook = Layer42Hook(model)
HEAD = nn.Linear(D_MODEL, D_MODEL, bias=True).to(dev, torch.float32); HEAD.load_state_dict(torch.load(os.path.join(args.modlens_ar, "head.pt"), map_location=dev)["head"]); HEAD.eval()
J = torch.load(args.jlens, map_location="cpu", weights_only=False)["J"][42].to(dev).float(); AMU = torch.tensor(np.load(args.amu)["mu"], device=dev).float()
T = F.normalize(H @ J.T - AMU, dim=-1)                                            # targets in the reward space
for i, d in enumerate([x for x in args.modlens.split(",") if x]): model.load_adapter(d, adapter_name=f"ml{i}")
model.load_adapter(args.karvonen, adapter_name="kar")
BULLET_RE = re.compile(r"^\s*[\*\-•]+\s*")
@torch.no_grad()
def ar_embed(phrases):                                                            # list[str] -> [n, D] AR vectors (their embed(): mean-pool over phrase tokens, head)
    model.set_adapter("ar"); out = []
    for a in range(0, len(phrases), 128):
        b = tok(phrases[a:a + 128], add_special_tokens=False, padding=True, truncation=True, max_length=14, return_tensors="pt").to(dev)
        h = ar_read(model, hook, b["input_ids"], b["attention_mask"], None).float(); mk = b["attention_mask"].unsqueeze(-1).float()
        out.append(HEAD((h * mk).sum(1) / mk.sum(1).clamp(min=1e-6)))
    return torch.cat(out)
def nnls_exact(B, t):
    k = B.shape[0]; best = -1.0; G = B @ B.T; c = B @ t
    for r in range(1, k + 1):
        for sup in itertools.combinations(range(k), r):
            idx = torch.tensor(sup, device=B.device); Gs = G[idx][:, idx] + 1e-6 * torch.eye(r, device=B.device)
            try: w = torch.linalg.solve(Gs, c[idx])
            except Exception: continue
            if bool((w < -1e-8).any()): continue
            rec = w.clamp(min=0) @ B[idx]; cs = F.cosine_similarity(rec, t, dim=0).item() if rec.norm() > 0 else -1.0
            best = max(best, cs)
    return best
def score(bullet_lists):                                                          # list (per activation) of list[str] -> matched, permuted
    uniq = sorted({b for row in bullet_lists for b in row}); emb = {p: i for i, p in enumerate(uniq)}; V = F.normalize(ar_embed(uniq), dim=-1) if uniq else None
    mt, pm = [], []
    for i, row in enumerate(bullet_lists):
        if not row: mt.append(0.0); pm.append(0.0); continue
        B = V[torch.tensor([emb[p] for p in row], device=dev)]; mt.append(max(nnls_exact(B, T[i]), 0.0)); pm.append(max(nnls_exact(B, T[(i + 1) % m]), 0.0))
    return float(np.mean(mt)), float(np.mean(pm))
def split_bullets(text, k=4, max_tok=12):
    out = []
    for line in (text or "").splitlines():
        s_ = BULLET_RE.sub("", line.strip()).strip()
        if not s_: continue
        ids = tok(s_, add_special_tokens=False)["input_ids"][:max_tok]; s_ = tok.decode(ids, skip_special_tokens=True).strip()
        if s_: out.append(s_)
        if len(out) == k: break
    return out
res = {"eval_file": args.eval_file, "n": m, "readers": {}}
# ---- modulation oracle adapters: replace-mode injection, greedy 4 bullets
class ReplaceInject:
    def __init__(self, mdl, layer=1): self.vec = None; self.ids = None; self.h = backbone(mdl).layers[layer].register_forward_hook(self)
    def set(self, vec, ids): self.vec, self.ids = vec, ids
    def off(self): self.vec = None
    def __call__(self, _m, _i, out):
        if self.vec is None: return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1: return out
        b, t = (self.ids == MARKER_ID).nonzero(as_tuple=True); h = h.clone(); h[b, t] = self.vec[b].to(h.dtype); return (h, *out[1:]) if isinstance(out, tuple) else h
rep = ReplaceInject(model)
for i, d in enumerate([x for x in args.modlens.split(",") if x]):
    prompt = tok.apply_chat_template([{"role": "user", "content": open(os.path.join(d, "prompt.txt")).read()}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    pids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]], device=dev); texts = []
    model.set_adapter(f"ml{i}")
    with torch.no_grad():
        for a in range(0, m, 16):
            hb = H[a:a + 16]; ids = pids.repeat(hb.shape[0], 1); rep.set(hb, ids)
            g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id); rep.off()
            texts += [tok.decode(x[ids.shape[1]:], skip_special_tokens=True) for x in g]
    bl = [split_bullets(t) for t in texts]; mt, pm = score(bl)
    res["readers"][f"modulation oracle {os.path.basename(d)}"] = {"matched": mt, "permuted": pm, "delta": mt - pm, "mean_bullets": float(np.mean([len(b) for b in bl])), "examples": texts[:3]}
    print(f"[modlens-metric] modulation oracle {os.path.basename(d)}: matched {mt:.4f} permuted {pm:.4f} delta {mt-pm:.4f} | bullets/activation {np.mean([len(b) for b in bl]):.2f}", flush=True)
# ---- our 27B Karvonen reader: 4 sampled spans (+ greedy K=1)
kinj = InjectL1(model); KIDS = torch.tensor([build_av_prompt(tok)], device=dev); model.set_adapter("kar"); spans = []; gre = []
with torch.no_grad():
    for a in range(0, m, 16):
        hb = H[a:a + 16]; ids = KIDS.repeat(hb.shape[0] * 4, 1); kinj.set(hb.repeat_interleave(4, 0), ids)
        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=12, min_new_tokens=12, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id); kinj.off()
        dec = [tok.decode(x[KIDS.shape[1]:]).strip() for x in g]; spans += [dec[j * 4:(j + 1) * 4] for j in range(hb.shape[0])]
        ids1 = KIDS.repeat(hb.shape[0], 1); kinj.set(hb, ids1); g1 = model.generate(input_ids=ids1, attention_mask=torch.ones_like(ids1), max_new_tokens=12, min_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id); kinj.off(); gre += [[tok.decode(x[KIDS.shape[1]:]).strip()] for x in g1]
mt, pm = score(spans); mg, pg = score(gre); res["readers"]["27B Karvonen reader (4 sampled spans)"] = {"matched": mt, "permuted": pm, "delta": mt - pm, "greedy_K1": {"matched": mg, "permuted": pg, "delta": mg - pg}, "examples": spans[:2]}
print(f"[modlens-metric] 27B Karvonen reader: 4 samples matched {mt:.4f} permuted {pm:.4f} delta {mt-pm:.4f} | greedy K=1 delta {mg-pg:.4f}", flush=True)
# ---- tiny readers (adapter P), 4 sampled spans (+ greedy K=1)
ad = torch.load(args.adapter, map_location=dev); P_, b_ = ad["P"].to(dev).float(), ad["b"].to(dev).float(); V_in = H @ P_ + b_
for ck in [x for x in args.tiny.split(",") if x]:
    rd = load_small(ck, args.frozen, dev)
    with torch.no_grad():
        gs = rd.generate(V_in.repeat_interleave(4, 0), 12, temperature=1.0); g1 = rd.generate(V_in, 12)
    spans = [[tok.decode(gs[i * 4 + j]).strip() for j in range(4)] for i in range(m)]; gre = [[tok.decode(g1[i]).strip()] for i in range(m)]
    mt, pm = score(spans); mg, pg = score(gre); name = ck.split("/ckpt/")[-1]
    res["readers"][f"tiny {name} (4 sampled spans, adapter)"] = {"matched": mt, "permuted": pm, "delta": mt - pm, "greedy_K1": {"matched": mg, "permuted": pg, "delta": mg - pg}, "examples": spans[:2]}
    print(f"[modlens-metric] tiny {name}: 4 samples matched {mt:.4f} permuted {pm:.4f} delta {mt-pm:.4f} | greedy K=1 delta {mg-pg:.4f}", flush=True)
    del rd; torch.cuda.empty_cache()
os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False); print("MODLENS_METRIC_DONE", flush=True)
