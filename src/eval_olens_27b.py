"""Paper-style eval for the FULL-LM reader (27B + LoRA + Karvonen layer-1 injection) on REAL activations, mirroring eval_olens_rl.py:
input v = real h42 (or P(h42) via --adapter) injected at the marker; sample K spans; map each through the AR (LoRA 'ar' adapter or affine);
NNLS in whitened space -> wFVE; plus greedy / mean-sampled / best-of-K, and the 27B's own NLL of the generated spans (naturalness).
Reader and reward AR are two adapters on ONE 27B (set_adapter switches).

    ... --script eval_olens_27b.py --args "--reader /vol/ckpt/av27b_karvonen_12m/final --reward lora --k 8 --n 256"
"""
import argparse, json, os
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import nnls
from peft import PeftModel
from common import D_MODEL, InjectL1, Layer42Hook, MARKER_ID, ar_read, build_av_prompt, load_base, load_tokenizer, Whitener

ap = argparse.ArgumentParser()
ap.add_argument("--reader", required=True, help="LoRA adapter dir of the 27B Karvonen-inject reader")
ap.add_argument("--reward", default="lora", choices=["lora", "affine"]); ap.add_argument("--ar", default="/vol_data/ckpt/ar_mse_v3/final"); ap.add_argument("--affine", default="/vol/ckpt/ar_affine_lastmean/affine.pt")
ap.add_argument("--adapter", default=None); ap.add_argument("--whitener", default="/vol_data/data/whitener_v2.pt")
ap.add_argument("--eval-file", default="/vol/data/harvest_v5/shard00_part0000.parquet"); ap.add_argument("--n", type=int, default=256); ap.add_argument("--k", type=int, default=8); ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--out", default="/vol/results/olens_eval_27b.json"); args = ap.parse_args(); dev = "cuda"; tok = load_tokenizer(); WHT = Whitener.load(args.whitener, dev)
tb = pq.ParquetFile(args.eval_file).read(columns=["h42", "roll_ids"]); m = min(tb.num_rows, args.n)
H = torch.tensor(tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, D_MODEL).astype(np.float32)[:m], device=dev)
R = torch.tensor(tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, 12).astype(np.int64)[:m], device=dev)
base = load_base(dev); model = PeftModel.from_pretrained(base, args.reader, adapter_name="default").eval(); inj = InjectL1(model)
if args.reward == "lora":
    model.load_adapter(args.ar, adapter_name="ar"); hook = Layer42Hook(model)
    vh = nn.Linear(D_MODEL, D_MODEL, bias=True).to(dev, torch.float32); vh.load_state_dict(torch.load(f"{args.ar}/value_head.pt", map_location=dev))
else:
    hook = Layer42Hook(model); vh = None; _a = torch.load(args.affine, map_location=dev); W_, b_, feat_ = _a["W"].to(dev).float(), _a["b"].to(dev).float(), _a["feat"]
model.set_adapter("default")
PROMPT = build_av_prompt(tok); PLEN = len(PROMPT); PROMPT_T = torch.tensor(PROMPT, dtype=torch.long, device=dev); pad_id = tok.eos_token_id
ADP = None
if args.adapter: _ad = torch.load(args.adapter, map_location=dev); ADP = (_ad["P"].to(dev).float(), _ad["b"].to(dev).float())
def pol_in(x): return x @ ADP[0] + ADP[1] if ADP is not None else x
@torch.no_grad()
def ar_vec(ids):
    out = []
    if args.reward == "lora": model.set_adapter("ar")
    else: model.disable_adapter_layers()
    for a in range(0, ids.shape[0], 256):
        t = ids[a:a + 256]; Hs = ar_read(model, hook, t, torch.ones_like(t), vh).float()
        out.append(vh(Hs[:, -1]) if vh is not None else (torch.cat([Hs[:, -1], Hs.mean(1)], -1) if feat_ == "last+mean" else Hs[:, -1]) @ W_ + b_)
    if args.reward == "lora": model.set_adapter("default")
    else: model.enable_adapter_layers()
    return torch.cat(out)
@torch.no_grad()
def lm_nll(ids):
    model.disable_adapter_layers(); out = []
    for a in range(0, ids.shape[0], 64):
        t = ids[a:a + 64]; lg = model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False).logits[:, :-1].float()
        out.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t[:, 1:].reshape(-1), reduction="none").view(t.shape[0], -1).mean(1))
    model.enable_adapter_layers(); return torch.cat(out)
@torch.no_grad()
def generate(v, do_sample):
    outs = []
    for a in range(0, v.shape[0], 64):
        vb = v[a:a + 64]; ids = PROMPT_T[None].repeat(vb.shape[0], 1); inj.set(vb, ids)
        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=12, min_new_tokens=12, do_sample=do_sample, temperature=args.temp if do_sample else None, top_p=1.0 if do_sample else None, top_k=0 if do_sample else None, pad_token_id=pad_id)
        inj.off(); outs.append(g[:, PLEN:PLEN + 12])
    return torch.cat(outs)
WH = WHT(H); ceiling = (F.cosine_similarity(WHT(ar_vec(R)), WH).clamp_min(0) ** 2).mean().item(); nll_true = lm_nll(R).mean().item()
def nnls_k(dirs, target):
    A = dirs.T.cpu().double().numpy(); y = target.cpu().double().numpy(); c, _ = nnls(A, y); return 1 - ((y - A @ c) ** 2).sum() / (y ** 2).sum()
g = generate(pol_in(H), False); wg = (F.cosine_similarity(WHT(ar_vec(g)), WH).clamp_min(0) ** 2).mean().item(); nll_g = lm_nll(g).mean().item()
samp = generate(pol_in(H).repeat_interleave(args.k, 0), True); ws = WHT(ar_vec(samp)).view(m, args.k, -1)
single = F.cosine_similarity(ws, WH[:, None, :].expand_as(ws), dim=-1).clamp_min(0) ** 2
res = {"reader": args.reader, "input": "P(h42)" if ADP is not None else "real h42", "reward_model": args.reward, "n": m, "k": args.k, "true_span_ceiling_wfve": ceiling, "true_span_lm_nll": nll_true,
       "greedy_wfve": wg, "mean_sampled_wfve": single.mean().item(), "best_of_k_wfve": single.max(1).values.mean().item(), "nnls_k_wfve": float(np.mean([nnls_k(ws[i], WH[i]) for i in range(m)])),
       "lm_nll_greedy": nll_g, "lm_nll_sampled": lm_nll(samp).mean().item(), "decodes": [{"true": tok.decode(R[i]), "greedy": tok.decode(g[i]), "samples": [tok.decode(samp[i * args.k + j]) for j in range(3)]} for i in range(4)]}
print(f"[olens-27b] {os.path.basename(os.path.dirname(args.reader.rstrip('/')))}/{os.path.basename(args.reader.rstrip('/'))} input={res['input']}: greedy {wg:.4f} | mean sampled {res['mean_sampled_wfve']:.4f} | best-of-{args.k} {res['best_of_k_wfve']:.4f} | NNLS({args.k}) {res['nnls_k_wfve']:.4f} | ceiling {ceiling:.4f} | NLL greedy {nll_g:.3f} true {nll_true:.3f}", flush=True)
for d in res["decodes"][:2]: print(f"    TRUE {d['true']!r}\n    GREEDY {d['greedy']!r}\n    SAMPLES {d['samples']}", flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False); print("OLENS27B_DONE", flush=True)
