"""Paper-style inference eval for oracle-lens readers on REAL activations: sample K phrases per activation, map each through
the AR, fit non-negative least squares in mean-centred WHITENED space, report the whitened FVE of the K-phrase combination
(plus greedy single-phrase wFVE, mean single-sample wFVE, best-of-K wFVE, and the true-span ceiling). Several checkpoints per run.

    ... --script eval_olens_rl.py --args "--ckpts a.pt,b.pt --reward lora --k 8 --n 512"
"""
import argparse, glob, json, os
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import nnls
from common import D_MODEL, load_base, load_tokenizer, Layer42Hook, ar_read, Whitener
from av1_model import load_small

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", required=True); ap.add_argument("--reward", default="lora", choices=["lora", "affine"])
ap.add_argument("--ar", default="/vol_data/ckpt/ar_mse_v3/final"); ap.add_argument("--affine", default="/vol/ckpt/ar_affine_lastmean/affine.pt")
ap.add_argument("--frozen", default="/vol/frozen/qwen36_27b_embed_head.pt"); ap.add_argument("--whitener", default="/vol_data/data/whitener_v2.pt")
ap.add_argument("--eval-file", default="/vol/data/harvest_v5/shard00_part0000.parquet"); ap.add_argument("--n", type=int, default=512); ap.add_argument("--k", type=int, default=8); ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--adapter", default=None, help="adapter.pt: feed the reader P(h42)+b instead of h42")
ap.add_argument("--out", default="/vol/results/olens_rl_eval.json"); args = ap.parse_args(); dev = "cuda"; tok = load_tokenizer(); WHT = Whitener.load(args.whitener, dev)
tb = pq.ParquetFile(args.eval_file).read(columns=["h42", "roll_ids"]); m = min(tb.num_rows, args.n)
H = torch.tensor(tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, D_MODEL).astype(np.float32)[:m], device=dev)
R = torch.tensor(tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(-1, 12).astype(np.int64)[:m], device=dev)
base = load_base(dev)
if args.reward == "lora":
    from peft import PeftModel
    AR = PeftModel.from_pretrained(base, args.ar, adapter_name="ar").eval(); hook = Layer42Hook(AR)
    vh = nn.Linear(D_MODEL, D_MODEL, bias=True).to(dev, torch.float32); vh.load_state_dict(torch.load(f"{args.ar}/value_head.pt", map_location=dev))
else:
    AR = base.eval(); hook = Layer42Hook(AR); vh = None; _a = torch.load(args.affine, map_location=dev); W_, b_, feat_ = _a["W"].to(dev).float(), _a["b"].to(dev).float(), _a["feat"]
@torch.no_grad()
def ar_vec(ids):
    out = []
    for a in range(0, ids.shape[0], 256):
        t = ids[a:a + 256]; Hs = ar_read(AR, hook, t, torch.ones_like(t), vh).float()
        out.append(vh(Hs[:, -1]) if vh is not None else (torch.cat([Hs[:, -1], Hs.mean(1)], -1) if feat_ == "last+mean" else Hs[:, -1]) @ W_ + b_)
    return torch.cat(out)
ADP = None
if args.adapter:
    _ad = torch.load(args.adapter, map_location=dev); ADP = (_ad["P"].to(dev).float(), _ad["b"].to(dev).float())
def pol_in(x): return x @ ADP[0] + ADP[1] if ADP is not None else x
BASE_LM = base.get_base_model() if hasattr(base, "get_base_model") else base
@torch.no_grad()
def lm_nll(ids):                                   # naturalness: the frozen 27B's mean per-token NLL of the bare span (no context, no adapter)
    if args.reward == "lora": AR.disable_adapter_layers()
    out = []
    for a in range(0, ids.shape[0], 128):
        t = ids[a:a + 128]; lg = BASE_LM(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False).logits[:, :-1].float()
        out.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), t[:, 1:].reshape(-1), reduction="none").view(t.shape[0], -1).mean(1))
    if args.reward == "lora": AR.enable_adapter_layers()
    return torch.cat(out)
WH = WHT(H); wt_ = WHT(ar_vec(R)); ceiling = (F.cosine_similarity(wt_, WH).clamp_min(0) ** 2).mean().item(); nll_true = lm_nll(R).mean().item()
def wfve1(wv): return (F.cosine_similarity(wv, WH).clamp_min(0) ** 2)          # single-direction NNLS = clipped cos^2
def nnls_k(dirs, target):                                                        # dirs [K, d], target [d] (whitened) -> FVE of NNLS combination
    A = dirs.T.cpu().double().numpy(); y = target.cpu().double().numpy(); c, _ = nnls(A, y); res = y - A @ c
    return 1 - (res ** 2).sum() / (y ** 2).sum(), c
results = {"eval_file": args.eval_file, "n": m, "k": args.k, "reward_model": args.reward, "true_span_ceiling_wfve": ceiling, "ckpts": {}}
print(f"[olens-eval] {m} real activations | reward model {args.reward} | true-span ceiling wFVE {ceiling:.4f} | 27B NLL/token of the TRUE spans (no context) {nll_true:.3f}", flush=True)
results["true_span_lm_nll"] = nll_true
for ck in args.ckpts.split(","):
    reader = load_small(ck, args.frozen, dev)
    with torch.no_grad():
        g = reader.generate(pol_in(H), 12); w_g = wfve1(WHT(ar_vec(g))).mean().item(); nll_g = lm_nll(g).mean().item()
        HK = pol_in(H).repeat_interleave(args.k, 0); samp = reader.generate(HK, 12, temperature=args.temp); nll_s = lm_nll(samp).mean().item()
        ws = WHT(ar_vec(samp)).view(m, args.k, -1); single = F.cosine_similarity(ws, WH[:, None, :].expand_as(ws), dim=-1).clamp_min(0) ** 2
        mean_single, best_k = single.mean().item(), single.max(1).values.mean().item()
        fve_k = float(np.mean([nnls_k(ws[i], WH[i])[0] for i in range(m)]))
    name = ck.split("/ckpt/")[-1]; results["ckpts"][name] = {"greedy_wfve": w_g, "mean_sampled_wfve": mean_single, "best_of_k_wfve": best_k, "nnls_k_wfve": fve_k, "lm_nll_greedy": nll_g, "lm_nll_sampled": nll_s}
    print(f"[olens-eval] {name}: greedy {w_g:.4f} | mean sampled {mean_single:.4f} | best-of-{args.k} {best_k:.4f} | NNLS({args.k} phrases) {fve_k:.4f} | 27B NLL/token greedy {nll_g:.3f} sampled {nll_s:.3f}", flush=True)
    for i in range(2): print(f"    TRUE {tok.decode(R[i])!r}\n    SAMPLES {[tok.decode(samp[i*args.k + j]) for j in range(3)]}", flush=True)
    del reader; torch.cuda.empty_cache()
os.makedirs(os.path.dirname(args.out), exist_ok=True); json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False); print("OLENS_EVAL_DONE", flush=True)
