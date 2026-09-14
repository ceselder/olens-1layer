"""GRPO/RLOO-style RL of an AR-inverter reader against the frozen LoRA AR (the reconstruction reward, contrastive):
    reward(sample) = cos(AR(sample), AR(span_true)) - cos(AR(sample), AR(other row))
Policy = reader from an SFT checkpoint (small / fullattn / linrnn), reference = frozen copy (token-level KL penalty).
Inputs stream from the AR cache (v_true = cached AR(span)); the reward model is the LoRA AR on the 27B (blocks 0..42 + value head).

    modal run --detach modal_app.py --task train --gpus 1 --script rl_inverter.py --args "--ckpt /vol/ckpt/small_d1024L1_30m/final.pt --out /vol/ckpt/rl_small_d1024L1 ..."
"""
import argparse, copy, glob, json, math, os, sys, time, traceback
def _hook(t, v, tb): print(f"[rank {os.environ.get('RANK','0')}] UNCAUGHT: " + ''.join(traceback.format_exception(t, v, tb)), flush=True)
sys.excepthook = _hook
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F
from common import D_MODEL, load_base, load_tokenizer, Layer42Hook, ar_read
from av1_model import load_small
from common import Whitener

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--frozen", default="/vol/frozen/qwen36_27b_embed_head.pt"); ap.add_argument("--ar", default="/vol_data/ckpt/ar_mse_v3/final")
ap.add_argument("--reward", default="lora", choices=["lora", "affine", "modlens"], help="reward model: the LoRA AR or the closed-form affine AR (--affine)")
ap.add_argument("--affine", default="/vol/ckpt/ar_affine_lastmean/affine.pt")
ap.add_argument("--modlens-ar", default="/vol_modlens/ar_l42_text2vec"); ap.add_argument("--jlens", default="/vol/jlens/qwen36_27b_jlens.pt"); ap.add_argument("--amu", default="/vol_modlens/data/natural_whitener_jspace.npz")
ap.add_argument("--data", default="/vol/data/harvest_v5/*.parquet"); ap.add_argument("--exclude", default="shard00_part0000"); ap.add_argument("--ar-cache", default="/vol/data/ar_v3")
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--group", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--kl", type=float, default=0.02); ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--policy", default="small", choices=["small", "lora27b"], help="lora27b = the 27B + LoRA Karvonen-injection reader (--ckpt = adapter dir) as the policy")
ap.add_argument("--recipe", default="scalerl", choices=["scalerl", "grpo"], help="scalerl = MAEMM's ScaleRL bundle: CISPO loss (eps_max 5), batch-level advantage normalisation with zero-variance groups dropped, prompt-level loss aggregation, fp32 lm_head; grpo = the earlier plain group-relative REINFORCE")
ap.add_argument("--input", default="h42", choices=["h42", "arcache"], help="policy input / reward target: h42 = the REAL layer-42 activation (oracle-lens RL, default); arcache = the cached AR(span) vector (inverter RL)")
ap.add_argument("--adapter", default=None, help="adapter.pt (P, b): feed the policy P(h42)+b (AR-space projection of the real activation) while the reward target stays the real h42")
ap.add_argument("--whitener", default="/vol_data/data/whitener_v2.pt", help="layer-42 whitener (mu, W=Sigma^-1/2) fitted on harvested activations")
ap.add_argument("--reward-fn", default="wfve", choices=["wfve", "fve", "cos", "cos_contrastive"], help="wfve (default, paper A.9.2) = fraction of WHITENED variance explained after a non-negative LS refit = max(0, cos_whitened)^2; fve = raw mean-centred FVE; cos = centred cosine; cos_contrastive = the earlier cos(v_true) - cos(v_other)")
ap.add_argument("--cispo-eps-max", type=float, default=5.0)
ap.add_argument("--no-centred", action="store_true", help="use raw cosine instead of cosine after subtracting the batch mean AR vector (default: centred)")
ap.add_argument("--sft-mix", type=float, default=0.0, help="weight of the teacher-forced CE on the true span added to the RL loss")
ap.add_argument("--n-tok", type=int, default=12); ap.add_argument("--eval-every", type=int, default=100); ap.add_argument("--heldout", type=int, default=512)
ap.add_argument("--wandb-name", default=None); ap.add_argument("--no-wandb", action="store_true")
args = ap.parse_args()
import torch.distributed as dist
RANK, WORLD = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)); is_dist = WORLD > 1; is_main = RANK == 0
LRANK = int(os.environ.get("LOCAL_RANK", 0))
if is_dist:
    dist.init_process_group("nccl", rank=RANK, world_size=WORLD)
dev = f"cuda:{LRANK}"; torch.cuda.set_device(dev); tok = load_tokenizer(); os.makedirs(args.out, exist_ok=True)   # explicit device: accelerate maps "cuda" to cuda:0
print(f"[phase] rank {RANK}/{WORLD} init ok, device {torch.cuda.current_device()}", flush=True)
def P(*a):
    if is_main: print(*a, flush=True)

# ---- policy + reference (small readers: frozen embed/norm/head from the frozen file; the 27B is loaded only for the AR) ----
from common import InjectL1, build_av_prompt
if args.policy == "small":
    sd0 = torch.load(args.ckpt, map_location="cpu")
    policy = load_small(args.ckpt, args.frozen, dev).train()
    ref = copy.deepcopy(policy).eval()
    for p_ in ref.parameters(): p_.requires_grad_(False)
    trainable = [p_ for p_ in policy.parameters() if p_.requires_grad]
    for p_ in trainable: p_.data = p_.data.float()
    base = load_base(dev)
else:                                               # 27B + LoRA Karvonen reader as the policy: adapter "default" trainable, on the SAME base as the reward AR
    from peft import PeftModel
    sd0 = {"block_type": "lora27b", "adapter": args.ckpt}
    base = PeftModel.from_pretrained(load_base(dev), args.ckpt, adapter_name="default", is_trainable=True); policy = base; ref = None
    trainable = [p_ for n_, p_ in policy.named_parameters() if p_.requires_grad and ".default." in n_]
    for p_ in trainable: p_.data = p_.data.float()
    INJ = InjectL1(policy); PROMPT_T = torch.tensor([build_av_prompt(tok)], device=dev); PLEN = PROMPT_T.shape[1]; PAD = tok.eos_token_id
    P(f"[rl] lora27b policy: {sum(p_.numel() for p_ in trainable)/1e6:.0f}M LoRA params trainable | prompt {PLEN} tokens")
print(f"[phase] rank {RANK} policy loaded", flush=True)
P(f"[rl] policy {sd0.get('block_type')} trainable {sum(p_.numel() for p_ in trainable)/1e6:.1f}M | world {WORLD} | per-rank {args.batch} prompts x {args.group} samples")
# ---- reward model: LoRA AR ----
def _attach(adir):
    from peft import PeftModel
    if args.policy == "lora27b": base.load_adapter(adir, adapter_name="ar"); base.set_adapter("default"); return base
    return PeftModel.from_pretrained(base, adir, adapter_name="ar").eval()
if args.reward == "lora":
    AR = _attach(args.ar); hook = Layer42Hook(AR)
    vh = nn.Linear(D_MODEL, D_MODEL, bias=True).to(dev, torch.float32); vh.load_state_dict(torch.load(f"{args.ar}/value_head.pt", map_location=dev))
elif args.reward == "modlens":                    # the modulation-lens AR: LoRA on blocks 0..42, MEAN-pool over the phrase tokens, Linear head -> J-space vector
    AR = _attach(args.modlens_ar); hook = Layer42Hook(AR)
    _hd = torch.load(os.path.join(args.modlens_ar, "head.pt"), map_location=dev); ML_HEAD = nn.Linear(D_MODEL, D_MODEL, bias=True).to(dev, torch.float32); ML_HEAD.load_state_dict(_hd["head"]); ML_HEAD.eval()
    ML_J = torch.load(args.jlens, map_location="cpu", weights_only=False)["J"][42].to(dev).float()
    ML_AMU = torch.tensor(np.load(args.amu)["mu"], device=dev).float(); vh = None
    P(f"[rl] modulation-lens reward: AR {args.modlens_ar}, J[42] {tuple(ML_J.shape)}, ||amu|| {ML_AMU.norm():.2f}")
else:
    AR = base.eval(); hook = Layer42Hook(AR); vh = None
    _a = torch.load(args.affine, map_location=dev); AFF_W, AFF_b, AFF_feat = _a["W"].to(dev).float(), _a["b"].to(dev).float(), _a["feat"]
@torch.no_grad()
def ar_vec(ids):                                   # ids [N, T] long -> [N, 5120] reward-model vectors
    out = []
    if args.policy == "lora27b": base.set_adapter("ar")
    for a in range(0, ids.shape[0], 1024):                     # bigger chunks -> better GPU utilisation of the 27B reward forward
        t = ids[a:a + 1024]; H_ = ar_read(AR, hook, t, torch.ones_like(t), vh).float()
        if args.reward == "lora": out.append(vh(H_[:, -1]))
        elif args.reward == "modlens": out.append(ML_HEAD(H_.mean(1)))                      # mean-pool (all 12 tokens real, no padding)
        else: out.append((torch.cat([H_[:, -1], H_.mean(1)], -1) if AFF_feat == "last+mean" else H_[:, -1]) @ AFF_W + AFF_b)
    if args.policy == "lora27b": base.set_adapter("default")
    return torch.cat(out)

# ---- data: cached AR(span) + spans ----
files = sorted(sum((glob.glob(g.strip()) for g in args.data.split(",")), [])); files = [f for f in files if args.exclude not in f]
def cpath(f): return args.ar_cache + "/" + f.replace("/vol_data/data/", "").replace("/vol/data/", "").replace("/", "__")
files = [f for f in files if args.input == "h42" or os.path.exists(cpath(f))]; rng = np.random.default_rng(0); order = rng.permutation(len(files))
held_fi = order[0]; order = np.array([fi for fi in order[1:]])[RANK::WORLD]      # rank-disjoint training files; held-out = file 0 (rank 0 evaluates)
def read(f):
    if args.input == "h42":                          # REAL activation: input to the policy and target of the reward
        tb = pq.ParquetFile(f).read(columns=["roll_ids", "h42"]); n = tb.num_rows
        return (tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float32),
                tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, 12).astype(np.int64)[:, : args.n_tok])
    tb = pq.ParquetFile(cpath(f)).read(columns=["roll_ids", "ar_vec"]); n = tb.num_rows
    return (tb.column("ar_vec").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float32),
            tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, 12).astype(np.int64)[:, : args.n_tok])
Vh, Rh = read(files[held_fi]); Vh, Rh = torch.tensor(Vh[: args.heldout], device=dev), torch.tensor(Rh[: args.heldout], device=dev)
def batches():
    for fi in order:
        V, R = read(files[fi]); p = np.random.permutation(len(V))
        for a in range(0, len(p) - args.batch + 1, args.batch):
            idx = p[a:a + args.batch]; yield torch.tensor(V[idx], device=dev), torch.tensor(R[idx], device=dev)
bit = batches()
print(f"[phase] rank {RANK} data ready: {len(order)} files, held {Vh.shape[0]}", flush=True)

print(f"[phase] rank {RANK} reward model loaded", flush=True)
WHT = Whitener.load(args.whitener, dev)
ADP = None
if args.adapter:
    _ad = torch.load(args.adapter, map_location=dev); ADP = (_ad['P'].to(dev).float(), _ad['b'].to(dev).float()); P('[rl] policy input = P(h42) via', args.adapter)
def pol_in(x): return x @ ADP[0] + ADP[1] if ADP is not None else x                 # whiten(x) = (x - mu) @ W^T ; all rewards live in this space (paper A.9.2)
HEAD_W32 = policy.head.weight.detach().float() if args.policy == "small" else None   # ScaleRL "fp32 at the LM head" (small readers)
def gen27b(v, n, temperature):
    outs = []
    for a in range(0, v.shape[0], 128):
        vb = v[a:a + 128]; ids = PROMPT_T.repeat(vb.shape[0], 1); INJ.set(vb, ids)
        g = policy.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, min_new_tokens=n, do_sample=temperature > 0, temperature=temperature if temperature > 0 else None, top_p=1.0 if temperature > 0 else None, top_k=0 if temperature > 0 else None, pad_token_id=PAD)
        INJ.off(); outs.append(g[:, PLEN:PLEN + n])
    return torch.cat(outs)
def seq_logp27b(v, toks):                          # log pi(span | prompt, injected v) [N, T]; fp32 log-softmax on the span positions
    ids = torch.cat([PROMPT_T.repeat(v.shape[0], 1), toks], 1); INJ.set(v, ids)
    out = policy(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False); INJ.off()
    lg = out.logits[:, PLEN - 1:PLEN - 1 + toks.shape[1]].float()
    return torch.log_softmax(lg, -1).gather(-1, toks[..., None])[..., 0]
def seq_logp(model, v, toks):
    if args.policy == "lora27b": return seq_logp27b(v, toks)
    """log pi(tok_i | ...) [N, T] with the head projection AND log-softmax in fp32 (--recipe scalerl); bf16 head for grpo."""
    if args.recipe != "scalerl":
        lg = model(v, toks)[:, :-1].float(); return torch.log_softmax(lg, -1).gather(-1, toks[..., None])[..., 0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.norm(model.hidden(v, toks))[:, :-1]                       # [N, T, 5120] pre-head hidden states
    out = []
    for a in range(0, h.shape[0], 256):                                    # chunked: [256, T, 248k] fp32 logits at a time
        lg = F.linear(h[a:a + 256].float(), HEAD_W32)
        out.append(torch.log_softmax(lg, -1).gather(-1, toks[a:a + 256, ..., None])[..., 0])
    return torch.cat(out)

@torch.no_grad()
def evaluate(step):
    policy.eval(); g = (gen27b(pol_in(Vh), args.n_tok, 0.0) if args.policy == "lora27b" else policy.generate(pol_in(Vh), args.n_tok)); vg = ar_vec(g); mu = 0 if args.no_centred else Vh.mean(0, keepdim=True)
    ct = F.cosine_similarity(vg - mu, Vh - mu).mean().item(); co = F.cosine_similarity(vg - mu, Vh.roll(1, 0) - mu).mean().item()
    fve_g = (1 - ((vg - Vh) ** 2).sum(-1) / ((Vh - mu) ** 2).sum(-1).clamp_min(1e-6)).mean().item()
    wfve_g = (F.cosine_similarity(WHT(vg), WHT(Vh)).clamp_min(0) ** 2).mean().item(); wfve_ceil = (F.cosine_similarity(WHT(ar_vec(Rh)), WHT(Vh)).clamp_min(0) ** 2).mean().item()
    if args.reward == "modlens":
        tJ = F.normalize(Vh @ ML_J.T - ML_AMU, dim=-1); aJ = F.normalize(vg, dim=-1); m_ = (aJ * tJ).sum(-1).clamp_min(0).mean().item(); p_ = (aJ * tJ.roll(1, 0)).sum(-1).clamp_min(0).mean().item()
        print(f"  [eval {step}] MODLENS metric (greedy, K=1): matched {m_:.4f} | permuted {p_:.4f} | delta {m_-p_:.4f}", flush=True)
    ce = -seq_logp(policy, pol_in(Vh), Rh).mean().item(); vt = ar_vec(Rh); ceil = F.cosine_similarity(vt - mu, Vh - mu).mean().item()
    policy.train()
    print(f"  [eval {step}] greedy: WHITENED FVE {wfve_g:.4f} (true-span ceiling {wfve_ceil:.4f}) | raw FVE {fve_g:.4f} | centred cos(AR(gen), v_true) {ct:.4f} | cos(AR(gen), v_other) {co:.4f} | gap {ct-co:.4f} | CE(true span) {ce:.3f} | true-span cos ceiling {ceil:.4f}", flush=True)
    for i in range(2): print(f"    TRUE {tok.decode(Rh[i])!r}\n    GEN  {tok.decode(g[i])!r}", flush=True)
    return {"wfve": wfve_g, "wfve_ceiling": wfve_ceil, "fve": fve_g, "cos_true": ct, "cos_other": co, "gap": ct - co, "ce_true": ce, "ceiling": ceil}

optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)   # MAEMM adam betas
wb = None
if is_main and not args.no_wandb and os.environ.get("WANDB_API_KEY"):
    import wandb; wb = wandb.init(project="olens-1layer", name=args.wandb_name or "rl_" + os.path.basename(args.out), config=vars(args))
ev0 = evaluate(0) if is_main else None
if is_main: json.dump(ev0, open(f"{args.out}/eval_0.json", "w"))
t0 = time.time(); best = -1e9
print(f"[phase] rank {RANK} entering loop", flush=True)
for step in range(1, args.steps + 1):
    v, r_true = next(bit); B, G = v.shape[0], args.group
    vG = v.repeat_interleave(G, 0)
    with torch.no_grad():
        policy.eval(); samp = (gen27b(pol_in(vG), args.n_tok, args.temp) if args.policy == "lora27b" else policy.generate(pol_in(vG), args.n_tok, temperature=args.temp)); policy.train()
        va = ar_vec(samp); v_other = v.roll(1, 0).repeat_interleave(G, 0)
        mu = 0 if args.no_centred else v.mean(0, keepdim=True)          # mean-centred: remove the shared mean direction of AR vectors
        cos_t = F.cosine_similarity(va - mu, vG - mu); cos_o = F.cosine_similarity(va - mu, v_other - mu)
        fve = 1 - ((va - vG) ** 2).sum(-1) / ((vG - mu) ** 2).sum(-1).clamp_min(1e-6)   # per-sample mean-centred FVE (raw space)
        wa, wt = WHT(va), WHT(vG); wfve = F.cosine_similarity(wa, wt).clamp_min(0) ** 2     # whitened FVE after NNLS refit (K=1)
        if args.reward == "modlens":
            tJ = F.normalize(vG @ ML_J.T - ML_AMU, dim=-1); tJo = F.normalize(v_other @ ML_J.T - ML_AMU, dim=-1); aJ = F.normalize(va, dim=-1)
            ml_fit = (aJ * tJ).sum(-1).clamp_min(0); ml_neg = (aJ * tJo).sum(-1).clamp_min(0); ml_rew = ml_fit - ml_neg   # fit(matched) - fit(other), K=1 NNLS = clipped cos
        wfve_raw = 1 - ((wa - wt) ** 2).sum(-1) / (wt ** 2).sum(-1).clamp_min(1e-6)         # whitened FVE without refit (reference)
        rew = ml_rew if args.reward == "modlens" else wfve if args.reward_fn == "wfve" else fve if args.reward_fn == "fve" else cos_t if args.reward_fn == "cos" else cos_t - cos_o
        rg = rew.view(B, G); adv = rg - rg.mean(1, keepdim=True)
        if args.recipe == "scalerl":                                     # MAEMM compute_advantages mode 'batch' + zero-variance filter
            nz = (rg.std(1) > 1e-6); keep = nz.repeat_interleave(G); advf = adv.view(-1) * keep
            stats = torch.tensor([advf[keep].double().pow(2).sum().item(), advf[keep].double().sum().item(), float(keep.sum())], dtype=torch.float64, device=dev)
            if is_dist: dist.all_reduce(stats)                          # ONE std over all surviving advantages of the GLOBAL batch
            n_all = stats[2].item(); std = math.sqrt(max(stats[0].item() / n_all - (stats[1].item() / n_all) ** 2, 0.0)) if n_all > 1 else 1.0
            adv = (advf / (std + 1e-6)).view(-1)
        else:
            keep = torch.ones(B * G, dtype=torch.bool, device=dev); adv = adv.view(-1)
        old_lp = seq_logp(policy, pol_in(vG), samp)                              # behaviour-policy logprobs (pre-update = the sampler's)
        lp_ref = seq_logp(ref, pol_in(vG), samp) if (args.kl > 0 and ref is not None) else old_lp
        T = samp.shape[1]
        if args.recipe == "scalerl":                                      # ScaleRL prompt-level weights: group 1/n_eff_groups, token 1/tokens-in-group
            gm = keep.float()[:, None].expand(B * G, T); m3 = gm.view(B, G, T); tok_g = m3.sum((1, 2)); n_eff_g = max(int((tok_g > 0).sum()), 1)
            w = (m3 / tok_g.clamp(min=1)[:, None, None] / n_eff_g).view(B * G, T)
        else:
            w = torch.full((B * G, T), 1.0 / (B * G), device=dev); n_eff_g = B
    # ---- chunked backward: the fp32 [chunk, T, 248k] log-softmax graph is built and freed per chunk; grads accumulate
    optim.zero_grad(set_to_none=True); pg_tot = 0.0; kl_tot = 0.0; CH = 256 if args.policy == "small" else 32
    for a in range(0, B * G, CH):
        lp = seq_logp(policy, pol_in(vG[a:a + CH]), samp[a:a + CH])                                  # [c, T] with grad
        if args.recipe == "scalerl":
            rho = torch.exp(lp.detach() - old_lp[a:a + CH]); is_w = rho.clamp(max=args.cispo_eps_max)   # CISPO truncated IS weight
            pg_c = (-(is_w * adv[a:a + CH, None] * lp) * w[a:a + CH]).sum()
        else:
            pg_c = -(adv[a:a + CH] * lp.sum(-1)).sum() / (B * G)
        kl_c = ((lp - lp_ref[a:a + CH]).sum(-1) * w[a:a + CH].sum(-1)).sum() if args.kl > 0 else torch.zeros((), device=dev)
        (pg_c + args.kl * kl_c).backward(); pg_tot += pg_c.item(); kl_tot += kl_c.item()
    if args.sft_mix > 0:
        (args.sft_mix * (-seq_logp(policy, pol_in(v), r_true).mean())).backward()
    kl = torch.tensor(kl_tot)
    if is_dist:                                                            # n_eff-weighted average of the rank gradients (MAEMM _sync_grads)
        wsum = torch.tensor([float(n_eff_g)], device=dev); dist.all_reduce(wsum)
        for p_ in trainable:
            if p_.grad is None: p_.grad = torch.zeros_like(p_)
            p_.grad.mul_(float(n_eff_g)); dist.all_reduce(p_.grad); p_.grad.div_(wsum.item())
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0); optim.step()
    if is_main:
        print(f"step {step:05d} | reward({args.reward_fn}) {rew.mean().item():.4f} | wfve {wfve.mean().item():.4f} (no-refit {wfve_raw.mean().item():.4f}) fve {fve.mean().item():.4f} cos_true {cos_t.mean().item():.4f} cos_other {cos_o.mean().item():.4f} | groups kept {int(keep.view(B, G)[:, 0].sum())}/{B} | kl {kl.item():.4f} | gn {float(gn):.2f} | {(time.time()-t0)/60:.1f}min", flush=True)
        if wb: wb.log({"step": step, "reward": rew.mean().item(), "wfve": wfve.mean().item(), "fve": fve.mean().item(), "cos_true": cos_t.mean().item(), "cos_other": cos_o.mean().item(), "groups_kept_frac": keep.view(B, G)[:, 0].float().mean().item(), "kl": kl.item(), "grad_norm": float(gn)})
    if step % args.eval_every == 0 and is_main:
        ev = evaluate(step); ev["step"] = step; json.dump(ev, open(f"{args.out}/eval_{step}.json", "w"))
        if wb: wb.log({"step": step, **{"eval/" + k: v_ for k, v_ in ev.items() if k != "step"}})
        if args.policy == "lora27b":
            policy.save_pretrained(f"{args.out}/step_{step:06d}", selected_adapters=["default"]); sd = None
        else:
            sd = dict(sd0); sd["trainable_sd"] = {n_: p_.detach().cpu() for n_, p_ in policy.named_parameters() if p_.requires_grad}; sd["rl"] = vars(args)
            torch.save(sd, f"{args.out}/step_{step:06d}.pt")
            if ev["wfve"] > best: best = ev["wfve"]; torch.save(sd, f"{args.out}/best.pt")
if is_main:
    if args.policy == "lora27b": policy.save_pretrained(f"{args.out}/final", selected_adapters=["default"])
    else: torch.save(sd, f"{args.out}/final.pt")
P("RL_DONE")
