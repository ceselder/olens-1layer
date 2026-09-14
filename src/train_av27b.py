"""FULL-LM BASELINE: Qwen3.6-27B + rsLoRA reader with Karvonen-style layer-1 injection (norm-matched add of the AR vector at a
marker token inside a chat prompt), trained as an AR-inverter on EXACTLY the same cached (AR(span), span) rows as the shallow
readers. Streams the AR cache (no AR forward), data-parallel over ranks, single pass.

    torchrun --nproc_per_node 8 train_av27b.py --data '...' --ar-cache /vol/data/ar_v3 --out /vol/ckpt/av27b_karvonen_12m
"""
import argparse, glob, json, os, time
import numpy as np, pyarrow.parquet as pq, torch, torch.nn.functional as F, torch.distributed as dist
from peft import LoraConfig, get_peft_model
from common import D_MODEL, InjectL1, MARKER_ID, build_av_prompt, load_base, load_tokenizer, lora_target_re

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--exclude", default=""); p.add_argument("--ar-cache", default="/vol/data/ar_v3"); p.add_argument("--max-files", type=int, default=0)
p.add_argument("--out", required=True); p.add_argument("--steps", type=int, default=100000); p.add_argument("--batch", type=int, default=32); p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--lora-r", type=int, default=64); p.add_argument("--lora-alpha", type=int, default=16); p.add_argument("--n-tok", type=int, default=12)
p.add_argument("--heldout", type=int, default=2000); p.add_argument("--eval-every", type=int, default=500); p.add_argument("--save-every", type=int, default=5000); p.add_argument("--warmup", type=int, default=100)
p.add_argument("--buffer-files", type=int, default=8); p.add_argument("--wandb-project", default="olens-1layer"); p.add_argument("--wandb-name", default=None); p.add_argument("--no-wandb", action="store_true")
args = p.parse_args()
RANK, WORLD, LRANK = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0)); is_dist = WORLD > 1; is_main = RANK == 0
if is_dist: dist.init_process_group("nccl", rank=RANK, world_size=WORLD)
dev = f"cuda:{LRANK}"; torch.cuda.set_device(dev); torch.manual_seed(0)
if is_main: os.makedirs(args.out, exist_ok=True)
tok = load_tokenizer(); model = load_base(dev)
lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, use_rslora=True, lora_dropout=0.0, bias="none", target_modules=lora_target_re(None), task_type="CAUSAL_LM")
model = get_peft_model(model, lcfg); model.print_trainable_parameters(); inj = InjectL1(model)
PROMPT = build_av_prompt(tok); PLEN = len(PROMPT); assert MARKER_ID in PROMPT; PROMPT_T = torch.tensor(PROMPT, dtype=torch.long); pad_id = tok.eos_token_id
# ---- data: rank-disjoint cache files, streamed
def cpath(f): return args.ar_cache + "/" + f.replace("/vol_data/data/", "").replace("/vol/data/", "").replace("/", "__")
files = sorted(sum((glob.glob(g.strip()) for g in args.data.split(",")), [])); files = [f for f in files if not any(x and x in f for x in args.exclude.split(",")) and os.path.exists(cpath(f))]
my_files = files[RANK::WORLD]; my_files = my_files[: args.max_files] if args.max_files > 0 else my_files
def read(f):
    tb = pq.ParquetFile(cpath(f)).read(columns=["roll_ids", "ar_vec"]); n = tb.num_rows
    return (tb.column("ar_vec").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float16),
            tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, 12).astype(np.int32)[:, : args.n_tok])
held_file, stream_files = my_files[0], my_files[1:]
Vh, Rh = read(held_file); Vh, Rh = Vh[: args.heldout], Rh[: args.heldout]
counts = [pq.ParquetFile(cpath(f)).metadata.num_rows for f in stream_files]; n_local = int(sum(counts))
cap = torch.tensor([n_local // args.batch - len(stream_files) // args.buffer_files - 1], device=dev)
if is_dist: dist.all_reduce(cap, op=dist.ReduceOp.MIN)
args.steps = min(args.steps, int(cap.item()))
if is_main: print(f"[data] rank0 {n_local} rows in {len(stream_files)} cache files | steps {args.steps} | eff batch {args.batch * WORLD}", flush=True)
def stream(rng):
    from concurrent.futures import ThreadPoolExecutor
    order = rng.permutation(len(stream_files)); pool = ThreadPoolExecutor(8); chunks = [order[a:a + args.buffer_files] for a in range(0, len(order), args.buffer_files)]
    load = lambda ch: list(pool.map(lambda fi: read(stream_files[fi]), ch)); nxt = pool.submit(load, chunks[0]); cv, cr = [], []
    for ci in range(len(chunks)):
        got = nxt.result(); nxt = pool.submit(load, chunks[ci + 1]) if ci + 1 < len(chunks) else None
        V = np.concatenate(cv + [g[0] for g in got]); R = np.concatenate(cr + [g[1] for g in got]); pm = rng.permutation(len(V)); V, R = V[pm], R[pm]
        nf = (len(V) // args.batch) * args.batch
        for a in range(0, nf, args.batch): yield V[a:a + args.batch], R[a:a + args.batch]
        cv, cr = [V[nf:]], [R[nf:]]
def make(V, R):
    B = len(R); ids = torch.full((B, PLEN + args.n_tok), pad_id, dtype=torch.long); lab = torch.full_like(ids, -100)
    ids[:, :PLEN] = PROMPT_T; ids[:, PLEN:] = torch.tensor(R.astype(np.int64)); lab[:, PLEN:] = ids[:, PLEN:]
    return ids.to(dev), torch.ones_like(ids).to(dev), lab.to(dev), torch.tensor(V.astype(np.float32), device=dev)
def ce_loss(V, R):
    ids, attn, lab, vec = make(V, R); inj.set(vec, ids)
    out = model(input_ids=ids, attention_mask=attn, use_cache=False); inj.off()
    lg = out.logits[:, :-1].float(); return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
@torch.no_grad()
def evaluate():
    model.eval(); ces = [ce_loss(Vh[a:a + args.batch], Rh[a:a + args.batch]).item() for a in range(0, min(len(Rh), 512), args.batch)]
    ids = PROMPT_T[None].repeat(4, 1).to(dev); inj.set(torch.tensor(Vh[:4].astype(np.float32), device=dev), ids)
    g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.n_tok, do_sample=False, pad_token_id=pad_id); inj.off(); model.train()
    return float(np.mean(ces)), [(tok.decode(Rh[i]), tok.decode(g[i, PLEN:])) for i in range(4)]
trainable = [q for q in model.parameters() if q.requires_grad]
optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1.0, s / max(1, args.warmup)) * max(0.05, 1 - s / max(1, args.steps)))
wb = None
if is_main and not args.no_wandb and os.environ.get("WANDB_API_KEY"):
    import wandb; wb = wandb.init(project=args.wandb_project, name=args.wandb_name or os.path.basename(args.out), config={**vars(args), "eff_batch": args.batch * WORLD, "gpus": WORLD})
it = stream(np.random.default_rng(1 + RANK)); t0 = time.time(); model.train()
for step in range(args.steps):
    V, R = next(it); optim.zero_grad(set_to_none=True); loss = ce_loss(V, R); loss.backward()
    if is_dist:
        for q in trainable:
            if q.grad is None: q.grad = torch.zeros_like(q)
            dist.all_reduce(q.grad); q.grad.div_(WORLD)
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0); optim.step(); sched.step()
    if is_main:
        if step % 25 == 0: print(f"step {step:06d} | ce {loss.item():.4f} | gn {float(gn):.2f} | {(time.time()-t0)/60:.1f}min", flush=True)
        log = {"step": step, "ce": loss.item(), "lr": sched.get_last_lr()[0]}
        if step > 0 and step % args.eval_every == 0:
            ev, samples = evaluate(); log["eval_ce"] = ev; print(f"  [eval {step}] ce {ev:.4f}", flush=True)
            for tr, ge in samples[:2]: print(f"    TRUE {tr!r}\n    GEN  {ge!r}", flush=True)
        if wb: wb.log(log)
        if step > 0 and step % args.save_every == 0: model.save_pretrained(f"{args.out}/step_{step:06d}")
if is_main:
    ev, samples = evaluate(); model.save_pretrained(f"{args.out}/final"); json.dump({"args": vars(args), "final_eval_ce": ev, "prompt_len": PLEN, "marker": MARKER_ID}, open(f"{args.out}/meta.json", "w"), indent=1)
    print(f"[final] eval ce {ev:.4f}", flush=True); print("AV27B_DONE", flush=True)
