"""Harvest (context, L42 activation, on-policy 12-token rollout, clean top-k teacher log-probs) rows
from the FineFineWeb token corpus on the maemm-data volume (50,287 x 512 Qwen3.6 token ids).

Per batch: pick ONE context length L ~ U[ctx_min, ctx_max] (all rows in the batch share L -> no
padding, no left-pad ambiguity for the GDN layers), draw B random (seq, t) with t >= L-1, context =
toks[seq, t-L+1 : t+1] (plain ids, no BOS). One HF generate call: prefill captures the block-42 output
at the last context token (= the activation the AR must reconstruct), then 12 sampled tokens at T=1
(top_p=1, top_k=0; EOS suppressed via min_new_tokens) with RAW logits at every step -> top-k log-probs
(full-vocab normalized) = the clean teacher distributions for the KL objective (gold patch == clean).

Output parquet columns:
  seq int32, t int32, ctx_len int32, ctx_ids list<int32>, h42 fixed[5120] float16,
  roll_ids fixed[N] int32, roll_lp fixed[N] float16 (log-prob of each sampled token),
  top_ids fixed[N*K] int32, top_lp fixed[N*K] float16

    modal run --detach modal_app.py --task train --gpus 1 --script harvest.py \
        --args "--shard 0 --nshards 4 --n-rows 75000 --out /vol/data/harvest_v1"
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from common import Layer42Hook, backbone, load_base, load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--n-rows", type=int, default=1000, help="rows for THIS shard")
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--ctx-min", type=int, default=32)
ap.add_argument("--ctx-max", type=int, default=256)
ap.add_argument("--n-fut", type=int, default=12)
ap.add_argument("--topk", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="/vol/data/harvest_smoke")
ap.add_argument("--corpus", default="/vol_maemm/acts27b")
ap.add_argument("--flush-every", type=int, default=4000, help="rows per parquet part file")
args = ap.parse_args()

dev = "cuda"
meta = json.load(open(f"{args.corpus}/meta.json"))
n_seq, T = meta["n_seq"], meta["seq_len"]
toks = np.memmap(f"{args.corpus}/toks.i32", dtype=np.int32, mode="r", shape=(n_seq, T))
rng = np.random.default_rng(args.seed * 1000 + args.shard)

tok = load_tokenizer()
model = load_base(dev)
bb = backbone(model)
hook = Layer42Hook(model)
eos = tok.eos_token_id
os.makedirs(args.out, exist_ok=True)
N, K = args.n_fut, args.topk

buf = {k: [] for k in ("seq", "t", "ctx_len", "ctx_ids", "h42", "roll_ids", "roll_lp", "top_ids", "top_lp")}
n_done, part, t_start = 0, 0, time.time()


def flush():
    global part, buf
    if not buf["seq"]:
        return
    n = len(buf["seq"])
    table = pa.table({
        "seq": pa.array(buf["seq"], pa.int32()),
        "t": pa.array(buf["t"], pa.int32()),
        "ctx_len": pa.array(buf["ctx_len"], pa.int32()),
        "ctx_ids": pa.array(buf["ctx_ids"], pa.list_(pa.int32())),
        "h42": pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate(buf["h42"]).astype(np.float16)), 5120),
        "roll_ids": pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate(buf["roll_ids"]).astype(np.int32)), N),
        "roll_lp": pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate(buf["roll_lp"]).astype(np.float16)), N),
        "top_ids": pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate(buf["top_ids"]).astype(np.int32)), N * K),
        "top_lp": pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate(buf["top_lp"]).astype(np.float16)), N * K),
    })
    path = f"{args.out}/shard{args.shard:02d}_part{part:04d}.parquet"
    pq.write_table(table, path + ".tmp", compression="zstd")
    os.replace(path + ".tmp", path)
    print(f"[flush] {path} rows={n} total={n_done} elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
    part += 1
    buf = {k: [] for k in buf}


while n_done < args.n_rows:
    B = min(args.batch, args.n_rows - n_done)
    L = int(rng.integers(args.ctx_min, args.ctx_max + 1))
    seqs = rng.integers(0, n_seq, size=B)
    ts = rng.integers(L - 1, T, size=B)                      # position of the read token
    ctx = np.stack([np.asarray(toks[s, t - L + 1:t + 1]) for s, t in zip(seqs, ts)]).astype(np.int64)
    ctx_t = torch.tensor(ctx, device=dev)
    with torch.no_grad():
        hook.capture()
        g = model.generate(ctx_t, attention_mask=torch.ones_like(ctx_t), do_sample=True, temperature=1.0,
                           top_p=1.0, top_k=0, max_new_tokens=N, min_new_tokens=N, output_logits=True,
                           return_dict_in_generate=True, pad_token_id=eos)
        hook.off()
        h42 = hook.captured[:, -1].float().cpu().numpy()                       # [B, d]  (prefill call)
        roll = g.sequences[:, L:L + N]                                           # [B, N]
        lp = torch.stack([F.log_softmax(l.float(), -1) for l in g.logits], 1)   # [B, N, V]
        tlp, tki = lp.topk(K, -1)                                               # [B, N, K]
        rlp = lp.gather(-1, roll[..., None])[..., 0]                            # [B, N]
    assert hook.captured.shape[1] == L, f"capture got seq_len {hook.captured.shape[1]} != {L}"
    for i in range(B):
        buf["seq"].append(int(seqs[i])); buf["t"].append(int(ts[i])); buf["ctx_len"].append(L)
        buf["ctx_ids"].append(ctx[i].astype(np.int32).tolist())
        buf["h42"].append(h42[i]); buf["roll_ids"].append(roll[i].cpu().numpy())
        buf["roll_lp"].append(rlp[i].cpu().numpy())
        buf["top_ids"].append(tki[i].reshape(-1).cpu().numpy()); buf["top_lp"].append(tlp[i].reshape(-1).cpu().numpy())
    n_done += B
    if n_done % (args.batch * 10) < B:
        el = time.time() - t_start
        print(f"[harvest] shard {args.shard} rows {n_done}/{args.n_rows} | {n_done / el:.1f} rows/s | "
              f"L={L} | topk mass {tlp.exp().sum(-1).mean():.3f} | sample: {tok.decode(roll[0])!r}", flush=True)
    if len(buf["seq"]) >= args.flush_every:
        flush()
flush()
json.dump({"shard": args.shard, "rows": n_done, "n_fut": N, "topk": K, "ctx_min": args.ctx_min,
           "ctx_max": args.ctx_max, "corpus": args.corpus, "seed": args.seed,
           "sampling": "temperature=1.0 top_p=1.0 top_k=0 min_new_tokens=N (EOS suppressed)",
           "convention": "plain ids, no BOS; h42 = block-42 output at last ctx token (hidden_states[43])",
           "elapsed_min": (time.time() - t_start) / 60},
          open(f"{args.out}/shard{args.shard:02d}_meta.json", "w"), indent=1)
print("HARVEST_DONE", flush=True)
