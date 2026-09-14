"""Precompute AR(span) = value_head(LoRA-AR block-42 state of the bare 12-token span) for every row of the given parquet
files, once, so inverter training never re-runs the 27B. Output mirrors each input file as a flat name under --out with
columns roll_ids [12] int32 and ar_vec [5120] fp16. Worker i of n handles files[i::n].

    modal run --detach modal_app.py --task train-many --gpus 1 --script precompute_ar.py --args "--worker 0 --nworkers 32 ...;;--worker 1 ..."
"""
import argparse, glob, os, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch, torch.nn as nn
from common import D_MODEL, load_base, Layer42Hook, ar_read

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--exclude", default="")
ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
ap.add_argument("--ar", default="/vol_data/ckpt/ar_mse_v3/final"); ap.add_argument("--out", default="/vol/data/ar_v3")
ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--n-tok", type=int, default=12)
ap.add_argument("--affine", default="/vol/ckpt/ar_affine_lastmean/affine.pt")
ap.add_argument("--mode", default="lora", choices=["lora", "frozen", "affine"], help="affine = the closed-form affine AR (W [last,mean] + b from --affine) on the frozen block-42 states; " + "lora = AR(span) via the LoRA AR + value head; frozen = the FROZEN 27B block-42 state at the span's last token (modulation-lens dictionary vector)")
args = ap.parse_args(); dev = "cuda"; os.makedirs(args.out, exist_ok=True)

def flat(f): return f.replace("/vol_data/data/", "").replace("/vol/data/", "").replace("/", "__")
files = sorted(sum((glob.glob(g.strip()) for g in args.data.split(",")), []))
files = [f for f in files if not any(x and x in f for x in args.exclude.split(","))][args.worker::args.nworkers]
todo = [f for f in files if not os.path.exists(f"{args.out}/{flat(f)}")]
print(f"[pre] worker {args.worker}/{args.nworkers}: {len(files)} files, {len(todo)} to do", flush=True)
if todo:
    base = load_base(dev)
    if args.mode == "lora":
        from peft import PeftModel
        AR = PeftModel.from_pretrained(base, args.ar, adapter_name="ar").eval(); hook = Layer42Hook(AR)
        vh = nn.Linear(D_MODEL, D_MODEL, bias=True).to(dev, torch.float32); vh.load_state_dict(torch.load(f"{args.ar}/value_head.pt", map_location=dev))
    else:
        AR = base.eval(); hook = Layer42Hook(AR); vh = None
        if args.mode == "affine":
            _aff = torch.load(args.affine, map_location=dev); AFF_W, AFF_b, AFF_feat = _aff["W"].to(dev).float(), _aff["b"].to(dev).float(), _aff["feat"]
    t0 = time.time(); n = 0
    for i, f in enumerate(todo):
        tb = pq.ParquetFile(f).read(columns=["roll_ids"]); m = tb.num_rows
        R = tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(m, 12).astype(np.int32)
        V = []
        with torch.no_grad():
            for a in range(0, m, args.batch):
                t = torch.tensor(R[a:a + args.batch, : args.n_tok].astype(np.int64), device=dev)
                H_ = ar_read(AR, hook, t, torch.ones_like(t), vh).float(); hh = H_[:, -1]
                if args.mode == "affine":
                    x = torch.cat([hh, H_.mean(1)], -1) if AFF_feat == "last+mean" else hh; out_v = x @ AFF_W + AFF_b
                else:
                    out_v = vh(hh) if vh is not None else hh
                V.append(out_v.half().cpu().numpy())
        V = np.concatenate(V)
        out = pa.table({"roll_ids": pa.FixedSizeListArray.from_arrays(pa.array(R.reshape(-1)), 12),
                        "ar_vec": pa.FixedSizeListArray.from_arrays(pa.array(V.reshape(-1)), D_MODEL)})
        tmp = f"{args.out}/{flat(f)}.tmp"; pq.write_table(out, tmp); os.replace(tmp, f"{args.out}/{flat(f)}"); n += m
        if i % 20 == 0: print(f"[pre] {i+1}/{len(todo)} files, {n} rows, {(time.time()-t0)/60:.1f} min", flush=True)
print("PRECOMPUTE_DONE", flush=True)
